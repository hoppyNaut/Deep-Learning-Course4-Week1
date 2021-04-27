import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from cnn_utils import load_dataset


torch.manual_seed(1)

# 载入数据
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(f'X_train_orig.shape:{X_train_orig.shape} Y_train_orig.shape:{Y_train_orig.shape}')
print(f'X_test_orig.shape:{X_test_orig.shape} Y_test_orig.shape:{Y_test_orig.shape}')

# 将输入数据集的格式变为(m,n_C,n_H,n_W)归一化数据集
X_train = np.transpose(X_train_orig,(0,3,1,2))/255
X_test = np.transpose(X_test_orig,(0,3,1,2))/255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

def data_loader(X_train, Y_train, batch_size = 64):
    # TensorDataset可以对tensor数据进行打包,对每一个tensor的第一维度进行索引,所以要保持传入tensor第一维度相等
    train_db = TensorDataset(torch.from_numpy(X_train).float(),torch.squeeze(torch.from_numpy(Y_train)))
    # 对训练数据分为随机batch
    train_loader = DataLoader(dataset=train_db, batch_size=batch_size, shuffle=True)
    return train_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # conv1 input:(3, 64, 64) output:(8, 9, 9)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8,stride=8,padding=4)
        )
        # conv2 input:(8, 9, 9) output:(16, 3, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 2, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4,padding=2)
        )
        # fullconnect output:(20, 1)
        self.fullconnect = nn.Sequential(
            nn.Linear(16 * 3 * 3, 20),
            # nn.ReLU()
        )
        # nn.LogSoftmax(dim = 1):对每一行的元素进行softmax运算后取log
        self.classifier = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fullconnect(x)
        output = self.classifier(x)
        return output

# 自定义权重初始化方法
def weigh_init(m):
    if isinstance(m,nn.Conv2d):
        # 使用xavier初始化一个服从均匀分布权重矩阵
        nn.init.xavier_uniform_(m.weight.data)
        # 使用0填充bias
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, mini_batch_size = 64, print_cost = True,is_plot = True):
    train_loader = data_loader(X_train,Y_train,mini_batch_size)
    cnn = CNN()
    # cnn.apply(weigh_init)
    # 定义损失函数
    cost_func = nn.NLLLoss()
    # 批量更新参数
    optimizer = torch.optim.Adam(cnn.parameters(),lr = learning_rate,betas=(0.9,0.999))
    # 保存每次迭代的cost列表
    costs = []

    m = X_train.shape[0]
    num_batch = m / mini_batch_size

    for epoch in range(num_epochs):
        epoch_cost = 0
        for step, (batch_x,batch_y) in enumerate(train_loader):
            # print(f'batch_x.shape:{batch_x.shape} batch_y.shape:{batch_y.shape}')
            # 前向传播
            output = cnn(batch_x)
            # 计算成本
            cost  = cost_func(output,batch_y)
            epoch_cost += cost.data.numpy() / num_batch
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            cost.backward()
            # 更新参数
            optimizer.step()

        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)
            print('Cost after epoch %i : %f' % (epoch, epoch_cost))

    # 绘制学习曲线
    if is_plot:
        plt.plot(costs)
        plt.xlabel('iteration per 5')
        plt.ylabel('cost')
        plt.show()

    # 保存学习后的参数
    torch.save(cnn.state_dict(), 'net_params.pkl')
    print('参数已保存到本地pkl文件')

    # 预测训练集
    cnn.load_state_dict(torch.load('net_params.pkl'))
    output_train = cnn(torch.from_numpy(X_train).float())
    pred_Y_train = torch.max(output_train, dim = 1)[1].data.numpy()
    # 预测测试集
    output_test = cnn(torch.from_numpy(X_test).float())
    pred_Y_test = torch.max(output_test, dim=1)[1].data.numpy()
    # 训练集准确率
    print('Train Accuracy: %.2f %%' % float(np.sum(np.squeeze(Y_train) == pred_Y_train) / m * 100))
    # 测试集准确率
    print('Test Accuracy: %.2f %%' % float(np.sum(np.squeeze(Y_test) == pred_Y_test) / X_test.shape[0] * 100))
    return cnn



model(X_train, Y_train, X_test, Y_test)