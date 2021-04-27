import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)      #指定随机种子


def zero_pad(X, pad):
    """
    把数据集X的图像边界用0来扩充pad个高度和宽度
    :return:
        X_paded - 扩充后的图像数据集，维度为(样本数,图像高度 + 2 * pad,图像宽度 + 2 * pad,图像通道数)
    """

    X_paded = np.pad(X,
                     ((0,0),
                      (pad,pad),
                      (pad,pad),
                      (0,0)),
                     'constant',constant_values=0)

    return X_paded

def conv_single_step(a_slice_prev, W, b):
    """
    在前一层激活输出的片段上应用一个W卷积核
    这里切片大小和卷积核大小相同

    :param a_slice_prev:输入数据的一个片段,维度(卷积核长,卷积核宽,上一层通道数)
    :param W:权重参数,维度同上
    :param b:偏执参数,维度为(1,1,1)
    :return:
        Z - 卷积后的结果
    """

    s = np.multiply(a_slice_prev,W) + b
    Z = np.sum(s)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积函数的前向传播
    :param A_prev:上一层的激活输出矩阵,维度为(m,n_H_prev,n_W_prev,n_C_prev)
    :param W:权重矩阵,维度为(卷积核长,卷积核宽,上一层输出的通道数,卷积核数量)
    :param b:偏置矩阵,维度为(1,1,1,卷积核数量)
    :param hparameters:包含超参数pad和stride的字典
    :return:
        Z - 卷积输出,维度为(m, n_H, n_W, n_C)
        cache - 缓存了一些反向传播所需要的数据
    """

    # 获取上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape
    # 获取超参数
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    #计算卷积后的图像宽高
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # 初始化卷积输出
    Z = np.zeros((m,n_H,n_W,n_C))

    # padding输入
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位当前的切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 开始切片
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, : ]
                    # 对切片进行卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])

    assert(Z.shape == (m, n_H, n_W, n_C))

    cache = (A_prev,W,b,hparameters)

    return (Z,cache)


def pool_forward(A_prev, hparameters, mode="max"):
    """
    实现池化层的前向传播
    :param hparameters:包含了f和stride的超参数字典
    :param mode:max|average
    :return:
        A - 池化层的输出,维度为(m,n_H,n_W,n_C)
        cache - 包含了输入和超参数字典的缓存
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f)/stride) + 1
    n_W = int((n_W_prev - f)/stride) + 1
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i,h,w,c] = np.average(a_slice_prev)
                    else:
                        print(f'mode:{mode}不合法,请输入合法的mode')

    assert(A.shape == (m,n_H,n_W,n_C))

    cache = (a_prev,hparameters)

    return (A,cache)


np.random.seed(1)

A_prev = np.random.randn(2,4,4,3)
hparameters = {"f":4 , "stride":1}

A , cache = pool_forward(A_prev,hparameters,mode="max")
A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print("----------------------------")
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)

