import common_header

import numpy as np


# x ** x        x>0
# (-x) ** (-x)  x <0
def fx(x):
    y = np.empty_like(x)
    i = x > 0
    y[i] = np.power(x[i], x[i])
    i = x < 0
    y[i] = np.power(-x[i], -x[i])
    # i = (x==0)
    # y[i]=1
    return y


if __name__ == '__main__':
    '''
        开场：
        numpy是非常好用的数据包，如：可以这样得到这个二维数组
        [[ 0  1  2  3  4  5]
         [10 11 12 13 14 15]
         [20 21 22 23 24 25]
         [30 31 32 33 34 35]
         [40 41 42 43 44 45]
         [50 51 52 53 54 55]] 
    '''
    ar = np.arange(0, 60, 10)
    print(ar, "\n")

    rar = ar.reshape((-1, 1))
    print(rar, "\n")

    a = + rar + np.arange(6)
    print(a, "\n")

    '''
        正式开始：
        标准Python的列表(list)中，元素本质是对象。
        如：L = [1, 2, 3]，需要3个指针和三个整数对象，对于数值运算比较浪费内存和CPU。
        因此，Numpy提供了ndarray(N-dimensional array object)对象：存储单一数据类型的多维数组。
    '''

    print("\n---------------------------\n")
    print("\n---------------------------\n")

    # 1.使用array
    # 通过array函数传递list对象
    L = [1, 2, 3, 4, 5, 6]
    print("L = ", L)

    a = np.array(L)
    print(" a = ", a)
    # 1x6
    # ndarray的元素之间是没有,
    print(type(a))

    print("\n---------------------------\n")
    # 若传递的是多层嵌套的list，将创建多维数组
    # 3x4
    b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(" b = ", b)

    # 数组大小可以通过其shape属性获得
    print("a.shape = ", a.shape)
    print("b.shape = ", b.shape)

    # 可以强制修改shape
    b.shape = 4, 3
    print("b.shape = ", b.shape)
    # 注：从(3,4)改为(4,3)并不是对数组进行转置，而只是改变每个轴的大小，数组元素在内存中的位置并没有改变
    print(" b = ", b)

    # 当某个轴为-1时，将根据数组元素的个数自动计算此轴的长度
    b.shape = 2, -1
    print("b.shape = ", b.shape)
    print(" b = ", b)

    b.shape = 3, 4

    print("\n---------------------------\n")
    # 使用reshape方法，可以创建改变了尺寸的新数组，原数组的shape保持不变
    c = b.reshape((4, -1))
    print(" b = \n", b)
    print(" c = \n", c)

    # 数组b和c共享内存，修改任意一个将影响另外一个
    b[0][1] = 20
    print(" b = \n", b)
    print(" c = \n", c)

    print("\n---------------------------\n")
    # 数组的元素类型可以通过dtype元素属性获得
    print(' a.dtype = ', a.dtype)
    print(' b.dtype = ', b.dtype)

    # 可以通过dtype参数在创建时指定元素类型
    d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    f = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)
    print(d)
    print(f)

    # 如果更改元素类型，可以使用astype安全的转换
    f = d.astype(np.int)
    print(f)

    # 但不要强制仅修改元素类型，如下面这句，将会以int来解释单精度float类型
    d.dtype = np.int
    # print(d) 数据出现问题

    '''
    
    '''

    print("\n---------------------------\n")
    print("\n---------------------------\n")

    # 2.使用函数创建
    # 如果生成一定规则的数据，可以使用NumPy提供的专门函数
    # arange函数类似于python的range函数：指定起始值、终止值和步长来创建数组
    # 和Python的range类似，arange同样不包括终值；但arange可以生成浮点类型，而range只能是整数类型

    # 从1开始，到10结束，步长0.5，左闭右开，即1~9.5
    a1 = np.arange(1, 10, 0.5)
    print(' a = ', a1)

    # linspace函数通过指定起始值、终止值和元素个数来创建数组，缺省包括终止值
    bl = np.linspace(1, 10, 19)
    print(' bl = ', bl)

    # 可以通过endpoint关键字指定是否包括终值
    c = np.linspace(1, 10, 19, endpoint=False)
    print(' c = ', c)

    # 和linspace等差数列类似，logspace可以创建等比数列
    # 下面函数创建起始值为10^1，终止值为10^2，10个数的等比数列
    d = np.logspace(1, 2, 10, endpoint=True)  # 起始值终止值默认10的n次方
    print('d = ', d)

    # 下面创建起始值为2^0，终止值为2^10(包括)，有10个数的等比数列
    f = np.logspace(0, 10, 11, endpoint=True, base=2)  # 基数是2，起始值终止值为2的n次方
    print('f = ', f)

    # 使用 frombuffer, fromstring, fromfile等函数可以从字节序列创建数组
    s = 'abcd'
    g = np.fromstring(s, dtype=np.int8)
    print('g = ', g)

    '''

    '''

    print("\n---------------------------\n")
    print("\n---------------------------\n")
    # 3.存取
    # 3.1常规办法：数组元素的存取方法和Python的标准方法相同
    a = np.arange(10)
    print(a)

    # 获取某个元素
    print(a[3])
    # 切片[3,6]，左闭右开,取3，6之间的元素，返回的是一个数组,不包含最后一个元素
    print(a[3:6])
    # 省略开始下标，表示从0开始
    print(a[:5])
    # 下标为负表示从后向前数
    print(a[3:])
    # 从后往前数3个
    print(a[-3:])
    # 步长为2
    print(a[1:9:2])
    # 步长为-1，即翻转
    print(a[::-1])
    # 切片数据是原数组的一个视图，与原数组共享内容空间，可以直接修改元素值
    a[1:4] = 10, 20, 30
    print(a)
    # 因此，在实践中，切实注意原始数据是否被破坏，如：
    b = a[2:5]
    b[0] = 200
    print(a)

    print("\n---------------------------\n")
    # 3.2 整数/布尔数组存取
    # 3.2.1
    # 根据整数数组存取：当使用整数序列对数组元素进行存取时，
    # 将使用整数序列中的每个元素作为下标，整数序列可以是列表(list)或者数组(ndarray)。
    # 使用整数序列作为下标获得的数组不和原始数组共享数据空间。
    a = np.logspace(0, 9, 10, base=2)
    print(a)
    i = np.arange(0, 10, 2)
    print(i)

    # 利用i取a中的元素
    b = a[i]
    print(b)

    # b的元素更改,a中的元素不受影响
    b[2] = 1.6
    print(b)
    print(a)

    print("\n---------------------------\n")
    # 3.2.2
    # 使用布尔数组i作为下标存取数组a中的元素：返回数组a中所有在数组b中对应下标为True的元素
    # 生成10个满足[0,1)中均匀分布的随机数
    a = np.random.rand(10)
    print(a)
    # 大于0.5的元素索引
    print(a > 0.5)  # 返回值是boolean数组，因为是对数组中的每一个元素对比
    # 大于0.5的元素
    b = a[a > 0.5]
    print(b)
    # 将原数组中大于0.5的元素截取成0.5
    a[a > 0.5] = 0.5
    print(a)
    # b不受影响
    print(b)

    print("\n---------------------------\n")
    # 3.3 二维数组的切片
    # [[ 0  1  2  3  4  5]
    #  [10 11 12 13 14 15]
    #  [20 21 22 23 24 25]
    #  [30 31 32 33 34 35]
    #  [40 41 42 43 44 45]
    #  [50 51 52 53 54 55]]
    a = np.arange(0, 60, 10)  # 行向量
    print(a)  # [ 0 10 20 30 40 50]
    # 转换成列向量
    b = a.reshape((-1, 1))
    """
    [
     [ 0]
     [10]
     [20]
     [30]
     [40]
     [50]
    ]
    """
    print(b)
    c = np.arange(6)
    print(c)
    f = b + c
    print(f)  # 行 + 列

    # 合并上述代码：
    a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    """
    [[ 0  1  2  3  4  5]
     [10 11 12 13 14 15]
     [20 21 22 23 24 25]
     [30 31 32 33 34 35]
     [40 41 42 43 44 45]
     [50 51 52 53 54 55]]
    """
    print(a)

    # 二维数组的切片，行列（0,2）（1,3）（2,4）（3,5）
    # [ 2 13 24 35]
    # 以二维数组中的行和列为索引，取数组a中的元素
    print(a[(0, 1, 2, 3), (2, 3, 4, 5)])
    # 以第3行到最后一行为行索引,以0,2,5为列索引，取数组中a的元素
    print(a[3:, [0, 2, 5]])
    i = np.array([True, False, True, False, False, True])
    # 为True取，为False不取
    """
    [[ 0  1  2  3  4  5]
     [20 21 22 23 24 25]
     [50 51 52 53 54 55]]
    """
    print(a[i])
    # 为True的行,第3列
    print(a[i, 3])

    '''

    '''
    print("\n---------------------------\n")

    import time
    import math

    print("\n---------------------------\n")
    print("\n---------------------------\n")
    # 4. numpy与Python数学库的时间比较
    for j in np.logspace(0, 7, 10):
        j = int(j)
        # 创建新数组
        x = np.linspace(0, 10, j)
        start = time.clock()
        y = np.sin(x)
        t1 = time.clock() - start

        x = x.tolist()
        start = time.clock()
        for i, t in enumerate(x):
            x[i] = math.sin(t)
        t2 = time.clock() - start
        print(j, ": ", t1, t2, t2 / t1)

    '''

    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import math

    print("\n---------------------------\n")
    print("\n---------------------------\n")
    # 5.绘图
    # 5.1 绘制正态分布概率密度函数
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False
    mu = 0
    sigma = 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    print(x.shape)
    print('x = \n', x)
    print(y.shape)
    print('y = \n', y)
    # plt.plot(x,y,'ro-',linewidth=2)
    plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
    plt.grid(True)
    # plt.show()
    plt.close()
    print("\n---------------------------\n")
# 5.2 损失函数：Logistic损失(-1,1)/SVM Hinge损失/ 0/1损失
    x = np.array(np.linspace(start=-2, stop=3, num=1001, dtype=np.float))
    y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0
    plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('1.png')
    # plt.show()
    plt.close()

    # 5.3 x^x
    x = np.linspace(-1.3, 1.3, 101)
    y = fx(x)
    plt.plot(x, y, 'g-', label='x^x', linewidth=2)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

