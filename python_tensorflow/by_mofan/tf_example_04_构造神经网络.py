import common_header
import tensorflow as tf

"""
激励函数通常来说就是将某一部分的神经元先激活起来
将激活起来的信息，传递到后面一层的神经系统

激励函数其实就是一个方乘

添加层 : 添加定义神经层

layer1 和 layer2
    ->  weights , biases, 激励函数
    
添加神经网络层
    和真实值进行一个对比
    误差的减少,提升预测值    
"""


# activation_function 如果没有默认就是一个线性函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # weights是一个随机变量，在生成初始weights的时候会比全部为0要好
    # 变量矩阵,in_size行 * out_size列，
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 类似列表的东东
    # biases在ML中推荐不为0,所以加的0.1
    # Weights为随机变量的值，biases全部为0.1,这两个值在每一步的训练中都会有变化
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # inputs * weights + biases 这里使用的是矩阵的乘法
    # 计算出来的值（还没有被激活的值存储在该Wx_plus_biases变量中)
    Wx_plus_biases = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        # 保持现状， 为线性
        outputs = Wx_plus_biases
    else:
        # 非线性
        outputs = activation_function(Wx_plus_biases)
    return outputs


import numpy as np

# 1个特性,300条数据(-1,1之间的数据)
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 噪音的添加
noise = np.random.normal(0, 0.05, x_data.shape)
# 真实值是基于x_data,2次方-0.5
y_data = np.square(x_data) - 0.5 + noise

"""
inputlayer:
    1个

hiddenlayer:
    10个神经元

outputlayer:
    1个    
"""
