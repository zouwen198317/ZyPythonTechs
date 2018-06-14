import common_header
import tensorflow as tf

import matplotlib.pyplot as plt

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

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 构造隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# 构造输出层
# 线性函数
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算prediction和y_data（预测值和真实值之间的一个差别)
# square 这个结果是对每一个例子的平方
# reduce_sum对每一个例子进行求和
# reduce_mean对每一个例子的和求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# 预测
# 从误差中去学习
# learning_rate 学习效率有多高,这个值通常是要小于1的
# minimize减少误差
# 第一个的训练都是通过优化器去减小误差
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 初始化所有的变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 结果可视化
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    # 连续输入
    plt.ion()
    plt.show()

    for i in range(1000):
        # 使用全部的数据进行运算
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            try:
                # 去除第一条线(为了避免异常所以try...catch
                ax.lines.remove(lines[0])
            except Exception:
                pass

            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # 把prediction_value以曲线的形式画上去
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # 去除第一条线
            # ax.lines.remove(lines[0])
            # 暂停0.1s
            plt.pause(1)
