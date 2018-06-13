# -*- coding:UTF-8 -*-

import tensorflow as tf

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create data(创建数据)
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure(创建tensorflow 结构) strt  ###
# Weights可能是二维的矩阵
# tf定义的变量，使用随机数列生成参数，1维随机数列，范围是-1到1
# 权重的初始值是从-1到1的一个数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 偏置的初始值为0
# 会通过学习将初始值接近0.1->0.3
biases = tf.Variable(tf.zeros([1]))
# tensorflow 就是从初始值不断的提升

# 要预测的数据
y = Weights * x_data + biases

# 我们需要提升y的准确率

# 计算y和实际的y_data的一个差别
# 在最初始的时候，Weights和biases的值是不确定的，所以y和y_data的之间的差别会是非常大
# 会导致loss很大
loss = tf.reduce_mean(tf.square(y - y_data))
# 创建优化器,对误差进行优化
# 学习效率是小于1的数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 使用优化器减少误差,每一步训练都需要做这个操作，使下一次训练误差减少
train = optimizer.minimize(loss)

# 在tf中初始化所有的变量
init = tf.global_variables_initializer()
### create tensorflow structure(创建tensorflow 结构) end  ###


sess = tf.Session()
# run的时候，就像指针一样，指到什么地方，该位置就被激活
# very importrant
sess.run(init)

# 训练201次
for step in range(201):
    sess.run(train)

    # 每隔20次，打印一下训练的结果
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

sess.close()
