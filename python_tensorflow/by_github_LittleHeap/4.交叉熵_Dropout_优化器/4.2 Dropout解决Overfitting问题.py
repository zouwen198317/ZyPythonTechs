#! /usr/bin/env python
# -*- coding:utf-8 -*-
#   @author: zzg
#   @contact: xfgczzg@163.com
#   @file: 4.2 Dropout解决Overfitting问题
#   @time: 2018/6/28
#   @desc:

import common_header

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
# one_hot为onhot编码
minist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = minist.train.num_examples // batch_size  # 整除

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 开始搭建神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L1_drop = tf.nn.dropout(L1, keep_prob)

W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)

# 将输出的信号转化为概率值
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 原先采用二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))

# 更换交叉熵作为代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_variables()
else:
    init = tf.global_variables_initializer()

# 结果存放到一个布尔类型的列表中，生成1*100的布尔矩阵
# argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(31):
        for batch in range(n_batch):  # 遍历所有数据集来进行训练
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

            # 测试准确度时，将神经网络全部激活
            test_acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: minist.train.images, y: minist.train.labels, keep_prob: 1.0})
            print("Iter " + str(epoch) + " , Testing Accuracy " + str(test_acc) + " , Training Accuracy: " + str(
                train_acc))
