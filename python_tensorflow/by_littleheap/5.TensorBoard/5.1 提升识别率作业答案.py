#! /usr/bin/env python
# -*- coding:utf-8 -*-
#   @author: zzg
#   @contact: xfgczzg@163.com
#   @file: 5.1 提升识别率作业答案
#   @time: 2018/6/29
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
lr = tf.Variable(0.001, dtype=tf.float32)

# 开始搭建神经网络
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
# 双曲正切激活函数计算某一层神经元的总输出
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# 1. 某一层神经元总输出，2。当前活跃神经元百分比
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)

# 将输出的信号转化为概率值
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 更换交叉熵作为代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度梯度下降法
# 训练
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

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

    for epoch in range(51):
        sess.run(tf.assign(lr, 0.01 * (0.95 ** epoch)))
        for batch in range(n_batch):  # 遍历所有数据集来进行训练
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        # 测试准确度时，将神经网络全部激活
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + " , Testing Accuracy " + str(acc) + " , Learning Rate: " + str(
            learning_rate))
