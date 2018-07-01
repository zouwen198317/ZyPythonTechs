#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   5.2 Tensorflow构建网络.py
# @Date    :   2018/6/29
# @Desc    :
from tensorflow.examples.tutorials.mnist import input_data

import common_header

import tensorflow as tf

# one_hot为onhot编码
minist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = minist.train.num_examples // batch_size  # 整除

# 定义两个placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('Weight'):
        W = tf.Variable(tf.zeros([784, 10]),name='W')
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([1, 10]),name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        # 将输出的信号转化为概率值
        prediction = tf.nn.softmax(wx_plus_b)

# 原先采用二次代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction))

# 使用梯度梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_variables()
else:
    init = tf.global_variables_initializer()

# 结果存放到一个布尔类型的列表中，生成1*100的布尔矩阵
# argmax返回一维张量中最大的值所在的位置
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 将图写入指定目录
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(10):
        for batch in range(n_batch):  # 遍历所有数据集来进行训练
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels})
        print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))

# tensorboard --logdir=log
