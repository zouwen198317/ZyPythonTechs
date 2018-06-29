#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   4.3优化器.py
# @Date    :   2018/6/29
# @Desc    :

import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# one_hot为onhot编码
minist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = minist.train.num_examples // batch_size  # 整除

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))

# 将输出的信号转化为概率值
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 原先采用二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))

# 更换交叉熵作为代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 此处使用Adam优化器对loss进行优化
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

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

    for epoch in range(21):
        for batch in range(n_batch):  # 遍历所有数据集来进行训练
            # 训练，得到模型，使用训练集中的数据
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        # 验证：测试集中的数据
        acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels})
        print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))
