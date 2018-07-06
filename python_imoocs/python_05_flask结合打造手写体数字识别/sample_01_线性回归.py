#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_线性回归.py
# @Date    :   2018/7/6
# @Desc    :

import os

import input_data

minist = input_data.read_data_sets(input_data.common_header.MINIST_FILE2, one_hot=True)
# print(minist)

import tensorflow as tf

import model

# create model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# # train
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.add(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(1000):
        batch_xs, batch_ys = minist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: minist.test.images, y_: minist.test.labels}))

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
                      write_meta_graph=False, write_state=False)

    print("Saved:", path)
