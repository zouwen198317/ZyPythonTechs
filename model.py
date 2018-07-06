#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   model.py
# @Date    :   2018/7/6
# @Desc    :


import tensorflow as tf


# Y = W*x+b
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]


def convolutional(x, keep_prob):
    def conv2d(x, W):
        return tf.nn.conv2d([1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def weigth_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = weigth_variable([5, 5, 1, 31])
    b_conv1 = bias_variable([32])

