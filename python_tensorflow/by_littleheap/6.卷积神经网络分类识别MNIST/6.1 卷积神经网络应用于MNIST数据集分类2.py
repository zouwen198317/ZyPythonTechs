# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   6.1 卷积神经网络应用于MNIST数据集分类.py
# @Date    :   2018/6/30
# @Desc    : 卷积神经网络的训练和测试（针对电脑内存比较小的，运行速度比较慢的）


import common_header

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def my_conv(input_image, out_dim, name, channel):
    with tf.variable_scope(name):
        w_conv1 = weight_variable([5, 5, channel, out_dim])
        b_conv1 = bias_variable([out_dim])
        h_conv1 = tf.nn.relu(conv2d(input_image, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    return h_pool1


def my_fc_layer(input_image, out_dim, name):
    with tf.variable_scope(name):
        w_fc1 = weight_variable([7 * 7 * 64, out_dim])
        b_fc1 = bias_variable([out_dim])
        h_pool2_flat = tf.reshape(input_image, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    return h_fc1


def build_net(input_data, keep_prob):
    conv1 = my_conv(input_image=input_data, out_dim=32, name='conv_layer1', channel=1)
    conv2 = my_conv(input_image=conv1, out_dim=64, name='conv_layer2', channel=32)
    fc1 = my_fc_layer(input_image=conv2, out_dim=1024, name='fc_layer1')

    h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    return y_conv


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

sum = tf.Variable(0.0, name="sum")
temp = tf.Variable(0.0, name="temp")

keep_prob = tf.placeholder(tf.float32)
y_conv = build_net(input_data=x_image, keep_prob=keep_prob)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()  # 启动Session
for i in range(500):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 下面是训练时发现电脑内存较小，只能把训练集拆分成多步完成

for i in range(200):
    testSet = mnist.test.next_batch(50)
    temp = accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})
    # print("test accuracy %g"%accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
    print("test accuracy %g" % temp)
    sum = tf.add(sum, temp)
s = sess.run(sum)
print(s / 200)
