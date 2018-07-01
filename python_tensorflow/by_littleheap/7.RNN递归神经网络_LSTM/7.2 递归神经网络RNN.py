# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   7.2 递归神经网络RNN.py
# @Date    :   2018/7/1
# @Desc    :

import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 输入图片是28*28
# 输入一行，共有28个数据
n_inputs = 28
max_time = 28
# 隐藏层单元
lstm_size = 100
# 10个分类
n_classes = 10
# 每批次50个样本
batch_size = 50
# 计算出一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 这里None表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签（真实的标签类型)
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
Weights = tf.Variable(tf.random_normal([lstm_size, n_classes]))
# 初始化偏值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(x, Weights, biases):
    # input = [batch_size,max_time,n_inputs]
    # 转化数据格式，-1对应一个批次的50
    inputs = tf.reshape(x, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL单元
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell_state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], Weights) + biases)
    return results


# 计算RNN返回值
prediction = RNN(x, Weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔列表中
# argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
# 把correct_prediction变为float32类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + " , Testing Accuracy = " + str(acc))
