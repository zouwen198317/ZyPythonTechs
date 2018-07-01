# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   8.2 save_restore读取
# @Date    :   2018/7/1
# @Desc    :

import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次100张照片
batch_size = 100
# 计算出一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
# 这里None表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签（真实的标签类型)
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
Weights = tf.Variable(tf.zeros([784, 10]))
# 初始化偏值
biases = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, Weights) + biases)

# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    print("Initial " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
    saver.restore(sess, 'net/my_net.ckpt')
    print("Trained " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
