# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   sample_02_CNN.py
# @Date    :   2018/7/8
# @Desc    :   参考代码: https://www.cnblogs.com/Ph-one/p/9074706.html

import model
import tensorflow as tf
import input_data
import os

mnist = input_data.read_data_sets(input_data.common_header.MINIST_FILE2, one_hot=True)

# 定义model
with tf.variable_scope("connvolutional"):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10], name="y")
# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 判断参数是否相等
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    merge_summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/minist_log/1', sess.graph)
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    # 10000-20000
    # -----------------     使用训练集对模型进行训练     -----------------
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d,training accuracy %g " % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # -----------------     使用训练集对模型进行训练     -----------------

    # -----------------     使用测试集对模型进行验证     -----------------
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # -----------------     使用测试集对模型进行验证     -----------------

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'),
                      write_meta_graph=False, write_state=False)

    print("Saved:", path)
