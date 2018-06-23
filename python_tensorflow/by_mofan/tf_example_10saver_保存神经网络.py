import common_header

"""
Saver只能保存变量，不能保存神经网络
"""

import tensorflow as tf

# # Save to file
# """
#     remember to define the same dtpye and shape when restore
# """
# W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name="weights")
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "my_net/save_net.ckpt")
#     print("Save to path: ", save_path)


# resotre variables
# redefine the same shape and same type for your variables

import numpy as np

# 维度为3，2行3列
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# 维度为1，1行3列
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# no need init step,不需要初始化的步骤
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
