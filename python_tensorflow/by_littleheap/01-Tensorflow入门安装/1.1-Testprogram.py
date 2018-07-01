# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# 1行5列的矩阵
a = tf.Variable([1, 0, 0, 1, 1])

b = tf.Variable([
    [1], [1], [1], [0], [0]
])

c = tf.Variable([
    [True], [True], [True], [False], [False]
])

cast_bool = tf.cast(b, dtype=tf.bool)
cast_float = tf.cast(b, dtype=tf.float32)

sess = tf.Session()
# 初始化所有的变量
sess.run(tf.initialize_all_variables())
print(sess.run(cast_float))
# 求平均值
print(sess.run(tf.reduce_mean(cast_float)))
