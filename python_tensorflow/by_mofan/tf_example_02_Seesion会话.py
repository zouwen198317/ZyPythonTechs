# -*- coding:UTF-8 -*-

import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1行2列的矩阵
matrix_1 = tf.constant([[3, 3]])
# 2行1列的矩阵
matrix_2 = tf.constant([
    [2], [2]
])

# import numpy as np
#
# # numpy 中的矩阵乘法
# n_result = np.multiply(matrix_1, matrix_2)
# print(n_result)

# tf的矩阵乘法
product = tf.matmul(matrix_1, matrix_2)

# 运行会话的两种方式
# 第一种方式：需要手动关闭会话
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 第二种方式:不需要手动关闭session
with tf.Session() as sess:
    result_2 = sess.run(product)
    print(result_2)
