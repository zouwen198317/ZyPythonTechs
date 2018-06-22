# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
可训练线性回归模型(完整代码)
"""

# Model parametrs
import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
# sum of the square
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
opeimizer = tf.train.GradientDescentOptimizer(0.01)
train = opeimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# trainning loop
init = tf.global_variables_initializer()
sess = tf.Session()
# reset values to wrong
sess.run(init)

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate traing accuracy
curr_w, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

print("W: %s b:%s loss: %s " % (curr_w, curr_b, curr_loss))

"""
损失是非常小的数字（非常接近零）
"""
