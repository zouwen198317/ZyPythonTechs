# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Tensorflow的基础用法
https://blog.csdn.net/lynskylate/article/details/54911295
"""
"""
Tensorflow是一个深度学习框架，它使用图（graph）来表示计算任务，使用tensor（张量）表示数据，图中的节点称为OP，在一个会话
（Session）的上下文中执行运算，最终产生tensor

tensor在数学中称为张量，表示空间，在计算图模型中就是基本的数据类型，如同我们在sklearn等机器学习框架中使用numpy的矩阵作为基本
运算的数据类型一样，无论几维的矩阵都是一个张量

Tensorflow的基本变量

tensor计算图谱的基本类型
    tensor 张量 
        Variable 变量
        Constant 常量
        Placeholder 占位符
    Graph 图
    Session 会话
"""

# Constant常量
# tf.constant(value, dtype=None, shape=None,name="const")

import tensorflow as tf

a = tf.constant(10)
print(a)
sess = tf.Session()
print(sess.run(a))
sess.close()

# Variable变量
v = tf.Variable(tf.zeros([1]))
print(v)

# 例子
state = tf.Variable(0, name="counter")
# 创建一个op,其作用使state增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，变量必须先经过初始化init op初始化
# 首先，必须增加一个初始化op到图中
init_op = tf.global_variables_initializer()

# 启动图，运行op
with tf.Session() as sess:
    # 运行init op
    sess.run(init_op)
    # 打印state的初始值
    print(sess.run(state))
    # 运行op 更新state,并打印state
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    sess.close()

"""
Placeholder

这个是暂时未定义的一个tensor 
在计算图运行时给予，类似函数的参数。 
python 
input1 = tf.placeholder(tf.float32) 

计算图

Tensorflow的程序一般分为两个阶段，构建阶段和执行极端。一般，构建阶段会创建一个图，来表示神经网络，在执行阶段在反复执行训练图。 
可以把图的构建看成定义一个复杂的函数。
"""

# 构建图
matrix = tf.constant([[3., 3]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix, matrix2)

# 启动图
# 方式A
sess = tf.Session()
result = sess.run(product)

# 运行图定义的product计算
print(result)
sess.close()

# 方式B
with tf.Session() as sess:
    result = sess.run([product])
    print(result)
    sess.close()

"""
Fetch取回

在sess的运算中不仅可以取回结果，还可以取回多个tensor，在神经网络中，我们可以取回中间层的结果
"""
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
# 相加
intermed = tf.add(input2, input3)
# 相乘
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul])
    print(result)
    sess.close()

"""
sess.run()接受的参数是个列表，输出会输出这个列表中的值

Feed供给

有时训练过程中我们会执行增量训练，这时就会使用前面介绍的Placeholder()
"""
input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)
output_v = tf.multiply(input1, input2)

with tf.Session() as sess:
    # result = sess.run([output_v], feed_dict={input_1: [7], input_2: [2.]})
    result = sess.run([output_v], feed_dict={input_1: 7, input_2: 2.})
    print(result)
    sess.close()
