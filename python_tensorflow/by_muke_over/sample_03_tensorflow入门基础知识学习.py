# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
tensorflow入门基础知识学习
https://www.imooc.com/article/30467
"""

import tensorflow as tf

# -----------------     查看tensorflow版本 -----------------
print(tf.__version__)

# -----------------     查看tensorflow版本 -----------------


# -----------------     图(graph)中的节点(Node) -----------------
"""
一些变量和常量，还有计算的操作都是一些实际的操作，而tensorflow这些都被作为图(graph)中的节点(Node)
"""
a = tf.constant(0)
b = tf.constant(1)
print(" a -> ", a)
print(" b -> ", b)

"""
其中，Const:0，是这个tensor的名称，shape=(),表示是标量，dtype=int32,表示其类型是int32.其并没有和相应的实际数值0关联起来。
"""
# -----------------     图(graph)中的节点(Node) -----------------

"""
在像计算图里面追加操作，并没有像其他语言一样会把其值覆盖，而是会在计算图里面新增加节点，现在a的名称已经变为Const_2:0
"""

a = tf.constant(0)
b = tf.constant(1)
print(" a -> ", a)
print(" b -> ", b)

d = a + b
e = a * b
print(" e -> ", d)
print(" e -> ", e)

# -----------------     以节点的方式操作 -----------------
mat_a = tf.constant([[1, 1, 1], [3, 3, 3]])
mat_b = tf.constant([[2, 2, 2], [5, 5, 5]], name='mat_b')
mat_a_b = mat_a * mat_b
tf_mult_a_b = tf.multiply(mat_a, mat_b)
tf_matmul_a_b = tf.matmul(mat_a, tf.transpose(mat_b), name="matmul_with_name")

this_graph = tf.get_default_graph()
this_graph_default = this_graph.as_graph_def()
# print(" this_graph_default - > ", this_graph_default)

print(" mat_a_b - > ", mat_a_b)
print(" tf_matmul_a_b - > ", tf_matmul_a_b)

"""
要使得得到的结果不是tensor，就需要用Session(),就是将计算图运行起来，得到结果。其关系就好比程序与进程之间的关系，进程就是程序
的一次执行，计算图就是计算的流程，就类比程序；程序跑起来就类似于session(
);通过sess.run()把计算图真正的运行起来
"""
sess = tf.Session()
mult_a_b_value, tf_mult_a_b_value, tf_matmul_a_b_value = sess.run([mat_a_b, tf_mult_a_b, tf_matmul_a_b])

print(" mult_a_b_value - > ", mult_a_b_value)
print(" tf_mult_a_b_value - > ", tf_mult_a_b_value)
print(" tf_matmul_a_b_value - > ", tf_matmul_a_b_value)

# -----------------     以节点的方式操作 -----------------

"""
当需要和外界进行数据交换的时候，不能只是用constant定义数据，不然怎么优化更新，需要用到Variable,placeholder
placeholder代表着从外界输入的数据，None代表了不确定的维度。
"""
# placeholder代表着从外界输入的数据，None代表了不确定的维度。
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Variable声明一个变量，在梯度下降中可以对它进行相应的更新（权重和偏置)
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

logits = tf.matmul(x, w) + b
output = tf.nn.sigmoid(logits)
cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)

print(" logits - > ", logits)
print(" output - > ", output)
print(" cross_entropy - > ", cross_entropy)

# # 梯度下降的优化器类
# # 其代表的是一个操作
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
#
# # 示例代码示贴出。运行会报错
x_value = tf.placeholder(dtype=tf.float32)
y_value = tf.placeholder(dtype=tf.float32)
#
# # 完成内存分配和初始化操作的动作
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
#
# """
# 需要输出其值，需要在，sess.run()里面传入相应的参数。其对应的值不会作相应的改变，原因是train_step没有放在其sess.run()里面
# """
logits_value, output_value, cross_entropy_value = sess.run([logits, output, cross_entropy],
                                                           feed_dict={
                                                               x: x_value,
                                                               y: y_value})
# print(" logits_value - > ", logits_value)
# print(" output_value - > ", output_value)
# print(" cross_entropy_value - > ", cross_entropy_value)

# 在sess.run()里面循环迭代优化，相当于计算了100次，train_steps
# for current_step in range(100):
#     output_value, cross_entropy_value = sess.run([cross_entropy, output, train_step],
#                                                  feed_dict={
#                                                      x: x_value,
#                                                      y: y_value})
#
# logits_value, output_value, cross_entropy_value = sess.run([cross_entropy, output, logits, w, b],
#                                                            feed_dict={
#                                                                x: x_value,
#                                                                y: y_value})
