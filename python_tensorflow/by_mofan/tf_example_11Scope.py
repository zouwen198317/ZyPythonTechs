from __future__ import print_function
import common_header

"""
Scope -> RNN中会用到
"""

import tensorflow as tf

# reproducible
tf.set_random_seed(1)

with tf.name_scope("a_name_scope"):
    # name_scope对于get_variable无效
    initializer = tf.constant_initializer(value=1)
    name_var1 = tf.get_variable(name="name_var1", shape=[1], dtype=tf.float32,
                                initializer=initializer)
    name_var2 = tf.Variable(name="name_var2", initial_value=[2], dtype=tf.float32)
    name_var21 = tf.Variable(name="name_var2", initial_value=[2.1], dtype=tf.float32)
    name_var22 = tf.Variable(name="name_var2", initial_value=[2.2], dtype=tf.float32)

with tf.variable_scope("a_variable_scope"):
    initializer = tf.constant_initializer(value=1)
    va_var1 = tf.get_variable(name="name_var1", shape=[1], dtype=tf.float32,
                              initializer=initializer)
    va_var2 = tf.Variable(name="name_var2", initial_value=[2], dtype=tf.float32)
    va_var21 = tf.Variable(name="name_var2", initial_value=[2.1], dtype=tf.float32)
    va_var22 = tf.Variable(name="name_var2", initial_value=[2.2], dtype=tf.float32)

with tf.variable_scope("a_variable_scope_2") as scope:
    initializer = tf.constant_initializer(value=1)
    va_var1_1 = tf.get_variable(name="name_var1_1", shape=[1], dtype=tf.float32,
                                initializer=initializer)
    # 变量重复利用
    scope.reuse_variables()

    # 不定义上面的语句这里会报错
    va_var2_1 = tf.get_variable(name="name_var1_1")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("name_var1.name : ", name_var1.name)
    print("name_var1.value : ", sess.run(name_var1))
    print("name_var2.name : ", name_var2.name)
    print("name_var2.value : ", sess.run(name_var2))
    print("name_var21.name : ", name_var21.name)
    print("name_var21.value : ", sess.run(name_var21))
    print("name_var22.name : ", name_var22.name)
    print("name_var22.value : ", sess.run(name_var22))
    print()

    print("va_var1.name : ", va_var1.name)
    print("va_var1.value : ", sess.run(va_var1))
    print("va_var2.name : ", va_var2.name)
    print("va_var2.value : ", sess.run(va_var2))
    print("va_var21.name : ", va_var21.name)
    print("va_var21.value : ", sess.run(va_var21))
    print("va_var22.name : ", va_var22.name)
    print("va_var22.value : ", sess.run(va_var22))
    print()

    print("va_var1_1.name : ", va_var1_1.name)
    print("va_var1_1.value : ", sess.run(va_var1_1))
    # 调用过就直接复用
    print("va_var2_1.name : ", va_var2_1.name)
    print("va_var2_1.value : ", sess.run(va_var2_1))
