import common_header
import tensorflow as tf

"""
placeholder 占位符

使用占位符，就意味着要在sess.run的时候通过feed_dict给占位符传值

placeholder和feed_dict是绑定的
"""

# type : 大部分只处理float32的形式
# [2,2] 可以规定结构
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 乘法运算
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # (非常重要)
    # 在run的时候，以feed_dict传值到占位值
    print(sess.run(output, feed_dict={input1: [3.], input2: [2.]}))
