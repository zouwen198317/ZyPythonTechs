import tensorflow as tf

import os

import log_utils as log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
五分钟带你入门TensorFlow
'''
# -----------------     创建图和运行图     -----------------
# -----------------     示例<矩阵乘法>     -----------------

"""
TensorFlow有几个概念需要进行明确：

1 图（Graph）：用来表示计算任务，也就我们要做的一些操作。

2 会话（Session）：建立会话，此时会生成一张空图；在会话中添加节点和边，形成一张图，一个会话可以有多个图，通过执行这些图得到结果。如果把每个图看做一个车床，那会话就是一个车间，里面有若干个车床，用来把数据生产成结果。

3 Tensor：用来表示数据，是我们的原料。

4 变量（Variable）：用来记录一些数据和状态，是我们的容器。

5 feed和fetch：可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据。相当于一些铲子，可以操作数据。

形象的比喻是：把会话看做车间，图看做车床，里面用Tensor做原料，变量做容器，feed和fetch做铲子，把数据加工成我们的结果。
"""
log.loge("矩阵乘法")
# 创建图和运行图
# 创建一个常量v1,它是一个1行2列的矩阵
v1 = tf.constant([[2, 3]])
# 创建一个常量v2,它是一个2行一列的矩阵
v2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法，这里要注意的是，创建了乘法后，是不会立即执行的，要在会话中运行才行
product = tf.matmul(v1, v2)
# 这个时候打印，得到的不是他们乘法之后的结果，而是得到乘法本身
print(product)

# 定义一个会话
sess = tf.Session()
# 运行乘法，得到结果
result = sess.run(product)
# 打印结果
print(result)
# 关闭会话
sess.close()
log.loge("矩阵乘法")
# -----------------     示例<矩阵乘法>     -----------------

# -----------------     创建一个变量，并用for循环对变量进行赋值操作     -----------------
#  创建一个变量，并用for循环对变量进行赋值操作
log.loge("for循环对变量进行赋值操作")

# 创建一个变量num
num = tf.Variable(0, name="count")
# 创建一个加法操作,把当前的数字+1
new_value = tf.add(num, 1)
# 创建一个赋值操作,把new_value赋值给num
op = tf.assign(num, new_value)

# 使用这种写法,在运行完毕后，会话会自动关闭
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 打印最初的值
    print(sess.run(num))
    # 创建一个for循环，每次给num+1,并打印出来
    for i in range(5):
        sess.run(op)
        print(sess.run(num))

log.loge("for循环对变量进行赋值操作")
# -----------------     创建一个变量，并用for循环对变量进行赋值操作     -----------------

# -----------------     通过feed设置placeholder的值     -----------------
log.loge("占位符的使用")
"""
有的时候，我们会在声明变量的时候不赋值，计算的时候才进行赋值，这个时候feed就派上用场了
"""
# 创建一个变量占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 创建一个加法操作,把input1和input2相乘
new_value = tf.multiply(input1, input2)

# 使用这种写法,在运行完毕后，会话会自动关闭
with tf.Session() as sess:
    # 打印new_value的值,在运算时，用feed设置两个输入值
    print(sess.run(new_value, feed_dict={input1: 23.0, input2: 11.0}))
    log.loge("占位符的使用")
# -----------------     通过feed设置placeholder的值     -----------------
