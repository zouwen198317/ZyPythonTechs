import common_header

import tensorflow as tf

# 创建一个常量op,产生一个1x2矩阵,这个op被作为一个节点
# 加到默认图中

# 构造器的返回值代表该常量op的返回值
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量op,产生一个2x1的矩阵
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵乘法 matmul op ,把matrix1和matrix2作为输入
# 返回值"product"代表矩阵乘法的结果
# 矩阵乘法:行*列
product = tf.matmul(matrix1, matrix2)

"""
默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 为了真正进行矩阵相乘运算, 并得到矩阵乘法的 结果, 你必须在会话里启动这个图.

在一个会话中启动图
构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
"""


def run_sess_normal():
    global sess
    # 启动默认图
    sess = tf.Session()
    # 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
    # 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
    # 矩阵乘法 op 的输出.
    # 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
    # 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
    # 返回值 'result' 是一个 numpy `ndarray` 对象.
    result = sess.run(product)
    print(result)
    # 关闭会话
    sess.close()


# run_sess_normal()

# Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

"""
Session 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.

with tf.Session() as sess:
  result = sess.run([product])
  print result
在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU). 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, 
TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.

如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行. with...Device 语句用来指派特定的 CPU 或 
GPU 执行操作:

with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
设备用字符串进行标识. 目前支持的设备包括:

"/cpu:0": 机器的 CPU.
"/gpu:0": 机器的第一个 GPU, 如果有的话.
"/gpu:1": 机器的第二个 GPU, 以此类推.
"""
