"""
交互式使用
文档中的 Python 示例使用一个会话 Session 来 启动图, 并调用 Session.run() 方法执行操作.

为了便于使用诸如 IPython 之类的 Python 交互环境, 可以使用 InteractiveSession 代替 Session 类, 使用 Tensor.eval() 和 Operation.run() 方法代替
Session.run(). 这样可以避免使用一个变量来持有会话.
"""

import common_header
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op的run方法初始化x
x.initializer.run()

# 增加一个减法 sub op ,从x减去a,运行减法op,输出结果
sub = tf.subtract(x, a)
print(sub.eval())
