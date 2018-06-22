"""
变量
Variables for more details. 变量维护图执行过程中的状态信息. 下面的例子演示了如何使用变量实现一个简单的计数器.
"""

import common_header
import tensorflow as tf

# 创建一个变量，初始化值为标量0
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)

# 启动图, 运行 op
with tf.Session() as sess:
    # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
    # 首先必须增加一个`初始化` op 到图中.
    # 运行 'init' op
    sess.run(tf.global_variables_initializer())
    sess.run(state)
    # 打印 'state' 的初始值
    print("state -> ", sess.run(state))
    for _ in range(3):
        # 运行 op, 更新 'state', 并打印 'state'
        sess.run(update)
        print("state ->> ", sess.run(state))

"""
代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式之前, 它并不会真正执行赋值操作.

通常会将一个统计模型中的参数表示为一组变量. 例如, 你可以将一个神经网络的权重作为某个变量存储在一个 tensor 中. 在训练过程中, 通过重复运行训练图, 更新这个 tensor.
"""
