import common_header
import tensorflow as tf

# 计数变量
state = tf.Variable(0, name="counter")
# print(state.name)

# 常量
one = tf.constant(1)

new_value = tf.add(state, one)

# 把new_value加载到state上，state的值就是new_value的值
update = tf.assign(state, new_value)

# 初始化所有的变量 (非常重要)
# 有定义变量一定要加这个
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # (非常重要)
    sess.run(init)

    for _ in range(5):
        sess.run(update)
        print("state = > ", sess.run(state))
