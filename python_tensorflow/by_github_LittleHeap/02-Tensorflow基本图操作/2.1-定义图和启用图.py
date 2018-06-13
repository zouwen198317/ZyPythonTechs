import common_header
import tensorflow as tf

# 定义1*2的矩阵常量
m1 = tf.constant([[3, 3]])

# 定义2*1的矩阵常量
m2 = tf.constant([[2], [3]])

# 创建一个矩阵乘法操作，把m1,m2传入
product = tf.matmul(m1, m2)

# 打印输出
print(product)

# # 定义一个会话，启动默认图
sess = tf.Session()

# 变量才需要初始化
# sess.run(tf.initialize_all_variables())

# 调用run方法执行op中的乘法操作
result = sess.run(product)

# 打印结果
print("第一种定义会话方式 => ", result)
sess.close()

# 第二种定义会话操作
with tf.Session() as sess:
    result2 = sess.run(product)
    print("第二种定义会话方式 => ", result2)
