import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
# 1.路径 2.把标签转换为01格式
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 数据分页，每页100条数据
batch_size = 100

# 计算一共有多少页
# batch_size 整除
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
# 将输出的信号转换为概率值
predictoin = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - predictoin))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存在一个布尔型列表中，生成1*100布尔矩阵
# argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predictoin, 1))
# 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        # 遍历所有数据集来进行训练
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuray, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))
