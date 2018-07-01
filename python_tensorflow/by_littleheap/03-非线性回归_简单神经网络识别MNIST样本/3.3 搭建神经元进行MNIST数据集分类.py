import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
# one_hot编码，只有一个标签为1其它标签均为0
minist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次大小
batch_size = 100
# 计算一共有多少个批次
# 使用训练集中的样本总数/每页数据的个数
n_batch = minist.train.num_examples / batch_size  # 整除

# 定义x,y，特征值和目标值
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# -----------------------------   简单的神经网络 -----------------------------
"""
# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 将输出的信号值转换为概率值
prediction = tf.nn.softmax(tf.matmul(x, W) + b)
"""
# -----------------------------   简单的神经网络 -----------------------------

"""
数据集分类
"""
# -----------------------------   数据集分类 -----------------------------
# 定义神经网络中间层(中间层是二十个神经元)
# 定义一个784*20的权值矩阵
Weights_L1 = tf.Variable(tf.random_normal([784, 20]))
# 定义一个1*20的偏值矩阵
biases_L1 = tf.Variable(tf.zeros([1, 20]))
# 定义网络总和函数
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 激活函数:用双曲正切函数作用于信号输出总和
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
# 定义一个20*10的权值矩阵
Weights_L2 = tf.Variable(tf.random_normal([20, 10]))
# 定义一个1*10的偏值矩阵
biases_L2 = tf.Variable(tf.zeros([1, 10]))
# 定义网络总和函数(输出层的输入就是中间层的输出)
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 激活函数:用双曲正切函数作用于信号输出总和,即预测值
prediction = tf.nn.tanh(Wx_plus_b_L2)

# -----------------------------   数据集分类 -----------------------------

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_variables()
else:
    init = tf.global_variables_initializer()

# 结果存放到一个布尔类型的列表中，生成1*100的布尔矩阵
# argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(100000):
        batch_xs, batch_ys = minist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        if epoch % 500 == 0:
            acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels})
            print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))
