import common_header
import tensorflow as tf

"""
    线性回归:
      要预测的值是一个线性连续性的值
      
    分类:
      数据分成指定的类型
      
    资料地址:https://my.oschina.net/dwqdwd/blog/1820103
    
    MNIST: http://yann.lecun.com/exdb/mnist/
    
    MNIST机器学习入门: http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
"""

"""
MNIST 数据 

首先准备数据（MNIST库）

数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。
"""
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
minist = input_data.read_data_sets(common_header.MINIST_FILE2, one_hot=True)


# 创建层
def add_layer(inputs, in_size, out_size, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b;
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


"""
搭建网络 
"""
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
"""
每张图片都表示一个数字，所以我们的输出是数字0到9，共10类。
"""
ys = tf.placeholder(tf.float32, [None, 10])

"""
调用add_layer函数搭建一个最简单的训练网络结构，只有输入层和输出层。
"""
# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

"""
其中输入数据是784个特征，输出数据是10个特征，激励采用softmax函数，网络结构图是这样子的
https://morvanzhou.github.io/static/results/tensorflow/5_01_2.png
"""

"""
Cross entropy loss

loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
train方法（最优化算法）采用梯度下降法。
"""
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


"""
训练 
现在开始train，每次只取100张图片，免得数据太多训练太慢。
"""
for i in range(1000):
    # 分批，每个批次100条数据

    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    """
    每训练50次输出一下预测精度
    """
    if i % 50 == 0:
        print(compute_accuracy(minist.test.images, minist.test.labels))
