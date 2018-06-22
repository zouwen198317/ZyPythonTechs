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

from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
minist = input_data.read_data_sets('../by_github_LittleHeap/03-非线性回归_简单神经网络识别MNIST样本/MNIST_data', one_hot=True)


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


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

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

for i in range(1000):
    # 分批，每个批次100条数据
    batch_xs, batch_ys = minist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(minist.test.images, minist.test.labels))
