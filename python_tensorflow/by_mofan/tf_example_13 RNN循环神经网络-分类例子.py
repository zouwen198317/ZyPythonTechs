from __future__ import print_function

import common_header

""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
"""

"""
设置 RNN 的参数 
这次我们会使用 RNN 来进行分类的训练 (Classification). 会继续使用到手写数字 MNIST 数据集. 让 RNN 从每张图片的第一行像素读到最后一行, 然后再进行分类判断. 
接下来我们导入 MNIST 数据并确定 RNN 的各种参数(hyper-parameters):
"""
import tensorflow as tf
from tensorflow.contrib import rnn

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# import MNIST data
# 导入数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("by_github_LittleHeap/03-非线性回归_简单神经网络识别MNIST样本/MNIST_data",
                                  one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters 训练参数
lr = 0.001  # learning rate
training_iters = 100000  # train step 上限
batch_size = 128

# Network Parameters
n_inputs = 28  # MNIST data input(img shape:28x28)
n_steps = 28  # 执行28次，即28行，因为执行一次，为一行28列的数据
n_hidden_units = 128  # hidden layer num of features 隐藏层128个特征 neurons in hidden layer
n_classes = 10  # MNIST totoal classes (0-9 digits) 10个分类

"""
接着定义 x, y 的 placeholder 和 weights, biases 的初始状况.
"""
# tf Graph input
# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
# 对 weights biases 初始值的定义
weights = {
    # shape (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

    # shape (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

"""
定义 RNN 的主体结构 
接着开始定义 RNN 主体结构, 这个 RNN 总共有 3 个组成部分 ( input_layer, cell, output_layer). 
首先我们先定义 input_layer:
"""


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch*28 steps,28 inputs)
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch*28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in = W*X + b
    # X_in = (128 batch*28 steps, 128 hidden)
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    """
    接着是 cell 中的计算, 有两种途径:

    使用 tf.nn.rnn(cell, inputs) (不推荐原因). 但是如果使用这种方法, 可以参考原因;
    使用 tf.nn.dynamic_rnn(cell, inputs) (推荐). 这次的练习将使用这种方式.
    
    因 Tensorflow 版本升级原因, state_is_tuple=True 将在之后的版本中变为默认.
     对于 lstm 来说, state可被分为(c_state, h_state).
    """
    # cell
    ########################################

    # 使用 basic LSTM Cell.
    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)

    # lstm cell is divided into two parts (c_state,h_state)
    # 初始化全零 state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    """
    如果使用tf.nn.dynamic_rnn(cell, inputs), 我们要确定 inputs 的格式. tf.nn.dynamic_rnn 中的 time_major 
    参数会针对不同 inputs 格式有不同的值.
    
    如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    """
    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks
    # /recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    """
    最后是 output_layer 和 return 的值. 因为这个例子的特殊性, 有两种方法可以求得 results.
    """
    # hidden layer for output as the final results
    #############################################
    # 方式一: 直接调用final_state 中的 h_state (final_state[1]) 来进行运算:
    results = tf.matmul(final_state[1], weights['out'] + biases['out'])

    # 方式二: 调用最后一个 outputs (在这个例子中,和上面的final_state[1]是一样的):
    # or
    # unpack to list [(batch, outputs)..] * steps

    # # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    # else:
    #     outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # 选取最后一个 output
    #
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
    return results


# 定义好了 RNN 主体结构后, 我们就可以来计算 cost 和 train_op:
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdadeltaOptimizer(lr).minimize(cost)

"""
训练 RNN 
训练时, 不断输出 accuracy, 观看结果:
"""
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1
