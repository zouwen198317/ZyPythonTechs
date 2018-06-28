from __future__ import division, print_function, absolute_import

import common_header
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import MINIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(common_header.MINIST_FILE2, one_hot=False)

"""
Autoencoder 

"""
# Visualize decoder setting
# parameters
learning_rate = 0.01
tarning_epochs = 5  # 五组训练
batch_size = 256
display_step = 1
example_to_show = 10

"""
我们的MNIST数据，每张图片大小是 28x28 pix，即 784 Features：
"""
# Network Parameters
n_input = 784  # Mnist data input: (img shape:28*28)

# tf graph input(only pictures)
X = tf.placeholder(tf.float32, [None, n_input])

""""
在压缩环节：我们要把这个Features不断压缩，经过第一个隐藏层压缩至256个 Features，再经过第二个隐藏层压缩至128个。
在解压环节：我们将128个Features还原至256个，再经过一步还原至784个。
在对比环节：比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，根据 cost 来提升我的 Autoencoder 的准确率，
下图是两个隐藏层的 weights 和 biases 的定义：
"""
# hidden layer settings
n_hidden_1 = 256  # 1lst layer num features
n_hidden_2 = 128  # 2lst layer num features
weights = {
    "encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    "encoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

    "decoder_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    "decoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    "encoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "encoder_b2": tf.Variable(tf.random_normal([n_hidden_2])),

    "decoder_b1": tf.Variable(tf.random_normal([n_hidden_1])),
    "decoder_b2": tf.Variable(tf.random_normal([n_input])),
}

"""
下面来定义 Encoder 和 Decoder ，使用的 Activation function 是 sigmoid， 压缩之后的值应该在 [0,1] 这个范围内。
在 decoder 过程中，通常使用对应于 encoder 的Activation function：
"""


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


"""
来实现 Encoder 和 Decoder 输出的结果：
"""
# Construct model
encoder_op = encoder(X)  # 128 Features
decoder_op = decoder(encoder_op)  # 784 Features

# Prediction
y_pred = decoder_op
# Targets(Lables) are the input data
y_true = X

"""
再通过我们非监督学习进行对照，即对 “原始的有 784 Features 的数据集” 和 “通过 ‘Prediction’ 得出的有 784 Features 的数据集” 
进行最小二乘法的计算，并且使 cost 最小化:
"""
# Defind loss and optimizer ,minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

"""
最后，通过 Matplotlib 的 pyplot 模块将结果显示出来， 注意在输出时MNIST数据集经过压缩之后 x 的最大值是1，而非255：
"""
# Lanuch the graph
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_variables()
    else:
        init = tf.global_variables_initializer()

    sess.run(init)
    # 数据分页加载
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(tarning_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # 分批数据的特征值和目标值
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x)=0
            # Run optimization op(backprop) and cost op(to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs pre epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), 'cost=', "{:.9f}".format(c))

    print("Optimization Finished!")

    ## Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:example_to_show]}
    )
    # Compare original images with their reconstrutions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()
