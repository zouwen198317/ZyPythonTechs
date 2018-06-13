# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
用TensorFlow玩转Kaggle的“手写识别

用TensorFlow实现一个手写数字识别功能，来展示TensorFlow如何用神经网络实现对图片的识别。google也为入门者提供了一个这样的例子，
也就是TensorFlow里的“hello world”，这个例子的名字叫“MNIST”
'''

"""
手写图片识别的实现，分为三步：

一、数据的准备

二、模型的设计

三、代码实现
"""

'''
1 载入数据，并对数据进行处理

在写代码的过程中，数据的预处理是最大的一块工作，做一个项目，60%以上的代码在做数据预处理。

这个项目的预处理，分为5步：

    1）把输入和结果分开

    2）对输入进行处理：把一维的输入变成28*28的矩阵

    3）对结果进行处理：把结果进行One-Hot编码

    4）把训练数据划分训练集和验证集

    5）对训练集进行分批
'''
# 1.加载数据，把对输入和结果进行分开
import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv("train.csv")
images = train.iloc[:, 1:].values
lables_flat = train['label'].values.ravel()

# 2。对输入进行处理
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print("输入数据的数量:(%g,%g)" % images.shape)

image_size = images.shape[1]
print("输入数据的维度=> {0}".format(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print("图片的长 => {0} \n图片的高 => {1}".format(image_width, image_height))

x = tf.placeholder('float', shape={None, image_size})


# 3. 对结果进行处理
lables_count = np.unique(lables_flat).shape[0]
print("结果的种类 => {0} ".format(lables_count))

y = tf.placeholder('float', shape=[None, lables_count])

# 进行One-hot编码
def dense_to_one_hot(lables_dense, num_classes):
    num_labels = lables_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    lables_one_hot = np.zeros((num_labels, num_classes))
    lables_one_hot.flat[index_offset + lables_dense.ravel()] = 1
    return lables_one_hot


labels = dense_to_one_hot(lables_flat, lables_count)
labels = labels.astype(np.uint8)

print("结果的数量: ({0[0]},{0[1]})".format(labels.shape))

# 4 把输入数据划分训练集和验证集

# 把40000个数据作为训练集，2000个数据作为验证集
VALIDATION_SIZE = 2000

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
# 5. 对训练集进行分批
batch_size = 100
n_batch = len(train_images) / batch_size

# 数据预处理好了，如果不需要显示结果，可以把Print语句都去掉，不影响建模。

#  建立神经网络，设置损失函数，设置梯度下降的优化参数
# 6 创建一个简单的神经网络来对图片进行识别
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x, weights) + biases
prediction = tf.nn.softmax(result)

# 7. 创建损失函数,以交叉熵的平均值为衡量
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 8. 使用梯度下降法优化参数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 3 初始化变量，设置好准确度的计算方法，在Session中运行
# 9. 初始化变量
init = tf.global_variables_initializer()

# 10 计算准确度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 循环50轮
    for epoch in range(50):
        for batch in range(n_batch):
            # 按照分片取出数据
            batch_x = train_images[batch * batch_size:(batch + 1) * batch_size]
            batch_y = train_labels[batch * batch_size:(batch + 1) * batch_size]
            # 进行训练
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        # 每一轮计算一次准确度
        accuracy_n = sess.run(accuracy, feed_dict={x: validation_images, y: validation_labels})
        print("第: " + str(epoch + 1) + "轮,准确度为:" + str(accuracy_n))
