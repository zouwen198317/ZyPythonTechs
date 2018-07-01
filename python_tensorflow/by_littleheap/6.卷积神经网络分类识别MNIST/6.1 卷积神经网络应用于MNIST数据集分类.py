# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   6.1 卷积神经网络应用于MNIST数据集分类.py
# @Date    :   2018/6/30
# @Desc    :

from tensorflow.examples.tutorials.mnist import input_data

import common_header
import tensorflow as tf

# one_hot为onhot编码
minist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = minist.train.num_examples // batch_size  # 整除


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 标准差
        tf.summary.scalar('stddev', stddev)
        # 最大值
        tf.summary.scalar('max', tf.reduce_max(var))
        # 最小值
        tf.summary.scalar('min', tf.reduce_min(var))
        # 直方图
        tf.summary.histogram('histogram', var)


# 初始化权值
def weight_variable(shape, name):
    # 生成一个截断的正态分布，标准差为0.1
    initial = tf.truncated_normal(shape, stddev=-.1)
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    '''
    # x是一个四维的tensor [batch, in_height, in_width, in_channels] 1.批次 2.图片高 3.图片宽 4.通道数：黑白为1，彩色为3
    # W是一个滤波器/卷积核 [filter_height, filter_width, in_channels, out_channels] 1.滤波器高 2.滤波器宽 3.输入通道数
    4.输出通道数
    # 固定 strides[0] = strides[3] = 1， strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    '''
    # x是一个四维的tensor [batch, in_height, in_width, in_channels] 1.批次 2.图片高 3.图片宽 4.通道数：黑白为1，彩色为3
    # ksize是窗口大小 [1,x,y,1] , 固定ksize[0] = ksize[3] = 1 ,  ksize[1]代表x方向的大小 , ksize[2]代表y方向的大小
    # 固定 strides[0] = strides[3] = 1， strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式为2维的向量[batch,in_height,in_width,in_channels] 1.批次 2.二维高 3.二维宽 4.通道数：黑白为1，彩色为3
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        # 5*5的采样窗口，32个卷积核从1个平面抽取特征
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    with tf.name_scope('b_conv1'):
        # 每一个卷积核一个偏置值
        b_conv1 = bias_variable([32], name='b_conv1')

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        # 5*5的采样窗口，64个卷积核从32个平面抽取特征
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    with tf.name_scope('b_conv2'):
        # 每一个卷积核一个偏置值
        b_conv2 = bias_variable([64], name='b_conv2')

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的平面
with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        # 输入层有7*7*64个列的属性，全连接层有1024个隐藏神经元
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
    with tf.name_scope('b_fc1'):
        # 1024个节点
        b_fc1 = bias_variable([1024], name='b_fc1')

    # 把池化层2的输出扁平化为1维，-1代表任意值
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # Dropout处理，keep_prob用来表示处于激活状态的神经元比例
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc2'):
        # 输入层有7*7*64个列的属性，全连接层有1024个隐藏神经元
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        # 1024个节点
        b_fc2 = bias_variable([10], name='b_fc2')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用梯度梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        # argmax返回一维张量中最大的值所在的位置
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        # 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
# merge = tf.summary.merge_all()

with tf.Session() as sess:
    # 初始化变量
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_variables()
    else:
        init = tf.global_variables_initializer()

    sess.run(init)
    # 将图写入指定目录
    # writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(5):
        for batch in range(n_batch):  # 遍历所有数据集来进行训练
            batch_xs, batch_ys = minist.train.next_batch(batch_size)
            # 此处可以更改dropout值
            # summary, _ = sess.run([merge, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6})
            summary = sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6})

        # writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: minist.test.images, y: minist.test.labels, keep_prob: 1.0})
        print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))
# tensorboard --logdir=log
