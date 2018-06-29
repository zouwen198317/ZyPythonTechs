#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   5.4 Tensorflow可视化.py
# @Date    :   2018/6/29
# @Desc    :

import common_header

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# 载入数据
mnist = input_data.read_data_sets(common_header.MINIST_FILE, one_hot=True)

# 运行次数
max_steps = 1001

# 图片数量
img_num = 3000

# 定义会话
sess = tf.Session()

# 载入图片(3000张图片打包)
embedding = tf.Variable(tf.stack(mnist.test.images[:img_num]), trainable=False, name='embedding')

projector_projector = common_header.ROJECTOR_DIR

metadata_file_name = projector_projector + "\metadata.tsv"


# 参数概要,计算各种数据值
def variable_summary(var):
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


# 定义两个placeholder
with tf.name_scope('input'):
    # None 表示第一个维度可以是任意的长度
    # 特征值
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 目标值
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片
with tf.name_scope('input_reshape'):
    # 784转换为28*28,维度是1(黑白是1，彩色是-1)
    # 一张图片是28*28,784是28*28像素
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    # 放进去10张图片
    tf.summary.image('input', image_shaped_input, 10)

# 创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('Weight'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summary(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summary(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        # 将输出的信号转化为概率值
        prediction = tf.nn.softmax(wx_plus_b)

# 采用交叉熵代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

# 使用梯度梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_variables()
else:
    init = tf.global_variables_initializer()

# 结果存放到一个布尔类型的列表中，生成1*100的布尔矩阵
# argmax返回一维张量中最大的值所在的位置
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
# 检查是否已经存在该文件，存在则删除
if tf.gfile.Exists(metadata_file_name):
    tf.gfile.DeleteRecursively(metadata_file_name)

# 参数w，文件不存在则会创建
with open(metadata_file_name, 'w') as f:
    # 获取测试集标签label
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    # 把3000张图片的label存入当前的文件中，每一行是一个标签
    for i in range(img_num):
        f.write(str(labels[i]) + '\n')

# 合并所有的summary
merge = tf.summary.merge_all()

# 读取图结构
projector_writer = tf.summary.FileWriter(common_header.ROJECTOR_DIR, sess.graph)
# 保存项
saver = tf.train.Saver()
# 配置项
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = metadata_file_name
embed.sprite.image_path = common_header.ROJECTOR_DATA + '\mnist_10k_sprite.png'

# 切片图片
embed.sprite.single_image_dim.extend([28, 28])
# 可视化
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    # 每个批次100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 固定配置写法
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # 把上面两个配置写入训练运行函数的options中
    summary, _ = sess.run([merge, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)

    # 记录参数
    projector_writer.add_run_metadata(run_metadata, 'step%30d' % i)
    projector_writer.add_summary(summary, i)

    # 每训练100次打印一次准确率
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

saver.save(sess, common_header.ROJECTOR_DIR + '\a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
