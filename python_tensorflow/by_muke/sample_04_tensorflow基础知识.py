# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
tensorflow基础知识
https://blog.csdn.net/a18852867035/article/details/53897102
"""

"""
tensorflow的运行流程主要有2步，分别是构造模型和训练。 
在这之前，讲几个概念： 
tensorflow中的有几概念:Tensor,Variable,placeholder,session 
"""

# 1.概念
# -----------------     1.1 Tensor -----------------
import tensorflow as tf

a = tf.zeros(shape=[1, 2])

"""
不过要注意，因为在训练开始前，所有的数据都是抽象的概念，也就是说，此时a只是表示这应该是个1*5的零矩阵，而没有实际赋值，
也没有分配空间，所以如果此时print,就会出现如下情况:
"""
print("未训练的时候a的值 - > ", a)

# 只有在训练过程开始后，才能获得a的实际值
sess = tf.InteractiveSession()
print("训练的时候a的实际值 - > ", sess.run(a))
# -----------------     1.1 Tensor -----------------
# 关闭会话
sess.close()

# -----------------     1.2 Tensor -----------------
"""
1.2 Variable 
故名思议，是变量的意思。一般用来表示图中的各计算参数，包括矩阵，向量等。例如，我要表示下图中的模型，那表达式就是 
y=Relu(Wx+b) 

relu是一种激活函数，具体可见这里）这里W和b是我要用来训练的参数，那么此时这两个值就可以用Variable来表示。Variable的初始函数有很多其他选项，这里先不提，只输入一个Tensor也是可以的。

W = tf.Variable(tf.zeros(shape=[1,2]))
1
注意，此时W一样是一个抽象的概念，而且与Tensor不同，Variable必须初始化以后才有具体的值。
"""
tensor = tf.zeros(shape=[1, 2])
variable = tf.Variable(tensor)
sess = tf.InteractiveSession()
# 初始化所有的变量 global_variables_initializer 不初始化会报错
sess.run(tf.global_variables_initializer())
print("variable = > ", sess.run(variable))
# 关闭会话
sess.close()

# -----------------     1.2 Tensor -----------------


"""
1.3 placeholder

又叫占位符，同样是一个抽象的概念。用于表示输入输出数据的格式。告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，不过我正式运行的时候会补上的！例如上式中的x和y。因为没有具体数值，所以只要指定尺寸即可。

x = tf.placeholder(tf.float32,[1, 5],name='input')
y = tf.placeholder(tf.float32,[None, 5],name='input')
上面有两种形式，第一种x，表示输入是一个[1,5]的横向量。 
而第二种形式，表示输入是一个[?,5]的矩阵。那么什么情况下会这么用呢?就是需要输入一批[1,5]的数据的时候。比如我有一批共10个数据，那我可以表示成[10,5]的矩阵。如果是一批5个，那就是[5,
5]的矩阵。tensorflow会自动进行批处理。

1.4 Session 
session，也就是会话。我的理解是，session是抽象模型的实现者。为什么之前的代码多处要用到session
？因为模型是抽象的嘛，只有实现了模型以后，才能够得到具体的值。同样，具体的参数训练，预测，甚至变量的实际值查询，都要用到session,看后面就知道了 
"""

# 2.模型构建

"""
这里我们使用官方tutorial中的mnist数据集的分类代码，公式可以写作
z=Wx+ba=softmax(z)

那么该模型的代码描述为：
"""
# -----------------     2 模型构建 -----------------
# 输入占位符
x = tf.placeholder(tf.float32, [None, 784])
# 输出占位符(预期输出)
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 模型的实际输出
a = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数和训练方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
# 梯度下降法,学习速率为0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练目标:最小化损失函数
train = optimizer.minimize(cross_entropy)

"""
可以看到这样以来，模型中的所有元素(图结构，损失函数，下降方法和训练目标)都已经包括在train里面。我们可以把train叫做训练模型。
那么我们还需要测试模型
"""
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))

# accuracy 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后看看他们是否一致，是就返回true,不是就返回false,
这样得到一个boolean数组。tf.cast将boolean数组转成int数组，最后求平均值，得到分类的准确率(怎么样，是不是很巧妙)
"""

# 实际训练
# train 训练模型
# 测试模型

sess = tf.InteractiveSession()
# 初始化所有的变量
tf.global_variables_initializer().run()

# Import data
from tensorflow.examples.tutorials.mnist import input_data
# 以下示例代码运行不了mnist
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)  # 获得一批100个数据
#     train.run({x: batch_xs, y: batch_ys})  # 给训练模型提供输入和输出
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

"""
可以看到，在模型搭建完以后，我们只要为模型提供输入和输出，模型就能够自己进行训练和测试了。中间的求导，求梯度，反向传播等等繁杂
的事情，tensorflow都会帮你自动完成。
"""
# -----------------     2 模型构建 -----------------

# 以下示例代码运行不了
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# # Import data
# from tensorflow.examples.tutorials.mnist import input_data
#
# import tensorflow as tf
#
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')  # 把数据放在/tmp/data文件夹中
#
# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  # 读取数据集
#
# # 建立抽象模型
# x = tf.placeholder(tf.float32, [None, 784])  # 占位符
# y = tf.placeholder(tf.float32, [None, 10])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# a = tf.nn.softmax(tf.matmul(x, W) + b)
#
# # 定义损失函数和训练方法
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
# optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降法，学习速率为0.5
# train = optimizer.minimize(cross_entropy)  # 训练目标：最小化损失函数
#
# # Test trained model
# correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # Train
# sess = tf.InteractiveSession()  # 建立交互式会话
# tf.initialize_all_variables().run()
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     train.run({x: batch_xs, y: batch_ys})
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
