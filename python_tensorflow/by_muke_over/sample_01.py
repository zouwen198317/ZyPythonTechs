# -*- coding:UTF-8 -*-

"""
资料地址
https://www.imooc.com/article/22279?block_id=tuijian_wz

"""
"""
计算图
你可能会想到TensorFlow核心程序由两个独立的部分组成：
构建计算图。
运行计算图。
甲计算图形是一系列排列成节点的图形TensorFlow操作。我们来构建一个简单的计算图。每个节点将零个或多个张量作为输入，并产生张量作
为输出。一种类型的节点是一个常量。像所有的TensorFlow
常量一样，它不需要输入，而是输出一个内部存储的值
"""

import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

"""
打印节点不会输出值3.0，4.0正如您所期望的那样。相反，它们是在评估时分别产生3.0和4.0的节点。为了实际评估节点，我们必须在会话中运
行计算图。会话封装了TensorFlow运行时的控制和状态。
"""
print(node1, node2)

# 创建一个Session对象，然后调用它的run方法来运行足够的计算图来评估node1和node2
sess = tf.Session()
print("预期值 => ", sess.run([node1, node2]))

# 我们可以通过组合Tensor节点和操作来构建更复杂的计算（操作也是节点）。例如，我们可以添加我们的两个常量节点，并产生一个新的图形如下：
node3 = tf.add(node1, node2)
print("node3:", node3)
print("node3 => ", sess.run(node3))

# 图形可以被参数化来接受称为占位符的外部输入。一个占位符是一个承诺后提供一个值。

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

"""
前面的三行有点像一个函数或lambda，我们在其中定义两个输入参数（a和b），然后对它们进行操作。我们可以通过使用run方法的feed_dict
参数将多个输入的 具体值提供给占位符来评估这个图形。
"""
print("adder_node => ", sess.run(adder_node, {a: 3, b: 4.5}))
print("adder_node => ", sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 我们可以通过添加另一个操作来使计算图更加复杂
add_and_triple = adder_node * 3
print("add_and_triple = > ", sess.run(add_and_triple, {a: 3, b: 4.5}))

"""
在机器学习中，我们通常需要一个可以进行任意输入的模型，比如上面的模型。为了使模型可训练，我们需要能够修改图形以获得具有相同输入
的新输出。 变量允许我们将可训练参数添加到图形中。它们被构造成一个类型和初始值
"""
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

"""
常量在调用时被初始化tf.constant，其值永远不会改变。相比之下，变量在调用时不会被初始化tf.Variable。要初始化TensorFlow程序中的
所有变量，您必须显式调用一个特殊的操作
"""
init = tf.global_variables_initializer()
sess.run(init)

"""
实现initTensorFlow子图的一个句柄是初始化所有的全局变量，这一点很重要。在我们调用之前sess.run，变量是未初始化的。
既然x是占位符，我们可以同时评估linear_model几个值
"""
print("linear_model = > ", sess.run(linear_model, {x: [1, 2, 3, 4]}))

"""
我们已经创建了一个模型，但我们不知道它有多好。为了评估培训数据模型，我们需要一个y占位符来提供所需的值，我们需要编写一个损失函数。
损失函数用于衡量当前模型距离提供的数据有多远。我们将使用线性回归的标准损失模型，它将当前模型和提供的数据之间的三角形的平方相加。
linear_model - y创建一个向量，其中每个元素是相应示例的错误增量。我们打电话tf.square来解决这个错误。然后，我们总结所有的平方误差来创建一个标量，
它使用下面的方法来抽象所有例子的错误tf.reduce_sum
"""
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print("产生损失价值: ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

"""
我们可以手动重新分配的值提高这W和b为-1和1变量的值，完美初始化为提供的价值tf.Variable，但可以使用操作等来改变tf.assign。例如，
 W=-1并且b=1是我们模型的最佳参数
"""
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print("fix loss => ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

"""
TensorFlow提供了优化器，可以逐渐改变每个变量，以最大限度地减少损失函数。最简单的优化器是梯度下降。它根据相对于该变量的损失导数
的大小来修改每个变量。一般来说，手动计算符号派生是繁琐和容易出错的。因此，TensorFlow可以自动生成衍生产品，仅使用该函数对模型进行描述tf.gradients。为了简单起见，优化程序通常会为您执行此操作
"""
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("最终的模型参数 => ", sess.run([W, b]))

"""
tf.estimator 是一个高级的TensorFlow库，它简化了机器学习的机制，包括以下内容：
运行训练循环
运行评估循环
管理数据集
tf.estimator定义了许多常见的模型。
基本用法
注意线性回归程序变得简单多了 tf.estimator：
"""
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)

"""
评估数据是如何有更高的损失，但仍然接近于零
"""

"""
自定义模型
tf.estimator不会将您锁定在预定义的模型中。假设我们想创建一个没有内置到TensorFlow中的自定义模型。我们仍然可以保留数据集，喂养，
培训等的高层次抽象 tf.estimator。为了说明，我们将展示如何实现我们自己的等价模型，以LinearRegressor使用我们对低级别TensorFlow
 API的知识。
要定义一个适用的自定义模型tf.estimator，我们需要使用 tf.estimator.Estimator。tf.estimator.LinearRegressor实际上是一个子类
tf.estimator.Estimator。Estimator我们只是简单地提供Estimator一个函数model_fn来说明 tf.estimator如何评估预测，训练步骤和损失，而不是分类
"""

# 代码见自定义模型.py