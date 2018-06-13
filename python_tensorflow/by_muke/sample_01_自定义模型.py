# -*- coding:UTF-8 -*-

"""
自定义模型
tf.estimator不会将您锁定在预定义的模型中。假设我们想创建一个没有内置到TensorFlow中的自定义模型。我们仍然可以保留数据集，喂养，培训
等的高层次抽象
tf.estimator。为了说明，我们将展示如何实现我们自己的等价模型，以LinearRegressor使用我们对低级别TensorFlow API的知识。
要定义一个适用的自定义模型tf.estimator，我们需要使用 tf.estimator.Estimator。tf.estimator.LinearRegressor实际上是一个子类
tf.estimator.Estimator。Estimator我们只是简单地提供Estimator一个函数model_fn来说明 tf.estimator如何评估预测，训练步骤和损失，而不是分类 。
"""

import numpy as np
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Declare list of features,we only have one real-valued feature
def model_fn(features, labels, mode):
    # Build a line model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Trainning sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # EstimatorSpec connects subgraphs we built to the appropriate funcationlity
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train
    )


estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
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

# train 训练
estimator.train(input_fn=input_fn, steps=1000)

# Here we eveluate how well our model did 对模型进行评估
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
