#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_构建一个非常简单的模型的示例.py
# @Date    :   2018/7/6
# @Desc    :

import common_header

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model

# load the diabetes dataset
diabetes = datasets.load_diabetes()
# print(len(diabetes))

# Use only one feature
diabetes_x = diabetes.data[:, np.newaxis, 2]
# print(diabetes_x)

# Split the data into training/testing sets 分割数据为训练集和测试集(特征值)
# 0-末尾-20条为训练集数据
diabetes_X_train = diabetes_x[:-20]
# 末尾20条为测试集数据
diabetes_X_test = diabetes_x[-20:]

# Split the targets into training/testing sets 拆分训练和测试集(目标值)
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

# Create linear regression object 创建一个线性回归对象
# A model is built based on Linear regression algorithm 建立基于回归算法的模型
regr = linear_model.LinearRegression()

# Train the model using the training sets 使用训练集数据训练模型
# The modl is trained using the training dataset 使用训练集训练模型
regr.fit(diabetes_X_train, diabetes_Y_train)

# The coefficients
# print("Coefficients: \n ", regr.coef_)
# 预测，得出准确率
regr_predict = regr.predict(diabetes_X_test)
# print("regr_predict: \n ", regr_predict)
# The mean squared error 均方误差
# 训练一个模型的目的是在所有例子中找到一组平均损失低的参数。
print("Mean squared error: %.2f" % np.mean((regr_predict - diabetes_Y_test) ** 2))

# Explained variance score: 1 is prefect prediction 解释方差评分：1是精确预测。
# 得到x和y之间的关系
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_Y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, regr_predict, color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.savefig('./data/模型回归示例.png')
# plt.show()
