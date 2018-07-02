#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   线性回归实践.py
# @Date    :   2018/7/2
# @Desc    : 没有数据集暂时无法验证

import common_header

import numpy as np

from numpy.linalg import inv
from numpy import dot
from numpy import mat
import pandas as pd

dataset = pd.read_csv('data.csv')

temp = dataset.iloc[:, 2:5]
temp['x0'] = 1
X = temp.iloc[:, [3, 0, 1, 2]]
Y = dataset.iloc[:, 1].values.reshape(150, 1)

theta = dot(dot(inv(dot(X.T, X)), X.T), Y)

theta = np.array([1., 1., 1., 1.]).reshape(4, 1)
alpha = 0.1

temp = theta

X0 = X.iloc[:, 0].values.reshape(150, 1)
X1 = X.iloc[:, 1].values.reshape(150, 1)
X2 = X.iloc[:, 2].values.reshape(150, 1)
X3 = X.iloc[:, 3].values.reshape(150, 1)

for i in range(1000):
    temp[0] = theta[0] + alpha.np.sum((Y - dot(X, theta)) * X0) / 150.
    temp[1] = theta[1] + alpha.np.sum((Y - dot(X, theta)) * X1) / 150.
    temp[2] = theta[2] + alpha.np.sum((Y - dot(X, theta)) * X2) / 150.
    temp[3] = theta[3] + alpha.np.sum((Y - dot(X, theta)) * X3) / 150.
    theta = temp

print(theta)
