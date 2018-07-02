#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   线性回归_梯度下降.py
# @Date    :   2018/7/2
# @Desc    :

import common_header

import numpy as np

# 求矩阵的逆
from numpy.linalg import inv
# 求矩阵的点乘
from numpy import dot
# 矩阵
from numpy import mat

#  y = 2X
# 1*3
# 需要reshape否则会出现问题
X = mat([[1, 2, 3]]).reshape(3, 1)
Y = 2 * X

# theta = (X'x)^-1X'Y
# theta = dot(dot(inv(dot(X.T, X)), X.T), Y)

# 使函数能够快速的收敛
# theta = theta-alpha*(theta*X-Y)*x
theta = 1.
alpha = 0.1

for i in range(100):
    # sum 加权求平均
    theta = theta + np.sum(alpha * (Y - dot(X, theta)) * X.reshape(1, 3)) / 3.0
print("theta = ", theta)
