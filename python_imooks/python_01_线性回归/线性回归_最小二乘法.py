#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   线性回归_最小二乘法.py
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
theta = dot(dot(inv(dot(X.T, X)), X.T), Y)
print("theta = ", theta)
