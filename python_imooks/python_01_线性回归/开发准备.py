#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   开发准备.py
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

# ----------------   矩阵   ----------------
# 1*2
A = np.mat([1, 1])
print('A:\n', A)

# 矩阵转置: [1,1] ===> [[1][1]]  从一个行向量转换为一个列向量
# print('A.T:\n', A.T)
print('A.reshape \n ', A.reshape(2, 1))
print('----------------')

# ----------------   数组   ----------------
B = np.array([1, 1])
print('B:\n', B)

'''
A:
 [[1 1]]
B:
 [1 1]
'''
print('----------------')
# 多行矩阵
# 2*3
C = mat([[1, 3], [2, 4]])
print('C的逆:\n', inv(C))
# 第一行所有列
print('第一行所有列\n', C[0, :])
# 第一列所有行
print('第一列所有行\n', C[:, 0])
print('C.reshape \n ', C.reshape(1, 4))

# 矩阵点乘
# A: 1*2,C 2*2
D = dot(A, C)
print('D:\n', D)

# C * A 的时候，A要转置要不然会报错
D1 = dot(C, A.T)
print('D1:\n', D1)
