#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_day0602_矩阵和数组的区别.py
# @Date    :   2018/7/18
# @Desc    :

import common_header

import numpy as np

a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
b = [2, 2, 2, 2]
result = np.multiply(a, b)
b = [[2], [2], [2], [2]]
print("数组运算结果 =  ", result)
# 矩阵必须是二维的
result = np.dot(a, b)
print("矩阵运算结果 =  ", result)
