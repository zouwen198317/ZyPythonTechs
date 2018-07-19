#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_KNN_01.py
# @Date    :   2018/7/18
# @Desc    :

import common_header

import numpy as np

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# 欧氏距离
resultA = np.linalg.norm(vec1 - vec2)
print(resultA)
resultB = np.sqrt(np.sum(np.square(vec1 - vec2)))
print(resultB)
print(resultA == resultB)
