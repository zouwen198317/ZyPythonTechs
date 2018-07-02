#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_05_广播.py
# @Date    :   2018/7/2
# @Desc    :

import common_header

import numpy as np

a = np.array([[1, 2, 3], [2, 3, 4], [12, 31, 45], [34, 43, 9]])
b = np.array([1, 2, 3])

# a每一行和b相加: 使用循环计算很慢,当矩阵值很大时就很慢
for i in range(4):
    a[i, :] += b
print(a)

a1 = a + np.tile(b, (4, 1))
print(' a1 = ', a1)

# 这就是广播，会在缺失位置为1的维度上进行
a2 = a + b
print(' a2 = ', a2)
