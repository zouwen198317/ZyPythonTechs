#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_索引操作.py
# @Date    :   2018/7/2
# @Desc    :   索引操作

import common_header

import numpy as np

# indexing
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# [-2:,1:3] -2表示倒数第2行，1:3 表示从第一列开始到第3列结束,不包含第3列
print(a[-2:, 1:3])
'''
[[ 6  7]
 [10 11]]
'''

print(a[1, -2])

print(a.shape)

# a数组的每行的第二个元素+10 (三种实现方式)
a[np.arange(3), 1] += 10
print(a)

a[np.arange(3), [1, 1, 1]] += 10
print(a)

a[[0, 1, 2], [1, 1, 1]] += 10
print(a)

# np.arange(3)会产生一个数组:[0 1 2]
# np.arange 产生一个指定范围的数组
print(np.arange(3))
print(np.arange(3, 7))

# 判断数组中所有>10的元素
result_index = a > 10
print(a[result_index])

print(a[a > 10])
