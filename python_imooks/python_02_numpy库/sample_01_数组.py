#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_数组.py
# @Date    :   2018/7/2
# @Desc    : 数组

import common_header

# rank 表示数组的维度

import numpy as np

a = np.array([1, 2, 3])
# 查看数组的类型
# print(type(a))
# 查看数组的维度
# print(a.shape)

# 明确指定数组是一个行向量或列向量
# [行，列] : 有1个为-1，表示为自动
a = a.reshape([1, -1])
# print(a.shape)

a = np.array([1, 2, 3, 4, 5, 6])
# print(a.shape)
a = a.reshape([2, -1])
# print(a.shape)

a = a.reshape([-1, 3])
# print(a.shape)

# print(a[1][2])
a[1][2] = 55

# print(a)

# zeros 创建元素全部为0的数组
a = np.zeros((3, 3))
# print(a)

# one 创建元素全部为1的数组
a = np.ones([2, 3])
# print(a)

# one和zeros全部可以用full
# full
a = np.full((3, 3), 0)
# print(a)

# eve 创建左上角和右下角元素为1，其它元素为0的矩阵,单位矩阵
a = np.eye(3)
# print(a)


# random.random 创建随机矩阵,(0-1)
a = np.random.random((3, 4))
print(a)
