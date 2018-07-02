#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_04_数学运算与常用函数.py
# @Date    :   2018/7/2
# @Desc    :

import common_header

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [6, 5]])

print('a + b = ', a + b)
print('np .add ( a,b ) = ', np.add(a, b))

print('a - b = ', a - b)
print('np .subtract ( a,b ) = ', np.subtract(a, b))

print('a * b = ', a * b)
print('np .multiply ( a,b ) = ', np.multiply(a, b))

print('a / b = ', a / b)
print('np . divide( a,b ) = ', np.divide(a, b))

# 数组中的元素开方
print('np.sqrt(a)', np.sqrt(a))

# 矩阵的乘法，np.dot

b = np.array([[1, 2, 3], [4, 5, 6]])
# 行*列
# [1*1+2*4,1*2+2*5,1*3+2*6] ==>> [9,12,15]
# [3*1+4*4,3*2+4*5,3*3+4*6] ==>> [19,26,33]
print('a.dot b = ', a.dot(b))
print('np.dot(a,a) = ', np.dot(a, b))

# 常用函数
# sum  求和
a1 = np.sum(a)
# 数组中的所有的元素求和
print("a1 = ", a1)  # a1 =  10

# 数组中的每一列进行求和操作
np_sum = np.sum(a, axis=0)
print("np_sum = ", np_sum)  # np_sum =  [4 6]

# 对数组中的每一行进行求和
np_sum = np.sum(a, axis=1)
print("np_sum = ", np_sum)  # np_sum =  [3 7]

# mean 所有和的平均值
np_mean = np.mean(a)
print("np_mean = ", np_mean)  # (1+2+3+4)/4=2.5

# 数组中的每一列进行求和操作
np_mean = np.mean(a, axis=0)
print("np_sum = ", np_mean)  # np_sum =  [2. 3.]  = (1+3)/2,(2+4)/2

# 对数组中的每一行进行求和
np_mean = np.mean(a, axis=1)
print("np_sum = ", np_mean)  # np_sum =  [1.5 3.5]  = (1+2)/2,(3+4)/2

# uniform 生成一个指定范围的随机数值
# 随机小数
random_uniform = np.random.uniform(3, 4)
print("random_uniform = ", random_uniform)

random_uniform = np.random.uniform(1, 100)
print("random_uniform = ", random_uniform)

# tile 将一个数组做为一个元素，重复指定的次数
# 在列上重复2次
a = np.tile(a, (1, 2))
print(a)
# [[1 2 1 2]
# [3 4 3 4]]

# 在行上重复2次
a = np.tile(a, (2, 1))
print(a)
# [[1 2 1 2]
#  [3 4 3 4]
#  [1 2 1 2]
#  [3 4 3 4]]

# argsort 将数组中的元素进行排序
a = np.array([[3, 6, 4, 11], [54, 2, 33, 17]])
# print(a.argsort())  # 从小到大排列,返回的是元素的下标
# print(a.argsort(axis=0))  # 从大到小排列,返回的是元素的下标

# 矩阵的转置,行-》列，列-》行
print(a)
print(a.T)
print(np.transpose(a))
