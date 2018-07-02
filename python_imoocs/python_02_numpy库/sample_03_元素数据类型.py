#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_03_元素数据类型.py
# @Date    :   2018/7/2
# @Desc    :

import common_header

import numpy as np

# numpy会根据创建的数组中的元素，自动判断元素的类型

a1 = np.array([1, 2, 3])
print('a1.dtype = ', a1.dtype)
a2 = np.array([1.1, 2.2, 3.3])
print('a2.dtype = ', a2.dtype)
a3 = np.array([1, 2.2])
print('a3.dtype = ', a3.dtype)
a4 = np.array([1, 2.2, False])
print('a4.dtype = ', a4.dtype)

a3_1 = np.array([1, 2.2], dtype=np.int64)
print('a3_1.dtype = ', a3_1.dtype)
print(a3_1)
# False: 0 ,Ture: 1
a4_1 = np.array([1, 2.2, False], dtype=np.int64)
print('a4_1.dtype = ', a4_1.dtype)
print(a4_1)

a5 = np.array([1, 2.2, 'zzg'])
print('a5.dtype = ', a5.dtype)
a6 = np.array([1, 2.2, 'zzg', False])
print('a6.dtype = ', a6.dtype)

b = np.array(a2, dtype=np.int64)
print(b)
