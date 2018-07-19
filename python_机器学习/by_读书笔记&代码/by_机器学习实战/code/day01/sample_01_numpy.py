#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_numpy.py
# @Date    :   2018/7/19
# @Desc    :

import common_header

from numpy import *

# 生成随机数组
randT = random.rand(4, 4)
print("randT = ", randT)

# mat将数组转换为矩阵
randMat = mat(randT)
print("randMat = ", randMat)

# .I操作符实现了矩阵求逆的运算
invRandMat = randMat.I
print("randMat.I = ", invRandMat)

# 矩阵乘法
myEye = randMat * invRandMat
print("myEye = ", myEye)
"""
结果应该是单位矩阵，除了对角线元素是1，4×4矩阵的其他元素应该全是0。实际输出结果略有
不同，矩阵里还留下了许多非常小的元素，这是计算机处理误差产生的结果
测试的结果和描述的有区别
"""
# 函数eye(4)创建4×4的单位矩阵
print("myEye = ", myEye - eye(4))
