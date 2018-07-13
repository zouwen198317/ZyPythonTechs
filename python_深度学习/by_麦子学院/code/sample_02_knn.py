#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_knn.py
# @Date    :   2018/7/13
# @Desc    :

import common_header

import math

# 计算两个点的距离
def CalcEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d
    pass


d_ag = CalcEuclideanDistance(3, 104, 18, 90)
print("d_ag = ", d_ag)
