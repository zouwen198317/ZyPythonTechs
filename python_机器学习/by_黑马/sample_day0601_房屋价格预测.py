#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_day0601_房屋价格预测.py
# @Date    :   2018/7/18
# @Desc    :

import common_header

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=[10, 10])

# 散点图
# x 是二维的,可以是一维的也可以是二维的
plt.scatter([60, 72, 75, 80, 83, 87, 90, 93], [126, 151.2, 157.5, 168, 174.3, 180, 192.2, 194])
plt.xticks(np.linspace(0, 100, 10, endpoint=True))  # 设置x轴刻度
plt.yticks(np.linspace(0, 200, 10, endpoint=True))  # 设置y轴刻度
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ylabel("房子价格(w)")
plt.xlabel("房子面积(m2)")
plt.savefig("imgs/test.png")
# plt.show()
