#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   Sample_02_KNN.py
# @Date    :   2018/7/19
# @Desc    :

import common_header
import kNN

import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.cretaeDataSet()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group, labels)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
