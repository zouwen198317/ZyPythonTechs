# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   sample_05_支持向量机（SVM）线性可分解决实例.py
# @Date    :   2018/7/14
# @Desc    :

import common_header

# 首先产生数据集
# 训练数据集
# 画图展示超平面
import numpy as np
import pylab as pl
from sklearn import svm

# we create 40 separable points
# 正太分布产生，均值分别为-2 2 可以线性可分
np.random.seed(0)

X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
# 权值
w = clf.coef_[0]
# 直线斜率
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors
# 前面的点位于左下角
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
# 后面的点位于右上角
b = clf.support_vectors_[1]
yy_up = a * xx + (b[1] - a * b[0])

print("w: ", w)
print("a: ", a)

print("xx: ", xx)
print("yy: ", yy)

# plot the line,the points ,and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=90, edgecolors='red')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
pl.axis('tight')
pl.savefig("data/SVM-线性可分.png")
pl.show()
