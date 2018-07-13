#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_iris.py
# @Date    :   2018/7/13
# @Desc    :

import common_header

from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()

# save data
# f = open('iris.data.csv', 'wb')
# f.write(str(iris))
# f.close()

print(iris)

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print("predictedLabel = " + str(predictedLabel))
