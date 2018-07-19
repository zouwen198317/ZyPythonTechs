#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_KNN_IRIS.py
# @Date    :   2018/7/18
# @Desc    : 鸢尾花KNN示例

import common_header

"""
目标-学习到一个可用于鸢尾花属性类别的分类器
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets

# 目标分类: 0-setosa(山鸢尾) 1-versicolor(变色鸢尾) 2-virginica(弗吉尼亚鸢尾)
iris_data = datasets.load_iris()

# 转换成pandas的DataFrame,方便观察数据
iris = pd.DataFrame(
    data=np.c_[iris_data['data'], iris_data['target']],
    columns=np.append(iris_data.feature_names, ['y'])
)
# print("iris.head = ", iris.head(2))

# 检查是否有缺失值
# print("iris.isnull().sum = ", iris.isnull().sum())
# 观察样本中按类别数量是否比较均衡
# print("iris.groupby('y').count = ", iris.groupby('y').count())

# -----------------------------   训练模型 -----------------------------
# 选择全部特征训练模型
X = iris[iris_data.feature_names]
# label
y = iris['y']

# 第一步，选择模型
from sklearn.neighbors import KNeighborsClassifier

# 模型参数为1
knn = KNeighborsClassifier(n_neighbors=1)

# 第二步,fit x,y
knn.fit(X, y)

# 第三步,predict新数据
predict = knn.predict([[3, 2, 2, 5]])
# print(predict)

# -----------------------------   训练模型 -----------------------------

# -----------------------------   评估模型 -----------------------------
"""
                正确分类的样本数
准确率 = ---------------------------
                    样本总数
"""

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 分割训练&测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# K= 15
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

y_pred_on_train = knn.predict(X_train)
y_pred_on_test = knn.predict(X_test)
# print("y_pred_on_train = ", y_pred_on_train, "y_pred_on_test = ", y_pred_on_test)

# 计算预测出来的值和实际的目标值之间的误差
# print(metrics.accuracy_score(y_train, y_pred_on_train))
print('accuracy: {}'.format(metrics.accuracy_score(y_test, y_pred_on_test)))
# -----------------------------   评估模型 -----------------------------


import matplotlib.pyplot as plt
# seaborn是一个matplotlib之上封装统计plot类库，这里我们只是使用seaborn的样式定义
import seaborn as sns
from MLLib import KNN

x = np.arange(-3.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, KNN.softmax(scores).T, linewidth=2)
plt.savefig("../data/KNN-softmax.png")

scores = np.array([2.0, 1.0, 0.1])
print(KNN.softmax(scores))
print(KNN.softmax(scores * 100))
print(KNN.softmax(scores / 100))
