#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_决策树.py
# @Date    :   2018/7/13
# @Desc    : 代码参考地址 https://blog.csdn.net/fukaixin12/article/details/79183175

import common_header

# 决策树算法
# 首先需要导入输入输出数据，并将输入输出数据转换为标准形式
# 然后使用sklearn的决策树tree进行处理
# 最后输出.dot文件结果，并用Graphviz输出决策树的图形
# 对已有的决策树对象，进行测试数据集predict测试

# sklearn支持的数据只能是integer类型的数据,即特征和目标的取值都得是整数类型
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'data/AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)

print("headers titles = ", headers)

# -----------------     数据预处理    -----------------
feaureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]  # headers[i]是key - 特征名称,row[i]是value -特征值
    feaureList.append(rowDict)

'''
[{'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'}, {'age': 'youth', 'income': 'high', 
'student': 'no', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'high', 'student': 'no', 
'credit_rating': 'fair'}, {'age': 'senior', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair'}, 
{'age': 'senior', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'low', 
'student': 'yes', 'credit_rating': 'excellent'}, {'age': 'middle_aged', 'income': 'low', 'student': 'yes', 
'credit_rating': 'excellent'}, {'age': 'youth', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair'}, 
{'age': 'youth', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 'income': 'medium', 
'student': 'yes', 'credit_rating': 'fair'}, {'age': 'youth', 'income': 'medium', 'student': 'yes', 'credit_rating': 
'excellent'}, {'age': 'middle_aged', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent'}, 
{'age': 'middle_aged', 'income': 'high', 'student': 'yes', 'credit_rating': 'fair'}, {'age': 'senior', 
'income': 'medium', 'student': 'no', 'credit_rating': 'excellent'}] 
使用json解析会看的更明显一些
'''
print("feaureList = ", feaureList)

# Vetorize features
vec = DictVectorizer()
# 把特征值全部转换成0,1的数据 变成数组
dummyX = vec.fit_transform(feaureList).toarray()
"""
[[0. 0. 1. 0. 1. 1. 0. 0. 1. 0.]
 [0. 0. 1. 1. 0. 1. 0. 0. 1. 0.]
 [1. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
 [0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]
 [0. 1. 0. 0. 1. 0. 1. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 1. 0. 0. 1.]
 [1. 0. 0. 1. 0. 0. 1. 0. 0. 1.]
 [0. 0. 1. 0. 1. 0. 0. 1. 1. 0.]
 [0. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
 [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
 [0. 0. 1. 1. 0. 0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 0. 0. 0. 1. 1. 0.]
 [1. 0. 0. 0. 1. 1. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 0. 0. 1. 1. 0.]]
 以上操作把数据转换成了sikilearn中可用的包含字典数据的list
"""
print("dummyX = ", str(dummyX))
print("vec.get_feature_names = ", vec.get_feature_names())
print("labelList = ", str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
# 把目标值全部转换成0,1的数据
dummyY = lb.fit_transform(labelList)
print("dummyY = ", str(dummyY))
# -----------------     数据预处理    -----------------

# Using decision tree for classification
# 构造决策树
# entropy 使用ID3信息熵算法
clf = tree.DecisionTreeClassifier(criterion="entropy")
# 填充特征值和目标值进行训练
clf = clf.fit(dummyX, dummyY)

print("clf = " + str(clf))

# Visualize model
with open("allElectronicInformationGainOri2.dot", 'w') as f:
    # get_feature_names 还原原来的特征值
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRow = ", oneRowX)

# 将第一组数据中年龄young改为middle_age
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX = ", newRowX)

# 需要使用reshape(1,-1)
predictY = clf.predict(newRowX.reshape(1, -1))
print("predictY = ", predictY)
