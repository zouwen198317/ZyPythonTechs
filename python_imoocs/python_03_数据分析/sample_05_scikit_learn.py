#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_05_scikit_learn.py
# @Date    :   2018/7/3
# @Desc    :

import common_header

import numpy as np

import pandas as pd

'''
决策树
'''


def sample_decision_tree():
    from sklearn.datasets import load_iris

    iris = load_iris()

    # print(iris)
    # print(len(iris['data']))

    from sklearn.model_selection import train_test_split
    # 数据分割成训练集和测试集数据
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
    # Model
    from sklearn import tree
    elf = tree.DecisionTreeClassifier(criterion="entropy")
    # 填充并训练
    # fit 建立模型
    elf.fit(train_x, train_y)
    # 预测
    # predict预测
    pred = elf.predict(test_x)

    # Verify验证: A 准确率 B 混淆矩阵
    from sklearn import metrics
    # y_true 真实值 ， y_pred预测值
    # accuracy_score预测真实值和预测值之前的一个误差
    print(metrics.accuracy_score(y_true=test_y, y_pred=pred))
    # 混淆矩阵
    print(metrics.confusion_matrix(y_true=test_y, y_pred=pred))
    '''
    [[11  0  0]
     [ 0 12  1]
     [ 0  0  6]]
     
     横轴为实际值，纵轴为预测值
    '''

    # 决策树输出到文件
    with open('./data/tree.dot', 'w') as fw:
        tree.export_graphviz(elf, out_file=fw)
    pass


def sample_split_data():
    import numpy as np
    from sklearn.model_selection import train_test_split

    X, y = np.arange(10).reshape((5, 2)), range
    X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    y = [0, 1, 2, 3, 4]
    print("X = ", X)
    print("y = ", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    print("X_train = ", X_train)
    print("y_train = ", y_train)
    print("X_test = ", X_test)
    print("y_test = ", y_test)
    pass


def export_internal_dataset():
    from sklearn.tree import export_graphviz
    from sklearn import tree
    from sklearn.datasets import load_breast_cancer

    from sklearn.datasets import load_iris

    data = load_iris()
    print(data.values())
    # C:\Program Files\Python36\Lib\site-packages\sklearn\datasets\data
    # C:\Program Files\Python36\Lib\site-packages\sklearn\datasets\data
    # 'C:\\Program Files\\Python36\\lib\\site-packages\\sklearn\\datasets\iris.csv'
    # 'C:\\Program Files\\Python36\\lib\\site-packages\\sklearn\\datasets\breast_cancer.csv'
    pass


if __name__ == '__main__':
    # sample_split_data()
    # sample_decision_tree()
    export_internal_dataset()
