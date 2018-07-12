#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_01_决策树_Classification.py
# @Date    :   2018/7/12
# @Desc    : http://scikit-learn.org/stable/modules/tree.html

import common_header

from sklearn import tree


def sample_test():
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)
    print(clf.predict([[2., 2.]]))
    print(clf.predict_proba([[2., 2.]]))
    pass


def sample():
    from sklearn.datasets import load_iris
    iris = load_iris()
    clfd = tree.DecisionTreeClassifier()
    clf = clfd.fit(iris.data, iris.target)

    import graphviz

    # 将数据保存到本地
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    # graph.render("iris")
    print(clfd.predict(iris.data[:1, :]))
    print(clfd.predict_proba(iris.data[:1, :]))

    pass


if __name__ == '__main__':
    # sample_test()
    sample()
