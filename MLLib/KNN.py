#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   KNN.py
# @Date    :   2018/7/19
# @Desc    :

import common_header

from MLLib import *

import common_header

from numpy import *
import numpy as np
import operator


def cretaeDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def score(x, w, b):
    """
        线性函数: y = wx+b
    """
    return np.dot(x, w) + b


def sigmoid(s):
    """
    sigmoid函数
                                     1
     Sigmoid(x)=       ----------------------------
                                  1+e(-x)
    :param s:
    :return:
    """
    return 1 / (1 + np.exp(-s))


def softmax(x):
    """
        softmax函数
    :param x:
    :return:
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(y, p):
    """
    交叉熵
    :param y: 真实标签
    :param p: 预测概率
    :return:
    """
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p, axis=1))
