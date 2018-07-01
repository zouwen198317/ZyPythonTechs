# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   1.2 calc_e.py
# @Date    :   2018/6/30
# @Desc    :

import common_header

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def calc_e_small(x):
    n = 10
    f = np.arange(1, n + 1).cumprod()
    b = np.array([x] * n).cumprod()
    return np.sum(b / f) + 1


def calc_e(X):
    reverse = False

    if X < 0:
        X = -X
        reverse = True
    ln2 = 0.69314718055994530941723212145818
    c = X / ln2
    a = int(c + 0.5)
    b = X - a * ln2
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1 / y
    return y


if __name__ == '__main__':
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 2, 20)
    t = np.concatenate((t1, t2))
    # 横轴数据
    print(t)

    y = np.empty_like(t)
    for i, X in enumerate(t):
        y[i] = calc_e(X)
        print('e^', X, ' = ', y[i], '(近似值)\t', math.exp(X), '(真实值)')

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2)
    plt.title(u'Taylor展式的应用', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True)
    plt.show()
