#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_matplotlib.py
# @Date    :   2018/7/3
# @Desc    :

import common_header

import matplotlib.pyplot as plt

import numpy as np


# 线性图
def draw_line():
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # cos 余弦， sin正弦
    c, s = np.cos(x), np.sin(x)
    plt.figure(1)
    # >>> plot(x, y, 'go--', linewidth=2, markersize=12)
    #         >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    #                 linewidth=2, markersize=12)
    plt.plot(x, c, color='blue', linewidth=0.5, linestyle='-', label='Cos', alpha=0.5)
    plt.plot(x, s, 'r*', label='Sin', linewidth=0.5)
    plt.title("Cos & Sin")
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # 将坐标移动到0点
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.2))

    plt.grid()
    plt.legend(loc='upper left')
    # plt.axis([-1, 1, -0.5, 1])
    plt.fill_between(x, np.abs(x) < 0.5, c, c > 0.5, color='green', alpha=0.25)

    t = 1
    plt.plot([t, t], [0, np.cos(t)], 'y', linewidth=3, linestyle='--')
    plt.annotate('cos(1', xy=(t, np.cos(1)), xycoords='data', xytext=(+10, +30),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.show()
    pass


# 散点图
def draw_scatter():
    ax = fig.add_subplot(3, 3, 1)
    n = 128
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)
    # plt.axes([0.025, 0.025, 0.95, 0.95])
    # plt.scatter(X, Y, s=75, c=T, alpha=.5)
    ax.scatter(X, Y, s=75, c=T, alpha=.5)
    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.axis()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('scatter')
    pass


# 直方图
def draw_bar():
    ax = fig.add_subplot(332)

    n = 10
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

    # plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    # plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x, y in zip(X, Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(X, Y2):
        plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')
    pass


# 饼图
def draw_pie():
    fig.add_subplot(333)

    n = 20
    Z = np.ones(n)
    Z[-1] *= 2
    plt.pie(Z, explode=Z * 0.05, colors=['%f' % (i / float(n)) for i in range(n)],
            labels=['%.2f' % (i / float(n)) for i in range(n)])
    plt.gca().set_aspect('equal')
    plt.xticks([]), plt.yticks([])
    pass


# 极坐标图
def draw_polar():
    # 不加参数默认是画折线
    fig.add_subplot(334, polar=True)
    n = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)
    radii = 10 * np.random.rand(n)
    plt.polar(theta, radii)
    # plt.plot(theta, radii)
    pass


# 热图
def draw_heatmap():
    fig.add_subplot(335)
    from matplotlib import cm

    data = np.random.rand(3, 3)
    cmap = cm.Blues
    map = plt.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=0, vmax=1)
    pass


# 3D图
def draw_3D():
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(336, projection='3d')

    ax.scatter(1, 1, 3, s=100)
    pass


# 热力图
def draw_hot_map():
    fig.add_subplot(313)

    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    pass


if __name__ == '__main__':
    fig = plt.figure()

    # draw_line()
    draw_scatter()
    draw_bar()
    draw_pie()
    draw_polar()
    draw_heatmap()
    draw_3D()
    draw_hot_map()
    plt.savefig('./data/fig.png')
    plt.show()
