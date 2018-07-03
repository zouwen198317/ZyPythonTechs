#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_03_scipy.py
# @Date    :   2018/7/3
# @Desc    :
from operator import le

import common_header

import numpy as np

import matplotlib.pyplot as plt

'''
积分
'''


def sample_integral():
    from scipy.integrate import quad, dblquad, nquad
    # print(quad(lambda x: np.exp(-x), 0, np.inf))
    '''
    (1.0000000000000002, 5.842606703608969e-11)
          值                   误差范围
    '''

    # print(dblquad(lambda t, x: np.exp(-x * t) / t ** 3, 0, np.inf, lambda x: 1, lambda x: np.inf))
    '''
    (0.33333333325010883, 1.3888461883425516e-08)
    '''

    def f(x, y):
        return x * y

    def bound_y():
        return [0, 0.5]

    def bound_x(y):
        return [0, 1 - 2 * y]

    print(nquad(f, [bound_x, bound_y]))  # (0.010416666666666668, 4.101620128472366e-16)
    pass


'''
优化器
'''


def sample_optimizer():
    from scipy.optimize import minimize
    def rosen(x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})

    # print("ROSE MINI", res)
    # 求函数全最小值
    # print("ROSE MINI", res.x)

    # 求函数在某个因素下的最小值
    def func(x):
        return (2 * x[0] * x[1] + 2 * x[0] - x[0] ** 2 - 2 * x[1] ** 2)

    def func_deriv(x):
        dfdx0 = (-2 * x[0] + 2 * x[1] + 2)
        dfdx1 = -(2 * x[0] - 4 * x[1])
        return np.array([dfdx0, dfdx1])

    cons = ({"type": "eq", "fun": lambda x: np.array([x[0] ** 3 - x[1]]),
             'jac': lambda x: np.array([3.0 * (x[0] ** 2.0), -1.0])},
            {"type": "ineq", "fun": lambda x: np.array([x[1] - 1]),
             'jac': lambda x: np.array([0.0, 1.0])})
    res = minimize(func, [-1.0, 1.0], jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
    print("RESTRICT", res)
    """
    success: True
       x: array([1., 1.])
    """

    from scipy.optimize import root
    def fuc_2(x):
        return x + 2 * np.cos(x)

    sol = root(fuc_2, 0.1)
    print("ROOT", sol.x, sol.fun)
    pass


"""
差值
"""


def sample_interpolation():
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)

    from scipy.interpolate import interp1d
    li = interp1d(x, y, kind='cubic')
    x_new = np.linspace(0, 1, 50)
    y_new = li(x_new)

    figure = plt.figure()
    plt.plot(x, y, 'r')
    plt.plot(x_new, y_new, 'k')
    plt.show()

    print(y_new)
    pass


"""
inv 求逆
solve 
det 行列式计算
norm 范式计算
"""
def sample_linear():
    from scipy import linalg as lg
    arr = np.array([[1, 2], [3, 4]])
    # print("Det:", lg.det(arr))
    # print("Inv:", lg.inv(arr))

    b = np.array([6, 14])
    print("Sol:", lg.solve(arr, b))
    print("Eig:", lg.eig(arr))
    print("Lu:", lg.lu(arr))
    print("Qr:", lg.qr(arr))
    print("Svd:", lg.svd(arr))
    print("Schur:", lg.schur(arr))

    pass


if __name__ == '__main__':
    # sample_integral()  # 积分
    # sample_optimizer()  # 优化器
    # sample_interpolation()  # 插值
    sample_linear()  # 线性计算
