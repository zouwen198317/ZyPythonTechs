# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   sample_01_numpy.py
# @Date    :   2018/7/2
# @Desc    :

import common_header
import numpy as np


def sample_ndarray():
    list = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print(type(list))

    # bool,int/int8/16/32/64/128,uint8/16/32/64/128,float/16/32/64,complex64/128
    np_list = np.array(list, dtype=np.float)
    print(type(np_list))
    print(np_list.shape)  # 形状
    print(np_list.ndim)  # 维度
    print(np_list.dtype)  # 数据类型
    print(np_list.itemsize)  # 每个元素的大小
    print(np_list.size)  # 数组元素个数
    pass


def sample_some_ndarray():
    print(np.zeros((3, 2)))
    print(np.ones((3, 2)))
    rand = np.random.rand(2, 4)
    print(rand)
    print(np.random.rand())

    print(np.random.randint(1, 10, 3))

    # 标准正态分布随机数
    print(np.random.randn(2, 4))
    # 随机数从可迭代的数组中去取
    print(np.random.choice([10, 20, 30, 40, 50]))

    # beta分布
    print(np.random.beta(1, 10, 100))
    pass


'''
    numpy常用操作
        矩阵函数	                    说明
        np.sin(a)	        对矩阵a中每个元素取正弦,sin(x)
        np.cos(a)	        对矩阵a中每个元素取余弦,cos(x)
        np.tan(a)	        对矩阵a中每个元素取正切,tan(x)
        np.arcsin(a)	    对矩阵a中每个元素取反正弦,arcsin(x)
        np.arccos(a)	    对矩阵a中每个元素取反余弦,arccos(x)
        np.arctan(a)	    对矩阵a中每个元素取反正切,arctan(x)
        np.exp(a)	        对矩阵a中每个元素取指数函数,ex
        np.sqrt(a)	        对矩阵a中每个元素开根号√x
'''


def sample_np_opes():
    np_list = np.arange(1, 11).reshape([2, -1])
    print(np_list)
    # 自然指数
    print('exp ---------- \n ', np.exp(np_list))
    print('exp2 ---------- \n ', np.exp2(np_list))
    print('sqrt ---------- \n ', np.sqrt(np_list))
    print('sin ---------- \n ', np.sin(np_list))
    print('log ---------- \n ', np.log(np_list))

    np_list2 = np.array([
        [[1, 2, 3, 4], [4, 5, 6, 7]],
        [[7, 8, 9, 10], [10, 11, 12, 13]],
        [[13, 14, 15, 16], [16, 17, 18, 19]]
    ])
    print(np_list2.sum())
    # 对最外层的数组进行操作
    print(np_list2.sum(axis=0))
    # 对第1层的数组进行操作
    print(np_list2.sum(axis=1))

    print(np_list2.max())
    print(np_list2.max(axis=0))
    print(np_list2.max(axis=1))

    print(np_list2.min(axis=0))

    np_list3 = np.array([10, 20, 30, 40])
    np_list4 = np.array([4, 3, 2, 1])
    print(np_list3 + np_list4)
    print(np_list3 - np_list4)
    print(np_list3 * np_list4)
    print(np_list3 / np_list4)
    print(np_list3 ** 2)

    print(np.dot(np_list3.reshape([2, 2]), np_list4.reshape([2, 2])))

    print(np.concatenate((np_list3, np_list4), axis=0))
    print(np.vstack((np_list3, np_list4)))
    print(np.hstack((np_list3, np_list4)))
    print(np.split(np_list3, 2))
    print(np.copy(np_list3))

    pass


'''
矩阵操作与线性方程组
'''

from numpy.linalg import *


def sample_lines_mat():
    # 3维矩阵
    # print(np.eye(3))

    # 矩阵求逆
    list = np.array([[1., 2.], [3., 4.]])
    # print(inv(list))

    # print(list.transpose())

    # print(det(list))

    # print(eig(list))

    y = np.array([[5.], [7.]])
    print(solve(list, y))

    pass


'''
    在其它领域的使用
'''


def sample_other_are_used():
    print("Fft \n ", np.fft.fft(np.array([1, 1, 1, 1, 1, 1, 1, 1])))
    # 相关系数:
    print("corrcoef \n ", np.corrcoef(np.array([[1, 0, 1], [0, 2, 1]])))
    # 一元多次函数
    print("poly1d \n ", np.poly1d([2, 1, 3]))
    pass


if __name__ == '__main__':
    # sample_ndarray()
    # sample_some_ndarray()
    # sample_np_opes()
    # sample_lines_mat()
    sample_other_are_used()
