#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_04_pandas.py
# @Date    :   2018/7/3
# @Desc    :
import datetime

import common_header

import numpy as np

import pandas as pd

from pylab import *


def sample_series_dataframe():
    s = pd.Series([i * 2 for i in range(1, 11)])
    print(type(s))

    data, pf = getSampleData()
    print(pf)
    """
                       A         B         C         D         E
        2017-03-01  1.324811 -1.018163  0.023407  0.039765 -0.649236
        2017-03-02  1.835024 -0.922693 -0.197869  1.216107  0.196652
        2017-03-03 -0.289364 -1.843176  0.874987  1.573981 -0.014803
        2017-03-04 -0.553138 -1.358500 -0.582780  0.539532 -0.192156
        2017-03-05  0.692719 -1.241383  0.866766 -0.235040 -0.634380
        2017-03-06 -0.547626  2.154662  0.303339  1.102116  0.604369
        2017-03-07 -0.930585 -0.393160  0.870119  0.457505 -0.070311
        2017-03-08 -0.151280  1.449013  0.577671  0.129399 -0.273984
    """

    getSampleData2()
    # print(pf2)
    """
       A          B    C    D        E
    0  1 2017-03-01  1.0  3.0   police
    1  1 2017-03-01  1.0  3.0  student
    2  1 2017-03-01  1.0  3.0  teacher
    3  1 2017-03-01  1.0  3.0   doctor
    """
    pass


def getSampleData2():
    pf = pd.DataFrame({
        'A': 1, 'B': pd.Timestamp('20170301'), 'C': pd.Series(1, index=list(range(4)), dtype='float32'), \
        'D': np.array([3] * 4, dtype='float32'), 'E': pd.Categorical(['police', 'student', 'teacher', 'doctor'])})
    return pf


def getSampleData():
    dates = pd.date_range('20170301', periods=8)
    df = pd.DataFrame(np.random.randn(8, 5), index=dates, columns=list('ABCDE'))
    return dates, df


def sample_basic_select_set():
    data, pf = getSampleData()

    # print(pf)

    # ----------------   基本操作   ----------------
    # 获取前3行
    # print(pf.head(3))
    # # 获取后3行
    # print(pf.tail(3))
    # # 获取索引
    # print(pf.index)
    # # 获取元素
    # print(pf.values)
    # # 转置
    # print(pf.T)
    # # 排序
    # print("排序 1 ->")
    # print(pf.sort_index(by="C"))
    # 通过索引性进行排序
    # print(pf.sort_index(axis=1, ascending=False))
    # ascending True:升序，False:降序
    # print(pf.sort_values('C', ascending=True))
    # print(pf.describe())

    # ----------------   基本操作   ----------------

    # ----------------   选择   ----------------
    # print(type(pf['A']))
    # 取从第一行到第三行的数据
    # print(pf[:3])
    # print(pf['20170301':])
    # print(pf['20170301':'20170304'])
    # print(pf.loc[data[0]])
    # print(pf.loc['20170301':'20170304', ['B', 'D']])
    # print(pf.at[data[0], 'C'])

    # 通过下标进行选择
    # print(pf.iloc[1:3, 2:4])
    # 索引从0开始
    # print(pf.iloc[1, 4])
    # print(pf.iat[1, 4])

    # 根据条件进行筛选
    # print(pf[pf.B > 0][pf.A < 0])
    # print(pf[pf > 0]) # 2017-03-01       NaN  0.046406  0.553590  1.848206       NaN(小于0则显示NaN)
    # print(pf[pf['E'].isin([1, 2])])
    # ----------------   选择   ----------------

    # ----------------   设置   ----------------
    s1 = pd.Series(list(range(10, 18)), index=pd.date_range("20170301", periods=8))
    pf['F'] = s1
    print(pf)
    print('----------------------------')

    pf.at[data[0], 'A'] = 0
    print(pf)
    print('----------------------------')

    pf.iat[1, 1] = 1
    pf.loc[:, 'D'] = np.array([4] * len(pf))
    print(pf)
    print('----------------------------')
    # ----------------   设置   ----------------
    print("--- pf2 ---")
    pf2 = pf.copy()
    pf2[pf2 > 0] = -pf2

    print(pf2)
    pass


"""
缺失值处理
"""


def sample_missing_data_processing():
    datas, pf = getSampleData()

    # print(pf)

    pf1 = pf.reindex(index=datas[:4], columns=list("ABCD") + ["G"])
    # 第0和第1行的，G列的值为1
    pf1.loc[datas[0]:datas[1], "G"] = 1
    print(pf1)

    """
                       A         B         C         D    G
    2017-03-01 -0.462885  0.344348  0.862179  0.956716  1.0
    2017-03-02  0.914244  1.209553 -0.488466 -3.176064  1.0
    2017-03-03  1.609201  1.053202  0.214807 -0.106187  NaN
    2017-03-04  1.930273 -0.054360 -2.070998 -0.257221  NaN
    """

    """
    两种处理方式：A 是丢弃，B 是往里面填充一个值
                 B 方式又分为两种:固定值和差值
    """
    # 丢弃处理
    print(pf1.dropna())
    """
                       A         B         C         D    G
    2017-03-01 -1.006260 -0.313930 -0.826146 -1.299872  1.0
    2017-03-02 -1.088339  1.100718  0.905799  1.361047  1.0
    """

    print(pf1.fillna(value=2))
    """
                       A         B         C         D    G
    2017-03-01 -1.006260 -0.313930 -0.826146 -1.299872  1.0
    2017-03-02 -1.088339  1.100718  0.905799  1.361047  1.0
    2017-03-03  0.915541 -2.702654  0.488660  2.000224  2.0
    2017-03-04  0.961771 -0.515850 -1.116270  1.132938  2.0
    """
    pass


"""
表拼接和图形整合
"""


def sample_merge_reshape():
    datas, pf = getSampleData()
    print("均值:")
    # 输出所有的均值
    # print(pf.mean())
    print('----------------------------')
    print("方差:")
    # 输出方差
    # print(pf.var())
    print('----------------------------')
    # s = pd.Series([1, 2, 4, np.nan, 5, 7, 9, 10], index=datas)
    print("Series:")
    # print(s)
    print("shift:")
    # print(s.shift(2))
    print('----------------------------')
    print("diff:")
    # print(s.diff())
    print('----------------------------')
    print("value_counts:")
    # print(s.value_counts())
    print('----------------------------')
    print("累加:")
    pf.loc[:, 'D'] = np.array([4] * len(pf))
    print(pf.apply(np.cumsum))
    """
                       A         B         C   D         E
    2017-03-01  0.410151  1.182577  2.880741   4 -2.530244
    2017-03-02  1.007475  3.046510  3.071617   8 -3.128351
    2017-03-03  2.191782  3.576113  4.346878  12 -2.820820
    2017-03-04  1.818895  2.385790  2.826615  16 -2.070287
    2017-03-05  3.575273  3.409887  3.125909  20 -1.568456
    2017-03-06  1.721258  3.975347  3.775062  24  0.652986
    2017-03-07  1.174253  4.605997  4.569078  28  1.612441
    2017-03-08  0.387342  2.686661  5.351802  32  1.176381
    """
    print('----------------------------')
    print("极差:")
    print(pf.apply(lambda x: x.max() - x.min()))
    print('----------------------------')

    # 表拼接
    # concat
    # 拼接前3行和后3行
    print("concat:")
    pieces = [pf[:3], pf[3:]]
    print(pieces)
    print('----------------------------')

    print("merge:")
    left = pd.DataFrame({"key": ['x', 'y'], 'value': [1, 2]})
    right = pd.DataFrame({"key": ['x', 'z'], 'value': [3, 4]})
    print("left ==> \n", left)
    print("right ==> \n", right)
    # 类似于sql中的join,left方式
    # how:默认是inner,outter可用
    print(pd.merge(left, right, on='key', how='left'))
    print('----------------------------')

    print("groupby:")
    pf2 = pd.DataFrame({"A": ['a', 'b', 'c', 'd'], "B": list(range(4))})
    print(pf2.groupby('A').sum())
    print('----------------------------')

    print("Reshape:")
    pf3 = pd.DataFrame({
        'A': ['one', 'one', 'two', 'three'] * 6,
        'B': ['a', 'b', 'c'] * 8,
        'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
        'D': np.random.randn(24),
        'E': np.random.randn(24),
        'F': [datetime.datetime(2017, i, 1) for i in range(1, 13)] +
             [datetime.datetime(2017, i, 15) for i in range(1, 13)]})
    # 数据透视表交叉表会使用到
    print(pd.pivot_table(pf3, values='D', index=['A', 'B'], columns=['C']))
    print('----------------------------')
    pass


'''
时间序列,图像和文件操作
'''


def sample_time_series_graph_files():
    # print("时间序列:")
    # time_s = pd.date_range("20170305", periods=10, freq='S')
    # print(time_s)
    # print('----------------------------')
    #
    # print("画图:")
    # ts = pd.Series(np.random.randn(1000), index=pd.date_range('20170301', periods=1000))
    # ts = ts.cumsum()
    # print(ts)
    # ts.plot()
    # # show()
    # print('----------------------------')

    # 这里有些问题需要解决
    print("文件操作:")
    t_xls_1 = pd.read_excel("./data/test.xls", "Sheet1")
    print(t_xls_1)
    print('----------------------------')
    pass


if __name__ == '__main__':
    # sample_series_dataframe()  # 数据结构
    # sample_basic_select_set()  # 基础操作
    # sample_missing_data_processing()  # 缺失值处理
    # sample_merge_reshape()  # 表拼接和图形整合
    sample_time_series_graph_files()  # 时间序列,图像和文件操作
