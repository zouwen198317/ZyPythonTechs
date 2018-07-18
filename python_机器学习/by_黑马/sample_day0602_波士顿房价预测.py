#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_day0602_波士顿房价预测.py
# @Date    :   2018/7/18
# @Desc    :

import common_header

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def myLinear():
    """
    线性回归
    :return:
    """

    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print("y_train = ", y_train)
    # print("y_test = ", y_test)

    # 进行标准化处理(特征值和目标值都需要进行标准化处理)
    # 特征值和目标值都必须进行标准化处理，实例化两个标准化API

    # 特征值
    # 新版本必须要传递二维的数据
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)

    # 目标值
    std_y = StandardScaler()
    # 新版本需要将数据转换成2维的，否则会报错
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.fit_transform(y_test.reshape(-1, 1))
    # print("y_train = ", y_train)
    # print("y_test = ", y_test)

    # estimator预测
    # 正规方程求解方式预测结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("LR 回归系数:", lr.coef_)

    # 预测只有误差
    # 可以预测测试集的房子价格
    # 转换成标准化之前的值,如果不转换，数据是标准化之后的值
    y_predict = std_y.inverse_transform(lr.predict(x_test))
    print("LR 测试集里面每个房子的预测价格:", y_predict)
    print("正规方程的均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_predict))

    # 梯度下降去进行房价预测
    """
      learning_rate : string, optional
        The learning rate schedule:

        - 'constant': eta = eta0
        - 'optimal': eta = 1.0 / (alpha * (t + t0)) [default]
        - 'invscaling': eta = eta0 / pow(t, power_t)
        
        eta0=0.01
        固定的值求的学习率
    """
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("SGD 回归系数:", sgd.coef_)

    # 预测只有误差
    # 可以预测测试集的房子价格
    # 转换成标准化之前的值,如果不转换，数据是标准化之后的值
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("SGD 测试集里面每个房子的预测价格:", y_sgd_predict)

    # 如果数据比较简单使用正规方程，准确率可能会高些
    # 如果数据量比较大比较复杂，使用梯度下降准确率会高些(几十万，几千万)

    print("SGD 的均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    pass


if __name__ == '__main__':
    myLinear()
