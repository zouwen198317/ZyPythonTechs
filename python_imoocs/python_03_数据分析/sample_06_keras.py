#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_06_keras.py
# @Date    :   2018/7/3
# @Desc    : 需要安装keras

import common_header

import numpy as np

import theano

# Sequential神经网络各层需要的库
from keras.models import Sequential
# Dense 求和的层，Activation激活函数
from keras.layers import Dense, Activation
# 优化器，随机梯度下降算法
from keras.optimizers import SGD


def main():
    from sklearn.datasets import load_iris
    iris = load_iris()

    print(iris['target'])

    from sklearn.preprocessing import LabelBinarizer
    # 标记化
    print(LabelBinarizer().fit_transform(iris['target']))

    from sklearn.model_selection import train_test_split

    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)

    # 训练标签(目标值)
    label_train = LabelBinarizer().fit_transform(train_y)
    # 测试标签(目标值)
    label_test = LabelBinarizer().fit_transform(test_y)

    # 建立模型
    model = Sequential(
        [
            Dense(5, input_dim=4),
            Activation('relu'),
            Dense(3),
            Activation('sigmoid')
        ]
    )
    # 还可以使用这种方式
    # model=Sequential()
    # model.add(Dense(5,input_dim=4))

    ssg = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=ssg, loss='categorical_crossentropy')
    model.fit(train_x, label_train, nb_epoch=200, batch_size=40)
    print(model.predict_classes(test_x))

    model.save_weights('./data/w')
    model.load_weights('./data/w')
    pass


if __name__ == '__main__':
    main()
