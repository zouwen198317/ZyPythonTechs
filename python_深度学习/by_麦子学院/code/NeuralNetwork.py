# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   NeuralNetwork.py
# @Date    :   2018/7/17
# @Desc    :

import common_header
import numpy as np


def tanh(x):
    return np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []

        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones(X.shape[0], X.shape[1] + 1)
        # add the bias unit to the input layer
        temp[:, 0:-1] = X
        X = temp
        y = np.array()

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # going forward network,for each layer
            for l in range(len(self.weights)):
                # cumpute the node value for each layer(o_i) using activation funcation
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            # compute the error at the top layer
            error = y[i] - a[-1]
            # for output layer , err calculation(delta is update error)
            # 减小误差
            deltas = [error * self.activation_deriv(a[-1])]

            # Staring backprobagation
            # we need to begin at the second to last layer
            for l in range(len(a) - 2, 0, 0, -1):
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[1].T) * self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(self.weights):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
