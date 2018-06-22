import common_header
import numpy as np
from sklearn.model_selection import train_test_split

"""
数据分割:
资料地址:https://www.jianshu.com/p/83c8ef18a1e8
"""
X, y = np.arange(10).reshape(5, 2), range

# 5行2列的矩阵
X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

y = [0, 1, 2, 3, 4]

print("X = -> ")
print(X)
print("X = <- ")
print()
print("y = -> ")
print(y)
print("y = -> ")
print()
# x 是特征 y 目标
# 80%做为训练集数据，后面20%做为测试集数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train = -> ")
print(X_train)
print("X_train = <- ")
print()

print("X_test = -> ")
print(X_test)
print("X_test = <- ")
print()

print("y_train = -> ")
print(y_train)
print("y_train = <- ")
print()

print("y_test = -> ")
print(y_test)
print("y_test = <- ")
print()