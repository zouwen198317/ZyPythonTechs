#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_03人脸识别_决策树.py
# @Date    :   2018/7/12
# @Desc    : 未验证

import common_header

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

''''
不加这现行。数据集会下载失败
'''
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target
print(data)
print(targets)

data = data.images.reshpe((len(data.images), -1))
train = data[targets < 30]
# Test on independent people
test = data[targets >= 30]

# Test on a subset of people
n_face = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_face,))
test = test[face_ids, :]

n_pixels = data.shape[1]

# Upper half of the faces
X_train = train[:, :(n_pixels + 1) // 2]
# lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

# Fit esimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV()
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    # 训练
    estimator.fit(X_train, y_train)
    # 预测
    y_test_predict[name] = estimator.predict(X_test)

# Plot the complete faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_face))
plt.suptitle('Face completion with multi-output estimators', size=16)

for i in range(n_face):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_face, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_face, n_cols, i * n_cols + 1, title="true faces")

    sub.asix("off")
    sub.imgshow(true_face.reshpe(image_shape), cmap=plt.cm.gray, interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_face, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_face, n_cols, i * n_cols + 2 + j, title="est")

        sub.asix("off")
        sub.imgshow(completed_face.reshpe(image_shape), cmap=plt.cm.gray, interpolation="nearest")

plt.show()
