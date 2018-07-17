# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @File    :   sample_06_人脸识别SVM算法实现.py
# @Date    :   2018/7/14
# @Desc    : https://blog.csdn.net/fukaixin12/article/details/79211747

import common_header

# 本例为人脸识别的SVM算法
# 首先fetch_1fw_pople导入数据
# 其次对数据进行处理，首先得到X，y,分割数据集为训练集和测试集,PCA降维,然后训练
# 最后查看正确率，classification_report以及confusion_matrix 以及绘制出特征图和预测结果
from time import time
# 程序进展信息
import logging
import matplotlib.pyplot as plt

import PIL
# 分割数据集
from sklearn.model_selection import train_test_split
# 下载数据集
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC

''''
不加这现行。数据集会下载失败
'''
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# download the data,if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw_people)
# introspect the images arrays to find the shapes(for plotting)
# 图像矩阵的行h,列w
n_samples, h, w = lfw_people.images.shape
print(n_samples, h, w)

# for machine learning we use the 2 data directly (as relative pixel positions into is ignored by this model)
# 图片数据
X = lfw_people.data
# 特征点数据
n_features = X.shape[1]

# the label to predict is the id of the person
# y是label,有7个目标时，0-6之间的取值
y = lfw_people.target
# 实际有哪些名字,字符串
target_names = lfw_people.target_names
# shape[0] -》 行维数 shape[1] 列维数
n_classes = target_names.shape[0]

print("Total dataset size:")

# print("target_names: %d " % target_names)
print("n_samples: %d " % n_samples)
print("n_features: %d " % n_features)
print("n_classes: %d " % n_classes)

# split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Compute a PCA(eigenfaces) on the face dataset(treated as unlabeled dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces " % (n_components, X_train.shape[0]))
t0 = time()

pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True)
# 训练如何降维
pca.fit(X, y)
print("done in %0.3fs " % (time() - t0))

# 三维
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonnormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs " % (time() - t0))

# train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
# 不停的缩小范围
pargram_grid = {'C': [1e3, 998, 1001, 999, 1002], 'gamma': [0.0025, 0.003, 0.0035]}
# GridSearchCV 第一个参数是分类器
clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), pargram_grid)
clf.fit(X_train_pca, y_train)
print("done in %0.3fs " % (time() - t0))
print("Baset estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs " % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# Qualitative evaluation of the predictions using atplotlib
def plot_gallery(images, titles, h, w, n_rows=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_rows * n_col):
        plt.subplot(n_rows, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks()
        plt.yticks()


# plot the result of the prediction on a protion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)
plt.savefig("data/svm_人脸识别1.png")

# plot the gallery of the most significative engenface
eigenface_titles = ['eigenface %d ' % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.savefig("data/svm_人脸识别2.png")
plt.show()
