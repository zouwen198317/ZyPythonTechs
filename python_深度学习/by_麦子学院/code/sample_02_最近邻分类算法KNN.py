#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_02_最近邻分类算法KNN.py
# @Date    :   2018/7/13
# @Desc    : https://blog.csdn.net/fukaixin12/article/details/79189132

import common_header
import csv
import random
import math
import operator


# 导入数据，并分成训练集和测试集
def loadDataSet(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    pass


# 求欧拉距离
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)


# 计算最近邻(K个数据集),testInstance是实例
def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        # testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # distance 是一个多个元组的list
        distance.append((trainingSet[x], dist))
        # 按照dist排序
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        # 要的是数据集
        neighbors.append(distance[x][0])
    return neighbors
    pass


# 投票法找出最近邻的结果哪种最多
def getResponse(neighbors):
    # key 花名字, value 个数
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
    pass


# 求出准确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 1 / 3
    loadDataSet(r'data/irisdata.txt', split, trainingSet, testSet)
    print("trainingSet = ", repr(len(trainingSet)))
    print("testSet = ", repr(len(testSet)))

    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(" > predicted = ", repr(result), ", actual = ", repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy = ", repr(accuracy) + " %")


if __name__ == '__main__':
    main()
