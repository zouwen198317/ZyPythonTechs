#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_00_理解决策树结构.py
# @Date    :   2018/7/13
# @Desc    :

import common_header

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)

# Using those arrays, we can parse the tree structure:
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

# The tree structure can be thraversed to compute various properties such
# as the depth of each node and wether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# seed is the root node id and its parent depth
stack = [(0, -1)]
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # if the have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has the flollowing tree structure:" % n_nodes)

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s X[:,%s] <=%s else to node %s."
              % (node_depth[i] * "\t", i, children_left[i], feature[i], threshold[i], children_right[i]))
print()

# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i,j) indicates that the sample i goes throuth the node j
node_indicator = estimator.decision_path(X_test)

# Similarly ,we can also have the leaves ids reached by each sample
leave_id = estimator.apply(X_test)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples.First,let's make it for the sample
sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]
print("Rules used to predict sample %s: " % sample_id)

for node_id in node_index:
    if leave_id[sample_id] != node_id:
        continue

    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print("decision id node %s : (X_test[%s,%s](=%s) %s %s )"
          % (node_id, sample_id, feature[node_id], X_test[sample_id, feature[node_id]], threshold_sign,
             threshold[node_id]))

# for a group of samples, we have the following common node
sample_id = [0, 1]
common_nodes = (node_indicator.toarray()[sample_id].sum(axis=0) == len(sample_id))
common_node_id = np.arange(n_nodes)[common_nodes]

print("\n The following samples %s share the node %s in the tree "
      % (sample_id, common_node_id))

print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
