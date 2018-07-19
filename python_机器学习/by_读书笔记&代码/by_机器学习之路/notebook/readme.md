# 第一章学习笔记  2018-7-18 #

机器学到的知识 == 训练好的模型

## 准确率 ##
				正确分类的数量
	准确率 = --------------------
				    总数 

## 机器学习的过程 ##
1. 训练模型
1. 优化模型
1. 应用模型


## 机器学习分类 ##
- 1. 监督学习(Supervised Learning)
> 有目标值

> 分类: 目标值离散(0,1,2,3分类)

> 回归: 目标值连续(可以在一个区间内取任意数)

- 2. 无监督学习(Unsupervised Learning)
> 没有目标值

- 3. 强化学习(Reinforcement Learning)


## KNN  ##
K-Nearest Neighbor , K-近邻算法 

分类问题就是给样本打标签

KNN可用于分类问题和回归问题

> KNN就是选出与待分类样本最相似的K个近邻,以确定样本所属的类别


### 欧氏距离 ###

a-b的平方,求和,开根号


	vec1 = np.array([1, 2, 3])
	vec2 = np.array([4, 5, 6])
	
	# 欧氏距离
	resultA = np.linalg.norm(vec1 - vec2)
	resultB = np.sqrt(np.sum(np.square(vec1 - vec2)))