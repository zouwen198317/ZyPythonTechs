	dirty
	
	英 [ˈdɜ:ti]   美 [ˈdɜrti] 
	
	He'd put his dirty laundry in the clothes basket.  
	他会把自己的脏衣服扔进洗衣篮里。

	incomplete
	英 [ˌɪnkəmˈpli:t]   美 [ˌɪnkəmˈplit] 
	
	
	an incomplete set of figures  
	一组不完整的数字

![](https://i.imgur.com/6R9kuLX.png)

# How to create machine learning models #
## 1. Split data into training & testing subsets 分割数据为训练集和测试集 ##
## 2. Train a model on tranning set 使用训练集训练模型 ##

1. Data Cleansing 数据清理
1. Feature Processing 特征处理
1. Model Selection 模型选择
1. Parameter Optimization 参数优化

## 3. make predictions on the testing set 对测试集进行预测 ##
## 4. compare predicted and true labels 比较预测和真实标签(比较预测值和真实值之间的一个误差) ##

![](https://i.imgur.com/ZfeJAgx.png)


# Typical machine learning tasks 典型机器学习任务  #

![](https://i.imgur.com/1mWpdQl.png)

![](https://i.imgur.com/SoYe9vo.png)

# Preprocessing data to avoid "garbage in, garbage out" #

![](https://i.imgur.com/oL1y331.png)

# 
the modern approaches for Feature Selection #

![](https://i.imgur.com/HoHldvv.png)



# google特征工程的定义 #

- Feature engineering means transforming raw data into a feature vector 特征工程意味着将原始数据转换为特征向量
- Process of creating features from raw data is feature engineering. Expect to spend significant time doing feature engineering. 从原始数据生成特征的过程是特征工程，期望花费大量时间进行特征工程

![](https://i.imgur.com/yg06UnE.png)


# The process of Machine Learning 机器学习过程#
![](https://i.imgur.com/6lqBOdI.png)


# What’s a model 什么是模型 #
## A model defines the relationship between features and label 模型定义了特征和标签之间的关系   ##

	relationship 
	 
	英 [rɪˈleɪʃnʃɪp]   美 [rɪˈleʃənˌʃɪp]  
	n.  关系; 联系; 浪漫关系; 血缘关系;

### An machine learning model is a mathematical model that generates predictions by finding patterns in your data 机器学习模型是通过在数据中找到模式来生成预测的数学模型 ###

	mathematical    
	英 [ˌmæθə'mætɪkl]   美 [ˌmæθəˈmætɪkəl]  
	adj.  数学的; 精确的; 绝对的; 可能性极小的;

### A machine learning algorithm use data to automatically learn the rules. It simplifies the complexity of the data into relationships described by rules 机器学习算法使用数据自动学习规则。它将数据的复杂性简化为规则描述的关系。  ###


	algorithm  
	英 [ˈælgərɪðəm]   美 [ˈælɡəˌrɪðəm]  
	    
	n.  运算法则; 演算法; 计算程序;

	automatically 
	英 [ˌɔ:tə'mætɪklɪ]   美 [ˌɔtəˈmætɪklɪ]  
	    
	adv.  自动地; 无意识地; 不自觉地; 机械地;

	simplifies    
	[ˈsimplifaiz]  
	    
	v.  使（某事物）简单[简明]，简化( simplify的第三人称单数 );

	complexity  
	英 [kəmˈpleksəti]   美 [kəmˈplɛksɪti]  
	     CET6 | TOEFL
	n.  复杂性，错综复杂的状态; 复杂的事物; 复合物;
	变形 复数: complexities

# How to develop a model 如何开发模型 #
![](https://i.imgur.com/FdwMyqc.png)

1. Predictive models are built or “trained” on historical data with a known outcome  预测模型是根据已知结果的历史数据建立或“训练”的。 
1. Once the model has been built, it’s applied onto new, more recent data which has an unknown outcome (because the outcome is in the future) 一旦建立了模型，就会将其应用于新的、最近的、结果未知的数据(因为结果是未来的)。
1. The algorithm will learn from the training data patterns that map the variables(features) to the target (i.e. labels), and it will output a model that captures these relationships 该算法将从将变量(特征)映射到目标(即标签)的训练数据模式中学习，并输出一个捕捉这些关系的模型


# Example of building the model ( Learning phase ) 建立模型的例子(学习阶段)  #
Predict if a customer is going to switch to another supplier  
预测客户是否将转入另一供应商 

![](https://i.imgur.com/apcEkrc.png)

# Example of using the model ( Applying phase )  使用模型的示例(应用阶段)  #
![](https://i.imgur.com/Ve0ZrQv.png)

# Train the model 训练模型 #

Let’s take a closer look at the model training 让我们仔细看看模特训练 

![](https://i.imgur.com/alDiupo.png)

# Train a model using algorithms and training data 利用算法和训练数据训练模型 #

Machine Learning uses algorithms to iteratively learn from data 机器学习使用算法迭代地从数据中学习

Algorithm is a self-contained set of rules used to solve problems through data processing, math, or automated reasoning. Machine learning algorithms use
computational methods to “learn” information directly from data without relying on a predetermined equation as a model.
算法是一组独立的规则集，用于通过数据处理、数学或自动推理来解决问题。机器学习算法使用计算方法来直接从数据中学习信息，而不依赖于预定的方程作为模型。 


The algorithm will learn from the training data patterns that map the variables(features) to the target (i.e. labels), and it will output a model that captures these relationships 该算法将从将变量(特征)映射到目标(即标签)的训练数据模式中学习，并输出一个捕捉这些关系的模型。

![](https://i.imgur.com/5ZNfh9a.png)

	iteratively 
	英 ['ɪtəˌreɪtɪvlɪ]   美 ['ɪtəˌreɪtɪvlɪ]  
	    
	[计] 迭代的;


# Let’s take a look at an example for regression problem 让我们来看一个回归问题的例子 #

## Example: Predict house’s price using linear regression 使用线性回归预测房子价格 ##

![](https://i.imgur.com/NXRPNYO.png)

## Example: Train a model using Linear regression algorithm 使用线性回归算法训练模型 ##

Use linear regression algorithm to approximate the relationship between x and y 用线性回归算法描述x与y的关系 

Take linear regression as example, the algorithm is trying to find a best-fit line to represent the relationship between the input feature x and target y. 以线性回归为例，该算法试图找到一条最佳拟合线来表示输入特征x与目标y之间的关系。 

![](https://i.imgur.com/MX7dmdv.png)

The straight line can be seen in the plot, showing how linear regression attempts to draw a straight line that will best minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.  在图中可以看到直线，显示线性回归如何试图绘制一条直线，使数据集中观察到的响应与线性近似预测的响应之间的残差平方和最小化。


# Usually there is deviation between the actual value and prediction 通常，实际值和预测值之间存在偏差。  #


## The deviations indicate how bad the model’s prediction was on the training examples  偏差表明模型的预测在训练示例中有多糟糕 ##

Loss (i.e. error) is a number indicating how bad the model‘s prediction was on a single example. If the model’s prediction is perfect, the loss is zero; otherwise, the loss is greater.  损失(即误差)是一个数字，表示模型的预测在单个例子中有多糟糕。如果模型的预测是完美的，损失是零；否则，损失就更大。 

![](https://i.imgur.com/8wbkC9b.png)

	The deviations from the fitted line to the observed values  从拟合线到观测值的偏差 
	
	Mean square error (MSE) is a commonly-used function to measure how large the loss is. It’s called as Loss function or Cost function  均方误差(MSE)是衡量损失大小的常用函数，称为损失函数或成本函数。
	
	Mean square error (MSE) is the average squared loss per example over the whole dataset. We will elaborate the details of loss function later on. 均方误差(MSE)是整个数据集中每个例子的平均平方损失，稍后我们将详细介绍损失函数。 
	
	The smaller the Mean square error, the better the fit of the line to the data. 均方误差越小，直线对数据的拟合越好。 



# Train the model to minimize the loss/error 训练模型使损失/误差最小化  #

## Training the model is an iterative process of finding the “best” parameters to minimize the error 训练模型是一个迭代过程，它可以找到“最佳”参数，以减少误差。  ##

Training a model simply means learning (determining) good values for all the parameters of the model from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss. 训练一个模型仅仅意味着从标记样本中学习(确定)模型所有参数的好值。在监督学习中，机器学习算法通过检查多个例子并试图找到一个损失最小的模型来建立一个模型。

![](https://i.imgur.com/HE7ETfy.png)

	The goal of training a model is to find a set of parameters that have low loss, on average, across all examples. 训练一个模型的目的是在所有例子中找到一组平均损失低的参数。 
	
	In this linear regression example, the goal of the training is to find estimated value for the parameters *0 and *1 which would provide the “best” fit for the data points within the training set. 在这个线性回归例子中，训练的目标是找到参数*0和*1的估计值，这将为训练集内的数据点提供“最佳”匹配。 
	
	If you don’t understand the linear regression equation, don’t worry about it. You can find the detailed explanation in Part 2 – popular algorithms 如果你不理解线性回归方程，就不用担心了

# How does the model find the “best” parameters 模型如何找到“最佳”参数？  #

## Gradient Descent is one of the most common algorithms to find the good parameters 梯度下降是最常用的参数提取算法之一 ##

A Machine Learning model is trained by starting with an initial guess for the parameters (e.g. weights and bias in neural network ) and iteratively adjusting those
guesses until learning parameters with the lowest possible loss  机器学习模型是从对参数(例如神经网络中的权重和偏差)的初始猜测开始，然后迭代地调整这些猜测，直到可能损失最小的参数为止

![](https://i.imgur.com/kOGtGuJ.png)


	Usually, you iterate until overall loss stops changing or at least changes extremely slowly. 通常，你会迭代，直到总损失停止变化，或者至少是变化非常缓慢。 
	
	When that happens, we say that the model has converged.2
	If you don’t get it, don’t worry. I will explain the details later in the Part 2 – popular algorithms 当这种情况发生时，我们说模型已经收敛


# Let’s take a look at an example for classification problem #

## Classification algorithm is another major types of machine learning algorithms 分类算法是机器学习算法的另一种主要类型。  ##


# Let’s take a look at an example for classification problem #

## Classification algorithm is another major types of machine learning algorithms ##
![](https://i.imgur.com/HZpBD7X.png)

# There is a wide range of algorithms … Looks #
![](https://i.imgur.com/fOovgyi.png)

# The categories of machine learning algorithms #
![](https://i.imgur.com/oMqFClu.png)

- Supervised Learning 有监督的学习
- Unsupervised Learning 无监督学习
- Semi Supervised Learning 半监督学习
- Reinforcement Learning 强化学习


# Some of Machine Learning Algorithms 一些机器学习算法 #
![](https://i.imgur.com/9ezPp3y.png)


# Which algorithm should I use #

## Selecting a machine learning algorithm is a process of trial and error 选择机器学习算法是一个反复试验的过程 ##

Choosing the right algorithm can seem overwhelming—there are dozens of supervised and unsupervised machine learning algorithms, and each takes a different
approach to learning. There is no best method or one size fits all. 选择正确的算法似乎是势不可挡的-有几十种有监督的和无监督的机器学习算法，每种算法都采用不同的学习方法。没有最好的方法，也没有一刀切的方法

Finding the right algorithm is partly just trial and error—even highly experienced data scientists can’t tell whether an algorithm will work without trying it out. 找到正确的算法在一定程度上只是一种尝试和错误-即使是经验丰富的数据科学家也无法判断一种算法在不尝试的情况下是否有效。 

### How to select the right algorithm ?  现在选择正确的算法 ###
The answer to the question varies depending on many factors, including: 

- The size, quality, and nature of data. 数据的大小、质量和性质。
- The available computational time. 可用计算时间
- The urgency of the task. 任务的紧迫性 
- What you want to do with the data. 你想用这些数据做什么 

![](https://i.imgur.com/k7P2MzC.png)

Clustering 聚类

Dimensionality Reduction 降维



# 06 Model Evaluation 模型评估 #

### Evaluate and interpret results 评估和解释结果  ###

1. Evaluate Performance 评估性能 
1. Optimize parameters 优化参数
1. Interpret results  解释结果

## What’s a good model ##

- Accurate 精确

	Are we making good predictions ? 我们做好预测了吗

- Interpretable 可解释的

	How easy is it to explain how the predictions are made ? 解释这些预测是如何作出的有多容易呢

- Fast 

	How long does it take to build a model and how long does the model take to make predictions? 建立一个模型需要多长时间？模型需要多长时间才能做出预测？ 

- Scalable 可升级的

	How much longer do we have to wait if we build/predict using a lot more data ? 如果我们使用更多的数据来构建/预测，我们还要等多久？ 
 

## The model might not generalize well to unseen new data 该模型可能不能很好地推广到未见的新数据。 ##

A model’s ability to generalize to new data is crucial for the success of a model  模型推广到新数据的能力是模型成功的关键 

Generalization refers to a model’s ability to perform well on new unseen data rather than only the training data. If the model is trained too well, it can fit perfectly the random fluctuations or noise in the training data but it will fail to predict accurately on new data.  泛化是指模型对新的未见数据(而不仅仅是训练数据)表现良好的能力，如果训练得太好，它可以很好地拟合训练数据中的随机波动或噪声，但不能准确地预测新的数据。 

![](https://i.imgur.com/tYKsu2m.png)


	Train a model to classify this data set as “blue” and “red” category 训练一个模型，将这个数据集分类为“蓝色”和“红色”类别。 
	
	During training model learns perfectly the inherent patterns in data. However, If a model has been trained too well on training data, it will be unable to generalize.  在训练过程中，模型很好地学习了数据中固有的模式，但是，如果模型在训练数据上训练得太好，就无法推广。
	
	The ”perfect” model performs badly on never-seen new data. This is called as “overfitting”. If the model is overfitting then it will not generalize well. “完美”模型在从未见过的新数据上表现不佳。这被称为“过度拟合”。如果模型过度拟合，那么它将不能很好地推广。 

## About the model fitting 关于模型拟合 ##

- When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting. 1 Underfitting is often a result of an excessively simple model.  当一个模型不能捕捉到数据中的重要区别和模式时，即使在训练数据方面，它的表现也很差，这就是所谓的欠拟合。不合适往往是一个过于简单的模型的结果。 
- The overfitting model performs perfectly on training data but fail to predict well on new data. In other words, it has low/no training error but has high test error 该模型对训练数据的拟合效果较好，但对新数据的预测效果不佳，即训练误差小/无训练误差，但测试误差大。 
- Both overfitting and underfitting lead to poor predictions on new data sets 过度拟合和不适当都会导致对新数据集的错误预测。 

![](https://i.imgur.com/4Jlmoue.png)

	This example demonstrates the problems of underfitting and overfitting and how we can use linear regression with polynomial features to approximate nonlinear functions. 这个例子说明了欠拟合和过拟合的问题，以及如何利用多项式特征的线性回归来逼近非线性函数。
	
	The models have polynomial features of different degrees. We can see that a linear function (polynomial with degree 1) is not sufficient to fit the training samples. This is called underfitting. A polynomial of degree 4 approximates the true function almost perfectly. However, for higher degrees the model will overfit the training data, i.e. it learns the noise of the training data. 该模型具有不同程度的多项式特征，可以看出，线性函数(1次多项式)不足以拟合训练样本，这称为次拟合，4次多项式几乎完全逼近了真实函数，但对于较高程度的训练数据，该模型会对训练数据进行拟合，即学习训练数据的噪声。 

## About the model fitting ##

![](https://i.imgur.com/19xxwG3.png)

## Model validation checks how well a model generalizes to new data ##

The only way to know how well a model will generalize to new cases is to actually try it out on new cases. It is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set. You train your model using the training set, and you test it using the test set. The error rate on new cases is called the generalization error, and by evaluating your model on the test set, you get an estimation of this error. This value tells you how well your model will perform on instances it has never seen before  要知道一个模型将如何推广到新的案例，唯一的方法就是尝试新的案例。这是常见的做法时，执行（监督）机器学习实验，以保持部分可用数据作为测试集。使用训练集训练模型，然后使用测试集测试它。新情况下的错误率称为泛化误差，通过对测试集上的模型进行评估，可以得到该错误的估计值。这个值告诉您模型在以前从未见过的情况下执行得怎么样。

## Metrics for evaluating classification model ##

- Accuracy  准确
- Precision 精确
- Recall  召回
- F-Score
- ROC 
- AUC 平均单位成本
- Log Loss 日志损失

### Example for Accuracy, Precision, Recall ###
![](https://i.imgur.com/1tEzM6I.png)

## Case study: Medical diagnosis ##
![](https://i.imgur.com/HxaoIqh.png)


# Linear Algebra Basic 线性代数基础 #

1 [Matrices and Vectors](https://youtu.be/Dft1cqjwlXE)

2 [Addition and Scalar Multiplication](https://youtu.be/4WP6jVGIn7M)

3 [Matrix Vector Multiplication](https://youtu.be/gPegoVYp64w)

4 [Matrix Matrix Multiplication](https://youtu.be/_lrHXJRukMw)

5 [Matrix Multiplication Properties](https://youtu.be/c7GhnL2N--I)

6 [Inverse and Transpose](https://youtu.be/7snro4M6ukk)
