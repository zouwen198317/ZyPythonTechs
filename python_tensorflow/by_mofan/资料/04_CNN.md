	https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-03-A-CNN/

# 什么是卷积神经网络 CNN (Convolutional Neural Network) #

 
## 学习资料: ##

- Tensorflow CNN [教程1](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-03-CNN1/)
- Tensorflow CNN [教程2](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-04-CNN2/)
- Tensorflow CNN [教程3](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-01-CNN/)
- PyTorch CNN [教程](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-01-CNN/)
- 方便快捷的 Keras CNN[教程](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-3-CNN/)


![](https://morvanzhou.github.io/static/results/ML-intro/cnn1.png)

卷积神经网络是近些年逐步兴起的一种人工神经网络结构, 因为利用卷积神经网络在图像和语音识别方面能够给出更优预测结果, 这一种技术也被广泛的传播可应用. 卷积神经网络最常被应用的方面是计算机的图像识别, 不过因为不断地创新, 它也被应用在视频分析, 自然语言处理, 药物发现, 等等. 近期最火的 Alpha Go, 让计算机看懂围棋, 同样也是有运用到这门技术.

## 卷积 和 神经网络  ##
![](https://morvanzhou.github.io/static/results/ML-intro/cnn2.png)

我们来具体说说卷积神经网络是如何运作的吧, 举一个识别图片的例子, 我们知道神经网络是由一连串的神经层组成,每一层神经层里面有存在有很多的神经元. 这些神经元就是神经网络识别事物的关键. 每一种神经网络都会有输入输出值, 当输入值是图片的时候, 实际上输入神经网络的并不是那些色彩缤纷的图案,而是一堆堆的数字. 就比如说这个. 当神经网络需要处理这么多输入信息的时候, 也就是卷积神经网络就可以发挥它的优势的时候了. 那什么是卷积神经网络呢?

![](https://morvanzhou.github.io/static/results/ML-intro/cnn3.png)

我们先把卷积神经网络这个词拆开来看. “卷积” 和 “神经网络”. 卷积也就是说神经网络不再是对每个像素的输入信息做处理了,而是图片上每一小块像素区域进行处理, 这种做法加强了图片信息的连续性. 使得神经网络能看到图形, 而非一个点. 这种做法同时也加深了神经网络对图片的理解. 具体来说, 卷积神经网络有一个批量过滤器, 持续不断的在图片上滚动收集图片里的信息,每一次收集的时候都只是收集一小块像素区域, 然后把收集来的信息进行整理, 这时候整理出来的信息有了一些实际上的呈现, 比如这时的神经网络能看到一些边缘的图片信息, 然后在以同样的步骤, 用类似的批量过滤器扫过产生的这些边缘信息, 神经网络从这些边缘信息里面总结出更高层的信息结构,比如说总结的边缘能够画出眼睛,鼻子等等. 再经过一次过滤, 脸部的信息也从这些眼睛鼻子的信息中被总结出来. 最后我们再把这些信息套入几层普通的全连接神经层进行分类, 这样就能得到输入的图片能被分为哪一类的结果了.

![](https://morvanzhou.github.io/static/results/ML-intro/cnn4.png)

我们截取一段 google 介绍卷积神经网络的视频, 具体说说图片是如何被卷积的. 下面是一张猫的图片, 图片有长, 宽, 高 三个参数. 对! 图片是有高度的! 这里的高指的是计算机用于产生颜色使用的信息. 如果是黑白照片的话, 高的单位就只有1, 如果是彩色照片, 就可能有红绿蓝三种颜色的信息, 这时的高度为3. 我们以彩色照片为例子. 过滤器就是影像中不断移动的东西, 他不断在图片收集小批小批的像素块, 收集完所有信息后, 输出的值, 我们可以理解成是一个高度更高,长和宽更小的”图片”. 这个图片里就能包含一些边缘信息. 然后以同样的步骤再进行多次卷积, 将图片的长宽再压缩, 高度再增加, 就有了对输入图片更深的理解. 将压缩,增高的信息嵌套在普通的分类神经层上,我们就能对这种图片进行分类了.

## 池化(pooling)  ##
![](https://morvanzhou.github.io/static/results/ML-intro/cnn5.png)

研究发现, 在每一次卷积的时候, 神经层可能会无意地丢失一些信息. 这时, 池化 (pooling) 就可以很好地解决这一问题. 而且池化是一个筛选过滤的过程, 能将 layer 中有用的信息筛选出来, 给下一个层分析. 同时也减轻了神经网络的计算负担 (具体细节参考). 也就是说在卷集的时候, 我们不压缩长宽, 尽量地保留更多信息, 压缩的工作就交给池化了,这样的一项附加工作能够很有效的提高准确性. 有了这些技术,我们就可以搭建一个属于我们自己的卷积神经网络啦.

## 流行的 CNN 结构  ##
![](https://morvanzhou.github.io/static/img/description/loading.gif)

比较流行的一种搭建结构是这样, 从下到上的顺序, 首先是输入的图片(image), 经过一层卷积层 (convolution), 然后在用池化(pooling)方式处理卷积的信息, 这里使用的是 max pooling 的方式. 然后在经过一次同样的处理, 把得到的第二次处理的信息传入两层全连接的神经层 (fully connected),这也是一般的两层神经网络层,最后在接上一个分类器(classifier)进行分类预测. 这仅仅是对卷积神经网络在图片处理上一次简单的介绍. 如果你想知道使用 python 搭建这样的卷积神经网络, 欢迎点击下面的内容.



# CNN 卷积神经网络 1 #

## 学习资料: ##

- Google 的 [CNN 教程](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/63796332430923)
- 机器学习-简介系列 [什么是 CNN](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-2-CNN/)
- 为 TF 2017 打造的[新版可视化教学代码](https://github.com/MorvanZhou/Tensorflow-Tutorial)



## CNN 简短介绍  ##

我们的一般的神经网络在理解图片信息的时候还是有不足之处, 这时卷积神经网络就是计算机处理图片的助推器. Convolutional Neural Networks (CNN) 是神经网络处理图片信息的一大利器. 有了它, 我们给计算机看图片,计算机理解起来就更准确. 强烈推荐观看我制作的短小精炼的 机器学习-简介系列 什么是 CNN

计算机视觉处理的飞跃提升，在图像和语音识别方面表现出了强大的优势，学习卷积神经网络之前，我们已经假设你对神经网络已经有了初步的了解，如果没有的话，可以去看看tensorflow第一篇视频教程哦~

卷积神经网络包含输入层、隐藏层和输出层，隐藏层又包含卷积层和pooling层，图像输入到卷积神经网络后通过卷积来不断的提取特征，每提取一个特征就会增加一个feature map，所以会看到视频教程中的立方体不断的增加厚度，那么为什么厚度增加了但是却越来越瘦了呢，哈哈这就是pooling层的作用喽，pooling层也就是下采样，通常采用的是最大值pooling和平均值pooling，因为参数太多喽，所以通过pooling来稀疏参数，使我们的网络不至于太复杂。

好啦，既然你对卷积神经网络已经有了大概的了解，下次课我们将通过代码来实现一个基于MNIST数据集的简单卷积神经网络。



# CNN 卷积神经网络 2  #
- 
这一次我们会说道 CNN 代码中怎么定义 Convolutional 的层和怎样进行 pooling.

基于上一次卷积神经网络的介绍，我们在代码中实现一个基于MNIST数据集的例子

## 定义卷积层的 weight bias  ##
首先我们导入

	import tensorflow as tf
采用的数据集依然是tensorflow里面的mnist数据集

我们需要先导入它

	python from tensorflow.examples.tutorials.mnist import input_data
本次课程代码用到的数据集就是来自于它

	mnist=input_data.read_data_sets('MNIST_data',one_hot=true)
接着呢，我们定义Weight变量，输入shape，返回变量的参数。其中我们使用tf.truncted_normal产生随机变量来进行初始化:
	
	def weight_variable(shape): 
		inital=tf.truncted_normal(shape,stddev=0.1)
		return tf.Variable(initial)
同样的定义biase变量，输入shape ,返回变量的一些参数。其中我们使用tf.constant常量函数来进行初始化:

	def bias_variable(shape): 
		initial=tf.constant(0.1,shape=shape) 
		return tf.Variable(initial)
定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。

	def conv2d(x,W):
		return tf.nn.conv2d(x,W,strides=[1,1,1,1]，padding='SAME') 

## 定义 pooling  ##
接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:

	def max_poo_2x2(x): 
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1])
好啦，如果你对本节课内容已经了解，下一次课我们将构建卷积神经网络的架构~



## CNN 卷积神经网络 3 ##

这一次我们一层层的加上了不同的 layer. 分别是:

convolutional layer1 + max pooling;
convolutional layer2 + max pooling;
fully connected layer1 + dropout;
fully connected layer2 to prediction.
我们利用上节课定义好的函数来构建我们的网络

# 图片处理#
首先呢，我们定义一下输入的placeholder

	xs=tf.placeholder(tf.float32,[None,784])
	ys=tf.placeholder(tf.float32,[None,10])
我们还定义了dropout的placeholder，它是解决过拟合的有效手段

	keep_prob=tf.placeholder(tf.float32)
接着呢，我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。

	x_image=tf.reshape(xs,[-1,28,28,1])

## 建立卷积层  ##
接着我们定义第一层卷积,先定义本层的Weight,本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap

	W_conv1=weight_variable([5,5,1,32])
接着定义bias，它的大小是32个长度，因此我们传入它的shape为[32]

	b_conv1=bias_variable([32])
定义好了Weight和bias，我们就可以定义卷积神经网络的第一个卷积层h_conv1=conv2d(x_image,W_conv1)+b_conv1,同时我们对h_conv1进行非线性处理，也就是激活函数来处理喽，这里我们用的是tf.nn.relu（修正线性单元）来处理，要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32

	h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
最后我们再进行pooling的处理就ok啦，经过pooling的处理，输出大小就变为了14x14x32

	h_pool=max_pool_2x2(h_conv1)
接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64

	W_conv2=weight_variable([5,5,32,64])
	b_conv2=bias_variable([64])
接着我们就可以定义卷积神经网络的第二个卷积层，这时的输出的大小就是14x14x64

	h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
最后也是一个pooling处理，输出大小为7x7x64

	h_pool2=max_pool_2x2(h_conv2)


## 建立全连接层  ##
好的，接下来我们定义我们的 fully connected layer,

进入全连接层时, 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.

	#[n_samples,7,7,64]->>[n_samples,7*7*64]
	h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) 
此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024

	W_fc1=weight_variable([7*7*64,1024]) 
	b_fc1=bias_variable([1024])
然后将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）

	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
如果我们考虑过拟合问题，可以加一个dropout的处理

	h_fc1_drop=tf.nn.dropout(h_fc1,keep_drop)
接下来我们就可以进行最后一层的构建了，好激动啊, 输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值

	W_fc2=weight_variable([1024,10]) b_fc2=bias_variable([10])
然后呢我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类

	prediction=tf.nn.softmax(tf.matmul(h_fc1_dropt,W_fc2),b_fc2)

## 选优化方法  ##
接着呢我们利用交叉熵损失函数来定义我们的cost function

	cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices=[1]))
我们用tf.train.AdamOptimizer()作为我们的优化器进行优化，使我们的cross_entropy最小

	train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
接着呢就是和之前视频讲的一样喽 定义Session

	sess=tf.Session()
初始化变量

	# tf.initialize_all_variables() 这种写法马上就要被废弃
	# 替换成下面的写法:
	sess.run(tf.global_variables_initializer())
好啦接着就是训练数据啦，我们假定训练1000步，每50步输出一下准确率， 注意sess.run()时记得要用feed_dict给我们的众多 placeholder 喂数据哦.

以上呢就是一个简单的卷积神经网络的例子代码


# Saver 保存读取 #



## 保存  ##
import所需的模块, 然后建立神经网络当中的 W 和 b, 并初始化变量.

	import tensorflow as tf
	import numpy as np
	
	## Save to file
	# remember to define the same dtype and shape when restore
	W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
	b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
	
	# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
	# 替换成下面的写法:
	init = tf.global_variables_initializer()

保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量. 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录 "my_net/save_net.ckpt".

	saver = tf.train.Saver()
	
	with tf.Session() as sess:
	    sess.run(init)
	    save_path = saver.save(sess, "my_net/save_net.ckpt")
	    print("Save to path: ", save_path)
	
	"""    
	Save to path:  my_net/save_net.ckpt
	"""

## 提取 ##
提取时, 先建立零时的W 和 b容器. 找到文件目录, 并用saver.restore()我们放在这个目录的变量.

	# 先建立 W, b 的容器
	W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
	b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
	
	# 这里不需要初始化步骤 init= tf.initialize_all_variables()
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    # 提取变量
	    saver.restore(sess, "my_net/save_net.ckpt")
	    print("weights:", sess.run(W))
	    print("biases:", sess.run(b))
	
	"""
	weights: [[ 1.  2.  3.]
	          [ 3.  4.  5.]]
	biases: [[ 1.  2.  3.]]
	"""