# -*- coding:UTF-8 -*-
"""
资料地址
https://www.imooc.com/article/24396

https://www.codementor.io/likegeeks/define-and-use-tensors-using-simple-tensorflow-examples-ggdgwoy4u
"""

"""
TensorFlow 的简单例子
"""
"""
TensorFlow 是什么?
TensorFlow 是 Google 为了解决复杂的数学计算耗时过久的问题而开发的一个库。
事实上，TensorFlow 能干许多事。比如：
求解复杂数学表达式
机器学习技术。你往其中输入一组数据样本用以训练，接着给出另一组数据样本基于训练的数据而预测结果。这就是人工智能了！
支持 GPU 。你可以使用 GPU（图像处理单元）替代 CPU 以更快的运算。TensorFlow 有两个版本： CPU 版本和 GPU 版本。

什么是张量？
张量tensor是 TensorFlow 使用的主要的数据块，它类似于变量，TensorFlow 使用它来处理数据。张量拥有维度和类型的属性。
维度指张量的行和列数，读到后面你就知道了，我们可以定义一维张量、二维张量和三维张量。
类型指张量元素的数据类型。
"""

# -----------------     定义一维张量    -----------------

"""
定义一维张量
可以这样来定义一个张量：创建一个 NumPy 数组（LCTT 译注：NumPy 系统是 Python 的一种开源数字扩展，包含一个强大的 N 维数组对象 
Array，用来存储和处理大型矩阵 ）或者一个 Python 列表 ，然后使用 tf_convert_to_tensor 函数将其转化成张量。

可以像下面这样，使用 NumPy 创建一个数组：
"""
import numpy as np

arr = np.array([1, 5.5, 3, 15, 20])
print("arr = > ", arr)
print("arr.ndim = > ", arr.ndim)
print("arr.shape = > ", arr.shape)
print("arr.dtype = > ", arr.dtype)

# 它和 Python 列表很像，但是在这里，元素之间没有逗号。
import tensorflow as tf

# 现在使用 tf_convert_to_tensor 函数把这个数组转化为张量。
tensor = tf.convert_to_tensor(arr, tf.float64)
print("张量 = > ", tensor)

"""
这次的运行结果显示了张量具体的含义，但是不会展示出张量元素。
要想看到张量元素，需要像下面这样，运行一个会话：
"""
sess = tf.Session()
# 显示张量的数据
print("一维张量中的数据 => ", sess.run(tensor))
# 显示张量中指定的数据
print("一维张量中角标为1的数据 => ", sess.run(tensor[1]))

# -----------------     定义一维张量    -----------------

# -----------------     定义二维张量    -----------------
# 定义二维张量，其方法和定义一维张量是一样的，但要这样来定义数组：
arr = np.array([(1, 5.5, 3, 15, 20), (10, 20, 30, 40, 50), (60, 70, 80, 90, 100)])

tensor = tf.convert_to_tensor(arr)
sess = tf.Session()
print("二维张量中的数据 => ", sess.run(tensor))

# -----------------     定义二维张量    -----------------

# -----------------     在张量上进行数学运算    -----------------
arr1 = np.array([(1, 2, 3), (4, 5, 6)])
arr2 = np.array([(7, 8, 9), (10, 11, 12)])

# 求和
arr3 = tf.add(arr1, arr2)
sess = tf.Session()
tensor = sess.run(arr3)
print("二维张量求和结果的数据 => ", tensor)

# -----------------     在张量上进行数学运算    -----------------

# -----------------     三维张量 -----------------
"""
而是用一张 RGB 图片。在这张图片上，每一块像素都由 x、y、z 组合表示。
这些组合形成了图片的宽度、高度以及颜色深度。
首先使用 matplotlib 库导入一张图片。如果你的系统中没有 matplotlib ，可以 使用 pip来安装它。
将图片放在 Python 文件的同一目录下，接着使用 matplotlib 导入图片
"""
import matplotlib.image as img
import matplotlib.pyplot as plot

myfile = "onepiece.png"
myimgage = img.imread(myfile)
print("myimgage.ndim => ", myimgage.ndim)
print("myimgage.shape => ", myimgage.shape)

# 使用TensorFlow处理图片
"""
使用 TensorFlow 生成或裁剪图片
首先，向一个占位符赋值：
myimage = tf.placeholder("int32",[None,None,3])
使用裁剪操作来裁剪图像：
cropped = tf.slice(myimage,[10,0,0],[16,-1,-1])
最后，运行这个会话：
result = sess.run(cropped, feed\_dict={slice: myimage})
然后，你就能看到使用 matplotlib 处理过的图像了
"""
slice = tf.placeholder("int32", [None, None, 3])
cropped = tf.slice(myimgage, [10, 0, 0], [16, -1, -1])
sess = tf.Session()
result = sess.run(cropped, feed_dict={slice: myimgage})
# plot.imshow(result)
# plot.show()

"""
用 TensorFlow 改变图像
在本例中，我们会使用 TensorFlow 做一下简单的转换。
首先，指定待处理的图像，并初始化 TensorFlow 变量值：
myfile = "likegeeks.png"
myimage = img.imread(myfile)
image = tf.Variable(myimage,name='image')
vars = tf.global_variables_initializer()
然后调用 transpose 函数转换，这个函数用来翻转输入网格的 0 轴和 1 轴。
sess = tf.Session()
flipped = tf.transpose(image, perm=[1,0,2])
sess.run(vars)
result=sess.run(flipped)
接着你就能看到使用 matplotlib 处理过的图像了
"""

myfile = "onepiece.png"
myimgage = img.imread(myfile)
image = tf.Variable(myimgage, name='image')
vars = tf.global_variables_initializer()
sess = tf.Session()
flipped = tf.transpose(image, perm=[1, 0, 2])
sess.run(vars)
result = sess.run(flipped)
plot.imshow(result)
plot.show()
# -----------------     三维张量 -----------------
