import common_header
import tensorflow as tf
from sklearn.datasets import load_digits
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

"""
dropout 解决 overfiting问题: 
dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃


过拟合 (Overfitting)

视频资料: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-A-overfitting/
"""

"""
建立 dropout 层 
"""
# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)

print("X = -> ")
print(X)
print("X = <- ")
print()

print("y = -> ")
print(y)
print("y = <- ")
print()

# 数据分割
# 验证集(测试集)占训练集30%
# 70%的数据做为训练集的数据，30%的数据做为测试集的数据
# X 为特征值， y 为目标值
# 其中X_train是训练数据, X_test是测试数据。 然后添加隐含层和输出层
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

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


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    # tensorboard histogram 可视化
    tf.summary.histogram(layer_name + "/outputs", outputs)
    return outputs


# define placeholder for inputs to network
"""
这里的keep_prob是保留概率，即我们要保留的结果所占比例，它作为一个placeholder，在run时传入， 当keep_prob=1的时候，相当于100%保留，
也就是dropout没有起作用。下面我们分析一下程序结构，首先准备数据，
"""
keep_prob = tf.placeholder(tf.float32)
# 8行8列
xs = tf.placeholder(tf.float32, [None, 64])  # 8*8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# 输入64个数据，输出50个数据,layer1 ,input layer
# 这里传入到add_layer的数据是占位符，在运行的时候需要进行赋值
l1 = add_layer(xs, 64, 50, "l1", activation_function=tf.nn.tanh)
# output 层
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
"""
loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
# tensorboard scalar 可视化
tf.summary.scalar('loss', cross_entropy)
# train方法（最优化算法）采用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# tensorboard 合并所有的summary,合并汇总
merged = tf.summary.merge_all()
# tensorboard summary write goes in here
# 训练集 -> 将汇总写入磁盘
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
# 测试集 -> 将汇总写入磁盘
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())

"""
训练 

最后开始train，总共训练500次。
"""
for i in range(500):
    # 保留0.4,这里就需要设置为0.6,0.6是被drop掉的数据
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        # 训练集的结果
        # 输出的时候，保持1
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        # 测试集的结果
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        # 将结果添加到summary中
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

"""
可视化结果 

训练中keep_prob=1时，就可以暴露出overfitting问题。keep_prob=0.5时，dropout就发挥了作用。 我们可以两种参数分别运行程序，对比一下结果。

当keep_prob=1时，模型对训练数据的适应性优于测试数据，存在overfitting，输出如下： 红线是 train 的误差, 蓝线是 test 的误差.
"""
