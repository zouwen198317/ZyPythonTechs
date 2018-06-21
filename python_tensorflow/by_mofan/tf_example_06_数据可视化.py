import common_header
import tensorflow as tf

"""
    图和histogram的使用
      有图才会显示图，有histogram才会显示如果没有则不会显示
"""

# activation_function 如果没有默认就是一个线性函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        # weights是一个随机变量，在生成初始weights的时候会比全部为0要好
        # 变量矩阵,in_size行 * out_size列，
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # 总结名字
            tf.summary.histogram(layer_name + "/weigths", Weights)
        # 类似列表的东东
        # biases在ML中推荐不为0,所以加的0.1
        # Weights为随机变量的值，biases全部为0.1,这两个值在每一步的训练中都会有变化
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + "/biases", biases)

        # inputs * weights + biases 这里使用的是矩阵的乘法
        # 计算出来的值（还没有被激活的值存储在该Wx_plus_biases变量中)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_biases = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            # 保持现状， 为线性
            outputs = Wx_plus_biases
        else:
            # 非线性
            outputs = activation_function(Wx_plus_biases)
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs


import numpy as np

# 1个特性,300条数据(-1,1之间的数据)
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 噪音的添加
noise = np.random.normal(0, 0.05, x_data.shape)
# 真实值是基于x_data,2次方-0.5
y_data = np.square(x_data) - 0.5 + noise

'''
定义input: input包括x_input,y_input
'''
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

# 构造隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

# 构造输出层
# 线性函数
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# 计算prediction和y_data（预测值和真实值之间的一个差别)
# square 这个结果是对每一个例子的平方
# reduce_sum对每一个例子进行求和
# reduce_mean对每一个例子的和求平均值
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(ys - prediction, name='tf.square'), reduction_indices=[1], name='reduce_sum'))
    # event中显示 不断的在减小说明是有学习到东西
    tf.summary.scalar('loss', loss)

# 预测
# 从误差中去学习
# learning_rate 学习效率有多高,这个值通常是要小于1的
# minimize减少误差
# 第一个的训练都是通过优化器去减小误差
# GradientDescentOptimizer 最基础的线性优化器
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# 初始化所有的变量
sess = tf.Session()

# 合并所有的summary,打包放到logs目录中
merged = tf.summary.merge_all()
# 收集的信息放到logs中
# tensorflow 新版取消了tf.train.SummaryWriter()，换成使用tf.summary.FileWriter()
writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

# 可视化训练
"""
1. log上一层目录 
2. tensorboard --logdir=logs
3. 电脑上浏览即可
"""

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        # merged 只有在run的时候才会真正的发挥它的作用
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        # i 是记录的步数
        writer.add_summary(result, i)