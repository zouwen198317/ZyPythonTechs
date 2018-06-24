# 什么是循环神经网络 RNN (Recurrent Neural Network) #


### 学习资料: ###

- [Tensorflow RNN 例子1](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-07-RNN1/) 
- [Tensorflow RNN 例子2 ](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)
- [Tensorflow RNN 例子3](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-09-RNN3/)
- [PyTorch RNN 例子1](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-RNN-classification/)
- [PyTorch RNN 例子2](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/)
- [Keras 快速搭建 RNN 1](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-4-RNN-classifier/)
- [Keras 快速搭建 RNN 2](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-5-RNN-LSTM-Regressor/)
- [RNN 作曲 链接](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)
	

## RNN 用途： ##
现在请你看着这个名字. 不出意外, 你应该可以脱口而出. 因为你很可能就用了他们家的一款产品 . 那么现在, 请抛开这个产品, 只想着斯蒂芬乔布斯这个名字 , 请你再把他逆序念出来. 斯布乔(
*#&, 有点难吧. 这就说明, 对于预测, 顺序排列是多么重要. 我们可以预测下一个按照一定顺序排列的字, 但是打乱顺序, 我们就没办法分析自己到底在说什么了.


## 序列数据  ##

![](https://morvanzhou.github.io/static/results/ML-intro/rnn2.png)

我们想象现在有一组序列数据 data 0,1,2,3. 在当预测 result0 的时候,我们基于的是 data0, 同样在预测其他数据的时候, 我们也都只单单基于单个的数据. 每次使用的神经网络都是同一个 NN. 不过这些数据是有关联 顺序的 , 就像在厨房做菜, 酱料 A要比酱料 B 早放, 不然就串味了. 所以普通的神经网络结构并不能让 NN 了解这些数据之间的关联.

## 处理序列数据的神经网络  ##
![](https://morvanzhou.github.io/static/results/ML-intro/rnn3.png)

那我们如何让数据间的关联也被 NN 加以分析呢? 想想我们人类是怎么分析各种事物的关联吧, 最基本的方式,就是记住之前发生的事情. 那我们让神经网络也具备这种记住之前发生的事的能力. 再分析 Data0 的时候, 我们把分析结果存入记忆. 然后当分析 data1的时候, NN会产生新的记忆, 但是新记忆和老记忆是没有联系的. 我们就简单的把老记忆调用过来, 一起分析. 如果继续分析更多的有序数据 , RNN就会把之前的记忆都累积起来, 一起分析.

![](https://morvanzhou.github.io/static/results/ML-intro/rnn4.png)

我们再重复一遍刚才的流程, 不过这次是以加入一些数学方面的东西. 每次 RNN 运算完之后都会产生一个对于当前状态的描述 , state. 我们用简写 S( t) 代替, 然后这个 RNN开始分析 x(t+1) , 他会根据 x(t+1)产生s(t+1), 不过此时 y(t+1) 是由 s(t) 和 s(t+1) 共同创造的. 所以我们通常看到的 RNN 也可以表达成这种样子.


## RNN 的应用  ##
RNN 的形式不单单这有这样一种, 他的结构形式很自由. 如果用于分类问题, 比如说一个人说了一句话, 这句话带的感情色彩是积极的还是消极的. 那我们就可以用只有最后一个时间点输出判断结果的RNN.

又或者这是图片描述 RNN, 我们只需要一个 X 来代替输入的图片, 然后生成对图片描述的一段话.

或者是语言翻译的 RNN, 给出一段英文, 然后再翻译成中文.

有了这些不同形式的 RNN, RNN 就变得强大了. 有很多有趣的 RNN 应用. 比如之前提到的, 让 RNN 描述照片. 让 RNN 写学术论文, 让 RNN 写程序脚本, 让 RNN 作曲. 我们一般人甚至都不能分辨这到底是不是机器写出来的.


### Python相关教程 ###

- [Tensorflow RNN 例子](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)
- [PyTorch RNN 例子](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-RNN-classification/)
- [Keras 快速搭建 RNN](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-4-RNN-classifier/)


# RNN 循环神经网络 #


 
### 学习资料: ###

- 机器学习-简介系列 [什么是RNN](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-3-RNN/)
- 机器学习-简介系列 [什么是LSTM RNN](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-4-LSTM/)
- Google CNN [视频介绍](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/64063017560923#)
- Google RNN [视频介绍](https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63770919610923#)
- 为 TF 2017 打造的[新版可视化教学代码](https://github.com/MorvanZhou/Tensorflow-Tutorial)


## 简介系列  ##
RNN recurrent neural networks 在序列化的预测当中是很有优势的. 我很先看看 RNN 是怎么工作的. 建议查看后续制作的浓缩版 RNN 简介 机器学习-简介系列 什么是RNN




# 什么是 LSTM 循环神经网络 #

今天我们会来聊聊在普通RNN的弊端和为了解决这个弊端而提出的 LSTM 技术. LSTM 是 long-short term memory 的简称, 中文叫做 长短期记忆. 是当下最流行的 RNN 形式之一.

注: 本文不会涉及数学推导. 大家可以在很多其他地方找到优秀的数学推导文章.

## RNN 的弊端  ##

之前我们说过, RNN 是在有顺序的数据上进行学习的. 为了记住这些数据, RNN 会像人一样产生对先前发生事件的记忆. 不过一般形式的 RNN 就像一个老爷爷, 有时候比较健忘. 为什么会这样呢?


想像现在有这样一个 RNN, 他的输入值是一句话: ‘我今天要做红烧排骨, 首先要准备排骨, 然后…., 最后美味的一道菜就出锅了’, shua ~ 说着说着就流口水了. 现在请 RNN 来分析, 我今天做的到底是什么菜呢. RNN可能会给出“辣子鸡”这个答案. 由于判断失误, RNN就要开始学习 这个长序列 X 和 ‘红烧排骨’ 的关系 , 而RNN需要的关键信息 ”红烧排骨”却出现在句子开头,


再来看看 RNN是怎样学习的吧. 红烧排骨这个信息原的记忆要进过长途跋涉才能抵达最后一个时间点. 然后我们得到误差, 而且在 反向传递 得到的误差的时候, 他在每一步都会 乘以一个自己的参数 W. 如果这个 W 是一个小于1 的数, 比如0.9. 这个0.9 不断乘以误差, 误差传到初始时间点也会是一个接近于零的数, 所以对于初始时刻, 误差相当于就消失了. 我们把这个问题叫做梯度消失或者梯度弥散 Gradient vanishing. 反之如果 W 是一个大于1 的数, 比如1.1 不断累乘, 则到最后变成了无穷大的数, RNN被这无穷大的数撑死了, 这种情况我们叫做剃度爆炸, Gradient exploding. 这就是普通 RNN 没有办法回忆起久远记忆的原因



## LSTM ##

LSTM 就是为了解决这个问题而诞生的. LSTM 和普通 RNN 相比, 多出了三个控制器. (输入控制, 输出控制, 忘记控制). 现在, LSTM RNN 内部的情况是这样.

他多了一个 控制全局的记忆, 我们用粗线代替. 为了方便理解, 我们把粗线想象成电影或游戏当中的 主线剧情. 而原本的 RNN 体系就是 分线剧情. 三个控制器都是在原始的 RNN 体系上, 我们先看 输入方面 , 如果此时的分线剧情对于剧终结果十分重要, 输入控制就会将这个分线剧情按重要程度 写入主线剧情 进行分析. 再看 忘记方面, 如果此时的分线剧情更改了我们对之前剧情的想法, 那么忘记控制就会将之前的某些主线剧情忘记, 按比例替换成现在的新剧情. 所以 主线剧情的更新就取决于输入 和忘记 控制. 最后的输出方面, 输出控制会基于目前的主线剧情和分线剧情判断要输出的到底是什么.基于这些控制机制, LSTM 就像延缓记忆衰退的良药, 可以带来更好的结果.

### Python相关教程 ###

- Tensorflow RNN [例子](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)
- PyTorch RNN [例子](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-02-RNN-classification/)
- Keras [快速搭建 RNN](https://morvanzhou.github.io/tutorials/machine-learning/keras/2-4-RNN-classifier/)