# 学习笔记
## constant使用
  # Constant 1-D Tensor populated with value list.
  tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

  # Constant 2-D tensor populated with scalar value -1.
  tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                               [-1. -1. -1.]]