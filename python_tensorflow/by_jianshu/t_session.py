import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])
print(tensor)

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# method1
session = tf.Session()
result = session.run(product)
print(result)
session.close()
