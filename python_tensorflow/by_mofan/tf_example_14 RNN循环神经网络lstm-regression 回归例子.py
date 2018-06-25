import common_header

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

BATCH_START = 0
TIME_STEPS = 0
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def get_batch():
    pass


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        with tf.variable_scope('out_hidden'):
            self.add_output_layer()

        with tf.name_scope('cost'):
            self.compute_cost()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdadeltaOptimizer(LR).minimize(self.cost)

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size],
                            name='2_2D')  # # (batch*n_step, in_size)
        # Ws(in_size,cell_size)
        WS_in = self._weight_variable([self.input_size, self.cell_size])
        # bs(cell_size,)
        bs_in = self._bias_variable(self.cell_size, )
        # l_in_y = (batch*n_steps,cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, WS_in) + bs_in
        # reshape l_in_y ==> (batch,n_steps,cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0,
                                                 state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_in_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outpus, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_in_state, time_major=False)
        pass

    def add_output_layer(self):
        # shape = (batch*steps,cell_size)
        l_out_x = tf.reshape(self.cell_outpus, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch*steps,output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        # 旧的seq2seq接口也就是tf.contrib.legacy_seq2seq下的那部分，新的接口在tf.contrib.seq2seq下。
        losses = tf.contrib.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def _weight_variable(self, shape, name='weights'):
        initizlizer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initizlizer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initizlizer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initizlizer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
