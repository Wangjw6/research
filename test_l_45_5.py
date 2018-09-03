import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
timestep = 9
predstep = 1
road =112 # one road to predict compared to route

def process(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*road])
    Y = np.array(data.iloc[:, timestep*road:])
    scaler = joblib.load('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_lstm_45min_5min.save')
    rescaledX = scaler.transform(X)
    return rescaledX, Y

def generateBatch(batchsize, X, Y, train=True, epoch=0):
    x_batches = []
    y_batches = []
    for i in range(batchsize):
        x_batch = []
        y_batch = []
        if train:
            index = random.randint(0, X.shape[0]-1)
            index = i + int(epoch) * batch_size
        else:
            index = i
        for k in range(timestep): # iterate according to timestep
            s = X[index][k*road:(k+1)*road]
            x_batch = np.append(x_batch, X[index][k*road:(k+1)*road]) # np.transpose(np.reshape(np.array(X[index][k*50:(k+1)*50]), (2,5,5)),(1,2,0))# 5 road * 5 road * 2 kind of direction
        y_batch = Y[index]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.2, shape=shape)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class LSTM:

    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.00002):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.input_size = timestep*road
        self.output_size = predstep*road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep


    def build_LSTM(self, rnn_size, num_layers,  batch_size): # rnn_size : output_size
        bottom = tf.reshape(self.bottom, [-1, road])
        W_input = weight_variable([road, 40])
        b_input = bias_variable([40])
        input_ = tf.nn.elu(tf.matmul(bottom, W_input) + b_input)
        input_ = tf.reshape(input_, [-1, self.timestep, 40])

        self.cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
        self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=True)
        # initial_state = self.cell.zero_state(batch_size, tf.float32)

        outputs, last_state = tf.nn.dynamic_rnn(self.cell, input_, dtype=tf.float32)
        output_last_step = outputs[:,-1,:]

        W_output = weight_variable([rnn_size, self.output_size])
        b_output = bias_variable([self.output_size])
        self.predict = tf.nn.elu(tf.matmul(output_last_step, W_output)+b_output)
        # self.predict = self.fc_layer(bottom=output_last_step, in_size=rnn_size, out_size=self.output_size, activation=tf.nn.elu, name='fc_pred_')
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.predict))

        global_step = tf.Variable(0, trainable=False)
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001, scope=None)
        self.weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, self.weights)

        self.learning_rate = tf.train.exponential_decay(0.00004, global_step, 500, 0.9, staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))+regularization_penalty
        self.accuracy = 1. - tf.reduce_mean(abs(self.target-self.predict)/self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)

with tf.Session() as sess:
    test_episode = 100
    X_test, Y_test = process(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_.csv')
    batch_size = len(X_test)
    model = LSTM()
    model.build_LSTM(rnn_size=1024, num_layers=1,  batch_size=batch_size)

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint('/home/administrator/pywork/DeepLearningXC/trainingDir/logs_l_45min_5min')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    if checkpoint:
        saver.restore(sess, checkpoint)
        print("## restore from the checkpoint {0}".format(checkpoint))

    trainloss = []
    trainacc = []
    test_step = []
    testloss = []
    testacc = []
    # for test after training
    acc = 0
    num = 0

    x_batch, y_batch = generateBatch(batchsize=batch_size, X=X_test, Y=Y_test, train=False)

    pre, loss_, acc = sess.run([model.predict, model.loss, model.accuracy],
                               feed_dict={model.bottom: x_batch, model.target: y_batch})

    print 'accuracy: %g loss: %g' % (acc, loss_)






