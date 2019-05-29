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
road =112
network=112
def process(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    # kick out intersection

    X = np.array(data.iloc[:, 0:(timestep+0)*road])

    Y = np.array(data.iloc[:, (timestep+0)*road:])

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

    scaler = StandardScaler().fit(X_train)
    scaler_name = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_fc_30min_15min.save'

    joblib.dump(scaler, scaler_name)

    rescaledX_train = scaler.transform(X_train)

    rescaledX_test = scaler.transform(X_test)

    return rescaledX_train, rescaledX_test, Y_train, Y_test

def generateBatch(batchsize, X, Y, act_roads=[82, 76, 35, 7],train=True,epoch=0 ):
    x_batches = []
    y_batches = []
    n = len(act_roads)
    act_roadsX = []

    # X = X[:,act_roadsX]
    # Y = Y[:,act_roads]
    for i in range(batchsize):
        x_batch = []
        y_batch = []
        if train:
            index = i+int(epoch)*batch_size
            # index = random.randint(0, X.shape[0]-1)
        else:
            index = i

        # for k in act_roads: # iterate according to timestep
        #     for t in range(timestep):
        #         x_batch = np.append(x_batch, X[index][k+t*network])
        # for t in range(timestep): # iterate according to network size
        #     for k in act_roads:
        #         x_batch = np.append(x_batch, X[index][k+t*network])
        for k in range(len(X[index])/road): # iterate according to timestep
            s = X[index][k*road:(k+1)*road]
            x_batch = np.append(x_batch, X[index][k*road:(k+1)*road])
        y_batch = Y[index]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.constant(0.2, shape=shape)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class FC:

    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.00002):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep

    def build_FC(self, ):

        W_fc1 = weight_variable([self.input_size,40])
        b_fc1 = bias_variable([40])
        fc1 = tf.nn.elu(tf.matmul(self.bottom, W_fc1)+b_fc1)

        W_fc2 = weight_variable([40, 40])
        b_fc2 = bias_variable([40])
        fc2 = tf.nn.elu(tf.matmul(fc1, W_fc2) + b_fc2)

        W_fc3= weight_variable([40, self.output_size])
        b_fc3 = bias_variable([self.output_size])
        self.predict = tf.nn.elu(tf.matmul(fc2, W_fc3) + b_fc3)

        global_step = tf.Variable(0, trainable=False)
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01, scope=None)
        self.weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, self.weights)

        self.learning_rate = 0.00004 # tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target-self.predict)/self.target)
        # self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        self.trainop = tf.train.RMSPropOptimizer(self.learning_rate,0.99,0.0,1e-6).minimize(self.loss, global_step=global_step)
        return self.predict

with tf.Session() as sess:
    train_episode = 20000
    test_episode = 100
    batch_size = 32
    X_train, X_test, Y_train, Y_test = process(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_road20180122-0413_.csv')

    model = FC()
    model.build_FC()

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint('/home/administrator/pywork/DeepLearningXC/trainingDir/logs_fc_45min_5min')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    if checkpoint:
        saver.restore(sess, checkpoint)
        print("## restore from the checkpoint {0}".format(checkpoint))

    print('## start training...')
    trainloss = []
    trainacc = []
    test_step = []
    testloss = []
    testacc = []
    # for test after training
    acc = 0
    num = 0
    hm_epochs = 20000
    i = 0
    lastepoch_loss = 0
    flag = 0
    #
    # for i in range(train_episode):
    #     flag = 0
    #     x_batch, y_batch = generateBatch(batchsize=batch_size, X=X_train, Y=Y_train)
    #     _, loss_, acc_train = sess.run([model.trainop, model.loss, model.accuracy], feed_dict={model.bottom:x_batch, model.target:y_batch})
    #     trainloss.append(loss_)
    #     trainacc.append(acc_train)
    #     if i % 10 == 0:
    #         print 'step:%d, loss:%g, accuracy:%g'%(i, loss_, acc_train)
    #         if loss_<10:
    #             saver.save(sess, os.path.join('/home/administrator/pywork/DeepLearning2.0/trainingDir/logs_fc_30min_15min/'), global_step=i)
    #     if i % 100 == 0:
    #         x_batch_test, y_batch_test = generateBatch(batchsize=batch_size, X=X_test, Y=Y_test)
    #         loss_test, acc_test = sess.run([model.loss, model.accuracy], feed_dict={model.bottom: x_batch_test, model.target: y_batch_test})
    #         test_step.append(i)
    #         print '######## test step:%d, loss:%g, accuracy:%g' % (i/10, loss_test, acc_test)
    #         testloss.append(loss_test)
    #         testacc.append(acc_test)

    for epoch in range(hm_epochs):

        epoch_loss = 0
        epoch_acc = 0
        for e in range(int(len(X_train) / batch_size)):
            i+=1
            x_batch, y_batch = generateBatch(batchsize=batch_size, X=X_train, Y=Y_train,epoch=e)
            _, loss_, acc_train = sess.run([model.trainop, model.loss, model.accuracy],
                                           feed_dict={model.bottom: x_batch, model.target: y_batch})
            trainloss.append(loss_)
            trainacc.append(acc_train)
            epoch_loss += loss_
            epoch_acc += acc_train
        saver.save(sess, os.path.join(
            '/home/administrator/pywork/DeepLearningXC/trainingDir/logs_fc_45min_5min/'), global_step=epoch)
        print('Process %d / %d, loss:%d, Accuracy:%g')%(epoch+1, hm_epochs, epoch_loss, float(epoch_acc)/int(len(X_train) / batch_size))

        x_batch_test, y_batch_test = generateBatch(batchsize=len(X_test), X=X_test, Y=Y_test)
        loss_test, acc_test = sess.run([model.loss, model.accuracy],
                                   feed_dict={model.bottom: x_batch_test, model.target: y_batch_test})
        test_step.append(i)
        print '     >>>test epoch:%d, loss:%g, accuracy:%g' % (epoch, loss_test, acc_test)
        testloss.append(loss_test)
        testacc.append(acc_test)
        if epoch_loss - lastepoch_loss >= 0:
            flag += 1

        if epoch_loss - lastepoch_loss < -5:
            flag = 0
        if flag >= 5:
            print 'early stop'
            break
        lastepoch_loss = epoch_loss
    plt.plot(trainloss,'r', label='Train Loss')
    plt.plot(test_step, testloss, 'b', label='Test Loss')
    plt.show()



