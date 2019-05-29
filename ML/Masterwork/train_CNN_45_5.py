import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import datetime

begin = datetime.datetime.now()
timestep = 9
predstep = 1
road = 112

def process(datapath):
    data = pd.DataFrame(pd.read_csv(datapath, error_bad_lines=False))
    X = np.array(data.iloc[:, 0:timestep*road])
    Y = np.array(data.iloc[:, timestep*road:])

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)

    scaler = StandardScaler().fit(X_train)
    scaler_name = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_CNN_45min_5min_road.save'
    joblib.dump(scaler, scaler_name)
    rescaledX_train = scaler.transform(X_train)
    X_ = scaler.transform(X)
    rescaledX_test = scaler.transform(X_test)

    return X_, rescaledX_test, Y, Y_test

def generateBatch(batchsize, X, Y, train=True):
    x_batches = []
    y_batches = []

    for i in range(batchsize):
        x_batch = []
        y_batch = []
        if train:
            index = random.randint(0, X.shape[0]-1)
        else:
            index = i
        # index=i
        for k in range(len(X[index])/road): # iterate according to timestep
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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=2, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


class CNN:

    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.00002):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep

    def build_CNN(self, ):

        # conv first
        bottom = tf.reshape(self.bottom, [-1, road, timestep, 1])
        W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.elu(conv2d(bottom, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # W_conv2 = weight_variable([3,3,256,128])
        # b_conv2 = bias_variable([128])
        # h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2)+b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
        #
        #
        # W_conv3 = weight_variable([3, 3, 128, 64])
        # b_conv3 = bias_variable([64])
        # h_conv3 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool3 = max_pool_2x2(h_conv3)


        h_flat3 = tf.reshape(h_pool1, [-1, 56 * 5 * 64])
        W_fc2 = weight_variable([56 * 5  * 64, 1200])
        b_fc2 = bias_variable([1200])
        h = tf.nn.elu(tf.matmul(h_flat3, W_fc2) + b_fc2)
        # h_flat3 = tf.reshape(h_pool3, [-1, 400])
        W_fc2 = weight_variable([1200, road * predstep])
        b_fc2 = bias_variable([road * predstep])
        self.predict = tf.nn.elu(tf.matmul(h, W_fc2) + b_fc2)


        # # fc first
        # W_fc1 = weight_variable([road * timestep, 784])
        # b_fc1 = bias_variable([784])
        # fc1 = tf.nn.elu(tf.matmul(self.bottom, W_fc1) + b_fc1)
        # fc1_conv = tf.reshape(fc1, [-1, 28, 28, 1])
        #
        # W_conv1 = weight_variable([5, 5, 1, 3])
        # b_conv1 = bias_variable([3])
        # h_conv1 = tf.nn.elu(conv2d(fc1_conv, W_conv1) + b_conv1)
        #
        # W_conv2 = weight_variable([3, 3, 3, 6])
        # b_conv2 = bias_variable([6])
        # h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2) + b_conv2)
        #
        #
        # W_conv3 = weight_variable([3, 3, 6, 6])
        # b_conv3 = bias_variable([6])
        # h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3) + b_conv3)
        #
        # h_flat1 = tf.reshape(h_conv3, [-1, 4704])
        # W_fc2 = weight_variable([4704, road*predstep])
        # b_fc2 = bias_variable([road*predstep])
        # self.predict = tf.nn.elu(tf.matmul(h_flat1, W_fc2) + b_fc2)

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.0002 #tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target-self.predict)/self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict
def process2(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*road])
    Y = np.array(data.iloc[:, timestep*road:])
    scaler = joblib.load('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_CNN_45min_5min_road.save')
    rescaledX = scaler.transform(X)
    return rescaledX, Y

def generateBatch2(batchsize, X, Y, train=True):
    x_batches = []
    y_batches = []
    for i in range(batchsize):
        x_batch = []
        y_batch = []
        if train:
            index = random.randint(0, X.shape[0]-1)
        else:
            index = i
        for k in range(len(X[index])/road): # iterate according to timestep
            s = X[index][k*road:(k+1)*road]
            x_batch = np.append(x_batch, X[index][k*road:(k+1)*road]) # np.transpose(np.reshape(np.array(X[index][k*50:(k+1)*50]), (2,5,5)),(1,2,0))# 5 road * 5 road * 2 kind of direction
        y_batch = Y[index]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches
with tf.Session() as sess:
    train_episode = 120000
    test_episode = 100
    batch_size = 32
    X_train, X_test, Y_train, Y_test = process(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_road20180122-0413_3.csv')

    model = CNN()
    model.build_CNN()

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint('/home/administrator/pywork/DeepLearningXC/trainingDir/logs_c_45min_5min_road/')

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
    hm_epochs = 1000000
    i = 0
    flag = 0
    lastepoch_loss = 0
    for epoch in range(hm_epochs):

        epoch_loss = 0
        for _ in range(int(len(X_train) / batch_size)):
            i+=1
            x_batch, y_batch = generateBatch(batchsize=batch_size, X=X_train, Y=Y_train)
            _, loss_, acc_train = sess.run([model.trainop, model.loss, model.accuracy],
                                           feed_dict={model.bottom: x_batch, model.target: y_batch})
            trainloss.append(loss_)
            trainacc.append(acc_train)
            epoch_loss += loss_

        saver.save(sess, os.path.join(
            '/home/administrator/pywork/DeepLearningXC/trainingDir/logs_c_45min_5min_road/'), global_step=epoch)
        print('Process %d / %d, loss:%d')%(epoch+1, hm_epochs, epoch_loss)
        x_, y_ = generateBatch(batchsize=12*24, X=X_train, Y=Y_train)
        pred = sess.run([model.predict],
                            feed_dict={model.bottom: x_, model.target: y_})

        pred = np.array(pred)
        y_ = np.array(y_)
        s=np.shape(y_)

        pred=pred[0,:,:]
        #
        # fig,ax=plt.subplots()
        # ax.plot(pred[0:12 * 24, 3], color='blue',label='preiction')
        # ax.plot(y_[:, 3], color='red',label='field data')
        # ax.legend()
        # plt.show()
        # task=pd.DataFrame()
        # task['predict']=pred[0:12 * 24, 3]
        # task['field']=y_[:, 3]
        # task.to_csv('cnnresult.csv')
        x_batch_test, y_batch_test = generateBatch(batchsize=len(X_test), X=X_test, Y=Y_test)
        loss_test, acc_test = sess.run([model.loss, model.accuracy],
                                   feed_dict={model.bottom: x_batch_test, model.target: y_batch_test})
        test_step.append(i)
        print '>>>test epoch:%d, loss:%g, accuracy:%g' % (epoch, loss_test, acc_test)
        testloss.append(loss_test)
        testacc.append(acc_test)
        if loss_test - lastepoch_loss >= 0:
            flag += 1

        if loss_test - lastepoch_loss < -0.05:
            flag = 0
        if flag >= 3:
            print 'early stop'
            X_test1, Y_test1 = process2(
                datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_3.csv')
            x_batch, y_batch = generateBatch2(batchsize=len(X_test1), X=X_test1, Y=Y_test1, train=False)

            pre, loss_, acc = sess.run([model.predict, model.loss, model.accuracy],
                                       feed_dict={model.bottom: x_batch, model.target: y_batch })
            y_batch = np.array(y_batch)
            pre = np.array(pre)

            ydf = pd.DataFrame(y_batch[:, :])
            predf = pd.DataFrame(pre[:, :])
            print '!! accuracy: %g loss: %g' % (acc, loss_)
            # ydf.to_csv(
            #     '/home/administrator/pywork/DeepLearningXC/trainingDir/real455.csv')
            # predf.to_csv(
            #     '/home/administrator/pywork/DeepLearningXC/trainingDir/predict_CNN455.csv')
            break
        lastepoch_loss = loss_test
        if epoch % 20 == 0:
            X_test1, Y_test1 = process2(
                datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_3.csv')
            x_batch, y_batch = generateBatch2(batchsize=len(X_test1), X=X_test1, Y=Y_test1, train=False)

            pre, loss_, acc = sess.run([model.predict, model.loss, model.accuracy],
                                       feed_dict={model.bottom: x_batch, model.target: y_batch })
            y_batch = np.array(y_batch)
            pre = np.array(pre)
            # plt.plot(pre[0:12 * 24, 2], color='red')
            # plt.plot(y_batch[0:12 * 24, 2])
            # plt.show()
            ydf = pd.DataFrame(y_batch[:, :])
            predf = pd.DataFrame(pre[:, :])
            print '!! accuracy: %g loss: %g' % (acc, loss_)
            # ydf.to_csv(
            #     '/home/administrator/pywork/DeepLearningXC/trainingDir/real455.csv')
            # predf.to_csv(
            #     '/home/administrator/pywork/DeepLearningXC/trainingDir/predict_CNN455.csv')

    end = datetime.datetime.now()
    print 'time cost: %s' % (end - begin)



