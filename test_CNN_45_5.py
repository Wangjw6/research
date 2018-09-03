import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from scipy.stats import gaussian_kde
timestep = 9
predstep = 1
road = 112
def process(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*road])
    Y = np.array(data.iloc[:, timestep*road:])
    plt.plot(Y[0:12*24,1])
    plt.show()
    scaler = joblib.load('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_CNN_45min_5min_road.save')
    rescaledX = scaler.transform(X)
    return rescaledX, Y

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
        for k in range(len(X[index])/road): # iterate according to timestep
            s = X[index][k*road:(k+1)*road]
            x_batch = np.append(x_batch, X[index][k*road:(k+1)*road]) # np.transpose(np.reshape(np.array(X[index][k*50:(k+1)*50]), (2,5,5)),(1,2,0))# 5 road * 5 road * 2 kind of direction
        y_batch = Y[index]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class CNN:
    def __init__(self, save_or_load_path=None, trainable=True, learning_rate=0.00002):
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
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.elu(conv2d(bottom, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_flat1 = tf.reshape(h_pool1, [-1, 8960])
        W_fc1 = weight_variable([8960, 2400])
        b_fc1 = bias_variable([2400])
        fc1 = tf.nn.elu(tf.matmul(h_flat1, W_fc1) + b_fc1)

        fc1_drop = tf.nn.dropout(fc1, keep_prob=0.8)
        W_fc2 = weight_variable([2400, road * predstep])
        b_fc2 = bias_variable([road * predstep])
        self.predict = tf.nn.elu(tf.matmul(fc1_drop, W_fc2) + b_fc2)

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
        self.learning_rate = 0.0002  # tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target - self.predict) / self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict

with tf.Session() as sess:
    test_episode = 100
    X_test, Y_test = process(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_.csv')
    batch_size = len(X_test)

    model = CNN()
    model.build_CNN()

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint('/home/administrator/pywork/DeepLearningXC/trainingDir/logs_c_45min_5min_road/')

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
    # plt.title('CNN: '+str(i))
    plt.plot([0,80],[0,80],'r')
    a = np.array(pre).reshape(-1,)
    b = np.array(y_batch).reshape(-1,)
    plt.xlabel('speed predicted (km/h)')
    plt.ylabel('speed detected (km/h)')
    xy=np.vstack([a, b])
    z=gaussian_kde(xy)(xy)
    plt.scatter(a, b, c=z)
    plt.grid(True,linewidth="0.3")
    # plt.savefig('CNN.jpg')
    plt.show()
    for i in range(112):
        plt.figure(i)
        # scatter
        plt.title('CNN: road '+str(i))
        plt.plot([20,70],[20,70],'r')
        roadindex=i
        a = np.array(pre)[:,roadindex]
        plt.xlabel('speed predicted (km/h)')
        plt.ylabel('speed detected (km/h)')
        plt.scatter(a, np.array(y_batch)[:,roadindex])
        plt.grid(True,linewidth="0.3")
        plt.savefig('CNN'+str(i)+'.jpg')
        # time series
        # plt.title('CNN: '+str(i))
        # roadindex=i
        # a = np.array(pre)[-12*24:,roadindex]
        # plt.xlabel('time tag')
        # plt.ylabel('speed')
        # plt.plot(a, color='blue')
        # plt.plot(np.array(y_batch)[-12*24:, roadindex], color='red')
        # # plt.grid(True,linewidth="0.3")
        # plt.savefig('CNN'+str(i)+'ts.jpg')
    # plt.show()
    loss = 0
    mselist=[]
    mapelist = []
    for i in range(len(pre)):
        for j in range(len(pre[i])):
            acc = acc + abs(pre[i][j] - Y_test[i][j])/Y_test[i][j]
            loss += (pre[i][j] - Y_test[i][j])*(pre[i][j] - Y_test[i][j])
            mselist.append(float((pre[i][j] - Y_test[i][j])*(pre[i][j] - Y_test[i][j])))
            mapelist.append(float(abs(pre[i][j] - Y_test[i][j])/Y_test[i][j]))
            num+=1
    print 'accuracy: %g' % (1-float(acc)/num)
    print 'mse var: %g' % (np.var(np.array(mselist)))
    print 'mape var: %g' % (np.var(np.array(mapelist)))
    # prediction
    road1 = []
    road2 = []
    road3 = []
    road4 = []
    road5 = []
    road6 = []
    road7 = []
    road8 = []
    road9 = []
    road10 = []
    road11 = []
    road12 = []
    road13 = []
    road14 = []
    road15 = []
    road16 = []
    road17 = []
    road18 = []
    road19 = []
    road20 = []
    road21 = []
    road22 = []
    road23 = []
    road24 = []

    # real
    road1_ = []
    road2_ = []
    road3_ = []
    road4_ = []
    road5_ = []
    road6_ = []
    road7_ = []
    road8_ = []
    road9_ = []
    road10_ = []
    road11_ = []
    road12_ = []
    road13_ = []
    road14_ = []
    road15_ = []
    road16_ = []
    road17_ = []
    road18_ = []
    road19_ = []
    road20_ = []
    road21_ = []
    road22_ = []
    road23_ = []
    road24_ = []

    prea = np.array(pre)
    y_batcha=np.array(y_batch)
    plt.title('BaseLine')
    p, = plt.plot(prea[:,1], 'ro-',  label='pred')
    r, = plt.plot(y_batcha[:,1], 'bh-', label='real')
    plt.legend([p, r], ['prediction', 'real'])
    plt.draw()
    plt.show()
    # plt.figure(figsize=(64, 64))
    # plt.ion()

    for i in range(batch_size):
        road1.append(pre[i][0])
        road2.append(pre[i][1])
        road3.append(pre[i][2])
        road4.append(pre[i][3])
        road5.append(pre[i][4])
        road6.append(pre[i][5])
        road7.append(pre[i][6])
        road8.append(pre[i][7])
        road9.append(pre[i][8])
        road10.append(pre[i][9])
        road11.append(pre[i][10])
        road12.append(pre[i][11])
        road13.append(pre[i][12])
        road14.append(pre[i][13])
        road15.append(pre[i][14])
        road16.append(pre[i][15])
        road17.append(pre[i][16])
        road18.append(pre[i][17])
        road19.append(pre[i][18])
        road20.append(pre[i][19])
        road21.append(pre[i][20])
        road22.append(pre[i][21])
        road23.append(pre[i][22])
        road24.append(pre[i][23])

        ax1 = plt.subplot2grid((8, 3), (0, 0))
        ax1.plot(road1, color='red', label='pred')
        ax1.plot(road1_, color='blue', label='real')

        ax2 = plt.subplot2grid((8, 3), (0, 1))
        ax2.plot(road2, color='red', label='pred')
        ax2.plot(road2_, color='blue', label='real')

        ax3 = plt.subplot2grid((8, 3), (0, 2))
        ax3.plot(road3, color='red', label='pred')
        ax3.plot(road3_, color='blue', label='real')

        ax4 = plt.subplot2grid((8, 3), (1, 0))
        ax4.plot(road4, color='red', label='pred')
        ax4.plot(road4_, color='blue', label='real')

        ax5 = plt.subplot2grid((8, 3), (1, 1))
        ax5.plot(road5, color='red', label='pred')
        ax5.plot(road5_, color='blue', label='real')

        ax6 = plt.subplot2grid((8, 3), (1, 2))
        ax6.plot(road6, color='red', label='pred')
        ax6.plot(road6_, color='blue', label='real')

        ax7 = plt.subplot2grid((8, 3), (2, 0))
        ax7.plot(road7, color='red', label='pred')
        ax7.plot(road7_, color='blue', label='real')

        ax8 = plt.subplot2grid((8, 3), (2, 1))
        ax8.plot(road8, color='red', label='pred')
        ax8.plot(road8_, color='blue', label='real')

        ax9 = plt.subplot2grid((8, 3), (2, 2))
        ax9.plot(road9, color='red', label='pred')
        ax9.plot(road9_, color='blue', label='real')

        ax10 = plt.subplot2grid((8, 3), (3, 0))
        ax10.plot(road10, color='red', label='pred')
        ax10.plot(road10_, color='blue', label='real')

        ax11 = plt.subplot2grid((8, 3), (3, 1))
        ax11.plot(road11, color='red', label='pred')
        ax11.plot(road11_, color='blue', label='real')

        ax12 = plt.subplot2grid((8, 3), (3, 2))
        ax12.plot(road12, color='red', label='pred')
        ax12.plot(road12_, color='blue', label='real')

        ax13 = plt.subplot2grid((8, 3), (4, 0))
        ax13.plot(road13, color='red', label='pred')
        ax13.plot(road13_, color='blue', label='real')

        ax14 = plt.subplot2grid((8, 3), (4, 1))
        ax14.plot(road14, color='red', label='pred')
        ax14.plot(road14_, color='blue', label='real')

        ax15 = plt.subplot2grid((8, 3), (4, 2))
        ax15.plot(road15, color='red', label='pred')
        ax15.plot(road15_, color='blue', label='real')

        ax16 = plt.subplot2grid((8, 3), (5, 0))
        ax16.plot(road16, color='red', label='pred')
        ax16.plot(road16_, color='blue', label='real')

        ax17 = plt.subplot2grid((8, 3), (5, 1))
        ax17.plot(road17, color='red', label='pred')
        ax17.plot(road17_, color='blue', label='real')

        ax18 = plt.subplot2grid((8, 3), (5, 2))
        ax18.plot(road18, color='red', label='pred')
        ax18.plot(road18_, color='blue', label='real')

        ax19 = plt.subplot2grid((8, 3), (6, 0))
        ax19.plot(road19, color='red', label='pred')
        ax19.plot(road19_, color='blue', label='real')

        ax20 = plt.subplot2grid((8, 3), (6, 1))
        ax20.plot(road20, color='red', label='pred')
        ax20.plot(road20_, color='blue', label='real')

        ax21 = plt.subplot2grid((8, 3), (6, 2))
        ax21.plot(road21, color='red', label='pred')
        ax21.plot(road21_, color='blue', label='real')

        ax22 = plt.subplot2grid((8, 3), (7, 0))
        ax22.plot(road22, color='red', label='pred')
        ax22.plot(road22_, color='blue', label='real')

        ax23 = plt.subplot2grid((8, 3), (7, 1))
        ax23.plot(road23, color='red', label='pred')
        ax23.plot(road23_, color='blue', label='real')

        ax24 = plt.subplot2grid((8, 3), (7, 2))
        ax24.plot(road24, color='red', label='pred')
        ax24.plot(road24_, color='blue', label='real')

        road1_.append(y_batch[i][0])
        road2_.append(y_batch[i][1])
        road3_.append(y_batch[i][2])
        road4_.append(y_batch[i][3])
        road5_.append(y_batch[i][4])
        road6_.append(y_batch[i][5])
        road7_.append(y_batch[i][6])
        road8_.append(y_batch[i][7])
        road9_.append(y_batch[i][8])
        road10_.append(y_batch[i][9])
        road11_.append(y_batch[i][10])
        road12_.append(y_batch[i][11])
        road13_.append(y_batch[i][12])
        road14_.append(y_batch[i][13])
        road15_.append(y_batch[i][14])
        road16_.append(y_batch[i][15])
        road17_.append(y_batch[i][16])
        road18_.append(y_batch[i][17])
        road19_.append(y_batch[i][18])
        road20_.append(y_batch[i][19])
        road21_.append(y_batch[i][20])
        road22_.append(y_batch[i][21])
        road23_.append(y_batch[i][22])
        road24_.append(y_batch[i][22])

        plt.pause(0.2)
        plt.show()




