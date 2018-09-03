import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import datetime
import multiprocessing as mp
import os


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

begin = datetime.datetime.now()
timestep = 9
predstep = 1
network = 112
path_to_clear = '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool'
def clean_dir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clean_dir(c_path)
        else:
            os.remove(c_path)


def process(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*network])

    Y = np.array(data.iloc[0:, timestep*network:])
    actroads = [88, 92, 90, 105, 107, 67, 65]
    # X_train = X[0:20000,:]
    # Y_train = Y[0:20000, :]
    # X_test = X[20000:21000,:]
    # Y_test = Y[20000:21000, :]
    # plt.imshow(Y_train[0:12*24,actroads], cmap='hot', interpolation='nearest', aspect='auto')

    # plt.axis('off')
    # plt.xlabel('location')
    # plt.ylabel('time')
    # plt.yticks([2.5,14.5,26.5,38.5,50.5,62.5,74.5,86.5,98.5,110.5,122.5,134.5,146.5,158.5,170.5,182.5,
    #             194.5,206.5,218.5,230.5,242.5,254.5,266.5,278.5],['0:00', '1:00','2:00','3:00','4:00','5:00','6:00',
    #                                               '7:00','8:00','9:00','10:00','11:00','12:00','13:00',
    #                                               '14:00','15:00','16:00','17:00','18:00','19:00',
    #                                               '20:00','21:00','22:00','23:00'])
    #     # plt.colorbar()
    # plt.savefig('real.jpg')
    # plt.show()

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(X_train)
    scaler_name = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_route_45min_5min.save'
    joblib.dump(scaler, scaler_name)
    rescaledX_train = scaler.transform(X_train)

    rescaledX_test = scaler.transform(X_test)

    return rescaledX_train, rescaledX_test, Y_train, Y_test
def testprocess(datapath):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*network])

    Y = np.array(data.iloc[:, timestep*network:])

    scaler = joblib.load('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/Scaler_route_45min_5min.save')
    rescaledX = scaler.transform(X)
    return rescaledX, Y

def generateBatch(batchsize, X, Y, act_roads,train=True,epoch=0 ):
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
            index = i+int(epoch)*batchsize
            # index = random.randint(0, X.shape[0]-1)
        else:
            index = i

        # for k in act_roads: # iterate according to timestep
        #     for t in range(timestep):
        #         x_batch = np.append(x_batch, X[index][k+t*network])
        for t in range(timestep): # iterate according to network size
            for k in act_roads:
                x_batch = np.append(x_batch, X[index][k+t*network])
        for t in range(predstep): # iterate according to network size
            for k in act_roads:
                y_batch = np.append(y_batch, Y[index][k+t*network])
        # y_batch = Y[index]
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    return x_batches, y_batches


def weight_variable(shape, name='w'):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape, name='b', value=0.):
    initial = tf.constant(0.01, shape=shape)
    # return initial
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return initial

class GLSTM:

    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.0002, road=3, rw = 0.):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.dropout = tf.placeholder(tf.float32, shape=[ ], name='dropout')
        self.step = road

    def build_GLSTM(self, rnn_size = 512, fb=1.):
        with tf.variable_scope(name_or_scope='s0'):
            flag=0
            routerelation=[]
            n_inputs = 1
            bilstmcelll_size = 32
            i = 0
            innersize = 32
            self.stateset=[]
            self.spatial=[]
            while i < timestep-1:
                st = []
                if i==0:
                    bottom0 = self.bottom[:,i*self.step : (i+1)*self.step]

                    for j in range(self.step):
                        with tf.variable_scope(name_or_scope='temporal' + str(i) + str(j)):
                            Wi = weight_variable([1, innersize], name='for input')
                            bi = bias_variable([innersize], name='for input')
                        input_ = tf.nn.elu(tf.matmul(tf.reshape(bottom0[:,i*self.step+j], (-1, 1)), Wi) + bi)
                        if j == 0:
                            bottom = input_
                        else:
                            bottom = tf.concat((bottom, input_), axis=1)
                    n_inputs = innersize
                    bottom = tf.reshape(bottom, [-1, self.step, n_inputs])
                    self.input1_ = tf.transpose(bottom, [1, 0, 2])
                    self.input1_ = tf.reshape(self.input1_, (-1, n_inputs))
                    x1 = tf.split(self.input1_, self.step)
                    with tf.variable_scope(name_or_scope='spatial'):
                        cell_fw1 = tf.contrib.rnn.BasicLSTMCell(bilstmcelll_size, forget_bias=fb)
                        cell_bw1 = tf.contrib.rnn.BasicLSTMCell(bilstmcelll_size, forget_bias=fb)
                        outputs_1, fws,bws= tf.contrib.rnn.static_bidirectional_rnn(cell_fw1, cell_bw1, x1, dtype=tf.float32)
                        self.spatial.append(outputs_1[:][0:bilstmcelll_size])
                    for j in range(self.step):
                        with tf.variable_scope(name_or_scope='temporal'+str(i)+str(j)):
                            Ws = weight_variable([bilstmcelll_size * 2, innersize], name='for state')
                            bs = bias_variable([innersize], name='for state')

                            Wi = weight_variable([1, innersize], name='for input')
                            bi = bias_variable([innersize], name='for input')

                            W = weight_variable([innersize, 1], name='for weight')
                            b = bias_variable([1], name='for weight', value=0.01)

                            Wo= weight_variable([innersize, 1], name='for output')
                            bo = bias_variable([1], name='for output', value=0.01)

                        last_state = tf.matmul(outputs_1[j], Ws) + bs
                        st.append(last_state)
                        input_ = tf.matmul(tf.reshape(self.bottom[:, (i + 1) * self.step + j], (-1, 1)), Wi) + bi
                        state =  tf.nn.elu(last_state + input_)
                        T = tf.nn.sigmoid(tf.matmul(state, W) + b)

                        if j == 0:
                            bottom = state
                        else:
                            bottom = tf.concat((bottom, state), axis=1)
                        j+=1
                if i>0 :
                    with tf.variable_scope(name_or_scope='spatial', reuse=True ):
                        cell_fw1 = tf.contrib.rnn.BasicLSTMCell(bilstmcelll_size, forget_bias=fb)
                        cell_bw1 = tf.contrib.rnn.BasicLSTMCell(bilstmcelll_size, forget_bias=fb)
                        bottom = tf.reshape(bottom, [-1, self.step, n_inputs])
                        self.input1_ = tf.transpose(bottom, [1, 0, 2])
                        self.input1_ = tf.reshape(self.input1_, (-1, n_inputs))
                        x1 = tf.split(self.input1_, self.step)
                        outputs_1, _,_ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw1, cell_bw1, x1, dtype=tf.float32)
                        self.spatial.append(outputs_1[:][0:bilstmcelll_size])
                    for j in range(self.step):
                        with tf.variable_scope(name_or_scope='temporal'+str(i)+str(j)):
                            Ws = weight_variable([bilstmcelll_size * 2, innersize], name='for state')
                            bs = bias_variable([innersize], name='for state')

                            Wi = weight_variable([1, innersize], name='for input')
                            bi = bias_variable([innersize], name='for input')

                            W = weight_variable([innersize, 1], name='for weight')
                            b = bias_variable([1], name='for weight')

                            Wo = weight_variable([innersize, 1], name='for output')
                            bo = bias_variable([1], name='for output')

                        last_state = (tf.matmul(outputs_1[j], Ws) + bs)
                        input_ = (tf.matmul(tf.reshape(self.bottom[:, (i + 1) * self.step + j], (-1, 1)), Wi) + bi)
                        state = tf.nn.elu(last_state + input_)
                        st.append(last_state)
                        if j == 0:
                            bottom = state
                        else:
                            bottom = tf.concat((bottom, state), axis=1)

                        j += 1
                self.stateset.append(st)
                i+=1




            self.Wp = weight_variable([innersize*self.step,self.step*predstep], name='for predict')
            self.bp = bias_variable([self.step*predstep], name='for predict')
            output_final = (tf.matmul(bottom,self.Wp)+self.bp) #+ self.bottom[:, (timestep-1) * self.step:]
            # W_input = weight_variable([self.step, 256])
            # b_input = bias_variable([256])
            # h1 = tf.nn.elu(tf.matmul(bottom, W_input)+b_input)
            # W_input2 = weight_variable([256, self.step])
            # b_input2 = bias_variable([self.step])
            # h2 = tf.nn.elu(tf.matmul(h1, W_input2) + b_input2)
            # input_ = tf.nn.elu(tf.matmul(routerelation, W_input) + b_input)
            # input_ = tf.reshape(input_, [-1, timestep, 800])
            #
            # self.cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
            # self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * 1, state_is_tuple=True)
            # # initial_state = self.cell.zero_state(batch_size, tf.float32)
            #
            # outputs, last_state = tf.nn.dynamic_rnn(self.cell, input_, dtype=tf.float32)

            self.predict = output_final


        global_step = tf.Variable(0, trainable=False)
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
        self.weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, self.weights)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.error =   tf.reduce_mean(abs(self.target-self.predict)/self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict

#####################################################################################################################################################################

#
# with tf.variable_scope('model1'):
#     # 23 22
#     m1 = GLSTM(road=2)
#     m1.build_GLSTM()
# with tf.variable_scope('model2'):
#     # 10 11
#     m2 = GLSTM(road=2)
#     m2.build_GLSTM()
# with tf.variable_scope('model3'):
#     # 17 19
#     m3 = GLSTM(road=2)
#     m3.build_GLSTM()
# with tf.variable_scope('model4'):
#     # 17 2
#     m4 = GLSTM(road=2)
#     m4.build_GLSTM()
# with tf.variable_scope('model5'):
#     # 20 11
#     m5 = GLSTM(road=2)
#     m5.build_GLSTM()
# with tf.variable_scope('model6'):
#     # 13 12
#     m6 = GLSTM(road=2)
#     m6.build_GLSTM()
# with tf.variable_scope('model7'):
#     # 14 19
#     m7 = GLSTM(road=2)
#     m7.build_GLSTM()
# with tf.variable_scope('model8'):
#     # 5 6
#     m8 = GLSTM(road=2)
#     m8.build_GLSTM()
# with tf.variable_scope('model9'):
#     # 4 13
#     m9 = GLSTM(road=2)
#     m9.build_GLSTM()
# with tf.variable_scope('model10'):
#     # 15 20
#     m10 = GLSTM(road=2)
#     m10.build_GLSTM()
# with tf.variable_scope('model11'):
#     # 19 10 11
#     m11 = GLSTM(road=3)
#     m11.build_GLSTM()
# with tf.variable_scope('model12'):
#     # 1 16
#     m12 = GLSTM(road=2)
#     m12.build_GLSTM()
# with tf.variable_scope('model13'):
#     # 0 1
#     m13 = GLSTM(road=2)
#     m13.build_GLSTM()
# with tf.variable_scope('model14'):
#     # 16 21
#     m14 = GLSTM(road=2)
#     m14.build_GLSTM()
# with tf.variable_scope('model15'):
#     # 8 3
#     m15 = GLSTM(road=2)
#     m15.build_GLSTM()
# with tf.variable_scope('model16'):
#     # 11 9
#     m16 = GLSTM(road=2)
#     m16.build_GLSTM()
# with tf.variable_scope('model17'):
#     # 7 2
#     m17 = GLSTM(road=2)
#     m17.build_GLSTM()
# with tf.variable_scope('model18'):
#     # 3 12
#     m18 = GLSTM(road=2)
#     m18.build_GLSTM()
# with tf.variable_scope('model19'):
#     # 8 6
#     m19 = GLSTM(road=2)
#     m19.build_GLSTM()
# with tf.variable_scope('model20'):
#     # 21 23
#     m20 = GLSTM(road=2)
#     m20.build_GLSTM()
# with tf.variable_scope('model21'):
#     # 5 3
#     m21 = GLSTM(road=2)
#     m21.build_GLSTM()
# with tf.variable_scope('model22'):
#     # 6 21
#     m22 = GLSTM(road=2)
#     m22.build_GLSTM()
# with tf.variable_scope('model23'):
#     # 15 6
#     m23 = GLSTM(road=2)
#     m23.build_GLSTM()
# with tf.variable_scope('model24'):
#     # 15 20 11
#     m24 = GLSTM(road=3)
#     m24.build_GLSTM()
# with tf.variable_scope('model25'):
#     # 18 20
#     m25 = GLSTM(road=2)
#     m25.build_GLSTM()

trainloss = []
trainerr = []
test_step = []
testloss = []
testerr = []
i=0
hm_epochs = 100
flag = 0
lastepoch_loss = 0
train_episode = 100
test_episode = 100
batch_size = 32
X_train, X_test, Y_train, Y_test = process(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_road20180122-0413_2.csv')
Xvalidate,Yvalidate = testprocess(datapath='/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_2.csv')
model = {}

# inedx of each road in the route accroding to the input record
# freq based
model_output = [[82, 76, 35, 7], [88, 92, 79, 82], [88, 92, 90, 105], [49, 28, 23, 21, 24], [56, 51, 45, 110], [74, 78, 80, 91, 87],  [88, 92, 79, 77, 73],  [88, 92, 90, 105, 107, 67, 65],
 [64, 66, 106, 104, 89, 91, 87], [34, 75, 81, 77],  [47, 25, 20, 22, 29, 49], [63, 96, 98, 100, 102],  [53, 51, 45, 55], [86, 89, 91, 87],  [74, 78, 80, 84],  [3, 1, 37, 107],
 [47, 109, 111, 44], [103, 101, 99, 97, 62], [47, 109, 111, 44, 50], [64, 66, 106, 104, 85],  [59, 103, 101, 99, 97, 62], [68, 25, 20, 22],  [35, 2, 9, 30],  [66, 106, 36, 33],
 [48, 28, 23, 21, 24], [101, 60, 50, 52],  [19, 23, 21, 24],  [7, 11, 14, 24],  [31, 8, 3, 34],  [88, 92, 90, 105, 107, 67, 70],  [103, 58, 42, 93],  [7, 17, 27, 55],
 [18, 8, 3, 34], [32, 4, 14, 24],  [55, 26, 16, 11],  [47, 109, 111, 44, 50, 57, 42],  [94, 43, 59, 103],  [35, 7, 17, 27, 44, 61],  [61, 45, 26, 16, 6, 34],
 [31, 13, 11, 14, 24],  [39, 40, 66, 106, 104, 85],  [66, 106, 36, 0, 2],  [7, 11, 5, 33], [68, 25, 15, 5, 33], [54, 26, 16, 6, 34],  [47, 25, 20, 16, 12],
 [71, 41, 38, 96], [56, 51, 45, 110, 108, 69], [65, 70, 68, 46],  [92, 90, 85, 32, 4, 10, 12, 19, 29, 49],  [83, 79, 77, 73],  [7, 12, 30, 72, 95, 94]]
model_output_weight = [14627,14007,13132,12866,12405,12375,11036,10053,10044,9889,9359,8885,8689,8127,7954,7359,7078,6930,6448,6405,5328,5307,5287,5149,5010,4934,4679,4310,
                       4649,4284,4257,4224,3984,3922,3748,3623,3497,3449,3445,3228,3183,2965,2959,2879,2518,2496,2314,2127,2076,1829,1779,1332]
# rp based
model_output = [
    [82, 76, 35, 7],
    [88, 92, 90, 105, 107, 67, 65],
    [49, 28, 23, 21, 24],
    [73, 86, 89, 84],
    [39, 40, 66, 106, 104, 85],
    [81, 80, 91, 87],
    [47, 109, 111, 44, 61],
    [78, 80, 91, 87],
    [47, 25, 20, 22, 29],
    [37, 107, 67, 65],
    [19, 23, 21, 24],
    [49, 53, 51, 45, 110],
    [88, 92, 90, 105, 107, 67, 65, 98, 100, 102],
    [3, 34, 75, 81, 80, 91, 87],
    [18, 8, 3, 7],
    [103, 101, 99, 97, 62],
    [56, 51, 45, 110],
    [27, 44, 61, 102],
    [35, 7, 17, 27],
    [92, 79, 77, 73],
    [39, 40, 70, 68, 25, 20, 22],
    [35, 7, 17, 27, 55],
    [60, 45, 26, 16, 6, 1],
    [101, 99, 64, 66, 106, 104, 89, 91, 87],
    [43, 93, 31, 13],
    [63, 96, 98, 100, 102],
    [54, 26, 16, 12],
    [32, 37, 107, 67, 70],
    [16, 6, 1, 33],
    [5, 33, 86, 89, 91, 87],
    [73, 32, 4, 14, 20],
    [32, 4, 10, 12, 30],
    [74, 78, 80, 84],
    [42, 48, 28, 23],
    [84, 90, 85, 32, 0, 2],
    [58, 56, 51, 45],
    [49, 28, 18, 13, 11],
    [80, 90, 105, 107, 67, 41],
    [47, 25, 15, 10],
    [58, 56, 52, 28],
    [55, 44, 50, 52],
    [21, 24, 69, 71, 65],
    [83, 90, 105, 107, 67, 65],
    [47, 109, 111, 44, 50, 57, 42],
    [53, 57, 59, 103],
    [49, 53, 51, 45, 110, 108],
    [88, 92, 90, 105, 36, 33],
    [51, 45, 110, 108, 46, 106, 104, 89, 84],
    [37, 107, 67, 41, 38, 62],
    [35, 2, 9, 30, 72],
    [94, 31, 13],
    [35, 2, 9, 30, 72, 95, 94, 43]]  # 1~3

model_output4 = [
    [88, 92, 90, 105, 107, 67, 65],
    [49, 28, 23, 21, 24],
    [39, 40, 66, 106, 104, 85],
    [65, 66, 106, 104, 89, 91, 87],
    [84, 90, 105, 107, 67, 65],
    [49, 53, 51, 45, 110],
    [103, 101, 99, 97, 62],
    [47, 25, 20, 22, 29],
    [88, 92, 90, 105, 107, 67, 65, 98, 100, 102],
    [47, 109, 111, 44, 61],
    [101, 99, 64, 66, 106, 104, 89, 91, 87],
    [60, 45, 26, 16, 6, 1],
    [35, 7, 17, 27, 55],
    [3, 34, 75, 81, 80, 91, 87],
    [63, 96, 98, 100, 102],
    [39, 40, 70, 68, 25, 20, 22],
    [5, 33, 86, 89, 91, 87],
    [55, 26, 16, 12, 8, 3, 34, 75],
    [21, 24, 69, 71, 65],
    [84, 90, 85, 32, 0, 2],
    [49, 28, 18, 13, 11],
    [4, 14, 20, 27, 44, 61],
    [80, 90, 105, 107, 67, 41],
    [73, 32, 4, 14, 20],
    [47, 25, 15, 5, 0, 2],
    [47, 25, 15, 10, 12, 30],
    [42, 48, 28, 23, 21, 24],
    [32, 37, 107, 67, 70],
    [88, 92, 90, 105, 36, 33],
    [84, 79, 82, 76, 35, 7, 17, 27, 55],
    [111, 44, 50, 52, 49],
    [83, 90, 105, 107, 67, 65],
    [15, 5, 33, 74, 78],
    [47, 109, 111, 44, 50, 57, 59],
    [35, 7, 12, 19, 29, 49],
    [53, 51, 45, 110, 108],
    [37, 107, 67, 41, 38, 62],
    [56, 51, 45, 110, 108, 46, 106, 104, 89],
    [35, 2, 9, 30, 72],
    [77, 73, 32, 4, 14, 20, 22, 29, 53, 57],
    [31, 8, 3, 1, 37, 104, 89],
    [43, 59, 103, 60, 45],
    [54, 26, 22, 18, 8],
    [94, 43, 59, 103, 101, 99, 97],
    [61, 50, 57, 42, 93],
    [58, 56, 51, 45, 55],
    [35, 2, 9, 30, 72, 95, 94, 43]]  # >4
model_output5 = [[88, 92, 90, 105, 107, 67, 65],
                 [39, 40, 66, 106, 104, 85],
                 [65, 66, 106, 104, 89, 91, 87],
                 [84, 90, 105, 107, 67, 65],
                 [101, 99, 64, 66, 106, 104, 89, 91, 87],
                 [88, 92, 90, 105, 107, 67, 65, 98, 100, 102],
                 [3, 34, 75, 81, 80, 91, 87],
                 [60, 45, 26, 16, 6, 1],
                 [39, 40, 70, 68, 25, 20, 22],
                 [80, 90, 105, 107, 67, 41],
                 [55, 26, 16, 12, 8, 3, 34, 75],
                 [84, 79, 82, 76, 35, 7, 17, 27, 55],
                 [5, 33, 86, 89, 91, 87],
                 [4, 14, 20, 27, 44, 61],
                 [47, 25, 20, 22, 18, 30],
                 [84, 90, 85, 32, 0, 2],
                 [39, 40, 70, 68, 25, 20, 22, 29],
                 [88, 92, 90, 105, 36, 33],
                 [83, 90, 105, 107, 67, 65],
                 [61, 45, 26, 21, 15, 5, 37],
                 [45, 110, 108, 69, 71, 41],
                 [47, 25, 15, 10, 12, 30],
                 [47, 109, 111, 44, 50, 57, 59],
                 [59, 103, 101, 99, 64, 66, 106, 104, 89, 91],
                 [49, 53, 51, 45, 110, 108],
                 [42, 48, 28, 23, 21, 24],
                 [94, 43, 59, 103, 101, 99, 97],
                 [37, 107, 67, 41, 38, 62],
                 [52, 28, 18, 8, 3, 1],
                 [107, 104, 85, 74, 78, 80],
                 [39, 40, 70, 68, 25, 20, 16, 11],
                 [51, 45, 110, 108, 46, 106, 104, 89, 84],
                 [77, 73, 32, 4, 10, 12],
                 [88, 92, 79, 77, 73, 32, 4, 10, 12, 19],
                 [56, 51, 45, 110, 108, 46, 106, 104, 89],
                 [31, 13, 17, 21, 24, 109],
                 [54, 110, 108, 69, 71, 65],
                 [63, 96, 64, 66, 106, 104, 85],
                 [77, 73, 32, 0, 2, 9, 19, 29],
                 [35, 2, 9, 30, 72, 95, 94, 43],
                 [103, 58, 42, 48, 28, 23, 21, 24],
                 [68, 109, 111, 44, 50, 52, 49, 93]]  # >5
model_output = [[88, 92, 90, 105, 107, 67, 65],
                 [65, 66, 106, 104, 89, 91, 87],
                 [88, 92, 90, 105, 107, 67, 65, 98, 100, 102],
                 [101, 99, 64, 66, 106, 104, 89, 91, 87],
                 [3, 34, 75, 81, 80, 91, 87],
                 [88, 92, 79, 82, 76, 35, 2],
                 [16, 6, 1, 33, 86, 89, 91],
                 [84, 79, 82, 76, 35, 7, 17, 27, 55],
                 [55, 26, 16, 12, 8, 3, 34, 75],
                 [59, 103, 101, 99, 64, 66, 106, 104, 89, 91],
                 [39, 40, 70, 68, 25, 20, 22, 29],
                 [51, 45, 110, 108, 46, 106, 104, 89, 84],
                 [47, 109, 111, 44, 50, 57, 59],
                 [39, 40, 70, 68, 25, 20, 22, 18, 30],
                 [83, 90, 105, 107, 47, 109, 111, 44, 50, 57],
                 [61, 45, 26, 21, 15, 5, 33],
                 [47, 109, 111, 44, 50, 57, 42],
                 [81, 80, 90, 105, 107, 67, 41],
                 [65, 70, 68, 109, 111, 44, 50, 52],
                 [56, 51, 45, 110, 108, 46, 106, 104, 89],
                 [39, 40, 70, 68, 25, 20, 16, 11],
                 [61, 45, 26, 21, 15, 5, 37],
                 [61, 45, 110, 14, 5, 33, 74, 78],
                 [23, 21, 24, 69, 71, 41, 38],
                 [28, 23, 21, 24, 69, 71, 41, 38],
                 [94, 43, 59, 103, 101, 99, 97],
                 [73, 32, 4, 14, 20, 27, 55],
                 [88, 92, 79, 77, 73, 32, 4, 10, 12, 19],
                 [77, 73, 32, 0, 2, 9, 19, 29],
                 [59, 103, 60, 45, 110, 108, 46, 106, 104, 89],
                 [104, 85, 32, 0, 2, 9, 30],
                 [53, 51, 45, 110, 108, 69, 71, 65],
                 [73, 32, 0, 7, 17, 22, 29, 49],
                 [35, 2, 9, 30, 72, 95, 94, 43],
                 [42, 48, 28, 23, 16, 6, 34],
                 [63, 96, 64, 66, 106, 104, 85],
                 [31, 13, 11, 14, 24, 69, 71, 65],
                 [64, 66, 106, 36, 0, 7, 11],
                 [97, 98, 54, 44, 50, 52, 49],
                 [84, 90, 105, 107, 67, 41, 38, 62],
                 [103, 58, 42, 48, 28, 23, 21, 24],
                 [68, 109, 111, 44, 50, 52, 49, 93]]
model_output=[[54, 110, 20, 27, 16, 11] ,
[78, 80, 90, 105, 107, 36, 4, 10, 17] ,
[48, 28, 23, 21, 24, 69, 71, 65, 97, 62] ,
[11, 5, 0, 2, 9, 19, 23, 27, 110] ,
[9, 30, 72, 95] ,
[84, 79, 77, 73, 32, 4, 14, 20, 16] ,
[13, 11, 5, 37, 107, 67, 65, 97, 62] ,
[8, 3, 1, 37, 107, 47, 69, 71, 41] ,
[5, 33, 74, 78, 82, 76, 35, 7, 11] ,
[42, 48, 28, 23, 21, 24, 109, 111, 44, 50] ,
[44, 50, 52, 28, 23, 21, 24, 109, 69] ,
[73, 32, 4, 10, 17, 27, 44, 50, 57, 59] ,
[44, 50, 57, 42, 48, 28, 23, 21, 15, 10] ,
[93, 2, 9, 19, 29] ,
[47, 109, 111, 44, 61] ,
[55, 44, 50, 57, 59, 103, 101, 99] ,
[57, 51, 45, 110, 108, 69] ,
[97, 62, 39, 46, 25, 20] ,
[78, 80, 91, 84, 90, 105, 36, 0] ,
[11, 14, 110, 44, 61, 102, 58] ,
[40, 70, 68, 109, 111, 44, 50, 52, 49] ,
[107, 104, 85, 32, 0, 2, 9] ,
[68, 25, 15, 10, 6, 1, 4, 16, 21] ,
[92, 90, 105, 85, 32, 0, 2] ,
[58, 56, 52, 28, 18, 8] ,
[43, 48, 28, 23] ,
[94, 102, 101, 54, 26] ,
[31, 8, 3, 1, 4, 10, 17, 21, 24] ,
[12, 30, 72, 95, 94, 48, 28, 18, 8, 3] ,
[74, 78, 32, 0, 34, 75, 81] ,
[66, 106, 36, 0, 2] ,
[60, 50, 52, 28, 18, 8, 3, 1, 4, 14] ,
[41, 70, 68, 109, 111, 26, 22, 18, 8] ,
[66, 106, 104, 89, 91, 79, 77, 73, 32] ,
[47, 69, 71, 65, 98, 100, 60] ,
[73, 86, 89, 79, 82] ,
[19, 29, 53, 51, 45, 26, 22, 8] ,
[83, 79, 77, 73, 86, 105, 107, 47, 109, 111] ,
[88, 92, 79, 82] ,
[80, 90, 105, 107, 67, 41, 38, 96, 98, 100] ,
[63, 96, 64, 66, 47, 25, 15, 5, 33, 86] ,
[84, 90, 85, 74, 91, 87]] #new
paths = ['/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs1/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs2/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs3/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs4/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs5/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs6/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs7/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs8/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs9/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs10/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs11/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs12/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs13/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs14/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs15/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs16/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs17/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs18/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs19/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs20/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs21/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs22/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs23/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs24/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs25/', '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs26/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs27/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs28/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs29/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs30/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs31/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs32/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs33/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs34/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs35/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs36/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs37/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs38/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs39/',
        '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs40/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs41/', '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs42/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs43/', '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs44/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs45/', '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs46/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs47/','/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs48/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs49/',  '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs50/',
         '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs51/',  '/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs52/'
         ]

checkpoints = []
for i in range(len(model_output)):
    checkpoints.append(tf.train.latest_checkpoint(paths[i]))

if False:
    for i in range(len(savers)):
        savers[i].restore(sess, checkpoints[i])
        print 'model%d is ready:'%(i+1)
        print("  restore from the checkpoint {0}".format(checkpoints[i]))


def train_Graph(r,k, epoch, train_loss_dict, test_loss_dict, train_loss, test_loss):
    i = 0
    trainloss = train_loss
    rw_ = 0
    lr = 0.001
    s = model_output[k]
    with tf.variable_scope('model'+str(k)):
        m_ = GLSTM(road=len(model_output[k]), learning_rate=lr, rw=rw_)
        m_.build_GLSTM()
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver_ = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'+str(k)))
    checkpoints[k] = tf.train.latest_checkpoint('/home/administrator/pywork/DeepLearningXC/trainingDir/45min_5min/pool/logs'+str(k+1)+'/')
    if epoch>-1:
        saver_.restore(sess, checkpoints[k])
    x_batch_test, y_batch_test = generateBatch(batchsize=len(X_test), X=X_test, Y=Y_test, act_roads=model_output[k],
                                              train=False)
    x_batch_v, y_batch_v = generateBatch(batchsize=len(Xvalidate), X=Xvalidate, Y=Yvalidate,
                                         act_roads=model_output[k], train=False)
    train_loss_dict[k] = train_loss
    test_loss_dict[k] = test_loss
    train_=[]
    test_=[]
    epoch=1
    if True:
        for e in range(epoch):
            trainloss = 0
            for ep in range(int(len(X_train) / batch_size)):
                i+=1
                x_batch, y_batch = generateBatch(batchsize=batch_size, X=X_train, Y=Y_train, act_roads=model_output[k], epoch=ep)
                # loss_, err_train,laststate,inppp = sess.run([m_.loss, m_.error,m_.last_state,m_.input_],
                #                             feed_dict={m_.bottom: x_batch, m_.target: y_batch, m_.dropout:1.})
                _,  = sess.run([m_.trainop],feed_dict={m_.bottom: x_batch, m_.target: y_batch, m_.dropout:0.6})

                loss_, err_train,pp = sess.run([m_.loss, m_.error,m_.predict],
                                            feed_dict={m_.bottom: x_batch, m_.target: y_batch, m_.dropout:0.6})
                trainloss+=loss_
            saver_.save(sess, os.path.join(paths[k]),global_step=epoch)

            testloss, errtest = sess.run([m_.loss,m_.error],feed_dict={m_.bottom: x_batch_test, m_.target: y_batch_test,m_.dropout:1.})
            train_loss_dict[k] = trainloss/ int(len(X_train) / batch_size)
            test_loss_dict[k] = testloss
            print '===>epoch %d model %d train loss: %g | test loss: %g | test accuracy:%g | gap: %g' % (e, k, trainloss / int(len(X_train) / batch_size), testloss,1-errtest,
                                                                             trainloss / int(len(X_train) / batch_size)- testloss)
            train_.append(trainloss / int(len(X_train) / batch_size))
            test_.append(testloss)
    pre, loss = sess.run([m_.predict, m_.loss], feed_dict={m_.bottom: x_batch_v, m_.target: y_batch_v, m_.dropout: 1.})
    # x_batch, y_batch = generateBatch(batchsize=1, X=X_train, Y=Y_train, act_roads=model_output[k], epoch=1)
    # spatial =  sess.run([m_.spatial,m_.loss ], feed_dict={m_.bottom: x_batch, m_.target: y_batch, m_.dropout:1.})
    # s = np.array(spatial).reshape(-1, len(model_output[k]))
    # plt.imshow(s,cmap='hot', interpolation='nearest', aspect='auto')
    # plt.show()
    pre = np.array(pre)
    date_state=[]
    real = []
    #
    # plt.figure(1)
    # plt.title('MR: 88')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=0
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr88.jpg')
    #
    # plt.figure(2)
    # plt.title('MR: 92')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=1
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr92.jpg')
    #
    # plt.figure(3)
    # plt.title('MR: 90')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=2
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr90.jpg')
    #
    # plt.figure(4)
    # plt.title('MR: 105')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=3
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr105.jpg')
    #
    # plt.figure(5)
    # plt.title('MR: 107')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=4
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr107.jpg')
    #
    # plt.figure(6)
    # plt.title('MR: 67')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=5
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr67.jpg')
    #
    # plt.figure(7)
    # plt.title('MR: 65')
    # plt.plot([20, 70], [20, 70], 'r')
    # roadindex=6
    # a = pre[:,roadindex]
    # plt.grid(True, linewidth="0.3")
    # plt.scatter(a, np.array(y_batch_v)[:,roadindex])
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # plt.savefig('mr65.jpg')
    # plt.show()
    # for x in range(7):
    #     date_state = []
    #     for p in range(12*24):
    #         x_batch, y_batch = generateBatch(batchsize=1, X=X_train, Y=Y_train, act_roads=model_output[k], epoch=p)
    #         real.append(y_batch)
    #         state, loss = sess.run([m_.stateset, m_.loss], feed_dict={m_.bottom: x_batch, m_.target: y_batch, m_.dropout: 1.})
    #         print loss
    #         state_road_at_onetime = state[x]
    #         state = np.array(state).reshape(timestep-1,len(model_output[k]))
    #         state_new=[]
    #         # for i in range(state.shape[0]):
    #         #     s = state[i]
    #         #     state_new.append(s[::-1])
    #         # state_new=np.array(state_new)
    #         date_state.append(state_road_at_onetime)
    #         # print state_new[2]
    #         # for i in range(state.shape[0]):
    #         #     s=[]
    #         #     for j in range(state.shape[1]):
    #         #         s.append(float(state[i][j])/sum(state[i]))
    #         #     ss.append(s)
    #         # plt.subplot(4,5,p+1)
    #     plt.figure(x)
    #     plt.title('layer '+str(x+1))
    #     date_state = np.array(date_state).reshape(-1, len(model_output[k]))
    #     # plt.figure(figsize=(12*24,len(model_output[k]))) np.array(real).reshape(-1,len(model_output[k]))
    #     plt.xlabel('location')
    #     plt.ylabel('time')
    #     plt.yticks([2.5, 14.5, 26.5, 38.5, 50.5, 62.5, 74.5, 86.5, 98.5, 110.5, 122.5, 134.5, 146.5, 158.5, 170.5, 182.5,
    #                 194.5, 206.5, 218.5, 230.5, 242.5, 254.5, 266.5, 278.5],
    #                ['0:00', '1:00','2:00','3:00','4:00','5:00','6:00', '7:00','8:00','9:00','10:00','11:00','12:00','13:00',
    #                                                   '14:00','15:00','16:00','17:00','18:00','19:00',
    #                                                   '20:00','21:00','22:00','23:00'])
    #     plt.imshow(date_state, cmap='hot', interpolation='nearest', aspect='auto')
    #     plt.savefig('beforetrain'+str(x)+'.jpg')
    #     # plt.savefig('aftertrain' + str(x) + '.jpg')
    #     # plt.axis('off')
    #
    #
    #     plt.colorbar()
    r[k] = pre

    # plt.show()
    # print 'OK'
#####################################################################################################################################################################

epoch_loss = 0.
epoch_err = 0.

begin_t = datetime.datetime.now()
ks = [i for i in range(len(model_output))]
Valve = [False for i in range(len(model_output))]

###########################################
train_loss_set = [100 for i in range(len(model_output))]
test_loss_set = [100 for i in range(len(model_output))]


bomb=0
for epoch in range(hm_epochs):
    print 'Round %d start trainning...'%(epoch+1)
    pres = []
    manager = mp.Manager()
    r = manager.dict()
    train_loss_dict = manager.dict()
    test_loss_dict = manager.dict()
    jobs = []
    for i in ks:
        p = mp.Process(target=train_Graph,args=(r,i,epoch, train_loss_dict, test_loss_dict, train_loss_set[i], test_loss_set[i]))
        jobs.append(p)
        p.start()
        # break
    for job in jobs:
        job.join()
        # break
    train_loss_set = [ ]
    test_loss_set = [ ]

    for k in range(len(model_output)):
        pre = np.array(r.values()[k])
        train_loss_set.append(train_loss_dict.values()[k])
        test_loss_set.append(test_loss_dict.values()[k])
        pre = np.reshape(pre, (len(Xvalidate), len(model_output[k])*predstep))
        pres.append(pre)

    end_t = datetime.datetime.now()
    print 'cost: %s'%(end_t-begin_t)
    x_batch_v, y_batch_v = generateBatch(batchsize=len(Xvalidate), X=Xvalidate, Y=Yvalidate,
                                               act_roads=[ar for ar in range(len(range(network)))], train=False)
    final_output = []
    for t in range(predstep):
        for i in range(network):
            avg = np.reshape(np.zeros(len(Xvalidate)),[len(Xvalidate),1])
            num = 0.
            for k in range(len(model_output)):
                if i in model_output[k]:
                    tag = model_output[k].index(i)
                    b=np.array(pres[k])
                    a= np.reshape(b[:,tag+t*len(model_output[k])],(len(Xvalidate),1))
                    avg+=a*model_output_weight[k]
                    num+=model_output_weight[k]
            if num==0:
                print i
            avg = (avg)/num
            final_output.append(avg)
    final_output = np.reshape(np.transpose(np.array(final_output)),(len(Xvalidate),network*predstep))

    x_batch_test, y_batch_test = generateBatch(batchsize=len(X_test), X=X_test, Y=Y_test,
                                               act_roads=[ar for ar in range(len(range(network)))], train=False)
    y_batch_v = np.array(y_batch_v)
    pre = np.array(final_output)
    ydf = pd.DataFrame(y_batch_v[ : , :])
    predf = pd.DataFrame(pre[ : , :])
    #
    # plt.plot(pre[0:12 * 24, 2], color='red')
    # plt.plot(y_batch_v[0:12 * 24, 2])
    # plt.show()
    loss_test = np.mean(np.square(final_output-y_batch_v))
    err_test = np.mean(abs(final_output -y_batch_v)/y_batch_v)
    test_step.append(i)
    print '>>>>>TEST epoch:%d, loss:%g, accuracy:%g' % (epoch+1, loss_test, 1-err_test)
    predf.to_csv(
        '/home/administrator/pywork/DeepLearningXC/trainingDir/predict_m455.csv')
    # plt.plot([0,80],[0,80],'r')
    # a = np.array(pre).reshape(-1,)
    # b = np.array(y_batch).reshape(-1,)
    # plt.xlabel('speed predicted (km/h)')
    # plt.ylabel('speed detected (km/h)')
    # xy=np.vstack([a, b])
    # z=gaussian_kde(xy)(xy)
    # plt.scatter(a, b, c=z)
    # plt.savefig('MR.jpg')
    # plt.show()


    testloss.append(loss_test)
    testerr.append(err_test)
    lastepoch_loss = loss_test
    if len(testloss)>20:
        if testloss[-1]-testloss[-2]>0:
            bomb+=1
    if bomb>5:
        break
end = datetime.datetime.now()
print 'time cost: %s' % (end - begin)
plt.plot(trainloss,'r', label='Train Loss')
plt.plot(range(len(testloss)), testloss, 'b', label='Test Loss')

plt.show()





