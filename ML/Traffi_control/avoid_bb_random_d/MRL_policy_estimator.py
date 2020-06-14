import tensorflow as tf
import numpy as np


def weight_variable(shape, train=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=train)


def bias_variable(shape, train=True):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial, trainable=train)


class policy_estimator():
    def __init__(self, sess, state_dim, policy_dim=1, name='estimator'):
        self.state_dim = state_dim
        self.policy_dim = policy_dim
        self.policy_holder = tf.placeholder(tf.float32, [None, self.policy_dim], name='policy')
        self.state_holder = tf.placeholder(tf.float32, [None, state_dim], name='state')
        self.sess = sess
        self.name = name

    def buildPENetwork(self, ):
        with tf.variable_scope('update-Critic_network' + self.name):
            w1 = weight_variable([self.state_dim, 512])
            b1 = bias_variable([512])
            w2 = weight_variable([512, self.policy_dim])
            b2 = bias_variable([self.policy_dim])

        self.c_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='update-Critic_network' + self.name)

        h1 = tf.nn.relu(tf.matmul(self.state_holder, w1) + b1)
        self.estimate = tf.nn.elu(tf.matmul(h1, w2) + b2)

        self.estimate_loss = tf.reduce_mean(tf.squared_difference(self.policy_holder, self.estimate))
        self.estimate_trainOp = tf.train.AdamOptimizer(learning_rate=0.0006).minimize(self.estimate_loss)
