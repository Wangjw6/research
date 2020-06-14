import tensorflow as tf
import numpy as np
import Env
from Env import Env
from BUS import bus
from BUS_STOP import bus_stop


def weight_variable(shape, train=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=train)


def bias_variable(shape, train=True):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial, trainable=train)


class agent_PPO():
    def __init__(self, sess, state_dim,state_dim_local,action_dim=1, action_bound=[], name='agent'):
        self.state_dim = state_dim
        self.state_dim_local = state_dim_local
        self.action_dim = action_dim
        self.action_holder = tf.placeholder(tf.float32, [None, action_dim], name='action')
        self.state_holder_local = tf.placeholder(tf.float32, [None, state_dim_local], name='local_state')
        self.state_holder = tf.placeholder(tf.float32, [None, state_dim-1], name='state')
        self.accumulated_reward_holder = tf.placeholder(tf.float32, [None, 1], name='reward_discounted_sum')
        self.epsilon_holder = tf.placeholder(tf.float32, None, name='epsilon')
        self.action_bound = action_bound
        self.sess = sess
        self.name = name


    def buildCriticNetwork(self, ):
        with tf.variable_scope('update-Critic_network'+self.name):
            w1 = weight_variable([self.state_dim-1, 256])
            b1 = bias_variable([256])
            w2 = weight_variable([256, 1])
            b2 = bias_variable([1])

        self.v_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='update-Critic_network'+self.name)

        h1 = tf.nn.elu(tf.matmul(self.state_holder, w1) + b1)
        self.v = tf.identity(tf.matmul(h1, w2) + b2)

        self.advantage = self.accumulated_reward_holder - self.v
        self.v_loss = tf.reduce_mean(tf.squared_difference(self.accumulated_reward_holder, self.v))
        self.v_trainOp = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.v_loss)

    def buildActorNetwork(self, ):
        with tf.variable_scope('update-Actor_network'+self.name):
            w1 = weight_variable([self.state_dim_local, 256])
            b1 = bias_variable([256])

            w2 = weight_variable([256, self.action_dim])
            b2 = bias_variable([self.action_dim])
            w3 = weight_variable([256, self.action_dim])
            b3 = bias_variable([self.action_dim])

        with tf.variable_scope('target-Actor_network'+self.name):
            w1_ = weight_variable([self.state_dim_local, 256], train=False)
            b1_ = bias_variable([256], train=False)

            w2_ = weight_variable([256, self.action_dim], train=False)
            b2_ = bias_variable([self.action_dim], train=False)
            w3_ = weight_variable([256, self.action_dim], train=False)
            b3_ = bias_variable([self.action_dim], train=False)

        self.p_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='update-Actor_network'+self.name)
        self.p_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-Actor_network'+self.name)

        self.h = tf.nn.elu(tf.matmul(self.state_holder_local, w1) + b1)
        self.action_mean = tf.nn.elu(tf.matmul(self.h, w2) + b2)
        self.action_sigma = tf.nn.softplus(tf.matmul(self.h, w3) + b3)

        self.h_old = tf.nn.elu(tf.matmul(self.state_holder_local, w1_) + b1_)
        self.action_mean_old = tf.nn.elu(tf.matmul(self.h_old, w2_) + b2_)
        self.action_sigma_old = tf.nn.softplus(tf.matmul(self.h_old, w3_) + b3_)

        self.pi = tf.distributions.Normal(loc=self.action_mean, scale=self.action_sigma)
        self.pi_old = tf.distributions.Normal(loc=self.action_mean_old, scale=self.action_sigma_old)

        self.action = tf.squeeze(tf.clip_by_value(self.pi.sample([1]),0., 1.), axis=0)
        # self.action = tf.squeeze(self.pi.sample([1]))
        self.p1 = self.pi.prob(self.action_holder)
        self.p2 = self.pi_old.prob(self.action_holder)
        # self.ratio = self.pi.prob(self.action_holder) / self.pi_old.prob(self.action_holder)
        self.ratio = tf.exp(self.pi.log_prob(self.action_holder) - tf.clip_by_value(self.pi_old.log_prob(self.action_holder), -20, 20))
        self.surrogate = self.ratio * self.advantage
        self.clip_surrogate = tf.clip_by_value(self.ratio, 1. - self.epsilon_holder, 1 + self.epsilon_holder) * self.advantage

        self.mim = tf.minimum(self.surrogate,self.clip_surrogate )

        self.p_loss = -tf.reduce_mean(tf.minimum(self.surrogate,self.clip_surrogate ))
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.p_loss, self.p_e_params), 5.)
        # grads = tf.gradients(self.p_loss, self.p_e_params)
        grads_and_vars = list(zip(grads, self.p_e_params))
        self.p_trainOp = tf.train.AdamOptimizer(learning_rate=0.0001).apply_gradients(grads_and_vars, name="apply_gradients")

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # gvs = optimizer.compute_gradients(self.p_loss, self.p_e_params)
        # capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        # self.p_trainOp = optimizer.apply_gradients(capped_gvs)

        self.Actor_network_update = [tf.assign(tar, eva) for tar, eva in zip(self.p_t_params, self.p_e_params)]

