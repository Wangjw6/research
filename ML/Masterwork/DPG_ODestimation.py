#encoding:utf-8
import Sim_Env
import tensorflow as tf
import os
import itertools
import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

FLAGS = tf.app.flags.FLAGS
env = Sim_Env.Env()
tf.app.flags.DEFINE_string('summary_dir', '/home/administrator/pywork/od/logs1_dpg/',
                           """Path of where to store the summary files.""")
tf.app.flags.DEFINE_string('state_dim', 28,
                           """state dimension.""")
tf.app.flags.DEFINE_string('action_dim', 1,
                           """action dimension.""")
tf.app.flags.DEFINE_string('reward_dim', 1,
                           """reward dimension.""")
tf.app.flags.DEFINE_string('R_average', 0,
                           """Average reward.""")
# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, FLAGS.state_dim], name='state')
with tf.name_scope('A'):
    A = tf.placeholder(tf.float32, shape=[None, FLAGS.action_dim], name='action')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, FLAGS.reward_dim], name='reward')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, FLAGS.state_dim], name='next_state')
	
###############################  Actor  ####################################

class Actor(object):
    def __init__(self, sess, action_dim, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()#tf.random_normal_initializer(0., 0.3)
            #init_b = tf.constant_initializer(0.1)
            net1 = tf.contrib.layers.fully_connected(
                s,
                6,    # number of hidden units
                activation_fn=None,
                weights_initializer=init_w,    # weights
                trainable=trainable
            )
            net2 = tf.contrib.layers.fully_connected(
                net1,
                6,    # number of hidden units
                activation_fn=tf.nn.tanh,
                weights_initializer=init_w,    # weights
                trainable=trainable
            )
            net3 = tf.contrib.layers.fully_connected(
                net2,
                12,    # number of hidden units
                activation_fn=tf.nn.relu,
                weights_initializer=init_w,    # weights
                trainable=trainable
            )
            with tf.variable_scope('a'):
                action = tf.contrib.layers.fully_connected(
                net3,
                self.a_dim,    # number of hidden units
                activation_fn=tf.nn.tanh,
                weights_initializer=init_w,    # weights
                trainable=trainable
            )

        #scaled_a = tf.clip_by_value(action, -1, 1)  # Scale output to -action_bound to action_bound
        return action

    def learn(self, s, a):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s, A: a})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(tar, eva) for tar, eva in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads_and_vars = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads_and_vars, self.e_params))#each gradient correspond to a parameter


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, t_replace_iter, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.q = self._build_net(S, A, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + 0.9 * self.q_
            #self.target_q = R + self.q_ - FLAGS.R_average

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

            tf.scalar_summary('value_loss', self.loss)

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, A)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()#tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 10
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                net2 = tf.contrib.layers.fully_connected(
                    net,
                    5,  # number of hidden units
                    activation_fn=None,
                    weights_initializer=init_w,  # weights
                    trainable=trainable
                )
            with tf.variable_scope('q'):
                q = tf.contrib.layers.fully_connected(
                net2,
                1,    # number of hidden units
                activation_fn=None,
                weights_initializer=init_w,    # weights
                trainable=trainable
            )# Q(s,a)
        return q

    def learn(self, s, a, r, s_, num):

        _, summary = self.sess.run([self.train_op, merged], feed_dict={S: s, A: a, R: r, S_: s_})

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(tar, eva) for tar, eva in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

        writer.add_summary(summary, num)

############################################  Memory  #########################################################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]



#####################  hyper parameters  ####################

LR_A = 0.0001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

# so we use replace_iter instead
REPLACE_ITER_A = 10
REPLACE_ITER_C = 5
BATCH_SIZE = 24
MEMORY_CAPACITY = 200
MAX_EPISODE_NUM = 500
EPISODE_FOR_TEST = 5
episode_length = 6

sess = tf.Session()
actor = Actor(sess, FLAGS.action_dim, LR_A, REPLACE_ITER_A)
critic = Critic(sess, FLAGS.state_dim, FLAGS.action_dim, LR_C, REPLACE_ITER_C, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
# initialize before summary
model_file = os.path.join(FLAGS.summary_dir, 'model.ckpt')
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

var = 1  # control exploration
num = 0
r_sum_max = -1000
flag = -1000
cumreward = []
reward_sum = []
M = Memory(MEMORY_CAPACITY, dims=2 * FLAGS.state_dim + FLAGS.action_dim + 1)
gradient=[]



for episode in range(1,MAX_EPISODE_NUM):

    start=2
    print 'new:%d ' % start
    env.start(start)
    state = env.pre_run(start)
    for t in itertools.count():
        print state
        action = actor.choose_action([state])
        action = np.array(action)
        print action
        action = np.clip(action + np.random.normal(0, var) * var, -1, 1)
        
        if episode > MAX_EPISODE_NUM - EPISODE_FOR_TEST:
             load_path = saver.restore(sess, model_file)
             action = actor.choose_action([state])
             action = np.array(action)
            # action = np.clip(action + np.random.normal(0, var) * var, -1, 1)

        print action
        next_state, reward= env.step(action*800, start+t, t, num) # start+t=timetag

        num+=1
        M.store_transition(state, action, reward, next_state)

        if M.pointer>MEMORY_CAPACITY :#and episode<MAX_EPISODE_NUM-EPISODE_FOR_TEST:
            var *= .998    # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :FLAGS.state_dim]
            b_a = b_M[:, FLAGS.state_dim: FLAGS.state_dim + FLAGS.action_dim]
            b_r = b_M[:, -FLAGS.state_dim - 1: -FLAGS.state_dim]
            b_s_ = b_M[:, -FLAGS.state_dim:]
            print 'update %d'%(num-100)

            actor.learn(b_s, b_a)
            critic.learn(b_s, b_a, b_r, b_s_, num)
            print sess.run(actor.policy_grads_and_vars[0][0][0],feed_dict={S:b_s,A:b_a,R:b_r})
            gradient.append(sess.run(actor.policy_grads_and_vars[0][0][0],feed_dict={S:b_s,A:b_a,R:b_r}))
            #FLAGS.R_average += 0.02*(critic.target_q - critic.q)
        t += 1

        print 'step %d in %d episode'%(t,episode)

        cumreward.append(reward)
        state = next_state
        if t>=episode_length:
        #cumreward.append(value_now)
            if r_sum_max < sum(cumreward):
                r_sum_max = sum(cumreward)
                save_path = saver.save(sess, model_file)
            reward_sum.append(round(sum(cumreward)))
            flag = sum(cumreward)
            cumreward=[]
            print 'next epoch!'
            break
    if abs(flag)/episode_length < 1.5:
         print 'episode %d good enough'%episode
         break


xmajorLocator = MultipleLocator(1)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
plt.plot(reward_sum)
##plt.plot(gradient)
plt.xlabel("episode")
plt.ylabel("cumulative reward")
plt.show()
ax.xaxis.set_major_locator(xmajorLocator)
env.end()

writer.close()