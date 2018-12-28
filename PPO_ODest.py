# encoding:utf-8
import tensorflow as tf
import random
import itertools
from tensorflow.contrib.distributions import Normal, kl_divergence
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import SimEnvPPO
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

EP_MAX = 500
EP_LEN = 4
GAMMA = 0.9
A_LR = 0.000002
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 8
C_UPDATE_STEPS = 6


begin=datetime.datetime.now()
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '/home/administrator/pywork/od/logs1_dpg/',
                           """Path of where to store the summary files.""")
# number of node in roadnetwork
tf.app.flags.DEFINE_string('node_dim', 10,
                           """state dimension.""")
tf.app.flags.DEFINE_string('action_dim', 6,
                           """action dimension.""")
tf.app.flags.DEFINE_string('reward_dim', 1,
                           """reward dimension.""")
tf.app.flags.DEFINE_string('keep_prob', 0.9,
                           """reward dimension.""")
tf.app.flags.DEFINE_string('SL_dir', '/home/administrator/pywork/od/trainSL.csv',
                           """od file for supervised learning.""")

def max_poo_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_var_maybe_avg(var_name, ema,  trainable, shape):
    if var_name=='V':
        # initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal(shape=shape,stddev=0.1)
        v = tf.get_variable(name=var_name, initializer=initializer, trainable=trainable )
    if var_name=='g':
        initializer = tf.constant_initializer(1.0)
        v = tf.get_variable(name=var_name, initializer=initializer, trainable=trainable, shape=[shape[-1]])
    if var_name=='b':
        initializer = tf.constant_initializer(0.1)
        v = tf.get_variable(name=var_name, initializer=initializer, trainable=trainable, shape=[shape[-1]])
    if ema is not None:
        v = ema.average(v)
    return v
def get_vars_maybe_avg(var_names, ema, trainable, shape):
    vars=[]
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, trainable=trainable, shape=shape))
    return vars

def conv2dWN(x,name, num_filters, shape, trainable='trainable',filter_size=[3,3], stride=[1,1], pad='SAME',  nonlinearity=None,  ema=None):
    with tf.variable_scope(name):
        V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema, trainable=trainable, shape=shape)
        # tf.assert_variables_initialized([V, g, b])
        W = tf.reshape(g, [1,1,1,num_filters])*tf.nn.l2_normalize(V,[0,1,2])
        # x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1]+stride+[1], pad), b)
        x = tf.nn.conv2d(x, W, [1] + stride + [1], pad)
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def denseWN(x, name, num_units, trainable, shape, nonlinearity=None, ema=None, **kwargs):
    with tf.variable_scope(name):
        V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema, trainable=trainable, shape=shape)
        # tf.assert_variables_initialized([V, g, b])
        x = tf.matmul(x, V)
        scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
        x = tf.reshape(scaler,[1,num_units])*x   #+ tf.reshape(b,[1,num_units])
        if nonlinearity is not None:
            x = nonlinearity(x)
        return x

def leaky_relu(x,alpha):
    return tf.minimum(x,2)

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, shape=[None, 3, FLAGS.node_dim*FLAGS.node_dim], name='state')
        self.batch_size = BATCH
        self.odnum = 6
        self.a_dim = FLAGS.action_dim
        self.q_cell_size = 120
        self.a_cell_size = 120
        self.input_size = 150
        # critic no target
        with tf.variable_scope('critic'):

            s = tf.reshape(self.tfs, [-1, 3, FLAGS.node_dim, FLAGS.node_dim])
            s = tf.transpose(s,[0,2,3,1])
            h_conv1 = conv2dWN(x=s, name='L1', num_filters=6,   nonlinearity=tf.nn.tanh, ema=None,
                               shape=[3, 3, 3, 6])
            h_conv1 = max_poo_2x2(h_conv1)
            h_flat = tf.reshape(h_conv1, [-1, 5 * 5 * 6])

            y = tf.stack([h_flat, h_flat, h_flat, h_flat, h_flat, h_flat], axis=0)

            y = tf.reshape(y, [-1, self.input_size])
            y = tf.split(axis=0, num_or_size_splits=self.odnum, value=y)
            # lstm cell
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.q_cell_size, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.tanh)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.q_cell_size, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.tanh)

            self.outputs_c, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                    inputs=y, dtype=tf.float32 )

            # final outout
            # out_x = tf.reshape(outputs, [-1])  # [batch*OD, cell_size]
            initializer = tf.truncated_normal(shape=[self.q_cell_size*2,1], stddev=0.01)
            weight_out = tf.get_variable(name='critic',initializer=initializer,trainable=True)
            # bias_out = tf.constant(0.1)
            transformed_outputs = [tf.matmul(output, weight_out) for output in self.outputs_c]
            q= tf.concat(transformed_outputs,0)
            Vset = tf.reshape(q, [-1, self.odnum])
            self.v = tf.reduce_sum(input_tensor=Vset, axis=1, keep_dims=True)

            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            tf.summary.scalar('value_loss', self.closs)
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        self.oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        with tf.variable_scope('sample_action'):
            self.action = tf.squeeze( self.pi.sample(1), axis=0)
            self.sample_op = tf.reshape(self.action, shape=(-1,self.odnum))

        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [ oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.odnum], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                tfa = tf.reshape(self.tfa, shape=(-1,1))
                tfadv = tf.reshape(self.tfadv, shape=(-1,))
                tfadv_ = tf.stack([tfadv,tfadv,tfadv,tfadv,tfadv,tfadv])
                tfadv_ = tf.transpose(tfadv_)
                self.tfadv_ = tf.reshape(tfadv_, shape=(-1,1))
                self.ssssss=self.pi.prob(tfa)

                self.ratio=((self.pi.prob(tfa))/(self.oldpi.prob(tfa)))

                self.surr_ = tf.minimum(self.ratio*self.tfadv_, tf.clip_by_value(self.ratio, 0.8,1.2)*self.tfadv_)

             # clipping method, find this is better
            self.aloss = -tf.reduce_mean(self.surr_)
            tf.summary.scalar('policy_value', self.aloss)

        with tf.variable_scope('atrain'):
             self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r, step):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        a_step=step
        c_step=step

        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        #update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # prepare input
            s = tf.reshape(self.tfs, [-1, 3, FLAGS.node_dim, FLAGS.node_dim])
            s = tf.transpose(s,[0,2,3,1])
            h_conv1=conv2dWN(x=s, name='L1', num_filters=6,  trainable=trainable, nonlinearity=tf.nn.tanh, ema=None, shape=[3,3,3,6])
            h_conv1 = max_poo_2x2(h_conv1)
            h_flat = tf.reshape(h_conv1, [-1, 5*5*6]) # [batch, input]
            # lstm input should be [batch*n_step, input_size], h_flat need n_step copy
            y = tf.stack([h_flat,h_flat,h_flat,h_flat,h_flat,h_flat], axis=0)

            y = tf.reshape(y, [-1, self.input_size])
            self.y = tf.split(axis=0, num_or_size_splits=self.odnum, value=y)
            # lstm cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.a_cell_size, forget_bias=1.0, activation=tf.nn.softplus)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.a_cell_size, forget_bias=1.0, activation=tf.nn.softplus)

            self.outputs_a, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=self.y, dtype=tf.float32)

            # final outout
            initializer = tf.truncated_normal(shape=[2*self.a_cell_size,1],stddev=0.01)

            self.weight_outmu = tf.get_variable(name='mu',initializer=initializer,trainable=trainable)
            bias_outmu = tf.constant(0.01)

            self.weight_outsigma = tf.get_variable(name='sigma',initializer= initializer ,trainable=trainable)
            bias_outsigma = tf.constant(0.01)

            transformed_outputsmu = [tf.nn.tanh(tf.matmul(output, self.weight_outmu)+bias_outmu)  for output in self.outputs_a]
            mu = tf.concat(transformed_outputsmu,0)
            mu = tf.reshape(mu,[-1, self.odnum])

            transformed_outputsigma = [tf.nn.softplus(tf.matmul(output, self.weight_outsigma)+bias_outsigma)  for output in self.outputs_a]
            sigma = tf.concat(transformed_outputsigma,0)
            sigma = tf.reshape(sigma,[-1, self.odnum])

            self.mu_reshape = tf.reshape(mu, shape=(-1,1))
            self.sigma_reshape  = tf.reshape(sigma, shape=(-1,1))

            norm_dist = Normal(loc=self.mu_reshape, scale=self.sigma_reshape) # for two dimmension mu[?,] each element is a mu for a distribution

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = [s]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return a

    def get_v(self, s):
        s = [s]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
r_sum_max = -1000

num = 0
all_ep_r = []
cumreward = []
reward_sum = []
episode_length = 4
ppo = PPO()
saver = tf.train.Saver()
model_file = os.path.join(FLAGS.summary_dir, 'model.ckpt')
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(FLAGS.summary_dir, ppo.sess.graph)

env = SimEnvPPO.Env()
step=0
pre_er = 0
datetag =23
timetag=25
for ep in range(EP_MAX):
    # s = env.reset()

    # timetag = random.randint(25,76)
    # datetag = random.randint(23,29)
    if timetag>76:
        datetag+=1
        timetag = 25
    if datetag>29:
        datetag = 23
        timetag = 25
    print 'new:%d ' % timetag
    print 'datetag:%d '% datetag
    env.start(timetag,datetag)
    state = env.pre_run(timetag, datetag)

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in itertools.count():    # in one episode
        # env.render()
        a = ppo.choose_action(state)
        print a
        # print ppo.sess.run(ppo.y, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print'******************************************************************'
        # print ppo.sess.run(ppo.outputs_a[0], {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print ppo.sess.run(ppo.ratio, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print ppo.sess.run(ppo.tfadv_, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print ppo.sess.run(ppo.surr_[0], {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print 'sur'
        # print ppo.sess.run(ppo.surr_, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print 'param'
        # print ppo.sess.run(ppo.mu_reshape, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print ppo.sess.run(ppo.sigma_reshape, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print ppo.sess.run(ppo.ratio, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print 'aloss'
        # print ppo.sess.run(ppo.aloss, {ppo.tfa: [a], ppo.tfs: [state], ppo.tfadv: [[ppo.get_v(state)]]})
        # print 'prob'
        # print ppo.sess.run(ppo.pi.prob(a), {ppo.tfa: [a], ppo.tfs:[state], ppo.tfadv:[[ppo.get_v(state)]]})

        next_state, reward = env.step(a, timetag + t, t, num, datetag)

        print 'reward: %g'%reward

        num += 1

        buffer_s.append(state)
        buffer_a.append(a)
        buffer_r.append(reward)    # normalize reward, find to be useful
        state = next_state
        ep_r += reward

        # update ppo
        if (t%3 == 0 and t!=0 ) :
            v_s_ = ppo.get_v(next_state)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []

            bs_ = np.reshape(bs, (-1, 3, 100))

            print 'updating times %d ' % (num)
            # print ppo.sess.run(ppo.ssssss, {ppo.tfa: ba, ppo.tfs: bs_, ppo.tfdc_r: br})
            # print ppo.sess.run(ppo.ratio, {ppo.tfa: ba, ppo.tfs: bs_, ppo.tfdc_r: br})
            # print  ppo.sess.run(ppo.tfdc_r - ppo.v,{ppo.tfa: ba, ppo.tfs: bs_, ppo.tfdc_r: br})

            # print ppo.sess.run(ppo.surr_, {ppo.tfa: ba, ppo.tfs: bs_, ppo.tfdc_r: br,ppo.tfadv:ppo.sess.run(ppo.tfdc_r - ppo.v, {ppo.tfa: ba, ppo.tfs: bs_, ppo.tfdc_r: br})})
            ppo.update(bs_, ba, br, step)

            print ba
            step+=4

        t += 1
        print 'step %d in %d episode' % (t, ep)
        # cumreward.append(reward*400)

        if t >= episode_length :
            # cumreward.append(value_now)
            if r_sum_max < sum(cumreward):
                r_sum_max = sum(cumreward)
                save_path = saver.save(ppo.sess, model_file)

            reward_sum.append(ep_r*400)
            plt.ion()
            plt.show()
            plt.plot(reward_sum)
            plt.draw()
            plt.pause(0.2)
            flag = sum(cumreward)

            print ep_r
            print round(sum(cumreward))
            print '-----------------------next epoch--------------------------'
            break
    timetag+=1

end = datetime.datetime.now()

plt.close()

print 'time cost: %s'%(end-begin)

xmajorLocator = MultipleLocator(1)

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111)

plt.plot(reward_sum)

plt.xlabel("episode")

plt.ylabel("cumulative reward")

plt.show()

ax.xaxis.set_major_locator(xmajorLocator)

env.end()

writer.close()
