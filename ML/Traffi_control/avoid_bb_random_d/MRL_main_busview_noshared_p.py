import tensorflow as tf
import numpy as np
from MRL_Env_busview_p import Env
from BUS import bus
from BUS_STOP import bus_stop
from MRL_brain_busview_p import agent_PPO
from MRL_policy_estimator import policy_estimator
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import math
import matplotlib
matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":14})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='medium', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)
# RL
batch_size_ = 32
batch_size_pe = 32
num_episodes = 300
updateStep = 1

tf.reset_default_graph()
sess_ = tf.Session()
pe = policy_estimator(sess=sess_, state_dim=18, policy_dim=6)
pe.buildPENetwork()

state_dim = 3+6 # local state + overall policy estimation
models=[]
for i in range(6):
    model = agent_PPO(sess=sess_, action_dim=1, state_dim=state_dim, state_dim_local=3, action_bound=[],name='agent'+str(i))
    model.buildCriticNetwork()
    model.buildActorNetwork()
    models.append(model)


# tf.get_default_graph().finalize()
records = [0 for i in range(6)]
reward_set = []
reward_each_agent=[[] for i in range(6)]
reward_set_r = []
reward_set1=[]
reward_set2=[]
v_loss_set = []
flag = 0
asets = []
catching_time = []
policy_buffer = []
pe_loss_set=[]
'''
train_mode: 
0 departure at same location & different emit time
1 departure at different location & same emit time
'''
w1 = 0.8
w2 = 1.

bus_stop_list = [np.pi * 2 / 12 * (i) for i in range(12)]

bus_stop_list = [0.,np.pi * 1.1 / 6 ,np.pi * 2. / 6 ,np.pi * 2.5 / 6 ,np.pi * 4.2 / 6 ,np.pi * 5 / 6 ,np.pi * 6 / 6,
                 np.pi * 7 / 6,np.pi * 8 / 6 ,np.pi * 9.5 / 6 ,np.pi *  10/ 6 ,np.pi * 11 / 6  ]
if False:
    init_op = tf.group(tf.global_variables_initializer())
    sess_.run(init_op)
    saver = tf.train.Saver()
    # saver.restore(sess_, "G:/mcgill/MAS/BB_testbed/tempmap/model.ckpt")
    # print('restore parameters...')
    state_collect=[ ]
    control_id = [i for i in range(12)]
    for i in range(num_episodes):
        env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120, emit_time_list=[0. * 3/30 for i in range(6)],
                  bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                  sim_horizon=60*60*3,train_mode=0,control_id=control_id)  # 36s = 3600s real world

        # env.reset()
        rewards = 0
        j = 0
        buffer_r_c = [[] for i in range(6)] # bus-wise view
        buffer_a_c = [[] for i in range(6)]
        buffer_s_c = [[] for i in range(6)] # state buffer for actor
        buffer_s_c_all = [[] for i in range(6)] # state buffer for critic
        temp_r = []
        temp_r1=[]
        temp_r2=[]
        while True:
            is_sim_over = env.sim()
            # simulation is over
            if is_sim_over<0:
                catching_time.append(env.catching_time)
                #
                if env.catching_time>90000000 :
                    state_collect = pd.DataFrame(state_collect)
                    # state_collect.to_csv('state_.csv')
                    f = plt.figure()
                    ax = plt.subplot(111)
                    plt.xlim([0, 60*60*3])
                    plt.ylim(0, 6.1)
                    plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
                               ['stop 1', 'stop 2', 'stop 3', 'stop 4', 'stop 5', 'stop 6', 'stop 7', 'stop 8',
                                'stop 9', 'stop 10', 'stop 11', 'stop 12'])

                    for b in (env.bus_list):
                        y = np.array(b.loc_set)
                        masky = np.ma.array(y, mask=y >= 6.15)
                        maskscatter = np.ma.array(np.array(b.hold_action), mask=np.array(b.hold_action) == 0.)
                        plt.scatter([i for i in range(len(b.hold_action))], maskscatter, c='black')
                        plt.plot([i for i in range(y.shape[0])], masky, '-', label='bus  ' + str(b.id))
                        # plt.scatter([i for i in range(y.shape[0])], masky, s=0.05, label='bus  ' + str(b.id))

                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                  fancybox=True, shadow=False, ncol=3)

                    csfont = {'fontname': 'Times New Roman'}
                    # plt.title('cumulative reward in each episode')
                    plt.xlabel('Time step', **csfont)
                    plt.ylabel('Stop', **csfont)
                    plt.yticks(fontname="Times New Roman")
                    plt.xticks(fontname="Times New Roman")
                    plt.show()
                cost = np.array(env.cost).reshape(-1, )
                if is_sim_over==-2 and env.flag==1:
                    r = np.array(env.reward).reshape(-1, )
                if is_sim_over==-3 and env.flag==1:
                    r = np.array(env.reward).reshape(-1, )

                a = np.array(a).reshape(-1, )

                # policy estimator
                s_temp = []
                for c in range(len(s)):
                    if len(s[c]) == 0:
                        s_temp.append([-1., -1., -1.])
                    else:
                        s_temp.append(s[c][:])
                policy_buffer.append(np.array(s_temp).reshape(-1).tolist()[:] + np.array(a).reshape(-1).tolist()[:])

                # policy estimator
                estimate = sess_.run(pe.estimate,
                                     feed_dict={pe.state_holder: np.array(s_temp).reshape(1, -1)})[0]

                estimate[estimate<0.01] = 0

                s_next = env.state[:]
                env.state = []
                v_s_=[0 for v_id in range(6)]
                for bus_id in range(6):
                    if len(s[bus_id][:]) > 0:  # and env.bus_list[bus_id].is_serving_rl==-1:

                        buffer_s_c[bus_id].append(s[bus_id][:])
                        proxy = estimate[:bus_id].tolist() + estimate[bus_id + 1:].tolist()
                        # proxy = estimate[:].tolist()
                        # proxy[bus_id] = a[bus_id]
                        buffer_s_c_all[bus_id].append(s[bus_id][:] + proxy )
                        buffer_a_c[bus_id].append(a[bus_id])

                    if len(buffer_s_c[bus_id])>len(buffer_r_c[bus_id]):
                        # buffer_r_c[bus_id].append(r)
                        # buffer_r_c[bus_id].append(np.exp(-(w*s_next[bus_id][-2]+abs(s_next[bus_id][-1] - s_next[bus_id][-2]))))
                        # print('id %d, fh:%g bh:%g'%(bus_id,s_next[bus_id][-2],s_next[bus_id][-1]))
                        # buffer_r_c[bus_id].append(
                        #     w1 * np.exp(-s_next[bus_id][-1] ) + w2 *np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                        # buffer_r_c[bus_id].append(
                        #     w * np.exp(-a[bus_id]) + np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                        buffer_r_c[bus_id].append(
                            w1 * np.exp(-buffer_a_c[bus_id][-1]) + w2 * np.exp(
                                - abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                        temp_r1.append(np.exp(-buffer_a_c[bus_id][-1]))
                        temp_r2.append(np.exp(
                                - abs(s_next[bus_id][-1] - s_next[bus_id][-2])))

                    else:
                        v_s_[bus_id]=0
                    temp_r.append(r)


                rewards2=0
                k=0
                temp_q=[]
                for c in range(6):
                    if len(buffer_r_c[c])>0:
                        discounted_r = []
                        rewards2+=sum(buffer_r_c[c])
                        k+=len(buffer_r_c[c])
                        reward_each_agent[c].append(sum(buffer_r_c[c])/len(buffer_r_c[c]))
                        # v_t = 1
                        # for r in buffer_r_c[c]:
                        #     if v_t<len(buffer_r_c[c]):
                        #         v_s_ = sess_.run(models[0].v, feed_dict={
                        #         models[0].state_holder: np.array(buffer_s_c_all[c][v_t]).reshape(-1, 8)})
                        #     else:
                        #         break
                        #     discounted_r.append(r + 0.9 * v_s_)
                        #     v_t += 1
                        for r in buffer_r_c[c][::-1]:
                            v_s_[c] = r + 0.9 * v_s_[c]
                            discounted_r.append(v_s_[c])
                        discounted_r.reverse()
                        records[c] += 1
                        _, _, v_loss,v,p_loss,ratio ,p1,p2,adv,sg1,sg2,mim= sess_.run([models[0].p_trainOp, models[0].v_trainOp, models[0].v_loss,models[0].v,models[0].p_loss,models[0].ratio,models[0].p1,models[0].p2,models[0].advantage,models[0].surrogate,models[0].clip_surrogate,models[0].mim],
                                         feed_dict={models[0].state_holder: np.array(buffer_s_c_all[c][:]).reshape(-1,8),
                                                    models[0].state_holder_local: np.array(buffer_s_c[c][:]).reshape(-1, 3),
                                                    models[0].action_holder: np.array(buffer_a_c[c][:]).reshape(-1,1),
                                                    models[0].accumulated_reward_holder: np.array(discounted_r).reshape(-1,1),
                                                    models[0].epsilon_holder: 0.2})

                        temp_q.append(v_loss)
                        buffer_r_c[c] = []
                        buffer_a_c[c] = []
                        buffer_s_c[c] = []
                        buffer_s_c_all[c] = []

                        print('1 bus:%d  vloss: %g ploss: %g p1:%g p2:%g adv:%g sg1：%g sg2：%g' % (c, v_loss,p_loss,np.mean(np.array(p1)),np.mean(np.array(p2)) ,np.mean(np.array(adv)) ,np.mean(np.array(sg1)) ,np.mean(np.array(sg2))))
                v_loss_set.append(sum(temp_q)/len(temp_q))
                if k>0:
                    rewards2=rewards2/k
                for m in range(len(models)):
                    if records[m] % updateStep == 0:
                        sess_.run(models[0].Actor_network_update)

                # update policy estimator
                if len(policy_buffer) >= 32:
                    _, pe_loss = sess_.run([pe.estimate_trainOp, pe.estimate_loss],
                                           feed_dict={pe.state_holder: np.array(policy_buffer)[-batch_size_pe:,0:18],
                                                      pe.policy_holder: np.array(policy_buffer)[-batch_size_pe:,18:]})

                    print('pe loss %g'%pe_loss)

                    pe_loss_set.append(pe_loss)
                    policy_buffer = []
                # state_collect.append(([-1,-1,-1]))
                break

            # control!
            if is_sim_over ==0:
                if env.flag==0:
                    bl = env.bus_loc_stop[:]
                    s=env.state[:]
                    state_collect.append((s[0][:] + [env.bus_list[0].travel_sum]))
                    a=[]
                    for c in range(6):
                        if len(s[c][:])>0:
                            a.append(sess_.run(models[0].action, feed_dict={models[0].state_holder_local: [s[c]]})[0][0])
                        else:
                            a.append(0.)
                    env.control(a)
                    env.state=[]
                    env.flag=1
                else:
                    cost = np.array(env.cost).reshape(-1, )
                    a = np.array(a).reshape(-1, )
                    r = np.array(env.reward).reshape(-1,)

                    s_temp=[]
                    for c in range(len(s)):
                        if len(s[c])==0:
                            s_temp.append([-1., -1., -1.])
                        else:
                            s_temp.append(s[c][:])
                    policy_buffer.append(np.array(s_temp).reshape( -1).tolist()[:] + np.array(a).reshape(-1).tolist()[:])

                    # policy estimator
                    estimate = sess_.run(pe.estimate,
                                         feed_dict={pe.state_holder: np.array(s_temp).reshape(1, -1)})[0]
                    estimate[estimate < 0.01] = 0

                    for bus_id in range(6):
                        if len(s[bus_id][:])<= 0:
                            estimate[bus_id]=0

                    s_next = env.state[:]
                    bl = env.bus_loc_stop[:]
                    for bus_id in range(6):
                        if len(s[bus_id][:])>0:# and env.bus_list[bus_id].is_serving_rl==-1:
                            buffer_s_c[bus_id].append(s[bus_id][:])
                            proxy = estimate[:bus_id].tolist() + estimate[bus_id + 1:].tolist()

                            # proxy = estimate[:].tolist()
                            # proxy[bus_id] = a[bus_id]
                            buffer_s_c_all[bus_id].append(s[bus_id][:] + proxy)
                            buffer_a_c[bus_id].append(a[bus_id])

                        if len(s_next[bus_id][:])>0 and len(buffer_s_c[bus_id])>0:
                            # buffer_r_c[bus_id].append(
                            #     w1*np.exp(-s_next[bus_id][-1] ) +w2 *np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                            # buffer_r_c[bus_id].append(
                            #     w * np.exp(-a[bus_id]) + np.exp(- abs(s_next[bus_id][-1] - s_next[bus_id][-2])))

                            # buffer_r_c[bus_id].append(np.exp(-(w*s_next[bus_id][-2]+abs(s_next[bus_id][-1] - s_next[bus_id][-2]))))
                            # buffer_r_c[bus_id].append(r)
                            buffer_r_c[bus_id].append(
                                w1 * np.exp(-buffer_a_c[bus_id][-1]) + w2 * np.exp(
                                    - abs(s_next[bus_id][-1] - s_next[bus_id][-2])))
                            temp_r.append(r)
                            temp_r1.append(np.exp(-buffer_a_c[bus_id][-1]))
                            temp_r2.append(np.exp(
                                - abs(s_next[bus_id][-1] - s_next[bus_id][-2])))


                    if (j + 1) % 8 == 0 and False:
                        if len(policy_buffer) >= 32:
                            _, pe_loss = sess_.run([pe.estimate_trainOp, pe.estimate_loss],
                                                   feed_dict={pe.state_holder: np.array(policy_buffer)[-batch_size_pe:,0:18],
                                                              pe.policy_holder: np.array(policy_buffer)[-batch_size_pe:,18:]})

                            print('pe loss %g' % pe_loss)

                            pe_loss_set.append(pe_loss)
                            policy_buffer = []
                        s_temp = []
                        for c in range(len(s_next)):
                            if len(s_next[c]) == 0:
                                s_temp.append([-1., -1., -1.])
                            else:
                                s_temp.append(s_next[c][:])
                        policy_buffer.append(
                            np.array(s_temp).reshape(-1).tolist()[:] + np.array(a).reshape(-1).tolist()[:])

                        # policy estimator
                        estimate = sess_.run(pe.estimate,
                                             feed_dict={pe.state_holder: np.array(s_temp).reshape(1, -1)})[0]
                        estimate[estimate < 0.01] = 0
                        for c in range(6):
                            if len(s_next[c])==0:
                                estimate[c]=0

                        for c in range(6):
                            if len(buffer_r_c[c]) > 16 and  len(s_next[c])>0:
                                proxy = estimate[:c].tolist() + estimate[c + 1:].tolist()
                                # proxy = estimate[:].tolist()
                                # proxy[c] = buffer_a_c[c][-1]
                                v_s_ = sess_.run(models[0].v, feed_dict={models[0].state_holder: np.array(s_next[c]+ proxy).reshape(1, -1)})[0][0]
                                discounted_r = []
                                for r in buffer_r_c[c][::-1]:
                                    v_s_ = r + 0.9 * v_s_
                                    discounted_r.append(v_s_)
                                discounted_r.reverse()
                                records[c]+=1
                                _, _, v_loss = sess_.run([models[0].p_trainOp, models[0].v_trainOp, models[c].v_loss],
                                                         feed_dict={
                                                             models[0].state_holder: np.array(buffer_s_c_all[c]).reshape(-1,8),
                                                             models[0].state_holder_local: np.array(
                                                                 buffer_s_c[c]).reshape(-1, 3),
                                                             models[0].action_holder: np.array(buffer_a_c[c]).reshape(
                                                                 -1, 1),
                                                             models[0].accumulated_reward_holder: np.array(
                                                                 discounted_r).reshape(-1, 1),
                                                             models[0].epsilon_holder: 0.2})
                                buffer_r_c[c]=[]
                                buffer_a_c[c] = []
                                buffer_s_c[c] = []
                                buffer_s_c_all[c] = []
                                print('2 bus:%d  vloss: %g'%(c,v_loss))

                        for m in range(len(models)):
                            if records[m] % updateStep == 0:
                                sess_.run(models[0].Actor_network_update)
                    a=[]
                    s = s_next[:]
                    state_collect.append((s[0][:] + [env.bus_list[0].travel_sum]))
                    for bus_id in range(6):
                        if len(s[bus_id])>0:
                            a.append(sess_.run(models[0].action, feed_dict={models[0].state_holder_local: [s[bus_id]]})[0][0])
                        else:
                            a.append(0.)

                    env.control(a)
                    env.state = []

                    j+=1

        print(' num_episodes:%d   r:%g realvar: %g' % (i, rewards2, sum(temp_r)/len(temp_r)))
        print(records)

        reward_set.append(sum(temp_r)/len(temp_r))
        reward_set1.append(sum(temp_r1)/len(temp_r1))
        reward_set2.append(sum(temp_r2)/len(temp_r2))
        reward_set_r.append(rewards2)
        if(i+0)%20==0 and i>0:
            save_path = saver.save(sess_, "G:/mcgill/MAS/BB_testbed/tempmap2/model.ckpt")
    csfont = {'size': 18}
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    # ax.tick_params(axis='x', which='major', labelsize=12)
    # ax.tick_params(axis='y', which='major', labelsize=12)
    save_path = saver.save(sess_, "G:/mcgill/MAS/BB_testbed/tempmap2/model.ckpt")
    plt.xlabel('Training episode')
    plt.ylabel('Mean squared error')
    smoothing_window = 10
    # plt.plot(v_loss_set,color='orange')
    v_loss_set_smoothed = pd.Series(v_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data = pd.DataFrame(v_loss_set)
    data.to_csv('v_loss_set.csv')
    plt.plot(v_loss_set, alpha=0.2)
    plt.plot(v_loss_set_smoothed, color='orange')
    plt.grid()
    plt.show()
    f.savefig("critic.pdf", bbox_inches='tight')

    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)

    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    plt.xlabel('Training episode' )
    plt.ylabel('Cumulative global reward' )
    smoothing_window = 10
    rewards_smoothed = pd.Series(reward_set_r).rolling(smoothing_window, min_periods=smoothing_window).mean()

    rewards_smoothed1 = pd.Series(reward_set1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(reward_set2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    plt.plot(reward_set_r, alpha=0.2)
    plt.plot(rewards_smoothed,color='orange',label='total reward')
    data = pd.DataFrame(reward_set_r)
    data.to_csv('reward_set_r.csv')

    plt.plot(reward_set1, alpha=0.2)
    plt.plot(rewards_smoothed1, color='red',label='reward for holding penalty')
    data = pd.DataFrame(reward_set1)
    data.to_csv('reward_set1.csv')

    plt.plot(reward_set2, alpha=0.2)
    plt.plot(rewards_smoothed2, color='green',label='reward for headway equalization')
    plt.grid()
    ax.legend(loc='best',  fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    data = pd.DataFrame(reward_set2)
    data.to_csv('reward_set2.csv')
    plt.show()
    f.savefig("actor_train.pdf", bbox_inches='tight')

    f=plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    plt.xlabel('Training episode' )
    plt.ylabel('Average of cumulative reward for each agent in each episode' )
    smoothing_window = 10
    for i in range(6):
        rewards_smoothed = pd.Series(reward_each_agent[i]).rolling(smoothing_window,
                                                                   min_periods=smoothing_window).mean()
        # plt.plot(reward_each_agent[i], alpha=0.2)
        plt.plot(rewards_smoothed, label='bus  ' + str(i))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  fancybox=True, shadow=False, ncol=3)
    plt.grid()
    plt.show()
    f.savefig("agent-gobal.pdf", bbox_inches='tight')

    # state_collect = pd.DataFrame(state_collect)
    # state_collect.to_csv('state_.csv')
    # train visualization
    f = plt.figure()
    # f.set_size_inches((8, 6))
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    smoothing_window = 10

    # plt.title('cumulative reward in each episode')
    plt.xlabel('Training episode',**csfont)
    plt.ylabel('Cumulative global reward in each episode',**csfont)

    rewards_smoothed = pd.Series(reward_set).rolling(smoothing_window, min_periods=smoothing_window).mean()

    plt.plot(reward_set, alpha=0.2)
    plt.plot(rewards_smoothed,color='orange')


    plt.grid()
    plt.show()

    f.savefig("train_var.pdf", bbox_inches='tight')



    f = plt.figure()
    # f.set_size_inches((8, 6))
    # plt.title('time to catch')
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training step'  )
    plt.ylabel('Mean squared error'  )
    # plt.yticks( size='12')
    # plt.xticks( size='12')
    # plt.plot(catching_time)
    pe_smoothed = pd.Series(pe_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    pe_loss_set_df = pd.DataFrame(pe_loss_set)
    pe_loss_set_df.to_csv('pe_loss_set.csv')
    plt.plot(pe_loss_set, alpha=0.2)
    plt.plot(pe_smoothed,color='orange')
    plt.grid()
    f.savefig("jat_train.pdf", bbox_inches='tight')
    plt.show()

# test
var=[]
loads=[]
wait_time_avg = []
catching=[]
hold=[]
slack=[]
wait_time_avgs=[]
load_set=[]
holds=[]


if True:
    control_id = [i for i in range(12)]
    init_op = tf.group(tf.global_variables_initializer())
    sess_.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess_, "G:/mcgill/MAS/BB_testbed/tempmap2/model.ckpt")
    print('restore parameters...')
    control_id = [i for i in range(12)]
    for i in range(10):
        print('run: %d'%i)
        env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                  emit_time_list=[0 * 3 / 30 for i in range(6)],
                  bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=[np.pi * 2 / 12 * (i) for i in range(12)],
                  sim_horizon=3600*8, train_mode=0, control_id=control_id) # 36s = 3600s real world

        while True:
            is_sim_over = env.sim()
            # simulation is over
            if is_sim_over < 0:
                holds.append(env.holds)
                load_set.append(env.load_set)
                wait_time_avgs.append(env.wait_time_avgs)
                print(np.mean(np.array(holds)))
                print(np.var(np.array(load_set)) / np.mean(np.array(load_set)))
                print(np.mean(np.array(wait_time_avgs)))

                # break
                # total_hold = 0
                # for b in (env.bus_list):
                #     loads.append(b.serve_list)
                #     holds.append([tt*30 for tt in b.hold_time_list])
                #
                # load_set.append(np.var(loads, axis=0))
                # loads = []
                # for b in (env.bus_list):
                #     total_hold += b.hold_time_sum
                # print('totoal hold time %g ' % (total_hold * 30/env.stop_visit))
                #
                # total_slack = 0
                # for b in (env.bus_list):
                #     total_slack += b.slack_time_sum
                # print('totoal slack time %g ' % (total_slack * 30))
                #
                # slack.append(total_slack*30)
                # catching.append(env.catching_time)
                # hold.append(total_hold*30/env.stop_visit)
                #
                # slack.append(env.slack)
                # # catching.append(env.catching_time)
                # hold.append(env.hold)
                # holds.append(env.holds)
                # load_set.append(np.var(env.load_set, axis=0))
                if False:
                    # csfont = {  'size': '18'}
                    f = plt.figure()
                    ax = plt.subplot(111)
                    for b in (env.bus_list):
                        plt.plot([i for i in range(len(b.serve_list))], b.serve_list, '-*', label='bus  ' + str(b.id))
                        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                        #           fancybox=True, shadow=False, ncol=3)
                    plt.xlabel('Bus station' )
                    plt.ylabel('Number of served passengers' )
                    # plt.yticks(fontname="Times New Roman", size='12')
                    plt.xticks([j for j in range(12)], [str(t + 1) for t in range(12)] )

                    f.savefig("MH_load.pdf", bbox_inches='tight')

                    plt.show()

                if env.catching_time>90:
                    f = plt.figure()
                    ax = plt.gca()
                    f.set_size_inches((3, 3))
                    plt.xlim([0, 3600 * 8 + 3])
                    plt.ylim(0, 6.1)
                    plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
                               ['1', '2', '3', '4', '5', '6', '7', '8',
                                '9', '10', '11', '12'])
                    ax.tick_params(length=4, width=0.5)
                    bus_trajectory = []
                    bus_hold_action = []
                    bus_hold_action_w = []
                    for b in (env.bus_list):
                        y = np.array(b.loc_set)
                        bus_trajectory.append(b.loc_set)
                        masky = np.ma.array(y, mask=y >= 6.2)
                        maskscatter = np.ma.array(np.array(b.hold_action), mask=np.array(b.hold_action)== 0.)
                        bus_hold_action.append(b.hold_action)
                        bus_hold_action_w.append(b.hold_action_w)
                        normalize = matplotlib.colors.Normalize(vmin=30, vmax=180)
                        plt.scatter([i for i in range(len(b.hold_action))], maskscatter, c=b.hold_action_w, norm=normalize,cmap='binary'  )
                        # plt.colorbar()
                        # plt.scatter([i for i in range(len(b.hold_action))], maskscatter, c='black')
                        plt.plot([i for i in range(y.shape[0])], masky, '-', label='bus  ' + str(b.id))
                        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                        #           fancybox=True, shadow=False, ncol=3)
                    bus_trajectory = pd.DataFrame(bus_trajectory)
                    bus_hold_action = pd.DataFrame(bus_hold_action)
                    bus_hold_action_w = pd.DataFrame(bus_hold_action_w)

                    bus_trajectory.to_csv('MH_bus_trajectory.csv')
                    bus_hold_action.to_csv('MH_bus_hold_action.csv')
                    bus_hold_action_w.to_csv('MH_bus_hold_action_w.csv')

                    # plt.title('cumulative reward in each episode')
                    plt.xlabel('Time step' )
                    plt.ylabel('Station' )
                    plt.yticks( )
                    plt.xticks( )
                    f.savefig("MH_tr.pdf", bbox_inches='tight')
                    plt.show()
                    # f=plt.figure()
                    # plt.ylim(0, np.pi * 6)
                    # plt.yticks([0 + n * np.pi * 2 / 6 for n in range(18)],
                    #            [str((i+1)*2) for i in range(18)])
                    # for b in (env.bus_list):
                    #     plt.plot(b.trajectory, '-', label='bus  ' + str(b.id))
                    #     plt.legend(loc='best')
                    # csfont = {'fontname': 'Times New Roman'}
                    # # plt.title('cumulative reward in each episode')
                    # plt.xlabel('Time step', **csfont)
                    # plt.ylabel('Trajectory', **csfont)
                    # plt.yticks(fontname="Times New Roman")
                    # plt.xticks(fontname="Times New Roman")
                    # f.savefig("trajectory_MAS.pdf", bbox_inches='tight')
                    # plt.show()

                # wait_time_avg = []
                # for k in range(len(env.bus_stop_list)):
                #     wait_time_avg.append(
                #         float(env.bus_stop_list[k].wait_time_sum) / float(env.bus_stop_list[k].wait_num_all))
                # wait_time_avgs.append(env.wait_time_avgs)

                break

            # control!
            if is_sim_over == 0:
                if env.flag == 0:
                    bl = env.bus_loc_stop[:]
                    s = env.state[:]
                    a = []

                    for c in range(6):
                        if len(s[c]) > 0:
                            a.append(sess_.run(models[0].action , feed_dict={models[0].state_holder_local: [s[c]]})[0][0])
                        else:
                            a.append(0.)
                    env.control(a)
                    env.state = []

                    env.flag = 1
                else:
                    # r = np.array(env.reward).reshape(-1, )
                    # print(r)
                    s_next = env.state[:]
                    a = []
                    s = s_next[:]
                    bl = env.bus_loc_stop[:]
                    for bus_id in range(6):
                        if len(s[bus_id]) > 0:
                            a.append(sess_.run(models[0].action , feed_dict={models[0].state_holder_local: [s[bus_id]]})[0][
                                    0])
                        else:
                            a.append(0.)

                    env.control(a)
                    # print(a)
                    env.state = []

        print('state:%d num_episodes:%d' % (is_sim_over, i))

csfont = {'size'   : 18}
f = plt.figure()
ax = plt.subplot(111)
# f.set_size_inches((8, 6))
ax = plt.gca()
ax.tick_params(length=4, width=0.5)
bus_load=[]
for b in (env.bus_list):
    print(b.serve_list)
    plt.xticks(  np.arange(len(b.serve_list)),
               ('1', '2', '3', '4', '5', '6', '7'
                , '8', '9', '10', '11', '12') )
    bus_load.append(b.serve_list)
    plt.plot(b.serve_list)
    # plt.xticks(2*np.arange(len(b.serve_list)),('stop 1','stop 2','stop 3','stop 4','stop 5','stop 6','stop 7'
    #                                            ,'stop 8','stop 9','stop 10','stop 11','stop 12'), rotation=30)
    # plt.bar(w+2*np.arange(len(b.serve_list)),b.serve_list,0.2,  label = 'bus  '+str(b.id))
    # plt.legend(loc='best')
    # w+=0.2
bus_load = pd.DataFrame(bus_load)
bus_load.to_csv('MH_load_1.csv')
plt.xlabel('Stations' )
plt.ylabel('Load (Pax.)'  )
f.savefig("MH_load.pdf", bbox_inches='tight')
# plt.show()
df_holds = pd.DataFrame(holds)
df_load_set = pd.DataFrame(load_set)
df_wait_time_avgs = pd.DataFrame(wait_time_avgs)

df_holds.to_csv('MH_hold.csv')
df_load_set.to_csv('MH_load.csv')
df_wait_time_avgs.to_csv('MH_wait.csv')
loads = np.array(loads)

print(catching)
print(slack)
print(hold)
print(np.mean(np.array(load_set),axis=0).tolist())
print(np.mean(wait_time_avgs,axis=0).tolist())
print(np.mean(holds,axis=0).tolist())

print(np.mean(np.array(hold)))
print(np.mean(np.array(load_set)))
print(np.mean(wait_time_avgs))

