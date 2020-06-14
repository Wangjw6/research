import tensorflow as tf
import numpy as np
from Env import Env
from BUS import bus
from BUS_STOP import bus_stop
import matplotlib.pyplot as plt

sess = tf.Session()
# 12 station 6 buses, corridor length 24 km ,r=120
catching=[]
slack=[]
i=0
loads=[]
wait_time_avgs = []
hold=[]
while i<20:
    env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120, emit_time_list=[0*3/30 for i in range(6)],
              bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=[np.pi*2/12*(i) for i in range(12)],
              sim_horizon=300) # 36s = 3600s real world
    env.run()
    total_hold = 0
    for b in (env.bus_list):
        total_hold += b.hold_time_sum * 30
    print('totoal hold time %g ' % (total_hold))
    hold.append(total_hold)
    total_slack = 0
    wait_time_avg=[]
    x=0
    for b in (env.bus_list):
        x+=(sum(b.serve_list))
        loads.append(b.serve_list)
    print(x)
    y=0
    for k in range(len(env.bus_stop_list)):
        y+=env.bus_stop_list[k].wait_num_all
        wait_time_avg.append(float(env.bus_stop_list[k].wait_time_sum) / float(env.bus_stop_list[k].wait_num_all))
    print(y)
    print('---')
    wait_time_avgs.append(wait_time_avg)
    if env.catching_time>90090:
        print(catching)
        print(slack)
        f = plt.figure()
        ax = plt.subplot(111)
        plt.xlim([0, 303])
        plt.ylim(0, np.pi * 2.1)
        plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
                   ['stop 1', 'stop 2', 'stop 3', 'stop 4', 'stop 5', 'stop 6', 'stop 7', 'stop 8',
                    'stop 9', 'stop 10', 'stop 11', 'stop 12'])

        for b in (env.bus_list):
            y = np.array(b.loc_set)
            d = np.ma.array(y, mask=y >= 6.2)
            plt.plot([i for i in range(y.shape[0])], d, '-', label='bus  ' + str(b.id))
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                      fancybox=False, shadow=False, ncol=3)

        csfont = {'fontname': 'Times New Roman'}
        # plt.title('cumulative reward in each episode')
        plt.xlabel('Time step', **csfont)
        plt.ylabel('Stop', **csfont)
        plt.yticks(fontname="Times New Roman")
        plt.xticks(fontname="Times New Roman")
        f.savefig("trajectory_MAS.pdf", bbox_inches='tight')
        plt.show()

    for b in (env.bus_list):
        total_slack+=b.slack_time_sum
    # print('totoal slack time %g '%(total_slack))
    catching.append(env.catching_time)
    slack.append(total_slack)
    i+=1

# trajectory visualization
# print(env.bus_stop_list[1].bus_arr_interval)
# print(env.bus_stop_list[1].actual_bus_arr)
# plt.ylim(0, np.pi*4.)
# plt.yticks([0+n*np.pi*2/12 for n in range(12)],['stop 1', 'stop 2', 'stop 3', 'stop 4','stop 5', 'stop 6', 'stop 7', 'stop 8','stop 9', 'stop 10', 'stop 11', 'stop 12'])
# for b in (env.bus_list):
#     print(len(b.trajectory))
#     plt.plot(b.trajectory,'-', label='bus  ' + str(b.id))
#     plt.legend(loc='best')
# plt.show()
#
# plt.title('headway deviation from schedule')
# for p in (env.bus_stop_list):
#     print(p.bus_arr_interval)
#     plt.plot([abs(interval-p.schedule_hw)/p.schedule_hw for interval in p.bus_arr_interval],  label = 'bus stop '+str(p.id))
#     plt.legend(loc='best')
# plt.show()

# plt.title('Accumulated number of person served in each stop')
w=0.2
for b in (env.bus_list):
    loads.append(b.serve_list)
    # plt.plot(b.serve_list)
    # # plt.xticks(2*np.arange(len(b.serve_list)),('stop 1','stop 2','stop 3','stop 4','stop 5','stop 6','stop 7'
    # #                                            ,'stop 8','stop 9','stop 10','stop 11','stop 12'), rotation=30)
    # # plt.bar(w+2*np.arange(len(b.serve_list)),b.serve_list,0.2,  label = 'bus  '+str(b.id))
    # plt.legend(loc='best')
    # w+=0.2
loads = np.array(loads)
wait_time_avgs=np.array(wait_time_avgs)
print(catching)
print(slack)
print(hold)
print(np.var(loads,axis=0).tolist())
print(np.mean(wait_time_avgs,axis=0).tolist())
plt.show()


