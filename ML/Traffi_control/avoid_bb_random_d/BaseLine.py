import tensorflow as tf
import numpy as np
from Env_schedule_based_control import Env
from BUS import bus
from BUS_STOP import bus_stop
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":12})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=4)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=4)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='medium', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)
sess = tf.Session()
catching=[]

bus_stop_list = [0.,np.pi * 1.1 / 6 ,np.pi * 2. / 6 ,np.pi * 2.5 / 6 ,np.pi * 4.2 / 6 ,np.pi * 5 / 6 ,np.pi * 6 / 6,
                 np.pi * 7 / 6,np.pi * 8 / 6 ,np.pi * 9.5 / 6 ,np.pi *  10/ 6 ,np.pi * 11 / 6  ]
tags = ['NH','FH','BH','SH','HH']
for tag in tags:
    hold = []
    slack = []
    loads = []
    wait_time_avgs = []
    load_set = []
    holds = []
    for i in range(1):
        print('episode:%d'%i)
        # 11 station 6 buses corridor length 24 km , r=120 scale=100:pi
        # f = plt.figure()
        # f.set_size_inches((3,3))
        # ax = plt.subplot(111)
        # arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
        #              1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        # plt.xticks(  np.arange(len(arr_rates)),
        #            ('1', '2', '3', '4', '5', '6', '7'
        #             , '8', '9', '10', '11', '12'))
        #
        # arr_rates = [a*60 for a in arr_rates]
        # plt.bar([s for s in range(12)],arr_rates ,color = 'lightseagreen',alpha=0.8)
        #
        # plt.xlabel('Stop' )
        # plt.ylabel('Passenger arrival rate (pax./min)' )
        # f.savefig('arr_rate.pdf')
        # plt.show()
        tag = 'SH'
        env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                  emit_time_list=[0 for i in range(6)],
                  bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                  sim_horizon=3600 * 8, FH=False, BH=False,
                  HH=False, SH=False, FB=False)
        if tag=='FH':
            env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                      emit_time_list=[0  for i in range(6)],
                      bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                      sim_horizon=3600*8,FH=True,BH=False,
                      HH = False, SH = False,FB = False)  # 36s = 3600s real world
        if tag=='BH':
            env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                      emit_time_list=[0  for i in range(6)],
                      bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                      sim_horizon=3600*8,FH=False,BH=True,
                      HH = False, SH = False,FB = False)
        if tag=='SH':
            env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                      emit_time_list=[0  for i in range(6)],
                      bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                      sim_horizon=3600*8,FH=False,BH=False,
                      HH = False, SH = True,FB = False)
        if tag=='HH':
            env = Env(state_dim=3, action_dim=1, bus_num=6, bus_stop_num=12, r=120,
                      emit_time_list=[0  for i in range(6)],
                      bus_dep_list=[0 for i in range(6)], bus_stop_loc_list=bus_stop_list,
                      sim_horizon=3600*8,FH=False,BH=False,
                      HH = True, SH = False,FB = False)
        env.run()

        if False:
            csfont = {'fontname': 'Times New Roman', 'size': '16'}
            f = plt.figure()
            ax = plt.subplot(111)
            for b in (env.bus_list):
                plt.plot([i for i in range(len(b.serve_list))],b.serve_list,'-*',  label='bus  ' + str(b.id))
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                          fancybox=True, shadow=False, ncol=3)
            plt.xlabel('Bus stop', **csfont)
            plt.ylabel('Number of served passengers', **csfont)
            plt.yticks(fontname="Times New Roman", size='12')
            plt.xticks([j for j in range(12)],[str(t+1) for t in range(12)],fontname="Times New Roman", size='12')

            # f.savefig("HH_load.pdf", bbox_inches='tight')
            # f.savefig("HH_load.pdf", bbox_inches='tight')
            # f.savefig("BH_load.pdf", bbox_inches='tight')
            f.savefig("FH_load.pdf", bbox_inches='tight')
            plt.show()


        holds.append(env.holds)

        load_set.append(env.load_set)

        wait_time_avgs.append(env.wait_time_avgs)

        print(np.mean(np.array(holds)))
        print(np.var(np.array(load_set))/np.mean(np.array(load_set)))
        print(np.mean(np.array(wait_time_avgs)))
        print(np.mean(np.array(load_set)))
        # for b in env.bus_list:
        #     plt.plot(b.serve_list)
        # plt.show()
        if env.catching_time>800 :
            f = plt.figure()
            ax = plt.gca()
            f.set_size_inches((3, 3))
            plt.xlim([0, 3600*8+3])
            plt.ylim(0, 6.1)
            plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
                       ['1', '2', '3', '4', '5', '6', '7',
                        '8', '9', '10', '11', '12'])
            ax.tick_params(length=4, width=0.5 )
            bus_trajectory=[]
            bus_hold_action=[]
            bus_hold_action_w=[]
            for b in (env.bus_list):
                y = np.array(b.loc_set)
                bus_trajectory.append(b.loc_set)
                masky = np.ma.array(y, mask=y >= 6.2)
                maskscatter = np.ma.array(np.array(b.hold_action), mask=np.array(b.hold_action) == 0.)
                bus_hold_action.append(b.hold_action)
                bus_hold_action_w.append(b.hold_action_w)
                normalize = matplotlib.colors.Normalize(vmin=30, vmax=180)
                plt.scatter([i for i in range(len(b.hold_action))], maskscatter, c=b.hold_action_w, norm=normalize,
                            cmap='binary')
                # plt.colorbar()
                # plt.scatter([i for i in range(len(b.hold_action))], maskscatter, c='black')
                plt.plot([i for i in range(y.shape[0])], masky, '-', label='bus  ' + str(b.id))
                # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                #           fancybox=True, shadow=False, ncol=3)
            # plt.colorbar(aspect=50)
            bus_trajectory = pd.DataFrame(bus_trajectory)
            bus_hold_action = pd.DataFrame(bus_hold_action)
            bus_hold_action_w = pd.DataFrame(bus_hold_action_w)

            bus_trajectory.to_csv(tag+'_bus_trajectory.csv')
            bus_hold_action.to_csv(tag+'_bus_hold_action.csv')
            bus_hold_action_w.to_csv(tag+'_bus_hold_action_w.csv')
            plt.xlabel('Time step' )
            plt.ylabel('Station' )
            f.savefig(tag+"_tr.pdf", bbox_inches='tight')

            plt.show()



    df_holds = pd.DataFrame(holds)
    df_load_set = pd.DataFrame(load_set)
    df_wait_time_avgs = pd.DataFrame(wait_time_avgs)

    df_holds.to_csv(tag+'_hold.csv')
    df_load_set.to_csv(tag+'_load.csv')
    df_wait_time_avgs.to_csv(tag+'_wait.csv')

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

    print('done')
    # # trajectory visualization
    #
    # print(env.bus_stop_list[1].bus_arr_interval)
    # print(env.bus_stop_list[1].actual_bus_arr)


    # plt.ylim(0, np.pi*8)

    # #
    # plt.title('headway deviation from schedule')
    # for p in (env.bus_stop_list):
    #     print(p.actual_bus_arr)
    #     plt.plot([abs(interval - p.schedule_hw) / p.schedule_hw for interval in p.bus_arr_interval],
    #              label='bus stop ' + str(p.id))
    #     plt.legend(loc='best')
    # plt.show()
    #

    # plt.title('Accumulated number of person served in each stop')
    w=0.2
    # csfont = {'size'   : 18}
    f = plt.figure()
    ax = plt.subplot(111)
    f.set_size_inches((3, 3))
    ax = plt.gca()
    ax.tick_params(length=4, width=0.5)
    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    bus_load=[]
    for b in (env.bus_list):
        print(b.serve_list)
        plt.xticks(  np.arange(len(b.serve_list)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b.serve_list)
        bus_load.append(b.serve_list)
        # plt.xticks(2*np.arange(len(b.serve_list)),('stop 1','stop 2','stop 3','stop 4','stop 5','stop 6','stop 7'
        #                                            ,'stop 8','stop 9','stop 10','stop 11','stop 12'), rotation=30)
        # plt.bar(w+2*np.arange(len(b.serve_list)),b.serve_list,0.2,  label = 'bus  '+str(b.id))
        # plt.legend(loc='best')
        # w+=0.2
    bus_load = pd.DataFrame(bus_load)
    bus_load.to_csv(tag+'_load_1.csv')
    plt.xlabel('Station' )
    plt.ylabel('Load (pax.)' )
    f.savefig(tag+'_load.pdf', bbox_inches='tight')
    # plt.show()