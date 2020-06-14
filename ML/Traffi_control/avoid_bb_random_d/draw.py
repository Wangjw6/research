import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import math
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
if False:
    f = plt.figure()
    f.set_size_inches((4,4))
    ax = plt.subplot(111)
    plt.xlabel('Training episode')
    plt.ylabel('Mean squared error')
    smoothing_window = 10

    plt.ylim(0, 160)
    plt.yticks([(i * 20) for i in range(8)], [str(i * 20) for i in range(8)])
    v_loss_set = np.array(pd.read_csv('v_loss_set.csv',index_col=0)).reshape(-1,)
    v_loss_set_smoothed = pd.Series(v_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(v_loss_set, alpha=0.2)
    plt.plot(v_loss_set_smoothed, color='black')
    plt.grid()
    f.savefig("critic.pdf", bbox_inches='tight')
    plt.show()

    f = plt.figure()
    f.set_size_inches((5,4))
    ax = plt.subplot(111)
    plt.xlabel('Training episode' )
    plt.ylabel('Cumulative reward' )
    smoothing_window = 10
    reward_set_r=np.array(pd.read_csv('reward_set_r.csv',index_col=0)).reshape(-1,)
    reward_set1=0.8*np.array(pd.read_csv('reward_set1.csv',index_col=0)).reshape(-1,)
    reward_set2=np.array(pd.read_csv('reward_set2.csv',index_col=0)).reshape(-1,)
    rewards_smoothed = pd.Series(reward_set_r).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed1 = pd.Series(reward_set1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(reward_set2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    plt.ylim(0, 1.6)
    plt.yticks([ (i*0.2) for i in range(9)],[0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6])
    plt.plot(reward_set_r, alpha=0.2)
    plt.plot(rewards_smoothed,color='black',label='Total reward')

    plt.plot(reward_set1, alpha=0.2)
    plt.plot(rewards_smoothed1, color='red',label='Reward for holding penalty')

    plt.plot(reward_set2, alpha=0.2)
    plt.plot(rewards_smoothed2, color='green',label='Reward for headway equalization')
    plt.grid()
    ax.legend(loc=4,  fancybox=True, shadow=False, ncol=1 )
    f.savefig("actor_train.pdf", bbox_inches='tight')
    # plt.show()

    f = plt.figure()
    f.set_size_inches((4,4))
    ax = plt.subplot(111)
    plt.xlabel('Training step'  )
    plt.ylabel('Mean squared error'  )
    pe_loss_set = np.array(pd.read_csv('pe_loss_set.csv',index_col=0)).reshape(-1,)
    pe_smoothed = pd.Series(pe_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()

    plt.ylim(0, 0.8)
    plt.yticks([ (i * 0.1) for i in range(8)], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    plt.plot(pe_loss_set, alpha=0.2)
    plt.plot(pe_smoothed,color='black')
    plt.grid()
    f.savefig("jat_train.pdf", bbox_inches='tight')
    # plt.show()

if True:
    f, ax = plt.subplots(nrows=2, ncols=3)

    f.set_size_inches((11, 8))

    ax = plt.subplot(231)
    bus_load_NH=np.array(pd.read_csv('NH_load_1.csv' ,index_col=0)).reshape(6,-1).tolist()
    plt.ylim(0, 1250)
    plt.yticks([0 + (n) * 200 for n in range(7)],
               ['0', '200', '400', '600', '800', '1000', '1200'])
    plt.title('NH')
    # plt.text(2, 5, 'NH', ha='left', rotation=0, wrap=True,
    #          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))

    # plt.text(0.5, 0.5, 'NH',horizontalalignment='right', verticalalignment='top',
    #      transform=ax.transAxes)

    for b in (bus_load_NH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b)

    plt.xlabel('Stop' )
    plt.ylabel('Load (pax.)' )

    ax = plt.subplot(232)

    plt.ylim(0, 1250)
    plt.yticks([0 + (n) * 200 for n in range(7)],
               ['0', '200', '400', '600', '800', '1000', '1200'])
    plt.title('HH')
    # ax.annotate('HH', xy=(1, 0), xycoords='axes fraction', fontsize=16,
    #             xytext=(-5, 160), textcoords='offset points',
    #             ha='right', va='bottom')
    bus_load_HH=np.array(pd.read_csv('HH_load_1.csv' ,index_col=0)).reshape(6,-1).tolist()
    for b in (bus_load_HH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b)

    plt.xlabel('Stop' )
    plt.ylabel('Load (pax.)' )
    ax = plt.subplot(233)
    plt.ylim(0, 1250)
    plt.yticks([0 + (n) * 200 for n in range(7)],
               ['0', '200', '400', '600', '800', '1000', '1200'])
    plt.title('SH')
    # ax.annotate('SH', xy=(1, 0), xycoords='axes fraction', fontsize=16,
    #             xytext=(-5, 160), textcoords='offset points',
    #             ha='right', va='bottom')
    bus_load_SH=np.array(pd.read_csv('SH_load_1.csv' ,index_col=0)).reshape(6,-1).tolist()
    for b in (bus_load_SH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b)
    plt.xlabel('Stop')
    plt.ylabel('Load (pax.)')

    ax = plt.subplot(235)
    plt.ylim(0, 850)
    plt.yticks([0 + (n) * 200 for n in range(5)],
               ['0', '200', '400', '600', '800'])
    plt.title('BH')
    # ax.annotate('BH', xy=(1, 0), xycoords='axes fraction', fontsize=16,
    #             xytext=(-5, 160), textcoords='offset points',
    #             ha='right', va='bottom')
    bus_load_BH = np.array(pd.read_csv('BH_load_1.csv', index_col=0)).reshape(6, -1).tolist()
    for b in (bus_load_BH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12'))
        plt.plot(b)
    plt.xlabel('Stop' )
    plt.ylabel('Load (pax.)' )

    ax = plt.subplot(234)
    plt.ylim(0, 850)
    plt.yticks([0 + (n) * 200 for n in range(5)],
               ['0', '200', '400', '600', '800'])
    plt.title('FH')
    # ax.annotate('FH', xy=(1, 0), xycoords='axes fraction', fontsize=16,
    #             xytext=(-5, 160), textcoords='offset points',
    #             ha='right', va='bottom')
    bus_load_FH=np.array(pd.read_csv('FH_load_1.csv' ,index_col=0)).reshape(6,-1).tolist()
    for b in (bus_load_FH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b)



    plt.xlabel('Stop' )
    plt.ylabel('Load (pax.)' )
    ax = plt.subplot(236)
    plt.ylim(0, 850)
    plt.yticks([0 + (n) * 200 for n in range(5)],
               ['0', '200', '400', '600', '800' ])
    plt.title('MH')
    bus_load_MH=np.array(pd.read_csv('MH_load_1.csv' ,index_col=0)).reshape(6,-1)
    # ax.annotate('MH', xy=(1, 0), xycoords='axes fraction', fontsize=16,
    #             xytext=(-5, 160), textcoords='offset points',
    #             ha='right', va='bottom')
    for b in (bus_load_MH):
        plt.xticks(np.arange(len(b)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.plot(b)

    plt.xlabel('Stop' )
    plt.ylabel('Load (pax.)' )
    f.tight_layout()
    f.subplots_adjust(hspace=0.52)
    f.savefig("load_comp.pdf", bbox_inches='tight')
    plt.show()


if False:
    # strategy_set=['NH_','HH_','SH_','FH_','BH_','MH_']
    strategy_set = [  'HH_', 'SH_', 'FH_', 'BH_', 'MH_']
    for i in range(len(strategy_set)):
        strategy= strategy_set[i]
        bus_trajectory = np.array(pd.read_csv(strategy+'bus_trajectory.csv', index_col=0)).reshape(6, -1).tolist()
        bus_hold_action = np.array(pd.read_csv(strategy+'bus_hold_action.csv', index_col=0)).reshape(6, -1).tolist()
        bus_hold_action_w = np.array(pd.read_csv(strategy+'bus_hold_action_w.csv', index_col=0)).reshape(6, -1).tolist()

        f = plt.figure()
        ax = plt.subplot(111)
        f.set_size_inches((5, 4))
        plt.xlim([0, 60 * 60 * 8 + 3])
        plt.ylim(0, 6.1)
        plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
                   ['1', '2', '3', '4', '5', '6', '7', '8',
                    '9', '10', '11', '12'])

        for b in range(6):
            y = np.array(bus_trajectory[b])
            masky = np.ma.array(y, mask=y >= 6.2)
            maskscatter = np.ma.array(np.array(bus_hold_action[b]), mask=np.array(bus_hold_action[b]) == 0.)
            normalize = matplotlib.colors.Normalize(vmin=30, vmax=180)
            plt.scatter([i for i in range(len(bus_hold_action[b]))], maskscatter, c=bus_hold_action_w[b], norm=normalize,
                        cmap='binary')

            plt.plot([i for i in range(y.shape[0])], masky, '-', label='bus  ' + str(b ))
        plt.colorbar(aspect=50)
        plt.xlabel('Time step ( '+ r'$\Delta$'+'t=1 sec)')
        plt.ylabel('Stop')
        plt.yticks()
        plt.xticks()
        f.savefig(strategy+"tr.pdf", bbox_inches='tight')
        print(strategy_set[i])
    plt.show()