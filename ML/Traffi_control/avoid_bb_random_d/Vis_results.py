import numpy as np
import pandas as pd
import  matplotlib
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import pickle
matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":12})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7,7])
matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size= 4)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size= 4)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='medium', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)

hold_NH = pd.DataFrame(pd.read_csv('NH_hold.csv',index_col=0))
hold_HH = pd.DataFrame(pd.read_csv('HH_hold.csv',index_col=0))
hold_SH = pd.DataFrame(pd.read_csv('SH_hold.csv',index_col=0))
hold_FH = pd.DataFrame(pd.read_csv('FH_hold.csv',index_col=0))
hold_BH = pd.DataFrame(pd.read_csv('BH_hold.csv',index_col=0))
hold_MH = pd.DataFrame(pd.read_csv('MH_hold.csv',index_col=0))

load_NH = pd.DataFrame(pd.read_csv('NH_load.csv',index_col=0))
load_HH = pd.DataFrame(pd.read_csv('HH_load.csv',index_col=0))
load_SH = pd.DataFrame(pd.read_csv('SH_load.csv',index_col=0))
load_FH = pd.DataFrame(pd.read_csv('FH_load.csv',index_col=0))
load_BH = pd.DataFrame(pd.read_csv('BH_load.csv',index_col=0))
load_MH = pd.DataFrame(pd.read_csv('MH_load.csv',index_col=0))

wait_NH = pd.DataFrame(pd.read_csv('NH_wait.csv',index_col=0))
wait_HH = pd.DataFrame(pd.read_csv('HH_wait.csv',index_col=0))
wait_SH = pd.DataFrame(pd.read_csv('SH_wait.csv',index_col=0))
wait_FH = pd.DataFrame(pd.read_csv('FH_wait.csv',index_col=0))
wait_BH = pd.DataFrame(pd.read_csv('BH_wait.csv',index_col=0))
wait_MH = pd.DataFrame(pd.read_csv('MH_wait.csv',index_col=0))

# t_set=[wait_time_NH_mean,wait_time_HH_mean,wait_time_BH_mean,wait_time_FH_mean,wait_time_MH_mean]
# t_set=[var_NH_mean,var_HH_mean,var_BH_mean,var_FH_mean,var_MH_mean]
# t_set = [hold_time_NH_mean, hold_time_HH_mean, hold_time_BH_mean, hold_time_FH_mean, hold_time_MH_mean]

labels=['HH','SH','BH','FH','MH']

i=0
# hold visulaization
if True:
    f=plt.figure()
    ax = plt.gca()
    f.set_size_inches((4, 4))
    a=np.array(hold_MH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean() )
    all_result=[[np.array(hold_HH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()) ],
                [ np.array(hold_SH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()) ],
                [ np.array(hold_BH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean() )],
                [ np.array(hold_FH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean() )],
                [ np.array(hold_MH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean() )]]
    # all_result = [[hold_HH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()],
    #               [hold_SH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()],
    #               [hold_BH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()],
    #               [hold_FH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()],
    #               [hold_MH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()]]
    hold_time_NH_mean = np.mean(np.array(hold_NH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()),axis=0)
    hold_time_HH_mean = np.mean(np.array(hold_HH.groupby((np.arange(len(hold_HH.columns)) // 72) + 1, axis=1).mean()),axis=0)
    hold_time_SH_mean = np.mean(np.array(hold_SH.groupby((np.arange(len(hold_SH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_BH_mean = np.mean(np.array(hold_BH.groupby((np.arange(len(hold_BH.columns)) // 72) + 1, axis=1).mean()),axis=0)
    hold_time_FH_mean = np.mean(np.array(hold_FH.groupby((np.arange(len(hold_FH.columns)) // 72) + 1, axis=1).mean()),axis=0)
    hold_time_MH_mean = np.mean(np.array(hold_MH.groupby((np.arange(len(hold_MH.columns)) // 72) + 1, axis=1).mean()),axis=0)
    a=np.mean(hold_time_MH_mean)
    hold_time_NH_std = np.std(np.array(hold_NH.groupby((np.arange(len(hold_NH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_HH_std = np.std(np.array(hold_HH.groupby((np.arange(len(hold_HH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_SH_std = np.std(np.array(hold_SH.groupby((np.arange(len(hold_SH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_BH_std = np.std(np.array(hold_BH.groupby((np.arange(len(hold_BH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_FH_std = np.std(np.array(hold_FH.groupby((np.arange(len(hold_FH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)
    hold_time_MH_std = np.std(np.array(hold_MH.groupby((np.arange(len(hold_MH.columns)) // 72) + 1, axis=1).mean()),
                                axis=0)

    t_set = [ hold_time_HH_mean, hold_time_SH_mean,hold_time_BH_mean, hold_time_FH_mean, hold_time_MH_mean]
    t_set_std = [hold_time_HH_std, hold_time_SH_std, hold_time_BH_std, hold_time_FH_std, hold_time_MH_std]

    plt.xlabel('Operation horizon (hr)'  )
    plt.ylabel('Average holding period (sec)'  )
    horizon=[3600*(i+1) for i in range(8)]
    i=0
    boxes=[]
    data1 = np.random.randn(40, 2)
    data2 = np.random.randn(30, 2)

    c = ['blue', 'orange', 'green', 'red', 'purple']
    for t in t_set:
        # t = [float(t[i])/horizon[i] for i in range(8)]
        plt.xticks(np.arange(len(t) - 1), ('2', '3', '4', '5', '6', '7'
                                           , '8'))
        dy = np.array(t_set_std[i]).reshape(-1,)[1:]*2

        up = [t[k] + t_set_std[i][k] for k in range(len(t))]
        down = [t[k] - t_set_std[i][k] for k in range(len(t))]
        plt.errorbar(np.arange(len(t) - 1),t[1:],yerr=dy,elinewidth=2,capsize=4,linestyle='',ecolor=c[i])
        plt.plot(t[1:], '-*', label=labels[i],color=c[i])

        # plt.fill_between(np.arange(len(t) - 1), down[1:], up[1:], alpha=0.5)
        # plt.boxplot(t[1:])
        # data=[]
        # h=1
        # while h<8:
        #     data.append(t[0][:,h])
        #     h+=1

        # plt.boxplot(data)
        # data=np.array(data)
        #
        # boxes.append(ax.boxplot(data.transpose(),  notch=True, widths=0.35,
        #                patch_artist=True, boxprops=dict(facecolor="C"+str(i))))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3)
        i+=1
    # ax.legend([boxes[0]["boxes"][0], boxes[1]["boxes"][0], boxes[2]["boxes"][0], boxes[3]["boxes"][0], boxes[4]["boxes"][0], boxes[5]["boxes"][0], boxes[6]["boxes"][0]], ['A', 'B', 'B', 'B', 'B', 'B'], loc='upper right')

    plt.grid()
    f.savefig("hold_system.pdf", bbox_inches='tight')
    # plt.show()
if True:
    i = 0
    f=plt.figure()
    f.set_size_inches((4, 4))
    ax = plt.gca()
    # ax.tick_params(length=4, width=0.5)
    wait_time_NH_mean = np.mean(np.array(wait_NH.groupby((np.arange(len(wait_NH.columns)) // 12) + 1, axis=1).mean()),axis=0)
    wait_time_HH_mean = np.mean(np.array(wait_HH.groupby((np.arange(len(wait_HH.columns)) // 12) + 1, axis=1).mean()),axis=0)
    wait_time_SH_mean = np.mean(np.array(wait_SH.groupby((np.arange(len(wait_SH.columns)) // 12) + 1, axis=1).mean()),
                               axis=0)
    wait_time_BH_mean = np.mean(np.array(wait_BH.groupby((np.arange(len(wait_BH.columns)) // 12) + 1, axis=1).mean()),axis=0)
    wait_time_FH_mean = np.mean(np.array(wait_FH.groupby((np.arange(len(wait_FH.columns)) // 12) + 1, axis=1).mean()),axis=0)
    wait_time_MH_mean = np.mean(np.array(wait_MH.groupby((np.arange(len(wait_MH.columns)) // 12) + 1, axis=1).mean()),axis=0)

    wait_time_NH_std = np.std(np.array(wait_NH.groupby((np.arange(len(wait_NH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)
    wait_time_HH_std = np.std(np.array(wait_HH.groupby((np.arange(len(wait_HH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)
    wait_time_SH_std = np.std(np.array(wait_SH.groupby((np.arange(len(wait_SH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)
    wait_time_BH_std = np.std(np.array(wait_BH.groupby((np.arange(len(wait_BH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)
    wait_time_FH_std = np.std(np.array(wait_FH.groupby((np.arange(len(wait_FH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)
    wait_time_MH_std = np.std(np.array(wait_MH.groupby((np.arange(len(wait_MH.columns)) // 12) + 1, axis=1).mean()),
                                axis=0)

    t_set = [ wait_time_HH_mean,wait_time_SH_mean, wait_time_BH_mean, wait_time_FH_mean, wait_time_MH_mean]
    t_set_std = [wait_time_HH_std, wait_time_SH_std, wait_time_BH_std, wait_time_FH_std, wait_time_MH_std]

    plt.xlabel('Operation horizon (hr)'  )
    plt.ylabel('Average waiting period (sec)'  )

    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    horizon=[3600*(i+1) for i in range(8)]
    c = ['blue', 'orange', 'green', 'red', 'purple']
    for t in t_set:
        plt.xticks(np.arange(len(t) - 1), ('2', '3', '4', '5', '6', '7'
                                           , '8'))
        dy = np.array(t_set_std[i]).reshape(-1, )[1:] * 2

        up = [t[k] + t_set_std[i][k] for k in range(len(t))]
        down = [t[k] - t_set_std[i][k] for k in range(len(t))]
        plt.errorbar(np.arange(len(t) - 1), t[1:], yerr=dy, elinewidth=2, capsize=4, linestyle='', ecolor=c[i])
        plt.plot(t[1:], '-*', label=labels[i], color=c[i])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3)
        i += 1
    # for t in t_set:
    #     plt.xticks(np.arange(len(t) - 1), ('2', '3', '4', '5', '6', '7'
    #                                        , '8'))
    #     plt.plot(t[1:], '-*', label=labels[i])
    #     up = [t[k] + t_set_std[i][k] for k in range(len(t))]
    #     down = [t[k] - t_set_std[i][k] for k in range(len(t))]
    #     plt.fill_between(np.arange(len(t) - 1), down[1:], up[1:], alpha=0.5)
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #               fancybox=True, shadow=False, ncol=3)
    plt.grid()
    f.savefig("wait_system.pdf", bbox_inches='tight')
    # plt.show()

if True:
    f=plt.figure()
    ax = plt.gca()
    f.set_size_inches((4, 4))
    var_NH_mean=[]
    var_NH_std = []
    for i in range(8):
        result=[]
        for j in range(10):
            data = np.array(load_NH.iloc[j,i*72:(i+1)*72]).reshape(6,12)
            var = np.var(data, axis=0) / np.mean(data, axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_NH_mean.append(sum(result)/10.)
        var_NH_std.append(np.std(np.array(result)))
    var_HH_mean=[]
    var_HH_std=[]
    for i in range(8):
        result=[]
        for j in range(10):
            data = np.array(load_HH.iloc[j,i*72:(i+1)*72]).reshape(6,12)
            var = np.var(data,axis=0)/np.mean(data,axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_HH_mean.append(sum(result)/10.)
        var_HH_std.append(np.std(np.array(result)))
    var_SH_mean = []
    var_SH_std=[]
    for i in range(8):
        result = []
        for j in range(10):
            data = np.array(load_SH.iloc[j, i * 72:(i + 1) * 72]).reshape(6, 12)
            var = np.var(data, axis=0) / np.mean(data, axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_SH_mean.append(sum(result) / 10.)
        var_SH_std.append(np.std(np.array(result)))
    var_BH_mean=[]
    var_BH_std=[]
    for i in range(8):
        result=[]
        for j in range(10):
            data = np.array(load_BH.iloc[j,i*72:(i+1)*72]).reshape(6,12)
            var = np.var(data, axis=0) / np.mean(data, axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_BH_mean.append(sum(result)/10.)
        var_BH_std.append(np.std(np.array(result)))

    var_FH_mean=[]
    var_FH_std=[]
    for i in range(8):
        result=[]
        for j in range(10):
            data = np.array(load_FH.iloc[j,i*72:(i+1)*72]).reshape(6,12)
            var = np.var(data, axis=0) / np.mean(data, axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_FH_mean.append(sum(result)/10.)
        var_FH_std.append(np.std(np.array(result)))
    var_MH_mean=[]
    var_MH_std=[]
    for i in range(8):
        result=[]
        for j in range(10):
            data = np.array(load_MH.iloc[j,i*72:(i+1)*72]).reshape(6,12)
            var = np.var(data, axis=0) / np.mean(data, axis=0)/ np.mean(data, axis=0)
            result.append(np.mean(var))
        var_MH_mean.append(sum(result)/10.)
        var_MH_std.append(np.std(np.array(result)))
    t_set = [ var_HH_mean, var_SH_mean, var_BH_mean, var_FH_mean, var_MH_mean]
    t_set_std = [var_HH_std, var_SH_std, var_BH_std, var_FH_std, var_MH_std]

    plt.xlabel('Operation horizon (hr)'  )
    plt.ylabel('Avearage occupancy dispersion' )
    horizon=[3600*(i+1) for i in range(8)]
    i = 0
    c = ['blue', 'orange', 'green', 'red', 'purple']
    for t in t_set:
        # t = [float(t[i])/horizon[i] for i in range(8)]
        plt.xticks(np.arange(len(t) - 1), ('2', '3', '4', '5', '6', '7'
                                           , '8'))
        dy = np.array(t_set_std[i]).reshape(-1, )[1:] * 2

        up = [t[k] + t_set_std[i][k] for k in range(len(t))]
        down = [t[k] - t_set_std[i][k] for k in range(len(t))]
        plt.errorbar(np.arange(len(t) - 1), t[1:], yerr=dy, elinewidth=2, capsize=4, linestyle='', ecolor=c[i])
        plt.plot(t[1:], '-*', label=labels[i], color=c[i])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3)
        i += 1
    # for t in t_set:
    #     plt.xticks(np.arange(len(t) - 1), ('2', '3', '4', '5', '6', '7'
    #                                        , '8'))
    #     plt.plot(t[1:], '-*', label=labels[i])
    #     up = [t[k] + t_set_std[i][k] for k in range(len(t))]
    #     down = [t[k] - t_set_std[i][k] for k in range(len(t))]
    #     plt.fill_between(np.arange(len(t) - 1), down[1:], up[1:], alpha=0.5)
    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #               fancybox=True, shadow=False, ncol=3)
    #     i += 1
    plt.grid()
    f.savefig("var_system.pdf", bbox_inches='tight')
    # plt.show()

if True:
    f=plt.figure()
    f.set_size_inches((4, 4))
    ax = plt.gca()
    w = 0.
    hold_time_NH = np.mean(np.array(hold_NH).reshape(-1,12),axis=0)
    hold_time_HH = np.mean(np.array(hold_HH).reshape(-1, 12), axis=0)
    hold_time_SH = np.mean(np.array(hold_SH).reshape(-1, 12), axis=0)
    hold_time_BH = np.mean(np.array(hold_BH).reshape(-1, 12), axis=0)
    hold_time_FH = np.mean(np.array(hold_FH).reshape(-1, 12), axis=0)
    hold_time_MH = np.mean(np.array(hold_MH).reshape(-1, 12), axis=0)

    t_set = [ hold_time_HH,hold_time_SH, hold_time_BH, hold_time_FH, hold_time_MH]
    plt.xlabel('Stop'  )
    plt.ylabel('Average holding period (sec)' )

    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    horizon=[3600*(i+1) for i in range(8)]
    i=0
    for t in t_set:
        plt.xticks(8 * np.arange(len(t)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.bar(w + 8 * np.arange(len(t)), t, 1, label=labels[i])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3 )
        w += 1.2
        i += 1
    # plt.grid()
    f.savefig("hold.pdf", bbox_inches='tight')
    # plt.show()

if True:
    f=plt.figure()
    f.set_size_inches((4, 4))
    ax = plt.gca()
    w = 0.
    wait_time_NH = np.mean(np.array(wait_NH).reshape(-1,12),axis=0)
    wait_time_HH = np.mean(np.array(wait_HH).reshape(-1, 12), axis=0)
    wait_time_SH = np.mean(np.array(wait_SH).reshape(-1, 12), axis=0)
    wait_time_BH = np.mean(np.array(wait_BH).reshape(-1, 12), axis=0)
    wait_time_FH = np.mean(np.array(wait_FH).reshape(-1, 12), axis=0)
    wait_time_MH = np.mean(np.array(wait_MH).reshape(-1, 12), axis=0)

    t_set = [ wait_time_HH, wait_time_SH, wait_time_BH, wait_time_FH, wait_time_MH]
    plt.xlabel('Stop'  )
    plt.ylabel('Average waiting time (sec)'  )

    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    horizon=[3600*(i+1) for i in range(8)]
    i = 0
    for t in t_set:
        plt.xticks(8 * np.arange(len(t)),
                   ('1', '2', '3', '4', '5', '6', '7'
                    , '8', '9', '10', '11', '12') )
        plt.bar(w + 8 * np.arange(len(t)), t, 1, label=labels[i])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3 )
        w += 1.2
        i += 1
    # plt.grid()
    f.savefig("wait.pdf", bbox_inches='tight')
    # plt.show()

if True:
    f=plt.figure()
    ax = plt.subplot(111)
    f.set_size_inches((4, 4))
    w = 0.
    i=0
    var_NH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1,12))
            for t in range(8):
                d+=np.array(load_NH.iloc[j,:]).reshape(-1,12)[b+t*6,:]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_NH.append(np.var(data,axis=0)/np.mean(data,axis=0))
    var_NH = np.mean(np.array(var_NH),axis=0).tolist()[0]

    var_HH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1,12))
            for t in range(8):
                d+=np.array(load_HH.iloc[j,:]).reshape(-1,12)[b+t*6,:]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_HH.append(np.var(data, axis=0) / np.mean(data, axis=0))
    var_HH = np.mean(np.array(var_HH), axis=0).tolist()[0]

    var_SH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1, 12))
            for t in range(8):
                d += np.array(load_SH.iloc[j, :]).reshape(-1, 12)[b + t * 6, :]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_SH.append(np.var(data, axis=0) / np.mean(data, axis=0))
    var_SH = np.mean(np.array(var_SH), axis=0).tolist()[0]

    var_BH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1,12))
            for t in range(8):
                d+=np.array(load_BH.iloc[j,:]).reshape(-1,12)[b+t*6,:]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_BH.append(np.var(data, axis=0) / np.mean(data, axis=0))
    var_BH = np.mean(np.array(var_BH), axis=0).tolist()[0]

    var_FH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1,12))
            for t in range(8):
                d+=np.array(load_FH.iloc[j,:]).reshape(-1,12)[b+t*6,:]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_FH.append(np.var(data, axis=0) / np.mean(data, axis=0))
    var_FH = np.mean(np.array(var_FH), axis=0).tolist()[0]

    var_MH = []
    for j in range(10):
        bus_serve = []
        for b in range(6):
            d = np.zeros((1,12))
            for t in range(8):
                d+=np.array(load_MH.iloc[j,:]).reshape(-1,12)[b+t*6,:]
            bus_serve.append(d)
        data = np.array(bus_serve)
        var_MH.append(np.var(data, axis=0) / np.mean(data, axis=0))
    var_MH = np.mean(np.array(var_MH), axis=0).tolist()[0]

    t_set = [ var_HH,var_SH, var_BH, var_FH, var_MH]
    plt.xlabel('Stop'  )
    plt.ylabel('Average occupancy dispersion'  )

    # ax.tick_params(axis='x', which='major', labelsize=14)
    # ax.tick_params(axis='y', which='major', labelsize=14)
    horizon=[3600*(i+1) for i in range(8)]
    i = 0
    for t in t_set:
        plt.xticks(8 * np.arange(len(t)),
                   ('1', '2', '3', '4', '5', '6', ' 7'
                    , '8', '9', '10', '11', '12') )
        plt.bar(w + 8 * np.arange(len(t)), t, 1, label=labels[i])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=False, ncol=3 )
        w += 1.2
        i += 1
    # plt.grid()
    f.savefig("var.pdf", bbox_inches='tight')
    # plt.show()