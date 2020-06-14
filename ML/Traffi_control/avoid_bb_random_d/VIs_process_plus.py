import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
# f = plt.figure()
# plt.title('title',**csfont)
# plt.ylim(0, np.pi * 4)
# plt.yticks([0 + n * np.pi * 2 / 12 for n in range(12)],
#            ['stop 1', 'stop 2', 'stop 3', 'stop 4', 'stop 5', 'stop 6', 'stop 7', 'stop 8',
#             'stop 9', 'stop 10', 'stop 11', 'stop 12'],**csfont)
# plt.plot(range(10), range(10), "o")
# plt.show()
# f.savefig("foo.pdf", bbox_inches='tight')

# catching time: No control, Target-schedule control（TC）, Dynamic control based on backward headway,

hold_time_NH_mean=[0.0,117,930,2458.5,4897.5,9378,16033,25776]
hold_time_HH_mean=[1097.,1690,2392,3285,4300,6389,9742,13947]
hold_time_BH_mean=[2915.,5791,8564,11242,14047,17197.4,19936.4,23382]
hold_time_FH_mean=[143.5,558.2,1202.,2435.3,3653.3,6463.39,8252.42,15787.8]
hold_time_MH_mean=[433,876,1400,2152,3169,3287.2,4246.1,4691]

var_NH_mean=[304.6,489,755.3,1155.67,2932,5126,7582,7874.4 ]
var_HH_mean=[370.8,604,836,1059.,1435.5,1750.,2672,3987]
var_BH_mean=[401.7,634.81,772.0,604.5,896.23,1023.20,1255.14,1265.2]
var_FH_mean=[331.7,516.2,739.6,1021.1,1322.6,1968.7,2418.9,4278.4]
var_MH_mean=[310.8,523,730,872,1063,1139.5,1198.6,1399]

wait_time_NH_mean=[14.62/2,12.1/2,12.78/2,14.49/2,16.4/2,19.8/2,23.7/2,30.69/2]
wait_time_HH_mean=[17.2/2,14.37/2,13.61/2,13.72/2,13.5/2,15.7/2,18.3/2,20.8/2]
wait_time_BH_mean=[17.4/2,14.22/2,12.95/2,12.4/2,12.11/2,11.97/2,11.8/2,11.98/2]
wait_time_FH_mean=[14.57/2,12.15/2,11.76/2,12.9/2,12.55/2,14.56/2,15.38/2,20.32/2]
wait_time_MH_mean=[14.2/2,11.4/2,10.8/2,10.4/2,10.4/2,10.2/2,10.45/2,10.18/2]


# t_set=[wait_time_NH_mean,wait_time_HH_mean,wait_time_BH_mean,wait_time_FH_mean,wait_time_MH_mean]
t_set=[var_NH_mean,var_HH_mean,var_BH_mean,var_FH_mean,var_MH_mean]
# t_set = [hold_time_NH_mean, hold_time_HH_mean, hold_time_BH_mean, hold_time_FH_mean, hold_time_MH_mean]
labels=['NC','HH','BH','FH','MH']
i=0
f=plt.figure()
csfont = {'fontname': 'Times New Roman', 'size': '16'}
plt.xlabel('Operation horizon (hr)', **csfont)
plt.ylabel('Normalized bus serving variance', **csfont)
# plt.ylabel('Average waiting time (min)', **csfont)
# plt.ylabel('Normalized average total hold period ', **csfont)
# plt.grid()
w=0.
ax = plt.subplot(111)
horizon=[3600*(i+1) for i in range(8)]
for t in t_set:
    t = [float(t[i])/horizon[i] for i in range(8)]
    plt.xticks(np.arange(len(t)),('1','2','3','4','5','6','7'
                                               ,'8'))
    plt.plot(t,'-*',  label = labels[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=False, ncol=5)
    i+=1
f.savefig("var_system.pdf", bbox_inches='tight')
plt.show()
# catching time with/without cost
