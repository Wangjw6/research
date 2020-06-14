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

catching_time_NC=[9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030, 9030]
slack_time_NC=[34080.0, 32670.0, 30210.0, 29550.0, 32970.0, 29430.0, 28680.0, 31080.0, 29490.0, 34710.0, 26910.0, 29820.0, 31320.0, 30120.0, 31800.0, 29010.0, 29340.0, 28740.0, 30180.0, 27480.0]
hold_time_NC=[132.0, 10.75, 34.0, 17.5, 46.25, 27.25, 82.0, 36.0, 158.5, 50.5, 45.0, 44.75]
wait_time_NC=[16.28150574063583, 15.031548386261477, 16.856765823378403, 15.562416862436653, 17.04450818761226, 15.56756169428167, 17.399037313596462, 14.821166195427438, 16.653807258867893, 15.323219937827067, 17.385570062194905, 14.892001364800493]
var_NC=[2422.0721250000015, 1613.898000000001, 1702.5153750000009, 1994.639625000001, 1861.6455000000012, 2060.2518750000013, 1912.692375000001, 1588.987125000001, 1547.3328750000012, 1869.5407500000013, 1949.1738750000015, 2409.4125000000013]

catching_time_FH=[18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030]
slack_time_FH=[35910.0, 32460.0, 30750.0, 31680.0, 29880.0, 32220.0, 35850.0, 32910.0, 32520.0, 33150.0, 30540.0, 30720.0, 32940.0, 32550.0, 30000.0, 32070.0, 35640.0, 29640.0, 28950.0, 30060.0]
hold_time_FH=[50.104999999996785, 60.38166666666366, 61.373333333330564, 55.71666666666399, 58.408333333330496, 59.49666666666341, 63.99166666666354, 64.4483333333301, 56.65999999999694, 59.85333333332997, 55.099999999996825, 53.75499999999714]
var_FH=[1278.077625000001, 1488.1185000000012, 1546.652250000001, 1360.841625000001, 1270.318500000001, 1272.768750000001, 1504.0451250000015, 1037.9531250000007, 1127.523375000001, 1265.418000000001, 1258.747875000001, 1319.187375000001]
wait_time_FH=[13.698547276241007, 12.0061272500821, 13.560808505705415, 12.006585788458713, 12.896632516852005, 11.712037468620231, 12.848811573299026, 11.652905532407093, 12.823966524203104, 11.476904179581968, 12.936537048107066, 12.023411381202397]


catching_time_BH=[18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030]
slack_time_BH=[42810.0, 43410.0, 44550.0, 43380.0, 42120.0, 42630.0, 42240.0, 44070.0, 42570.0, 43290.0, 43860.0, 43470.0, 43110.0, 44580.0, 44010.0, 43320.0, 42210.0, 43230.0, 43950.0, 43530.0]
hold_time_BH=[214.00000000000009, 195.3125, 185.12499999999994, 192.49999999999986, 197.4999999999999, 205.49999999999994, 197.56250000000006, 191.8125, 195.24999999999997, 202.24999999999997, 203.4375, 199.37500000000003]
var_BH=[1270.5907500000012, 757.9440000000006, 1070.759250000001, 812.8023750000004, 967.5765000000007, 817.0222500000007, 1128.3401250000009, 956.6865000000007, 833.4933750000007, 709.6196250000006, 753.8602500000005, 916.5296250000008]
wait_time_BH=[12.432535189281491, 11.756931578366439, 13.021134470595294, 11.431994505557443, 12.989144435798641, 11.57232203677957, 12.946056337038518, 11.302506209696606, 12.9755733105299, 11.597450825803936, 12.872444264015863, 11.473189590556895]

catching_time_TS=[18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030]
slack_time_TS=[33900.0, 32070.0, 37050.0, 34440.0, 37770.0, 33330.0, 33090.0, 35340.0, 36630.0, 38580.0, 31620.0, 33180.0, 33360.0, 31770.0, 31830.0, 31770.0, 32850.0, 31170.0, 33570.0, 30690.0]
hold_time_TS=[242.75000000000006, 44.50000000000004, 49.49999999999996, 62.50000000000004, 47.25000000000001, 61.0, 52.99999999999997, 59.999999999999986, 64.24999999999999, 60.25000000000002, 47.25000000000005, 59.000000000000036]
var_TS=[1165.230000000001, 1544.202000000001, 1547.8773750000014, 1866.2737500000014, 1432.171125000001, 1939.1006250000016, 1319.1873750000009, 1355.2605000000008, 1518.2021250000014, 1363.8363750000008, 1124.937000000001, 1083.2827500000008]
wait_time_TS=[14.637998941913803, 14.678999269346429, 16.561441195347673, 12.99626765206248, 15.22773575568809, 13.425447550781666, 15.452750522201388, 13.852257451984949, 15.393420920625875, 14.243681681678225, 16.470802081928724, 13.612071490555355]

catching_time_MAS=[18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030, 18030]
slack_time_MAS=[35160.0, 32310.0, 31890.0, 32580.0, 32880.0, 33060.0, 34230.0, 33420.0, 33480.0, 36720.0, 33240.0, 32730.0, 30840.0, 33030.0, 33480.0, 32130.0, 33210.0, 34680.0, 32160.0, 32850.0]
hold_time_MAS=[36.472167674452066, 32.98436205461621, 32.62374420184642, 39.09982818365097, 33.17094546929002, 42.52769573777914, 38.27595044672489, 39.61566421575844, 33.984071142971516, 34.74918906390667, 44.312577821314335, 38.92064518854022]
var_MAS=[1163.052000000001, 1235.606625000001, 1033.4610000000007, 1175.7116250000008, 1327.8993750000013, 1000.7910000000005, 1067.7645000000007, 964.1733750000009, 1002.8328750000007, 964.4456250000006, 1203.2088750000007, 1003.6496250000006]
wait_time_MAS=[11.020008626558006, 9.719729135960725, 11.15951342089861, 9.54616028375101, 11.127638602995496, 9.834904551421058, 11.028310012041738, 9.818031914720692, 11.362676069218281, 9.813032262725784, 11.196572419978725, 9.821216488015894]

# var draw
t_set=[hold_time_NC,hold_time_TS,hold_time_BH,hold_time_FH,hold_time_MAS]
wait_time_NC=[t*0.5 for t in (wait_time_NC)]
wait_time_TS=[t*0.5 for t in (wait_time_TS)]
wait_time_BH=[t*0.5 for t in (wait_time_BH)]
wait_time_FH=[t*0.5 for t in (wait_time_FH)]
wait_time_MAS=[t*0.5 for t in (wait_time_MAS)]

# t_set=[wait_time_NC,wait_time_TS,wait_time_BH,wait_time_FH,wait_time_MAS]
# t_set=[var_NC,var_TS,var_BH,var_FH,var_MAS]
t_set=[hold_time_NC,hold_time_TS,hold_time_BH,hold_time_FH,hold_time_MAS]
labels=['NC','HH','BH','FH','MH']
i=0
f=plt.figure()
csfont = {'fontname': 'Times New Roman', 'size': '16'}
# plt.xlabel('Holding strategies', **csfont)
# plt.ylabel('Bus serving variance', **csfont)
plt.ylabel('Average hold time (sec)', **csfont)
# plt.ylabel('Average waiting time (min)', **csfont)
# plt.grid()
w=0.
ax = plt.subplot(111)
for t in t_set:
    plt.xticks( 8*np.arange(len(t)),('station 1','station 2','station 3','station 4','station 5','station 6','station 7'
                                               ,'station 8','station 9','station 10','station 11','station 12'), rotation=30)
    plt.bar(w+ 8*np.arange(len(t)),t,1,  label = labels[i])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              fancybox=True, shadow=False, ncol=5)
    w+=1.2
    i+=1
f.savefig("hold.pdf", bbox_inches='tight')
plt.show()
# catching time with/without cost

# wait time draw
print('wait time average')
print(np.mean(np.array(wait_time_NC)))
print(np.mean(np.array(wait_time_TS)))
print(np.mean(np.array(wait_time_BH)))
print(np.mean(np.array(wait_time_FH)))
print(np.mean(np.array(wait_time_MAS)))
print('hold time')
print(np.mean(np.array(hold_time_NC)))
print(np.mean(np.array(hold_time_TS)))
print(np.mean(np.array(hold_time_BH)))
print(np.mean(np.array(hold_time_FH)))
print(np.mean(np.array(hold_time_MAS)))
f=plt.figure()
csfont = {'fontname': 'Times New Roman', 'size': '16'}
plt.xlabel('Holding strategies', **csfont)
plt.ylabel('Holding time (sec)', **csfont)
# plt.grid()
plt.xticks([i for i in range(5)],['NH','HH','BH','FH','MH'],fontname="Times New Roman", size='12')
plt.bar([i for i in range(5)],[np.mean(np.array(hold_time_NC)),np.mean(np.array(hold_time_TS)),np.mean(np.array(hold_time_BH)),np.mean(np.array(hold_time_FH)),np.mean(np.array(hold_time_MAS))],0.2,color=['darkred','darkorange','darkcyan','darkviolet','lime'])
f.savefig("hold.pdf", bbox_inches='tight')
plt.show()
print('catching time')
print(np.mean(np.array(catching_time_NC)))
print(np.mean(np.array(catching_time_TS)))
print(np.mean(np.array(catching_time_BH)))
print(np.mean(np.array(catching_time_FH)))
print(np.mean(np.array(catching_time_MAS)))
plt.plot(wait_time_NC, '-', label='wait_time_NC  '  )
plt.plot(wait_time_TS, '-', label='wait_time_TS '  )
plt.plot(wait_time_BH, '-', label='wait_time_BH '  )
plt.plot(wait_time_FH, '-', label='wait_time_FH '  )
plt.plot(wait_time_MAS, '-', label='wait_time_MAS '  )
plt.legend(loc='best')
plt.show()