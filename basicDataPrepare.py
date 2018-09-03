import pandas as pd
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
from skimage import data, io
import scipy
from mpl_toolkits.mplot3d import Axes3D
# aggregate demand
def OD(readpath, writepath):
    road=[1024,1020,1001,1023,1002,1022,1021,1003,1019,1006,1004,1007,1005,1018,1013,1008,
          1011,1009,1012,1010,1015,1016,1014,1017]
    outerroad = [1024,1020,1023,1022,1021,1018,1015,1017,1014,1016,1013,1019]
    node=[4694,4689,4662,4651,4632,4724,4854,4961,5029,5010,5021,4893,4927,4944,4808,4806,4835,
          4697,4706,4879,4822]
    dataset = pd.DataFrame(pd.read_csv(readpath))[['FROADID' ,'FTIME', 'TTIME', 'TROADID','TDIRARCID']]
    dataset['FTIME'] = pd.to_datetime(dataset['FTIME'])
    dataset['TTIME'] = pd.to_datetime(dataset['TTIME'])
    demandSet=[]
    od=[]
    beginTime = datetime.datetime.strptime('2017/12/4 0:00:00', '%Y/%m/%d %H:%M:%S')
    endTime = datetime.datetime.strptime('2017/12/4 23:55:00', '%Y/%m/%d %H:%M:%S')
    timetag = 96
    #
    # demandFlow=[]
    # while beginTime <= endTime:
    #     s = 0
    #     data_ = dataset[(dataset['FTIME'] >= beginTime) &
    #                  (dataset['FTIME'] <= (beginTime + datetime.timedelta(seconds=900)))]
    #     # for i in range(len(road)):
    #     #     demand=[]
    #     #     for j in range(len(road)):
    #     #         data = dataset[(dataset['FROADID'] == road[i]) & (dataset['TROADID'] == road[j])]
    #     #         data_ = data[(dataset['FTIME'] >= beginTime) &
    #     #                   (dataset['FTIME'] <= (beginTime+datetime.timedelta(seconds=900)))]
    #     #         demand.append(data_.shape[0])
    #     #     s+=sum(demand)
    #     #     demandSet.append(demand)
    #     demandFlow.append(data_.shape[0])
    #     # img = np.array(demandSet)
    #     demandSet=[]
    #     # scipy.misc.imsave('/home/administrator/pywork/test/img'+'/'+str(timetag)+'.jpg', img)
    #     #s = scipy.misc.imread('/home/administrator/pywork/test/img'+'/'+str(timetag)+'.jpg')
    #     print timetag
    #     timetag+=1
    #     beginTime = beginTime + datetime.timedelta(seconds=900)
    # plt.plot(demandFlow)
    # plt.show()

    dataset['origin'] = dataset.apply(lambda x:str(x['TDIRARCID'])[0:4],axis=1)
    dataset['dest'] = dataset.apply(lambda x: str(x['TDIRARCID'])[-4:], axis=1)

    for i in range(len(node) ):
        for j in range(len(node)):
            if road[i] != road[j]:
                sum = 0
                data = dataset[(dataset['origin']==str(node[i])) & (dataset['dest']==str(node[j]))]
                if data.empty:
                    continue
            # aggregate according to interval
                beginTime = datetime.datetime.strptime('2017/12/4 0:00:00', '%Y/%m/%d %H:%M:%S')
                endTime = datetime.datetime.strptime('2017/12/4 23:55:00', '%Y/%m/%d %H:%M:%S')

                od = str(node[i])+'_'+str(node[j])
                demand = []
                demand.append(od)
                while beginTime <= endTime:
                    data_ = data[(dataset['FTIME'] >= beginTime) &
                               (dataset['FTIME'] <= (beginTime+datetime.timedelta(seconds=900)))]
                    beginTime = beginTime+datetime.timedelta(seconds=900)
                    demand.append(data_.shape[0])
                    sum+=(data_.shape[0])
                   # print '*****'
                if sum>0:
                    print od + ' works'
                    demandSet.append(demand)
    df = pd.DataFrame(demandSet)
    df.to_csv(writepath2, index=False, sep=',', mode='a',header=False)



def statePreprocess1(readpath, writepath, begintime, endtime):
    tianArcDir = [[4893,4927],
[4927,4893],
[4927,4944],
[4944,4927],
[4893,4808],
[4808,4893],
[4806,4927],
[4927,4806],
[4835,4944],
[4944,4835],
[4808,4806],
[4806,4808],
[4806,4835],
[4835,4806],
[4808,4694],
[4694,4808],
[4697,4806],
[4806,4697],
[4706,4835],
[4835,4706],
[4694,4697],
[4697,4694],
[4697,4706],
[4706,4697],
[4694,4689],
[4689,4694],
[4651,4697],
[4697,4651],
[4724,4706],
[4706,4724],
[4835,4854],
[4854,4835],
[5021,4893],
[4893,5021],
[4927,5010],
[5010,4927],
[4879,4893],
[4893,4879],
[4680,4630],
[4630,4680],
[4680,4671],
[4671,4680],
[4634,4732],
[4732,4634],
[4651,4635],
[4635,4651],
[4689,4819],
[4819,4689],
[4732,4724],
[4724,4732],
[4635,4664],
[4664,4635],
[4664,4724],
[4724,4664],
[4562,4651],
[4651,4562],
[4634,4664],
[4664,4634],
[4554,4634],
[4634,4554],
[4536,4635],
[4635,4536],
[4630,4650],
[4650,4630],
[4605,4671],
[4671,4605],
[4671,4819],
[4819,4671],
[4660,4689],
[4689,4660],
[4671,4660],
[4660,4671],
[4854,4863],
[5082,5021],
[5021,5082],
[5010,5126],
[5126,5010],
[5123,5082],
[5082,5123],
[5122,5123],
[5123,5122],
[5126,5123],
[5123,5126],
[5100,5122],
[5122,5100],
[5012,5021],
[5021,5012],
[5190,5217],
[5217,5190],
[5012,5122],
[5122,5012],
[5122,5190],
[5190,5122],
[4732,4881],
[4881,4732],
[4863,4881],
[4630,4605],
[4605,4630],
[4605,4562],
[4562,4605],
[4562,4536],
[4536,4562],
[4536,4554],
[4554,4536],
[4879,5012],
[5012,4879],
[4819,4879],
[4879,4819],
[4662,4689],
[4689,4662],
[4651,4662],
[4662,4651]]
    road = len(tianArcDir)
    print '%s - %s'%(begintime, endtime)
    lossNum = 0
    total = 0
    remedyNum = 0
    csvfile = open(writepath, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['BEGINTIME','ROADID','FROMNODE','TONODE','SPEED','TRAVELTIME'])
    d=pd.DataFrame(pd.read_csv(readpath))
    dataset = pd.DataFrame(pd.read_csv(readpath))[['BEGINTIME', 'ARCID', 'SPEED', 'TRAVELTIME', 'FROMNODE', 'TONODE', 'VEHICLECOUNT','MIN_TRAVELTIME','LONGARCLEN']]
    dataset['BEGINTIME'] = pd.to_datetime(dataset['BEGINTIME'])
    dataset = dataset.drop_duplicates()
    # data = dataset[(dataset['FROMNODE'] == tianArcDir[22][0]) & (dataset['TONODE'] == tianArcDir[22][1])]

    # data['hour'] = data['BEGINTIME'].dt.hour
    # data = data[(data['hour']<5) | (data['hour']>23)]
    # n = np.array(data.iloc[:,6])
    # plt.plot(n)
    # plt.show()
    # hr = data.iloc[:,-1]
    freespeedSet = []
    sampleNeeded=[]

    for i in range((len(tianArcDir))):
        data = dataset[(dataset['FROMNODE'] == tianArcDir[i][0]) & (dataset['TONODE'] == tianArcDir[i][1])]
        speed_set = np.array(data.iloc[:,2])
        sigma = np.var(speed_set)
        sampleNeeded.append(max(int(1.96*1.96*sigma/9),5))
        speed_85 = data.nlargest(int(data.shape[0]*0.85), 'MIN_TRAVELTIME').iloc[-1]
        freespeedSet.append(float(data.iloc[0,-1])/float(speed_85.iloc[-2])*3.6)

    for i in range(len(tianArcDir)): # for each arc
        remedyArc = 0
        totalArc = 0
        lossArc = 0
        data = dataset[(dataset['FROMNODE'] == tianArcDir[i][0]) & (dataset['TONODE'] == tianArcDir[i][1])]
        validated_num = 0
        # recover
        if data.empty :
            print "%d - %d is null" % (tianArcDir[i][0], tianArcDir[i][1])
            continue
        arcid = data.iloc[0,1]
        fromnode = data.iloc[0,4]
        tonode = data.iloc[0,5]
        j = 0
        data = data.sort_values(by='BEGINTIME')
        beginTime = begintime
        endTime = endtime
        while beginTime<=endTime:

            hr = beginTime.hour

            if j<data.shape[0] and beginTime==data.iloc[j,0]:

                vehiclecount = int(data.iloc[j,6])
                data_match_vc = data[data['VEHICLECOUNT']==vehiclecount]
                speed_mean = np.array(data.iloc[:,2]).mean()
                # basic_speed = (float(data.iloc[0, -1]) / float(speed_median.iloc[-2]) * 3.6)

                # the more sample is, the more confidence on current estimation, the less need on basic_speed
                if vehiclecount>=sampleNeeded[i]:
                    speed = float(data.iloc[j,2])

                if vehiclecount<sampleNeeded[i] :
                    speed = speed_mean+np.random.normal(0,3)



                # if  (hr>=21) or (hr<=5):
                #     speed=38+np.random.normal(6,2)
                # if  (hr>=10 and hr<=13) or (hr>=15 and hr<=16):
                #     speed=35+np.random.normal(0,5)
                # if hr<6 or (hr>10 and hr<14) or (hr>21):
                #     speed = data_match_vc['SPEED'].max()*0.9+np.random.normal(0,2)
                # if speed<10:
                #     speed = 10+np.random.normal(0,2)

                writer.writerow([beginTime, arcid, fromnode, tonode, speed, data.iloc[j, -1]/speed*3.6])
                validated_num+=1
                j += 1

            else:
                lossArc+=1
                lossNum = lossNum + 1

                if ~data[data['VEHICLECOUNT']==1].empty:
                    speed = data[data['VEHICLECOUNT']==1]['SPEED'].max()+np.random.normal(0,2)

                writer.writerow([beginTime, arcid, fromnode, tonode,freespeedSet[i], 1])

            total = total + 1
            totalArc+=1
            beginTime = beginTime + datetime.timedelta(seconds=300)

        print "%s, %d - %d, total: %d, validate :%d, lossrate:%g" % (arcid,tianArcDir[i][0], tianArcDir[i][1], totalArc, validated_num, float(lossArc)/totalArc)

    csvfile.close()
    print 'loss rate is %g'%(float(lossNum)/float(total))
    print 'loss rate after recover is %g' % (float(lossNum - remedyNum) / float(total))


def statePreprocess2(readpath, writepath, begintime,endtime):
    tianArcDir = [[4893, 4927],
                  [4927, 4893],
                  [4927, 4944],
                  [4944, 4927],
                  [4893, 4808],
                  [4808, 4893],
                  [4806, 4927],
                  [4927, 4806],
                  [4835, 4944],
                  [4944, 4835],
                  [4808, 4806],
                  [4806, 4808],
                  [4806, 4835],
                  [4835, 4806],
                  [4808, 4694],
                  [4694, 4808],
                  [4697, 4806],
                  [4806, 4697],
                  [4706, 4835],
                  [4835, 4706],
                  [4694, 4697],
                  [4697, 4694],
                  [4697, 4706],
                  [4706, 4697],
                  [4694, 4689],
                  [4689, 4694],
                  [4651, 4697],
                  [4697, 4651],
                  [4724, 4706],
                  [4706, 4724],
                  [4835, 4854],
                  [4854, 4835],
                  [5021, 4893],
                  [4893, 5021],
                  [4927, 5010],
                  [5010, 4927],
                  [4879, 4893],
                  [4893, 4879],
                  [4680, 4630],
                  [4630, 4680],
                  [4680, 4671],
                  [4671, 4680],
                  [4634, 4732],
                  [4732, 4634],
                  [4651, 4635],
                  [4635, 4651],
                  [4689, 4819],
                  [4819, 4689],
                  [4732, 4724],
                  [4724, 4732],
                  [4635, 4664],
                  [4664, 4635],
                  [4664, 4724],
                  [4724, 4664],
                  [4562, 4651],
                  [4651, 4562],
                  [4634, 4664],
                  [4664, 4634],
                  [4554, 4634],
                  [4634, 4554],
                  [4536, 4635],
                  [4635, 4536],
                  [4630, 4650],
                  [4650, 4630],
                  [4605, 4671],
                  [4671, 4605],
                  [4671, 4819],
                  [4819, 4671],
                  [4660, 4689],
                  [4689, 4660],
                  [4671, 4660],
                  [4660, 4671],
                  [4854, 4863],
                  [5082, 5021],
                  [5021, 5082],
                  [5010, 5126],
                  [5126, 5010],
                  [5123, 5082],
                  [5082, 5123],
                  [5122, 5123],
                  [5123, 5122],
                  [5126, 5123],
                  [5123, 5126],
                  [5100, 5122],
                  [5122, 5100],
                  [5012, 5021],
                  [5021, 5012],
                  [5190, 5217],
                  [5217, 5190],
                  [5012, 5122],
                  [5122, 5012],
                  [5122, 5190],
                  [5190, 5122],
                  [4732, 4881],
                  [4881, 4732],
                  [4863, 4881],
                  [4630, 4605],
                  [4605, 4630],
                  [4605, 4562],
                  [4562, 4605],
                  [4562, 4536],
                  [4536, 4562],
                  [4536, 4554],
                  [4554, 4536],
                  [4879, 5012],
                  [5012, 4879],
                  [4819, 4879],
                  [4879, 4819],
                  [4662, 4689],
                  [4689, 4662],
                  [4651, 4662],
                  [4662, 4651]]
    road = len(tianArcDir)

    dataset = pd.DataFrame(pd.read_csv(readpath)[['BEGINTIME', 'FROMNODE', 'TONODE', 'SPEED']])

    speedSet=[['BEGINTIME','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112']]
    dataset['BEGINTIME'] = pd.to_datetime(dataset['BEGINTIME'])
    # dataset['judge'] = dataset['FROMNODE'].map(str)+' '+ dataset['TONODE'].map(str)+' '+dataset['BEGINTIME'].map(str)
    total = 0
    loss = 1
    beginTime = begintime
    endTime = endtime
    process_round=1
    while beginTime <= endTime: # for each timetag
        speed=[]
        speed.append(str(beginTime))
        flag = 0
        total+=1
        for i in range(len(tianArcDir)): # for each arc
            data = dataset[dataset['BEGINTIME'] == beginTime]
            data = data[data['TONODE'] == tianArcDir[i][1]]
            data = data[data['FROMNODE'] == tianArcDir[i][0]]
            # data = dataset[dataset['judge']==(str(tianArcDir[i][0])+' '+str(tianArcDir[i][1])+' '+str(begintime))]
            if data.empty:
                flag = 1
                break
            else:
                speed.append(str(data.iloc[0,3]))
        dataset=dataset[dataset['BEGINTIME'] > beginTime]
        beginTime = beginTime + datetime.timedelta(seconds=300)
        if flag==1:
            loss+=1
            continue
        else:
            print beginTime
            speedSet.append(speed)
        if len(speedSet)>12:
            df = pd.DataFrame(speedSet)
            df.to_csv(writepath, index=False, sep=',', mode='a', header=False)
            speedSet=[]
            print 'finishing %g'%(float(process_round*12.0)/((endtime-begintime).days*24*12))
            process_round+=1
    print 'loss rate is %g'%(float(loss)/float(total))


def statePreprocess3(readpath, writepath, writepath2,timestep, predictHorizon):
    dataset = pd.DataFrame(pd.read_csv(readpath,))
    tianArcDir = [[4893, 4927],
                  [4927, 4893],
                  [4927, 4944],
                  [4944, 4927],
                  [4893, 4808],
                  [4808, 4893],
                  [4806, 4927],
                  [4927, 4806],
                  [4835, 4944],
                  [4944, 4835],
                  [4808, 4806],
                  [4806, 4808],
                  [4806, 4835],
                  [4835, 4806],
                  [4808, 4694],
                  [4694, 4808],
                  [4697, 4806],
                  [4806, 4697],
                  [4706, 4835],
                  [4835, 4706],
                  [4694, 4697],
                  [4697, 4694],
                  [4697, 4706],
                  [4706, 4697],
                  [4694, 4689],
                  [4689, 4694],
                  [4651, 4697],
                  [4697, 4651],
                  [4724, 4706],
                  [4706, 4724],
                  [4835, 4854],
                  [4854, 4835],
                  [5021, 4893],
                  [4893, 5021],
                  [4927, 5010],
                  [5010, 4927],
                  [4879, 4893],
                  [4893, 4879],
                  [4680, 4630],
                  [4630, 4680],
                  [4680, 4671],
                  [4671, 4680],
                  [4634, 4732],
                  [4732, 4634],
                  [4651, 4635],
                  [4635, 4651],
                  [4689, 4819],
                  [4819, 4689],
                  [4732, 4724],
                  [4724, 4732],
                  [4635, 4664],
                  [4664, 4635],
                  [4664, 4724],
                  [4724, 4664],
                  [4562, 4651],
                  [4651, 4562],
                  [4634, 4664],
                  [4664, 4634],
                  [4554, 4634],
                  [4634, 4554],
                  [4536, 4635],
                  [4635, 4536],
                  [4630, 4650],
                  [4650, 4630],
                  [4605, 4671],
                  [4671, 4605],
                  [4671, 4819],
                  [4819, 4671],
                  [4660, 4689],
                  [4689, 4660],
                  [4671, 4660],
                  [4660, 4671],
                  [4854, 4863],
                  [5082, 5021],
                  [5021, 5082],
                  [5010, 5126],
                  [5126, 5010],
                  [5123, 5082],
                  [5082, 5123],
                  [5122, 5123],
                  [5123, 5122],
                  [5126, 5123],
                  [5123, 5126],
                  [5100, 5122],
                  [5122, 5100],
                  [5012, 5021],
                  [5021, 5012],
                  [5190, 5217],
                  [5217, 5190],
                  [5012, 5122],
                  [5122, 5012],
                  [5122, 5190],
                  [5190, 5122],
                  [4732, 4881],
                  [4881, 4732],
                  [4863, 4881],
                  [4630, 4605],
                  [4605, 4630],
                  [4605, 4562],
                  [4562, 4605],
                  [4562, 4536],
                  [4536, 4562],
                  [4536, 4554],
                  [4554, 4536],
                  [4879, 5012],
                  [5012, 4879],
                  [4819, 4879],
                  [4879, 4819],
                  [4662, 4689],
                  [4689, 4662],
                  [4651, 4662],
                  [4662, 4651]]
    road = len(tianArcDir)
    sampleSet = []
    sampleSet_for_test = []
    test=[]
    sampleTravelTimeSet=[]
    dates=[]

    dataset['BEGINTIME'] = pd.to_datetime(dataset['BEGINTIME'])
    dataset.sort_values(by='BEGINTIME')
    # df2 = dataset.unstack('BEGINTIME').resample('1W',how='mean')
    for i in range(dataset.shape[0]):
        sample = []
        sampleTT=[]
        step = 0
        beginTime = dataset.iloc[i,0]


        month =int( beginTime.month)
        date = int(beginTime.day)
        # add to find history
        day = int(beginTime.weekday())
        hour = int(beginTime.hour)
        min = int(beginTime.minute)

        tag = i
        timeset = []
        while beginTime == dataset.iloc[tag,0]:
            timeset.append(beginTime)
            # produce state for intersection cell and the null cell in favor of matrix construction
            dir1 = dataset.iloc[tag,1:]
            simplearray1_onlyRoad = np.zeros(road)
            outlier_flag = 0
            for index in range(road):
               simplearray1_onlyRoad[index] = float(dir1[index])
            sample = sample + [day, hour, min]
            sample = sample + [x for x in simplearray1_onlyRoad]

            step = step + 1
            beginTime = beginTime + datetime.timedelta(seconds=300)
            tag += 1
            if outlier_flag==1:
                break
            if tag >= dataset.shape[0]-predictHorizon: # in case of excess of index
                break
            # get prediction target once historical data done

            if step == timestep:
                predictSet = []
                predictTag = 0
                beginTimeTemp = beginTime
                for j in range(predictHorizon):
                    if beginTimeTemp == dataset.iloc[i+j+step,0]:
                        if i+j+step-1>0:
                            predictSet += pd.Series.tolist(dataset.iloc[i+j+step,1:]*0.5+dataset.iloc[i+j+step-1,1:]*0.5)
                        else:
                            predictSet += pd.Series.tolist(dataset.iloc[i + j + step, 1:])
                        predictTag = predictTag + 1
                        beginTimeTemp = beginTimeTemp + datetime.timedelta(seconds=300)
                if predictTag == predictHorizon:
                    sample = sample + predictSet
                    if -1 in sample:
                        sample=[]
                    else:
                        if i>(dataset.shape[0])-12*24*7:
                            sampleSet_for_test.append(sample)
                        else:
                            sampleSet.append(sample)
                        dates.append(str(dataset.iloc[i,0]))
                        print 'sample counts: %d time: %s'%(len(sampleSet), str(dataset.iloc[i,0]))
                break


    df = pd.DataFrame(sampleSet)
    df2 = pd.DataFrame(sampleSet_for_test)
    df.to_csv(writepath, index=False, sep=',', mode='a',header=False)
    df2.to_csv(writepath2, index=False, sep=',', mode='a', header=False)


def sampleGenerate(readpath, writepath, writepath2,timestep, predictHorizon): # requirement: data is continuoud along the temporal dimension
    begin=datetime.datetime.now()
    dataset = pd.DataFrame(pd.read_csv(readpath,header=None))
    sampleSet=[]
    sampleSet_for_test=[]
    for i in range(dataset.shape[0]):
        sample=[]
        inputs=[]
        predicts=[]
        if timestep+i+predictHorizon>dataset.shape[0]:
            break
        for inputlen in range(timestep):
            if len(inputs)==0:
                inputs=np.array(dataset.iloc[inputlen+i,:]).tolist()
            else:
                inputs+=np.array(dataset.iloc[inputlen+i,:]).tolist()
        for predictlen in range(predictHorizon):
            if len(predicts)==0:
                predicts=np.array(dataset.iloc[timestep+i+predictlen,:]).tolist()
            else:
                predicts+=np.array(dataset.iloc[timestep+i+predictlen,:]).tolist()

        sample=inputs+predicts
        if i > (dataset.shape[0]) - 12 * 24 * 7:
            sampleSet_for_test.append(sample)
        else:
            sampleSet.append(sample)
    end = datetime.datetime.now()
    print 'timecost: %s'%(end-begin) # about 4 min
    df = pd.DataFrame(sampleSet)
    df2 = pd.DataFrame(sampleSet_for_test)
    df.to_csv(writepath, index=False, sep=',', mode='a',header=False)
    df2.to_csv(writepath2, index=False, sep=',', mode='a', header=False)


readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speed_data20180122-0413_visual.csv'
writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_road20180122-0413_3.csv'
writepath2 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_3.csv'
sampleGenerate(readpath1, writepath1,writepath2, timestep=9, predictHorizon=1)

#
# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speed_data20180122-0413.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/raw_20180122-0413LL.csv'
# statePreprocess1(readpath1, writepath1,datetime.datetime.strptime('2018/01/22 00:00:00', '%Y/%m/%d %H:%M:%S'),
#                  datetime.datetime.strptime('2018/04/13 23:55:00', '%Y/%m/%d %H:%M:%S'))
# #
# # #
# # # print 'process1-1 done'
# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/raw_20180122-0413LL.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speedMat_20180122-0413_2.csv'
# statePreprocess2(readpath1, writepath1,datetime.datetime.strptime('2018/01/22 00:00:00', '%Y/%m/%d %H:%M:%S'),
#                  datetime.datetime.strptime('2018/04/13 23:55:00', '%Y/%m/%d %H:%M:%S'))
#
# print 'process2 done'



# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speedMat_20180122-0413_2.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_road20180122-0413_3.csv'
# writepath2 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_45min_5min_roadtest_3.csv'
# statePreprocess3(readpath1, writepath1,writepath2, timestep=9, predictHorizon=1)
# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speedMat_20171001-1231L.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_road20171001-1231.csv'
# statePreprocess3(readpath1, writepath1, timestep=6, predictHorizon=1)
# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speedMat_20180122-0320L.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_roadBig18L.csv'
# statePreprocess3(readpath1, writepath1, timestep=6, predictHorizon=1)
# readpath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/speedMat_20180321-0411L.csv'
# writepath1 = '/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_testL.csv'
# statePreprocess3(readpath1, writepath1, timestep=6, predictHorizon=1)
# print 'process3 done'
#
# data2017 = pd.DataFrame(pd.read_csv('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_roadBig17L.csv',header=None))
# data2018 = pd.DataFrame(pd.read_csv('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_roadBig18L.csv',header=None))
# data = pd.concat((data2017,data2018),axis=0)
# data = pd.DataFrame(data)
# data.to_csv('/home/administrator/pywork/DeepLearningXC/dataProcess/dataPool/sample_30min_5min_road1718L.csv', sep=",",index=False, header=None)
#
# print data2017.shape[0]
# print data2018.shape[0]
# print data.shape[0]
print('OK!')
