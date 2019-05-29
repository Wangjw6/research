#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import numpy as np
import utm
import math
import csv
import datetime
from shapely.geometry import LineString
from shapely.geometry import Point

# 基础输入
maxBias = 0.1
dataFCD = pd.DataFrame(pd.read_csv('QD_FCD_to_match.csv',encoding='gbk'))
print dataFCD.shape[0]
carplates=pd.DataFrame(dataFCD.values[:,0])
carplates=carplates.drop_duplicates()
network_df = pd.DataFrame(pd.read_csv('network_sd-topo.csv')[['id','beginJD','beginWD','endJD','endWD','fromRoadID','toRoadID']])


# 数据结构
class road:
    def __init__(self, fromroadid, toroadid, beginJD, beginWD, endJD, endWD):
        self.fromroadid = fromroadid
        self.toroadid = toroadid
        self.beginJD = beginJD
        self.beginWD = beginWD
        self.endJD = endJD
        self.endWD = endWD

network_df.fillna('*',inplace=True)
network = {}
for i in range(network_df.shape[0]):
    fromroadid = network_df.iloc[i, 5].split('|')
    toroadid = network_df.iloc[i, 6].split('|')
    beginJD = network_df.iloc[i, 1]
    beginWD = network_df.iloc[i, 2]
    endJD = network_df.iloc[i, 3]
    endWD = network_df.iloc[i, 4]
    road_ = road(fromroadid, toroadid, beginJD, beginWD, endJD, endWD)
    network[int(network_df.iloc[i, 0])] = road_


# 空间距离
def point_to_line_dist(dx, dy, lx1, ly1, lx2, ly2):
    # s1 = utm.from_latlon(dy, dx)
    # s2 = utm.from_latlon(ly1, lx1)
    # s3 = utm.from_latlon(ly2, lx2)
    # l = math.sqrt((s2[0]-s3[0])**2+(s2[1]-s3[1])**2)
    # dy = s1[0]
    # dx = s1[1]
    # ly1 = s2[0]
    # lx1 = s2[1]
    # ly2 = s3[0]
    # lx2 = s3[1]
    lcr1 = roadLen(dx, dy, lx1, ly1)
    lcr2 = roadLen(dx, dy, lx2, ly2)
    lr1lr2 = roadLen(lx1, ly1, lx2, ly2)
    costheta = (lcr1**2+lr1lr2**2-lcr2**2)/(2*lcr2)
    distance = lcr1*costheta #km
    return abs(distance)


def roadLen(lx1, ly1, lx2, ly2):
    l = 6370*np.arccos(np.cos(ly1*np.pi/180) * np.cos(ly2*np.pi/180) * np.cos((lx1-lx2)*np.pi/180) + np.sin(ly1*np.pi/180) * np.sin(ly2*np.pi/180))
    return l


def findShortestLen(p1, p2):

    current_to_roadID = network[p1].toroadid+[str(p1)] # the next road is probably the current road or its downstream
    roadset = []
    p2toidlist = network[int(p2)].toroadid
    if str(p1) in p2toidlist:
        return 9999,[]
    # 如果上下游直接拓扑相连，则下游就是最短路段
    if p1==p2:
        return 0,[p2]
    if str(p2) in current_to_roadID:
        return roadLen(network[p1].beginJD, network[p1].beginWD, network[p1].endJD,network[p1].endWD), [p2]
    if (network[int(p1)].endJD == network[int(p2)].endJD and network[int(p1)].endWD == network[p2].endWD) \
            or (network[int(p1)].beginJD == network[int(p2)].beginJD and network[int(p1)].beginWD == network[p2].beginWD):
        return 9999,[]
    if network[int(p1)].endJD == network[int(p2)].beginJD and network[int(p1)].endWD == network[p2].beginWD\
            and network[int(p1)].beginJD == network[int(p2)].endJD and network[int(p1)].beginWD == network[p2].endWD:
        return 9999,[]

    # 否则，上下游之间还存在其他路段，需要找出连接上下游的最短路径
    else:
        lens = [roadLen(network[p1].beginJD, network[p1].beginWD, network[p1].endJD,network[p1].endWD)]
        paths = [[p1]]
        minlen = 9999
        while len(paths) > 0:
            temp_paths = []
            temp_len = []
            for i in range(len(paths)):
                toidlist = network[int(paths[i][-1])].toroadid
                for tid in toidlist:
                    if  tid in paths[i][:]: # avoid loop trip
                        continue
                    if tid=='' or tid=='*':
                        break
                    tid = int(tid)
                    if network[int(paths[i][-1])].beginJD==network[tid].endJD and network[int(paths[i][-1])].beginWD==network[tid].endWD:
                        continue

                    # should be closer to dest and away from origin after each step
                    dist_from_origin_current = roadLen(network[int(paths[i][-1])].endJD, network[int(paths[i][-1])].endWD, network[p1].beginJD,
                                                       network[p1].beginWD)
                    dist_from_origin_moved = roadLen(network[tid].endJD, network[tid].endWD, network[p1].beginJD,
                                                       network[p1].beginWD)
                    dist_from_dest_current = roadLen(network[int(paths[i][-1])].endJD, network[int(paths[i][-1])].endWD, network[p2].beginJD,
                                                       network[p2].beginWD)
                    dist_from_dest_moved = roadLen(network[tid].endJD, network[tid].endWD, network[p2].beginJD,
                                                     network[p2].beginWD)
                    if (dist_from_origin_current <= dist_from_origin_moved and dist_from_dest_moved <= dist_from_dest_current):
                        pass
                    else:
                        if tid!=p2:
                            continue
                    if tid > 0 and lens[i] + roadLen(network[tid].beginJD, network[tid].beginWD, network[tid].endJD,network[tid].endWD)<= minlen:
                        s = paths[i][:]
                        s += [tid]
                        temp_paths.append(s)
                        temp_len.append(
                            lens[i] + roadLen(network[tid].beginJD, network[tid].beginWD, network[tid].endJD,network[tid].endWD))
                        if tid == p2:

                            minlen = temp_len[-1]
                            roadset = temp_paths[-1][:]

            paths = temp_paths[:]
            lens = temp_len[:]
    if len(roadset) == 0:
        minlen=99999999
    return minlen, roadset


# 候选路段 目前只有邻域筛选，加上：拓扑、驾驶习惯？
def findCandidate(c, maxBias, network):
    candidadateRoadSet = []
    for key,road in network.items():
        c_m = utm.from_latlon(c[1], c[0])
        r1_m = utm.from_latlon(road.beginWD,road.beginJD)
        r2_m = utm.from_latlon(road.endWD, road.endJD)

        # p = Point(c[0], c[1])
        # circle = p.buffer(maxBias).boundary
        # l = LineString([(road.beginJD, road.beginWD), (road.endJD, road.endWD)])
        # i = circle.intersection(l)
        p = Point(c_m[0], c_m[1])
        circle = p.buffer(100).boundary
        l = LineString([(r1_m[0], r1_m[1]), (r2_m[0], r2_m[1])])
        i = circle.intersection(l)

        l1 = roadLen(c[0], c[1], road.beginJD, road.beginWD)
        l2 = roadLen(c[0], c[1], road.endJD, road.endWD) #km
        if (i.is_empty==False)  or l1 <=maxBias or l2 <= maxBias:
            candidadateRoadSet.append(key)
    return candidadateRoadSet


# 第一个定位点的匹配路段概率
def firstPP(c, road, maxBias):
    road =  network[road]
    distance = point_to_line_dist(c[0], c[1], road.beginJD, road.beginWD, road.endJD, road.endWD)
    prob = np.exp(-(float(distance) / maxBias))
    return prob


def normlize(s):
    sum_ = 0.0
    temp = []
    for i in s:
        sum_ += i
    for i in s:
        temp.append(float(i) / sum_)
    return temp


# 非首个定位点的匹配概率
def followPP(candidateroad, c, maxBias, t, v, lastroadid):                 #c: longititude and latitude of car; candidataroad: road evaluating to match; lastroadid: last road of exising path
    distance = point_to_line_dist(c[0], c[1], network[candidateroad].beginJD, network[candidateroad].beginWD, network[candidateroad].endJD,network[candidateroad].endWD)
    prob1 = np.exp(-(float(distance) / maxBias))

    l, rset = findShortestLen(int(lastroadid), int(candidateroad))
    if v==0:
        if candidateroad==lastroadid:
            prob2 = 1
        else:
            prob2 = 0
    else:
        prob2 = np.exp(-l / (v * (t+0.00036)))
    if prob1*prob2==0:
        a=1
    return prob1 * prob2, rset


# 逐车匹配
def route(rawrecord,writer):
    pathresult = []
    probresult = []
    paths = []
    probsOfPath = []
    scanPoints=[]
    scanPointIndicator=[]
    flag = 0 # indicate whether there is a begin point
    picknum=0
    i=0
    while i < rawrecord.shape[0]:
    # for i in range(rawrecord.shape[0]):
        cp = findCandidate([rawrecord.iloc[i, 2], rawrecord.iloc[i, 3]], maxBias, network)
        if len(cp)==0:
            picknum+=1
            i+=1
            continue
        scanPoints.append(rawrecord.iloc[i, :])
        if flag == 0:
            for j in range(len(cp)):
                # 对于每个候补路段，找出连接它的概率最大的路径，第一条路段就是自身，此时是一条路径的诞生
                paths.append([cp[j]])
                probsOfPath.append(firstPP([rawrecord.iloc[i, 2], rawrecord.iloc[i, 3]], cp[j], maxBias))
                scanPointIndicator.append([1])
                flag = 1
            i+=1

        else:
            tempPaths = []
            tempProbs = []
            tempScanInd = []
            if scanPoints[-1].iloc[4] != scanPoints[-2].iloc[4]:  # 根据车辆载客状态划分匹配起点吗？
                flag = 0
                index = probsOfPath.index(max(probsOfPath))
                pathresult.append(paths[index])
                probresult.append(probsOfPath[index])
                indicator = scanPointIndicator[index]
                # 判断路径是否是连接两端的最短路径，作为匹配正确的一个检验
                # l, shortespath = findShortestLen(paths[index][0], paths[index][-1])
                r = 0
                scanpointindex=0
                while r < len(paths[index]):
                    if indicator[r]==1:
                        writer.writerow([scanPoints[scanpointindex].iloc[0],scanPoints[scanpointindex].iloc[1],scanPoints[scanpointindex].iloc[2],scanPoints[scanpointindex].iloc[3],
                                    scanPoints[scanpointindex].iloc[4],scanPoints[scanpointindex].iloc[5],scanPoints[scanpointindex].iloc[6],
                                         scanPoints[scanpointindex].iloc[7],paths[index][r],max(probsOfPath)])
                        scanpointindex+=1
                        picknum+=1
                    r+=1
                scanPoints = []
                paths = []
                probsOfPath = []
                scanPointIndicator=[]
                continue

            for k in range(len(cp)):
                # 对于每个候补路段，找出连接它的概率最大的路径
                maxprob = 0.05
                index = -1
                temprset=[]
                for j in range(len(paths)):
                    p,rset = followPP(cp[k], [rawrecord.iloc[i, 2], rawrecord.iloc[i, 3]], maxBias, (scanPoints[-1].iloc[1]-scanPoints[-2].iloc[1]).total_seconds()/3600,
                                 (scanPoints[-1].iloc[6] + scanPoints[-2].iloc[6]) / 2, int(paths[j][-1]))
                    if p * probsOfPath[j] > maxprob:
                        maxprob = p * probsOfPath[j]
                        index = j
                        temprset = rset[:]
                if maxprob>0.05:
                    s = paths[index][:]
                    s += temprset
                    ind = scanPointIndicator[index][:]
                    for ti in range(len(temprset)):
                        if temprset[ti]!=cp[k]:
                            ind.append(0)
                        else:
                            ind.append(1)
                    tempPaths.append(s)
                    tempProbs.append(maxprob)
                    tempScanInd.append(ind)


            if len(tempPaths)==0:
                flag = 0
                index = probsOfPath.index(max(probsOfPath))
                pathresult.append(paths[index])
                probresult.append(probsOfPath[index])
                indicator = scanPointIndicator[index]
                # 判断路径是否是连接两端的最短路径，作为匹配正确的一个检验
                # l, shortespath = findShortestLen(paths[index][0], paths[index][-1])
                r = 0
                # countrealnum=0
                # for indicatorindex in range(len(indicator)):
                #     if indicator[indicatorindex]==1:
                #         countrealnum+=1
                # # print countrealnum,
                # # print len(paths[index]),
                # # print len(scanPoints)
                scanpointindex=0
                while r < len(paths[index]):
                    if indicator[r]==1:
                        writer.writerow([scanPoints[scanpointindex].iloc[0],scanPoints[scanpointindex].iloc[1],scanPoints[scanpointindex].iloc[2],scanPoints[scanpointindex].iloc[3],
                                    scanPoints[scanpointindex].iloc[4],scanPoints[scanpointindex].iloc[5],scanPoints[scanpointindex].iloc[6],
                                         scanPoints[scanpointindex].iloc[7],paths[index][r],max(probsOfPath)])
                        scanpointindex+=1
                        picknum += 1
                    r+=1
                scanPoints = []
                paths = []
                probsOfPath = []
                scanPointIndicator=[]
                continue

            i+=1
            scanPointIndicator = tempScanInd[:]
            paths = tempPaths[:]
            probsOfPath = tempProbs[:]
            probsOfPath = normlize(probsOfPath)  # 假设a车的最大概率路径对应的概率为0.01，b车最大概率路径对应的概率为0.1，如何区分这两条路径的置信度

    if len(scanPoints)>0:
        index = probsOfPath.index(max(probsOfPath))
        pathresult.append(paths[index])
        probresult.append(probsOfPath[index])
        indicator = scanPointIndicator[index]
        # 判断路径是否是连接两端的最短路径，作为匹配正确的一个检验
        # l, shortespath = findShortestLen(paths[index][0], paths[index][-1])
        r = 0
        scanpointindex=0
        while r < len(paths[index]):
            if indicator[r] == 1:
                writer.writerow([scanPoints[scanpointindex].iloc[0], scanPoints[scanpointindex].iloc[1],
                                 scanPoints[scanpointindex].iloc[2], scanPoints[scanpointindex].iloc[3],
                                 scanPoints[scanpointindex].iloc[4], scanPoints[scanpointindex].iloc[5],
                                 scanPoints[scanpointindex].iloc[6],
                                 scanPoints[scanpointindex].iloc[7], paths[index][r], max(probsOfPath)])
                scanpointindex += 1
                picknum += 1
            r += 1
        scanPoints = []
        paths = []
        probsOfPath = []
        scanPointIndicator = []

    # 判断路径是否是连接两端的最短路径，作为匹配正确的一个检验
    #     l, shortespath = findShortestLen(paths[index][0], paths[index][-1])

    return pathresult, probresult

if __name__ == '__main__':
    # 逐车匹配
    pathset = []
    probset = []
    csvfile = open('FCD_matched0.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['CARPLATE', 'RTIME', 'JDZB', 'WDZB', 'ZKZT', 'YYZT','GPPSD','FX','roadID','prob'])
    begin = datetime.datetime.now()
    rr=0
    while rr <(carplates.shape[0]):

        begineach = datetime.datetime.now()

        carplate=carplates.iloc[rr].tolist()[0]
        data_=dataFCD[dataFCD['CARPLATE']==carplate]
        l = data_.shape[0]
        if l>1000:
            rr+=1
            continue
        print '%s %d out of %d records:%g' % (carplates.iloc[rr, 0], rr + 1, carplates.shape[0], l)

        data_['RTIME'] = pd.to_datetime(data_['RTIME'])
        data_=data_.sort_values(by='RTIME')

        pathresult,probresult = route(data_,writer)

        pathset+=pathresult
        probset+=probresult
        pathset_df = pd.DataFrame()
        pathset_df['path']=pathset
        pathset_df['prob']=probset
        print pathset_df.shape[0]
        pathset_df.to_csv('pathset0.csv')
        endeach = datetime.datetime.now()
        print 'this carplate cost:%g s'%(endeach-begineach).total_seconds()
        rr+=1
    end = datetime.datetime.now()
    print 'cost in all:%g s'%(float((end-begin).total_seconds())/3600)


