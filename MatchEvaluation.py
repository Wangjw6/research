#coding:utf-8
import pandas as pd
import numpy as np

networkpath='network_sd-topo.csv'
matchresultpath='path.csv'

network_df = pd.DataFrame(pd.read_csv(networkpath)[['id','beginJD','beginWD','endJD','endWD','fromRoadID','toRoadID']])
paths_df = pd.DataFrame(pd.read_csv(matchresultpath))
pathnum = paths_df.shape[0]

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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

# evaluate according to topology
def topovalidate(path,network):
    path = path.split('-')
    path = f7(path)
    if len(path)==1:
        return 1
    i = 0
    while i<len(path)-1:

        if path[i+1] in network[int(path[i])].toroadid:
            pass
        else:
            return 0
        i+=1
    return 1
topoValidateResult=0
for j in range(paths_df.shape[0]):
    p = paths_df.iloc[i,1]
    topoValidateResult+=topovalidate(p,network)
print 'topology valid: %g'%(float(topoValidateResult)/pathnum)

# evaluate according to fartest route
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

def shortestvalidate(path,network):
    path = path.split('-')
    path = f7(path)
    if len(path)==1:
        return 1
    else:
        origin = path[0]
        dest = path[-1]
        l,fastestP = findShortestLen(int(origin), int(dest))
        path = [int(p) for p in path]
        if fastestP==path:
            return 1
        else:
            return 0
shortestValiResult=0
for k in range(pathnum):
    p = paths_df.iloc[i, 1]
    shortestValiResult += shortestvalidate(p, network)
print 'shortest valid: %g'%(float(shortestvalidate)/pathnum)

