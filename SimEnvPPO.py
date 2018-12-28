import numpy as np
import tensorflow as tf
import cx_Oracle as orcl
import socket
import random
global num
num = 0
episode_length = 4
node_dim = 10
detector_num = 9
class Env():
    def __init__(self):
        self.username = 'sa'
        self.passwd = 'orclitssaf'
        self.host = '192.168.8.13'
        self.port = '1521'
        self.sid = 'ORCL'
        self.dsn = orcl.makedsn(self.host, self.port, self.sid)
        self.con = orcl.connect(self.username, self.passwd, self.dsn)
        print ("Oracle Connected!")

        self.HOST = '192.168.8.20'
        self.PORT = 8080
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print ("Socket Established!")
        self.sock.connect((self.HOST, self.PORT))

    def step(self, action, tag, t, num, datetag):
        # put in oracle
        reward = 0
        # sql_his = 'select * from RL_PriorOD t where t.TIMETAG=' + '\'' + str(
        #     tag) + '\''
        sql_his = 'select * from RL_RealOD t where t.TIMETAG='+'\''+str(tag)+'\'' + ' and t.datetag='+'\''+str(datetag)+'\''
        cursor_his = self.con.cursor()
        cursor_his.execute(sql_his)
        row_his = cursor_his.fetchone()
        od_adj = []
        od_num = 6 # num of od pair
        # prepare od set
        record_turn = []
        record = 1
        temp_turn = 0
        minus=0
        avg=[4, 1310, 18, 1838, 27, 447]
        for od in range(od_num):
            #temp = int(float(avg[od]) * (1+float(action[od])))
            temp = int(float(row_his[3+od]))#*float(action[od])) # +int(action[od])
            # temp = int(float(row_his[4 + od])+float(row_his[4 + od])*float(action[od])) # +int(action[od])
            #temp = int(float(row_his[4 + od]) + float(row_his[4 + od]) * float(action[0]))  # +int(action[od])
            # temp = int(1000.*float(action[od]))
            if temp<=0:
                temp=1
            od_adj.append(temp)

        sql1 = 'INSERT into RL_NB(id, timetag, OD_117_899, OD_117_180, OD_117_808, OD_117_128, OD_35_808, OD_35_128, datetag)VALUES(' \
               + str(num) + ',' + str(tag) + ','

        # if tag==48:
        #     od_adj=[7,1,16,1073,35,1]
        # if tag==49:
        #     od_adj=[1,1367,14,2067,1,1]
        # if tag==50:
        #     od_adj=[9,1666,8,824,30, 171]
        # if tag==51:
        #     od_adj=[4,611,22,1754,42,273]

        for od in range(od_num):
            if od > 0:
                sql1 += ','
            sql1 += str(od_adj[od])
        sql1+=','
        sql1+=str(datetag)
        sql1 += ')'
        print sql1
        cursor = self.con.cursor()
        cursor.execute(sql1)
        cursor.execute('commit')
        cursor.close()
        # send trigger to simulator

        # first run inform the simulator of time to begin
        send2sim = str(tag) + '\n'
        self.sock.sendall(send2sim.decode())

        print ('wait for next state and reward...')
        # wait for trigger from simulator read from oracle
        data = self.sock.recv(1024)
        if data == "next\r\n":
            sql2 = 'select * from RL_TRANSTATE_MORE t where TIMETAG ='+'\''+str(tag)+'\' '+' and datetag='+'\''+str(datetag)+'\' order by id DESC,STATION_ID ' # read newest result
            # result column index start from 0
            cursor = self.con.cursor()
            cursor.execute(sql2)

            result = cursor.fetchall()

            next_state = []
            sql_his2 = 'select * from RL_priorOD t where t.TIMETAG=' + '\'' + str(tag + 1) + '\''#+' and datetag='+'\''+str(datetag)+'\''
            cursor_his2 = self.con.cursor()
            cursor_his2.execute(sql_his2)
            row_his2 = cursor_his2.fetchone()
            hisod=[]
            flow=[]
            speed=[]

            for i in range(6):
                hisod.append(float(row_his2[i+4])/3000)

            k=0
            reward_set=[]
            for row in result:
                if int(row[6]) == 1:
                    flow.append(float(row[2]) / 1000)
                    speed.append(float(row[4]) / 60)
                    reward_set.append(float(row[3]))
                    k += 1
                if int(row[6]) == 0:
                    flow.append(float(row[2]) / 1000)
                    speed.append(float(row[4]) / 60)
                    reward_set.append(float(row[3]))
                    k += 1
                if k >= detector_num:
                    break # break after two newest record

            #next_state.append(float(tag/5))
            cursor.close()
            cursor_his2.close()
            dicfs = {1: 1, 2: 12, 3: 13, 4: 34, 5: 36, 6: 56, 7: 67, 8: 78, 9: 79}
            dicod = {1: 2, 2: 4, 3: 8, 4: 9, 5: 58, 6: 59}
            flow_ = [0 for i in range(node_dim * node_dim)]
            speed_ = [0 for i in range(node_dim * node_dim)]
            od_ = [0 for i in range(node_dim * node_dim)]
            for i in range(1,10):
                flow_[dicfs[i]] = flow[i-1]
                speed_[dicfs[i]] = speed[i-1]
            for i in range(1,6):
                od_[dicod[i]] = hisod[i-1]
            next_state.append(flow_)
            next_state.append(speed_)
            next_state.append(od_)

            reward = float(sum(reward_set))#+minus
            return next_state, reward

    def pre_run(self, tag,datetag):

        #clear database before each episode
        cursorclear = self.con.cursor()
        sqlclear1 = 'truncate table RL_Transtate_more'
        sqlclear2 = 'truncate table RL_NB'
        sqlclear3 = 'truncate table RL_REAL_TRANSTATE_MORE'
        cursorclear.execute(sqlclear1)
        cursorclear.execute(sqlclear2)
        cursorclear.execute('commit')

        cursor = self.con.cursor()
        sql = 'select * from RL_REAL_TRANSTATE_MORE t ' +  ' order by STATION_ID '  # read newest result

        cursor.execute(sql)
        result = cursor.fetchall()
        state = []
        flow=[]
        speed=[]
        hisod=[]

        sql_his = 'select * from RL_priorOD t' +' where timetag= '+str(tag)
        cursor_his = self.con.cursor()
        cursor_his.execute(sql_his)
        row_his = cursor_his.fetchone()
        j = 0
        for i in range(6):
            hisod.append(float(row_his[j + 4])/3000) # 4 form which od appears
            j += 1

        k = 0
        for row in result:
            if int(row[6]) == 1:
                flow.append(float(row[2]) / 1000)
                speed.append(float(row[4]) / 60)
                k += 1
            if int(row[6]) == 0:
                flow.append(float(row[2]) / 1000)
                speed.append(float(row[4]) / 60)
                k += 1
            if k >= detector_num:
                break  # break after two newest record

        cursor.close()
        cursor_his.close()
        dicfs = {1:1,2:12,3:13,4:34,5:36,6:56,7:67,8:78,9:79}
        dicod = {1:2,2:4,3:8,4:9,5:58,6:59}
        flow_=[0 for i in range(node_dim * node_dim)]
        speed_=[0 for i in range(node_dim * node_dim)]
        od_=[0 for i in range(node_dim * node_dim)]
        for i in range(1,10):
            flow_[dicfs[i]] = flow[i-1]
            speed_[dicfs[i]] = speed[i-1]
        for i in range(1,6):
            od_[dicod[i]] = hisod[i-1]

        state.append(flow_)
        state.append(speed_)
        state.append(od_)
        cursorclear.execute(sqlclear3)
        cursorclear.execute('commit')
        return state

    def end(self):
        end = "end"
        self.sock.sendall(end.decode())
        self.sock.close()

    def start(self, tag, date):
        send2sim = str(tag) + '_'+str(date)+ '\n'
        self.sock.sendall(send2sim.decode())
        data = self.sock.recv(1024)
        print '<<<warm up done>>>'
