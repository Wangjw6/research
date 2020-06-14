import numpy as np
from BUS import bus
from BUS_STOP import bus_stop
import datetime


class Env():
    '''
    state_dim :        dimension of state space
    action_dim:        dimension of action space
    bus_num:           number of bus
    bus_stop_num:      number of bus stop
    r:                 radius of bus corridor
    emit_time_list:    schedule of bus departure
    bus_dep_list:      bus departure location list
    bus_stop_loc_list: bus stop location list
    arrival_schedule:  schedule arrival time of next stop for a bus. Rolling update
    update_step:       simulation update step, update_step=0.01 indiactes simulation run every 0.01 second
    sim_horizon:       length of simulation /seconds
                       Noted: 1 second sim_horizon = 1/update_step seconds in reality
    cooridor_radius:   the radius of the bus corridor
    check_step :       check system state every 3 second(5 min in real world),includng service level of bus
    arrival_bias:      bias between actural arrival and arrival schedule for each bus on every stip
    now:               record interrupt time once RL brain prepare control
    flag:              1 indicating reward and next state have been prepared
    state:             state observation: how long since last bus left each stop
    action:            control for each stop
    reward:            minimize mean of headway and variance of headway
    max_holding:       max holding time
    control_id:        designated control id
    is_Train:          put bus at stop randomly for training
    cost:              record action for slack time imputation
    bus_loc_stop:      record in which stop the bus is located, -1 for no stop
    stop_visit         record how many stops all buses have visited

    '''

    def __init__(self, state_dim, action_dim, bus_num, bus_stop_num, r, emit_time_list,
                 bus_dep_list, bus_stop_loc_list, update_step= 1, sim_horizon=36,control_id=[1,3,5],train_mode=0):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = []
        self.action = -1
        self.reward = 0

        self.now = 0
        self.bus_stop_num = bus_stop_num
        self.bus_num = bus_num
        self.bus_list = []
        self.bus_stop_list = []
        self.bus_dep_list = bus_dep_list
        self.bus_stop_loc_list = bus_stop_loc_list
        self.emit_time_list = emit_time_list
        self.sim_horizon = sim_horizon
        self.update_step = update_step
        self.cooridor_radius = r

        self.max_holding = 180
        self.flag = 0
        self.control_id_set = control_id
        self.catching_time = 0
        self.train_mode = train_mode
        self.start_time = datetime.datetime.now()
        self.update_time = datetime.datetime.now()
        self.check_time = datetime.datetime.now()
        self.stop_time = datetime.datetime.now() + datetime.timedelta(seconds=self.sim_horizon)
        self.wait_time_avgs = []
        self.load_set = []
        self.holds = []
        self.stop_visit = 0
        self.cost = []
        self.bus_loc_stop=[-1 for _ in range(6)]
        self.check_time = 3600
        self.check_step = 3600


        if train_mode==1:
            self.init_stopindex = [np.random.multinomial(1, [1. / 12 for i in range(12)], 1).tolist()[0].index(1) for j in
                          range(6)]

            self.init_stopindex.sort(reverse=True)

            for i in range(bus_num):
                dispatch_loc = self.bus_stop_loc_list[self.init_stopindex[i]]
                b = bus(w=np.pi/(3.6*400) , capacity=72, radius=r, id=i, stop_nums=bus_stop_num,
                        color_='blue', emit_time=emit_time_list[i],dispatch_loc=dispatch_loc)
                self.bus_list.append(b)

        if train_mode == 0:
            dispatch_loc = np.pi * 2 / 12 * (self.bus_num - 0 - 1) * 2
            for i in range(bus_num):
                b = bus(w=np.pi/(3.6*400) , capacity=72, radius=r, id=i, stop_nums=bus_stop_num,
                        color_='blue', emit_time=emit_time_list[i],dispatch_loc=dispatch_loc)

                dispatch_loc-=np.pi*2/12*2
                self.bus_list.append(b)
        arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
                     1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        for i in range(bus_stop_num):
            bs = bus_stop(alpha=bus_stop_loc_list[i], radius=r, id=i,wait_num=0,arr_rate=arr_rates[i])#arr_rates[i])
            self.bus_stop_list.append(bs)

    def sim(self, ):
        # begin = self.start_time
        # now = datetime.datetime.now()
        # if self.now!=0:
        #     now = self.now
        #
        # if now >= self.update_time:
        #     # print("simulation: {:.2f} s| {:.2f} s".format((now - begin).total_seconds(), (self.sim_horizon)))
        #     self.update_time = now + datetime.timedelta(seconds=self.update_step)
        # else:
        #     self.now = datetime.datetime.now()
        #     return 1


        # quit if simulation is over
        if len(self.bus_list[0].trajectory) > self.sim_horizon:
            # print('Full-horizon simulation done!')
            self.catching_time=len(self.bus_list[0].trajectory)*30
            # RL control whenever a bus pass the control point
            self.state = [[] for _ in range(6)]
            # agent self info:
            for bb in self.bus_list:
                if bb.id == 0:
                    front_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                            np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[0].travel_sum)
                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                            self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc)

                elif bb.id == len(self.bus_list) - 1:
                    front_hw = self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                            bb.travel_sum + bb.dispatch_loc)
                    back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                            np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[0].travel_sum)
                else:
                    front_hw = self.bus_list[bb.id - 1].travel_sum + self.bus_list[
                        bb.id - 1].dispatch_loc - (
                                       bb.travel_sum + bb.dispatch_loc)
                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                            self.bus_list[bb.id + 1].travel_sum + self.bus_list[bb.id + 1].dispatch_loc)

                self.state[bb.id] += [0, abs(front_hw), abs(back_hw)]


            return -3
        p_flag = [0 for i in range(self.bus_stop_num)]  # note bus stop whether is being served

        for b in self.bus_list:
            now = len(b.trajectory)
            b.special_state=0
            # each sim step renew bus running state, to determine whether bus is moving, stop for holding or serving
            b.is_serving=0
            if len(b.trajectory) >= self.check_time:
                # print('hour:%d'%(self.check_time/120))
                for b in (self.bus_list):
                    self.load_set+=b.serve_list
                    self.holds += [b.hold_time_list[tt] * 1/max(b.stop_visit[tt],1) for tt in range(len(b.hold_time_list))]
                    b.stop_visit=[0 for tt in range(12)]
                    b.hold_time_list=[0 for tt in range(12)]

                for k in range(len(self.bus_stop_list)):
                    self.wait_time_avgs.append(
                        float(1 * self.bus_stop_list[k].wait_time_sum) / float(self.bus_stop_list[k].wait_num_all))
                    self.bus_stop_list[k].wait_time_sum = 0
                    self.bus_stop_list[k].wait_num_all = 0


                self.check_time = self.check_time + self.check_step

            if len(b.trajectory) >= b.emit_time/self.update_step:
                if b.is_emit == False:
                    b.arrival_schedule = now + 60*6
                    b.is_emit = True

            if b.is_emit == False:
                b.is_serving=-2
                continue

            hold_time = b.hold_time
            is_stop = 0

            hold_time -= self.update_step

            for p in range(self.bus_stop_num):
                # No waiting at the origin station at the first departure
                if p == 0 and b.travel_sum == 0:
                    break

                # judge whether bus has arrived a stop-(1)
                if b.dispatch_loc > self.bus_stop_loc_list[p] \
                        and abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[p]) < 0.001:

                    if hold_time > 0 and b.is_emit == True:
                        b.hold_stop = p
                        is_stop = 1

                    p_flag[p] = 1
                    # record when each bus arrive first time at each round
                    if self.bus_stop_list[p].is_served == 0:
                        b.stop_visit[p] += 1
                        self.bus_stop_list[p].actual_bus_arr.append(len(b.trajectory)*self.update_step)
                        if len(self.bus_stop_list[p].actual_bus_arr) >= 2:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - self.bus_stop_list[p].actual_bus_arr[-2]))
                        else:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - 0))
                        self.bus_stop_list[p].wait_time_analysis()
                        self.bus_stop_list[p].is_served = 1
                        b.arrival_bias.append(abs((now - b.arrival_schedule)))
                        b.arrival_schedule = now + 6 * 60

                    # alight and board
                    if b.alight_list[p] > 0 and b.is_close==0:
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop = 1
                        b.is_serving = 1
                        b.serve_stop = p
                        # b.trip_cost[p][].append(len(b.trajectory)-b.trip_record[p][])

                    if self.bus_stop_list[p].wait_num > 0 and b.is_close==0:

                        is_stop = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p] += self.bus_stop_list[p].board_rate

                        self.bus_stop_list[p].board()

                        # assign passgeners' destination
                        s = np.random.randint(1, self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index] += self.bus_stop_list[p].board_rate
                        # b.trip_record[p][]=len(b.trajectory)
                        b.is_serving = 1
                        b.serve_stop = p


                    if is_stop==1:
                        break

                # judge whether bus has arrived a stop-(2)
                if abs(self.bus_stop_loc_list[p] - b.loc) < 0.001:
                    p_flag[p] = 1

                    if hold_time > 0 and b.is_emit == True:
                        is_stop = 1
                        b.hold_stop = p
                    # record when each bus arrive the stop first time at each round
                    if self.bus_stop_list[p].is_served == 0:
                        b.stop_visit[p]+=1
                        self.bus_stop_list[p].actual_bus_arr.append(len(b.trajectory)*self.update_step)
                        if len(self.bus_stop_list[p].actual_bus_arr) >= 2:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - self.bus_stop_list[p].actual_bus_arr[-2]))
                        else:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - 0))

                        self.bus_stop_list[p].wait_time_analysis()

                        self.bus_stop_list[p].is_served = 1
                        b.arrival_bias.append(abs((now - b.arrival_schedule)))
                        b.arrival_schedule = now + 4 * 60

                    # alight and board
                    if b.alight_list[p] > 0 and b.is_close == 0:
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop = 1
                        b.is_serving = 1
                        b.serve_stop = p
                    if self.bus_stop_list[p].wait_num > 0 and b.is_close == 0:
                        is_stop = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p] += self.bus_stop_list[p].board_rate

                        self.bus_stop_list[p].board()

                        # assign passgeners' destination
                        s = np.random.randint(1, self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index] += self.bus_stop_list[p].board_rate

                        b.is_serving = 1
                        b.serve_stop = p
                        # avoid boarding when being held


                    if is_stop == 1:
                        break

            b.hold_time = hold_time

            if b.hold_time>0 and b.is_serving==0:
                is_stop=1
                b.is_serving = -1

            if  is_stop!=1:
                b.is_serving = 0
                b.is_close = 0

        # Activate RL control after catching
        is_catch_happen=0
        for b in self.bus_list:
            #find out whether b is catching its leading bus
            p = None
            min_dist = 10000
            for pp in range(self.bus_stop_num):
                # judge whether bus has arrived a stop-(1)
                if b.dispatch_loc > self.bus_stop_loc_list[pp]:
                    if min_dist > abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[pp]):
                        min_dist = abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[pp])
                        p = pp
                else:
                    if min_dist > abs(self.bus_stop_loc_list[pp] - b.loc):
                        min_dist = abs(self.bus_stop_loc_list[pp] - b.loc)
                        p = pp

            if b.id > 0:
                if (self.bus_list[b.id - 1].travel_sum + self.bus_list[b.id - 1].dispatch_loc - self.bus_list[
                    b.id].dispatch_loc - self.bus_list[b.id].travel_sum) <= 0.01 \
                        and (self.bus_list[b.id].is_emit == True and self.bus_list[b.id - 1].is_emit == True):
                    b.special_state=1
                    b.hold_stop=self.bus_list[b.id - 1].serve_stop
                    if b.hold_stop==None:
                        b.hold_stop=self.bus_list[b.id-1].hold_stop
                    if b.hold_stop == None:
                        b.hold_stop = p
                        b.serve_stop = p

                    is_catch_happen=1

            if b.id == 0:
                if (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                        np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[0].travel_sum) <= 0.01 \
                        and (self.bus_list[-1].is_emit == True and self.bus_list[0].is_emit == True):

                    b.hold_stop = self.bus_list[  - 1].serve_stop
                    if b.hold_stop == None:
                        b.hold_stop = self.bus_list[ - 1].hold_stop
                    if b.hold_stop == None:
                        b.hold_stop = p
                        b.serve_stop = p

                    b.special_state=1
                    is_catch_happen=1

        if is_catch_happen==1 :
            ss=len(self.bus_list[-1].trajectory)
            self.state = [[] for _ in range(6)]
            # agent self info:
            for bb in self.bus_list:
                if bb.id == 0:
                    front_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                            np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[0].travel_sum)
                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                            self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc)
                elif bb.id == len(self.bus_list) - 1:
                    front_hw = self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                            bb.travel_sum + bb.dispatch_loc)
                    back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                            np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[0].travel_sum)
                else:
                    front_hw = self.bus_list[bb.id - 1].travel_sum + self.bus_list[
                        bb.id - 1].dispatch_loc - (
                                       bb.travel_sum + bb.dispatch_loc)
                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                            self.bus_list[bb.id + 1].travel_sum + self.bus_list[bb.id + 1].dispatch_loc)

                for p in range(self.bus_stop_num):
                    if abs(self.bus_stop_loc_list[p] - bb.loc) < 0.001 or (bb.dispatch_loc > self.bus_stop_loc_list[p] \
                                                                          and abs(
                                np.pi * 2 - bb.loc + self.bus_stop_loc_list[p]) < 0.001):
                        self.state[bb.id] += [self.bus_stop_list[p].wait_num/10, abs(front_hw), abs(back_hw)]


            if self.flag == 1:
                headway = []
                for bb in self.bus_list:
                    if bb.id + 1 < len(self.bus_list):
                        headway.append(abs(
                            self.bus_list[bb.id].travel_sum + self.bus_list[bb.id].dispatch_loc - self.bus_list[
                                bb.id + 1].dispatch_loc -
                            self.bus_list[bb.id + 1].travel_sum))
                headway.append(abs(self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                        np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                       0].travel_sum))
                headway = np.array(headway)
                # base =max(0,-(np.var(headway)-np.var(self.headway)) )#1.0 / (np.var(headway)+0.1) / 10.
                self.reward = 1 / (1. + np.var(headway))

            return 0

        # update state for each stop, including new arrival and wheter it is being served by a bus
        for p in range(self.bus_stop_num):
            if len(self.bus_list[-1].trajectory)>60:
                self.bus_stop_list[p].arrive()
            if p_flag[p] == 0:
                self.bus_stop_list[p].is_served = 0

        if len(self.bus_list[-1].trajectory) == 2*30:
            for p in range(self.bus_stop_num):
                self.bus_stop_list[p].wait_num = 10

        # Activate RL control whenever bus arrives stop
        if self.bus_list[-1].is_emit==True and len(self.bus_list[-1].trajectory)>25*60:
             # RL control whenever a bus pass the control point
             if sum(p_flag)>0:
                is_control=0
                for c in self.control_id_set:
                    for bb in self.bus_list:
                        if self.bus_loc_stop[bb.id]<0 and bb.hold_time<=0 and bb.is_emit == True and bb.is_serving != -1 \
                            and len(self.bus_stop_list[c].bus_arr_interval) > 0 and ((bb.dispatch_loc > self.bus_stop_loc_list[c] and abs(
                                np.pi * 2 - bb.loc + self.bus_stop_loc_list[c]) < 0.001) \
                                or (abs(self.bus_stop_loc_list[c] - bb.loc) < 0.001)) and bb.hold_stop==None:
                            is_control=1

                if is_control==1:
                    self.state = [[] for _ in range(6)]
                    for bb in self.bus_list:
                        for c in self.control_id_set:
                            if self.bus_loc_stop[bb.id] ==-1 and ((bb.dispatch_loc > self.bus_stop_loc_list[c] and abs(
                                    np.pi * 2 - bb.loc + self.bus_stop_loc_list[c]) < 0.001) \
                                    or (abs(self.bus_stop_loc_list[c] - bb.loc) < 0.001)):
                                self.bus_loc_stop[bb.id] = c
                                if self.bus_list[bb.id].is_emit == False:
                                    self.state[bb.id] = []
                                    break
                                if bb.id == 0:
                                    front_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                                    0].travel_sum)
                                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                                            self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc)
                                elif bb.id == len(self.bus_list) - 1:
                                    front_hw = self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                                            bb.travel_sum + bb.dispatch_loc)
                                    back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                                   0].travel_sum)
                                else:
                                    front_hw = self.bus_list[bb.id - 1].travel_sum + self.bus_list[
                                        bb.id - 1].dispatch_loc - (
                                                       bb.travel_sum + bb.dispatch_loc)
                                    back_hw = (bb.travel_sum + bb.dispatch_loc) - (
                                            self.bus_list[bb.id + 1].travel_sum + self.bus_list[bb.id + 1].dispatch_loc)

                                self.state[bb.id] += [self.bus_stop_list[c].wait_num / 10, abs(front_hw), abs(back_hw)]
                                break

                        if self.bus_loc_stop[bb.id] < 0:
                            self.state[bb.id] = []


                    if self.flag<1:
                        headway = []
                        for bb in self.bus_list:
                            if bb.id + 1 < len(self.bus_list):
                                headway.append(abs(bb.travel_sum - self.bus_list[bb.id + 1].travel_sum + abs(
                                    self.bus_list[bb.id].dispatch_loc - self.bus_list[bb.id + 1].dispatch_loc)))
                        self.headway = np.array(headway)

                    if self.flag==1:
                        headway = []
                        for bb in self.bus_list:
                            if bb.id + 1 < len(self.bus_list):
                                headway.append(abs(self.bus_list[bb.id].travel_sum + self.bus_list[bb.id].dispatch_loc - self.bus_list[bb.id + 1].dispatch_loc -
                    self.bus_list[bb.id + 1].travel_sum))
                        headway.append(abs(self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                                   0].travel_sum))
                        headway = np.array(headway)
                        base = 0.
                        # base =max(0,-(np.var(headway)-np.var(self.headway)) )#1.0 / (np.var(headway)+0.1) / 10.
                        base = np.exp(-np.var(headway))
                        # base=1/(np.var(headway)+1.)
                        self.reward = 1/(1.+np.var(headway))
                        self.headway = headway[:]
                    self.now = len(b.trajectory)
                    return 0


        for b in self.bus_list:

            if b.is_serving==0:
                self.bus_loc_stop[b.id] = -1
                b.is_close = 0
                b.hold_stop = None
                b.serve_stop = None
                b.move()
            else:
                stop_id = b.serve_stop
                if stop_id == None:
                    stop_id = b.hold_stop
                if stop_id==None:
                    min_dist = 1000
                    for pp in range(self.bus_stop_num):
                        # judge whether bus has arrived a stop-(1)
                        if min_dist > abs(self.bus_stop_loc_list[pp] - b.loc):
                            min_dist = abs(self.bus_stop_loc_list[pp] - b.loc)
                            stop_id = pp


                b.hold_stop = stop_id
                b.serve_stop = stop_id
                b.loc = self.bus_stop_loc_list[stop_id]
                b.stop()

        self.now = datetime.datetime.now()
        return 1

    def run(self): # run in RL main
        self.reset()
        while self.sim() > 0:
            flag = 1

    # RL control execute
    def control(self, action):
        # set a certain conrol point
        # self.action = action*self.max_holding
        hold_times = [a*self.max_holding for a in action]

        self.cost = [0 for i in range(6)]#
        self.cost = [1. for i in range(6)]
        maxhold_time=0

        for b in self.bus_list:
            b.is_serving_rl = 0
            if b.special_state==1:
                b.is_serving=-1
                b.hold_time = max(1,hold_times[b.id])
                b.is_close = 0
                continue

            if self.bus_loc_stop[b.id]==-1 or b.is_emit==False:
                continue
            else:
                # mark unhold action to store corresponding experience
                if hold_times[b.id]<=1:
                    self.bus_list[b.id].is_serving_rl=-1
                    self.cost[b.id]=1.
                if b.hold_time+self.update_step<=0 and hold_times[b.id]>0 and b.hold_stop==None:
                    b.hold_time = hold_times[b.id]
                    b.hold_stop = self.bus_loc_stop[b.id]
                    b.is_close = 0

        for b in self.bus_list:
            if b.is_serving==0 and b.hold_time>0:
                b.is_serving=-1

                # avoid boarding when being held
                b.is_close=1
            if b.is_serving==0:
                b.hold_stop=None
                b.serve_stop=None
                self.bus_loc_stop[b.id]=-1
                b.is_close=0
                b.move()

            else:
                # force to aligh with the location of the stop
                stop_id = b.serve_stop
                if stop_id==None:
                    stop_id = b.hold_stop

                if stop_id==None:
                    min_dist = 1000
                    for pp in range(self.bus_stop_num):
                        if min_dist > abs(self.bus_stop_loc_list[pp] - b.loc):
                            min_dist = abs(self.bus_stop_loc_list[pp] - b.loc)
                            stop_id = pp

                b.hold_stop = stop_id
                b.serve_stop = stop_id
                b.loc = self.bus_stop_loc_list[stop_id]
                b.stop()







