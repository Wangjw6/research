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

    '''

    def __init__(self, state_dim, action_dim, bus_num, bus_stop_num, r, emit_time_list,
                 bus_dep_list, bus_stop_loc_list, update_step=1, sim_horizon=36,FH=False,BH=False,
              HH = False, SH = False,FB = False):
        self.FH = FH
        self.BH = BH
        self.HH = HH
        self.SH = SH
        self.FB = FB
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        self.check_step = 3600.

        self.catching = []
        self.hold = []
        self.slack = []

        self.wait_time_avgs = []
        self.load_set = []
        self.holds = []
        arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
                     1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        for i in range(bus_num):
            b = bus(w=np.pi/(400*3.6)*1,capacity=72,radius=r, id=i,stop_nums=bus_stop_num,
                    color_='blue',emit_time=emit_time_list[i],dispatch_loc=np.pi*2/12*(self.bus_num-i-1)*2)
            self.bus_list.append(b)
        for i in range(bus_stop_num):
            bs = bus_stop(alpha=bus_stop_loc_list[i], radius=r, id=i,wait_num=0,arr_rate=arr_rates[i])
            self.bus_stop_list.append(bs)

    def sim(self, ):
        FH = False
        BH = False
        HH = False
        SH = False
        FB = False
        if len(self.bus_list[-1].trajectory) > 25*60:
            thr=180
            FH = self.FH
            BH = self.BH
            alpha=0.5
            HH = self.HH
            SH = self.SH
        begin = self.start_time

        i = 0

        p_flag = [0 for i in range(self.bus_stop_num)]  # note bus stop whether is being served

        for b in self.bus_list:
            now = len(b.trajectory)
            if len(b.trajectory) >= b.emit_time / self.update_step:
                if b.is_emit == False:
                    b.arrival_schedule = now + 6*60
                b.is_emit = True

            if b.is_emit == False:
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
                # b.loc = self.bus_stop_loc_list[stop_id]
                b.stop()
                continue

            b.is_serving=0

            if len(b.trajectory) >= self.check_time:
                # print('hour:%d'%(self.check_time/120))
                for b in (self.bus_list):
                    self.load_set+=b.serve_list
                    # print(b.stop_visit)
                    self.holds += [b.hold_time_list[tt] * 1/max(b.stop_visit[tt],1) for tt in range(len(b.hold_time_list))]
                    b.stop_visit=[0 for tt in range(12)]
                    b.hold_time_list=[0 for tt in range(12)]
                for k in range(len(self.bus_stop_list)):
                    self.wait_time_avgs.append(
                        float(1 * self.bus_stop_list[k].wait_time_sum) / float(self.bus_stop_list[k].wait_num_all))
                    self.bus_stop_list[k].wait_time_sum = 0
                    self.bus_stop_list[k].wait_num_all = 0

                self.check_time = self.check_time + self.check_step

            if b.id>0:
                if (self.bus_list[b.id-1].travel_sum+self.bus_list[b.id-1].dispatch_loc-self.bus_list[b.id].dispatch_loc-self.bus_list[b.id].travel_sum)<=0.01\
                        and (self.bus_list[b.id].is_emit==True and self.bus_list[b.id-1].is_emit==True):
                    p = None
                    min_dist = 1000
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

                    if b.id == len(self.bus_list) - 1:
                        front_hw = (self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                                b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                       0].travel_sum)/ b.omega
                    else:
                        front_hw = (self.bus_list[b.id - 1].travel_sum + self.bus_list[
                            b.id - 1].dispatch_loc - (
                                           b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = ((b.travel_sum + b.dispatch_loc) - (
                                self.bus_list[b.id + 1].travel_sum + self.bus_list[b.id + 1].dispatch_loc))/ b.omega

                    # FH
                    if FH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0 :
                        b.hold_time = min(thr,max(0, 5.8*60  + 0.8 * (self.bus_stop_list[p].schedule_hw - front_hw)))
                        if b.hold_time>0:
                            is_first_hold = 1
                            b.hold_stop = p
                            is_stop = 1
                            b.is_close = 0
                    # BH
                    if BH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0 :
                            b.hold_time = min(thr,alpha *  (back_hw))
                            is_first_hold = 1
                            b.hold_stop = p
                            is_stop = 1
                            b.is_close = 0
                    #HH
                    if HH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]>0:
                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]))

                        b.is_close = 0
                    if SH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(b.arrival_schedule-len(b.trajectory)))

                        b.is_close = 0

                    if FB and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = max(0, (front_hw-back_hw)/2.)

                        b.is_close = 0

                    b.hold_time_sum+=1
                    b.hold_time_list[p] += 1

                    b.stop()
                    b.hold_action.append(b.loc_set[-1])
                    b.hold_action_w.append(1)
                    continue
            if b.id==0:
                if (self.bus_list[-1].travel_sum+self.bus_list[-1].dispatch_loc+(np.pi*2-self.bus_list[0].dispatch_loc)-self.bus_list[0].travel_sum)<=0.01\
                        and (self.bus_list[-1].is_emit==True and self.bus_list[0].is_emit==True):
                    p=None
                    min_dist=1000
                    for pp in range(self.bus_stop_num):
                        # judge whether bus has arrived a stop-(1)
                        if b.dispatch_loc > self.bus_stop_loc_list[pp]:
                            if min_dist>abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[pp]):
                                min_dist=abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[pp])
                                p=pp
                        if  abs(self.bus_stop_loc_list[pp] - b.loc) < 0.001:
                            if min_dist>abs(self.bus_stop_loc_list[pp] - b.loc):
                                min_dist=abs(self.bus_stop_loc_list[pp] - b.loc)
                                p=pp

                    front_hw = ((self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                            np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                    0].travel_sum))/ b.omega
                    back_hw = ((b.travel_sum + b.dispatch_loc) - (
                            self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc))/ b.omega

                    # FH
                    if FH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0 :
                        b.hold_time = min(thr,max(0, 5.8*60 + 0.8 * (self.bus_stop_list[p].schedule_hw - front_hw)))
                        if b.hold_time>0:
                            is_first_hold = 1
                            b.hold_stop = p
                            is_stop = 1
                            b.is_close = 0
                    # BH
                    if BH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0 :
                        b.hold_time =  min(thr,alpha *  (back_hw))
                        if b.hold_time>0:
                            is_first_hold = 1
                            b.hold_stop = p
                            is_stop = 1
                            b.is_close = 0

                    # HH
                    if HH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]>0:

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]))

                        b.is_close = 0

                    if SH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(b.arrival_schedule-len(b.trajectory)))

                        b.is_close = 0

                    if FB and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = max(0, (front_hw-back_hw)/2.)

                        b.is_close = 0

                    b.hold_time_sum += 1
                    b.hold_time_list[p] += 1


                    b.stop()
                    b.hold_action.append(b.loc_set[-1])
                    b.hold_action_w.append(1)
                    continue


            hold_time = b.hold_time
            is_stop = 0
            is_first_hold = 0
            # put this here in case of a vehicle being held multiple times continuously
            #
            if hold_time>0:
                hold_time -= self.update_step

            for p in range(self.bus_stop_num):
                # # No waiting at the origin station at the first departure
                # if p == 0 and b.travel_sum == 0:
                #     break
                # judge whether bus has arrived a stop-(1)
                if b.dispatch_loc > self.bus_stop_loc_list[p] \
                        and abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[p]) < 0.001:

                    if self.bus_stop_list[p].is_served == 0 and b.hold_time<=0:
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
                        b.arrival_schedule = now + 6*60

                    if b.id == 0:
                        front_hw = ((self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                        0].travel_sum))/ b.omega
                        back_hw = ((b.travel_sum + b.dispatch_loc) - (
                                self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc))/ b.omega
                    elif b.id == len(self.bus_list) - 1:
                        front_hw = (self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                                b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                       0].travel_sum)/ b.omega
                    else:
                        front_hw = (self.bus_list[b.id - 1].travel_sum + self.bus_list[
                            b.id - 1].dispatch_loc - (
                                           b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = ((b.travel_sum + b.dispatch_loc) - (
                                self.bus_list[b.id + 1].travel_sum + self.bus_list[b.id + 1].dispatch_loc))/ b.omega

                    # FH
                    if FH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0:

                        b.hold_time = min(thr,max(0, 5.8*60 + 0.8 * (self.bus_stop_list[p].schedule_hw - front_hw)))
                        if b.hold_time>0:
                            is_first_hold = 1
                            b.hold_stop = p
                            b.is_close = 0
                    # BH
                    if BH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0 :

                        b.hold_time = min(thr,alpha *  (back_hw))
                        if b.hold_time > 0:
                            is_first_hold = 1
                            b.hold_stop = p
                            b.is_close = 0

                    # HH
                    if HH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]>0:

                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]))
                        b.is_close = 0

                    if SH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(b.arrival_schedule-len(b.trajectory)))

                        b.is_close = 0

                    if FB and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = max(0, (front_hw-back_hw)/2.)

                        b.is_close = 0

                    # alight and board
                    if b.alight_list[p] > 0  and b.is_close==0:
                        p_flag[p] = 1
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop = 1
                        b.is_serving = 1
                        b.serve_stop = p

                    if self.bus_stop_list[p].wait_num > 0 and b.is_close==0:
                        p_flag[p] = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p] += self.bus_stop_list[p].board_rate
                        # assign passgeners' destination
                        s = np.random.randint(1, self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index] += self.bus_stop_list[p].board_rate
                        is_stop = 1
                        self.bus_stop_list[p].board()
                        b.is_serving = 1
                        b.serve_stop = p



                    if hold_time > 0 and b.is_emit == True and b.is_serving==0:
                        b.is_serving = -1
                        is_stop = 1

                    # avoid boarding when being held
                    if b.is_serving==-1:
                        b.is_close = 1

                    if is_stop == 1:
                        break

                # judge whether bus has arrived a stop-(2)
                if abs(self.bus_stop_loc_list[p] - b.loc) < 0.001:
                    if self.bus_stop_list[p].is_served == 0  and b.hold_time<=0:
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
                        b.arrival_schedule = now + 6*60
                    # (*)threshold holding control when all buses have departured
                    if b.id == 0:
                        front_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                        0].travel_sum)/ b.omega
                        back_hw = ((b.travel_sum + b.dispatch_loc) - (
                                self.bus_list[1].travel_sum + self.bus_list[1].dispatch_loc))/ b.omega
                    elif b.id == len(self.bus_list) - 1:
                        front_hw = (self.bus_list[-2].travel_sum + self.bus_list[-2].dispatch_loc - (
                                b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = (self.bus_list[-1].travel_sum + self.bus_list[-1].dispatch_loc + (
                                np.pi * 2 - self.bus_list[0].dispatch_loc) - self.bus_list[
                                       0].travel_sum)/ b.omega
                    else:
                        front_hw = (self.bus_list[b.id - 1].travel_sum + self.bus_list[
                            b.id - 1].dispatch_loc - (
                                           b.travel_sum + b.dispatch_loc))/ b.omega
                        back_hw = ((b.travel_sum + b.dispatch_loc) - (
                                self.bus_list[b.id + 1].travel_sum + self.bus_list[b.id + 1].dispatch_loc))/ b.omega

                    # FH
                    if FH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0:
                        b.hold_time = min(thr,max(0,5.8*60+0.8*(self.bus_stop_list[p].schedule_hw-front_hw)))
                        if b.hold_time>0:
                            b.is_close = 0
                            is_first_hold = 1
                            b.hold_stop = p
                    # BH
                    if BH and b.hold_time <= 0 and self.bus_list[-1].is_emit == True and b.is_serving != -1 \
                            and len(self.bus_stop_list[p].bus_arr_interval) > 0:
                        b.hold_time =  min(thr,alpha *  (back_hw))
                        if b.hold_time > 0:
                            is_first_hold = 1
                            b.hold_stop = p
                            b.is_close = 0

                    # HH
                    if HH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]>0:
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(self.bus_stop_list[p].schedule_hw-self.bus_stop_list[p].bus_arr_interval[-1]))

                        b.is_close = 0
                    if SH and b.hold_time<=0 and self.bus_list[-1].is_emit==True and b.is_serving!=-1\
                            and len(self.bus_stop_list[p].bus_arr_interval)>0 and b.arrival_schedule>len(b.trajectory):

                        is_stop = 1
                        is_first_hold = 1
                        b.hold_stop = p
                        b.hold_time = min(thr,(b.arrival_schedule-len(b.trajectory)))

                        b.is_close = 0
                    # alight and board

                    if b.alight_list[p] > 0  and b.is_close==0:
                        p_flag[p] = 1
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop = 1
                        b.is_serving = 1
                        b.serve_stop = p

                    if self.bus_stop_list[p].wait_num > 0 and b.is_close==0:
                        p_flag[p] = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p] += self.bus_stop_list[p].board_rate
                        # assign passgeners' destination
                        s = np.random.randint(1, self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index] += self.bus_stop_list[p].board_rate
                        is_stop = 1
                        self.bus_stop_list[p].board()
                        b.is_serving = 1
                        b.serve_stop = p

                    if hold_time > 0 and b.is_emit == True and b.is_serving==0:
                        b.is_serving = -1
                        is_stop = 1
                    # avoid boarding when being held
                    if b.is_serving==-1:
                        b.is_close = 1
                    if is_stop == 1:
                        break

                if b.is_serving==-1:
                    break

            if is_first_hold == 0:
                b.hold_time = hold_time

            if b.hold_time>0 and b.is_serving==0:
                is_stop=1
                b.is_serving = -1

            if is_stop!=1:
                b.is_serving=0
                b.is_close=0
                b.hold_stop = None
                b.serve_stop = None
                b.move()
            else:
                stop_id = b.hold_stop
                if stop_id == None:
                    stop_id = b.serve_stop

                if stop_id==None:
                    min_dist = 1000
                    for pp in range(self.bus_stop_num):
                        # judge whether bus has arrived a stop-(1)
                        if min_dist > abs(self.bus_stop_loc_list[pp] - b.loc):
                            min_dist = abs(self.bus_stop_loc_list[pp] - b.loc)
                            stop_id = pp


                b.hold_stop = stop_id
                b.serve_stop = stop_id
                b.stop()


        # update state for each stop, including new arrival and wheter it is being served by a bus

        for p in range(self.bus_stop_num):
            if len(self.bus_list[-1].trajectory)>2*30  :
                self.bus_stop_list[p].arrive()
            if p_flag[p] == 0:
                self.bus_stop_list[p].is_served = 0

        if len(self.bus_list[-1].trajectory) == 2*30:
            for p in range(self.bus_stop_num):
                self.bus_stop_list[p].wait_num = 10

        # quit if simulation is over
        if len(self.bus_list[0].trajectory)>self.sim_horizon:#  or (len(self.bus_list[0].trajectory)>0 and self.bus_list[0].trajectory[-1]>=np.pi*2):
            self.catching_time = 1 * len(self.bus_list[0].trajectory)
            print("finishing time: {:.2f} s".format( 1*len(self.bus_list[0].trajectory)))
            return -1

        return 1

    def run(self):
        self.reset()
        while self.sim() > 0:
            flag = 1

        print('simulation done!')

    def reset(self, ):
        self.start_time = datetime.datetime.now()
        self.update_time = datetime.datetime.now()+datetime.timedelta(seconds=1.2)
        self.check_time = 3600
        self.stop_time = datetime.datetime.now()+  datetime.timedelta(seconds=self.sim_horizon)
        self.bus_list = []
        self.bus_stop_list = []

        for i in range(self.bus_num):
            b = bus(w=np.pi/(3.6*400)*1,capacity=72,radius=self.cooridor_radius, id=i,stop_nums=self.bus_stop_num,
                    color_='blue',emit_time=self.emit_time_list[i],dispatch_loc=np.pi*2/12*(self.bus_num-i-1)*2)
            self.bus_list.append(b)
        arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
                     1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        for i in range(self.bus_stop_num):

            bs = bus_stop(alpha=self.bus_stop_loc_list[i], radius=self.cooridor_radius, id=i,wait_num=0,arr_rate=arr_rates[i])
            self.bus_stop_list.append(bs)





