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
                 bus_dep_list,bus_stop_loc_list,update_step=0.01, sim_horizon=36):

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
        self.check_step = 3.

        for i in range(bus_num):
            b = bus(w=np.pi/(400*3.6)*30,capacity=72,radius=r, id=i,stop_nums=bus_stop_num,
                    color_='blue',emit_time=emit_time_list[i],dispatch_loc=np.pi*2/12*(self.bus_num-i-1)*2)
            self.bus_list.append(b)
        for i in range(bus_stop_num):

            bs = bus_stop(alpha=bus_stop_loc_list[i], radius=r, id=i,wait_num=0)
            self.bus_stop_list.append(bs)

    def sim(self,):
        begin = self.start_time
        now = datetime.datetime.now()

        # check if the leading bus is cathced on by its following ones
        # if len(self.bus_list[0].trajectory)>self.bus_list[0].emit_time/self.update_step:
        #     i = 0
        #
        #     while i<len(self.bus_list)-1:
        #         if (self.bus_list[i].travel_sum+self.bus_list[i].dispatch_loc-self.bus_list[i+1].dispatch_loc-self.bus_list[i+1].travel_sum)<=0.1\
        #                 and (self.bus_list[i].is_emit==True and self.bus_list[i+1].is_emit==True):
        #             self.catching_time = 30*len(self.bus_list[i].trajectory)
        #             print("{} is cathced , catching time: {:.2f} s".format(i, 30*len(self.bus_list[i].trajectory)))
        #             self.bus_list[i + 1]
        #         i+=1
        #     if (self.bus_list[-1].travel_sum+self.bus_list[-1].dispatch_loc+(np.pi*2-self.bus_list[0].dispatch_loc)-self.bus_list[0].travel_sum)<=0.1\
        #             and (self.bus_list[-1].is_emit==True and self.bus_list[0].is_emit==True):
        #         self.catching_time = 30*len(self.bus_list[0].trajectory)
        #         print("{} is cathced , catching time: {:.2f} s".format(self.bus_list[-1].id, 30*len(self.bus_list[i].trajectory)))
        #         return -1
        # set update interval
        if now>=self.update_time:
            # print("simulation: {:.2f} s| {:.2f} s".format((now-begin).total_seconds(),(self.sim_horizon)))
            self.update_time = now+datetime.timedelta(seconds=self.update_step)
        else:
            return 1

        p_flag = [0 for i in range(self.bus_stop_num)] # note bus stop whether is being served
        for b in self.bus_list:
            if len(b.trajectory) >= b.emit_time / self.update_step:
                if b.is_emit==False:
                    b.arrival_schedule = now + datetime.timedelta(seconds=np.pi*2/12/b.omega)
                b.is_emit = True
            if b.is_emit == False:
                b.stop()
                continue

            if b.id>0:
                if (self.bus_list[b.id-1].travel_sum+self.bus_list[b.id-1].dispatch_loc-self.bus_list[b.id].dispatch_loc-self.bus_list[b.id].travel_sum)<=0.1\
                        and (self.bus_list[b.id].is_emit==True and self.bus_list[b.id-1].is_emit==True):
                    b.stop()
                    b.hold_time_sum+=1
                    continue
            if b.id==0:
                if (self.bus_list[-1].travel_sum+self.bus_list[-1].dispatch_loc+(np.pi*2-self.bus_list[0].dispatch_loc)-self.bus_list[0].travel_sum)<=0.1\
                        and (self.bus_list[-1].is_emit==True and self.bus_list[0].is_emit==True):
                    b.hold_time_sum += 1
                    b.stop()
                    continue

            b.is_serving = 0
            if now>=self.check_time:
                b.serve_level.append(float(sum(b.onboard_list))/b.capacity)
                self.check_time = self.check_time+datetime.timedelta(seconds=self.check_step)

            for p in range(self.bus_stop_num):
                # No waiting at the origin station at the first departure
                if p==0 and b.travel_sum==0:
                    break
                # judge whether bus has arrived a stop-(1)
                if b.dispatch_loc > self.bus_stop_loc_list[p] \
                        and abs(np.pi * 2 - b.loc + self.bus_stop_loc_list[p]) < 0.001:
                    p_flag[p] = 1
                    # record when each bus arrive first time at each round
                    if self.bus_stop_list[p].is_served == 0:
                        self.bus_stop_list[p].actual_bus_arr.append((now-begin).total_seconds())
                        if len(self.bus_stop_list[p].actual_bus_arr) >= 2:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - self.bus_stop_list[p].actual_bus_arr[-2]))
                        else:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - 0))
                        self.bus_stop_list[p].wait_time_analysis()
                        self.bus_stop_list[p].is_served = 1
                        b.arrival_bias.append(abs((now-b.arrival_schedule).total_seconds()))
                        b.arrival_schedule = now + datetime.timedelta(seconds=np.pi * 2 / 12 / b.omega)

                    # alight and board
                    is_stop=0
                    if b.alight_list[p] > 0:
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop=1
                        b.is_serving = 1
                    if self.bus_stop_list[p].wait_num > 0:
                        is_stop = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p]+=self.bus_stop_list[p].board_rate
                        self.bus_stop_list[p].board()
                        # assign passgeners' destination
                        s = np.random.randint(1,self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index]+= self.bus_stop_list[p].board_rate
                        b.is_serving = 1
                    if is_stop==1:
                        b.stop()
                        break

                # judge whether bus has arrived a stop-(2)

                if abs( self.bus_stop_loc_list[p] - b.loc) < 0.001:
                    p_flag[p] = 1
                    if self.bus_stop_list[p].is_served == 0:
                        self.bus_stop_list[p].actual_bus_arr.append((now - begin).total_seconds())
                        if len(self.bus_stop_list[p].actual_bus_arr) >= 2:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - self.bus_stop_list[p].actual_bus_arr[-2]))
                        else:
                            self.bus_stop_list[p].bus_arr_interval.append(
                                abs(self.bus_stop_list[p].actual_bus_arr[-1] - 0))
                        self.bus_stop_list[p].wait_time_analysis()
                        self.bus_stop_list[p].is_served = 1
                        b.arrival_bias.append(abs((now - b.arrival_schedule).total_seconds()))
                        b.arrival_schedule = now + datetime.timedelta(seconds=np.pi * 2 / 12 / b.omega)

                    # alight and board
                    is_stop = 0
                    if b.alight_list[p] > 0:
                        b.alight_list[p] -= b.alight_rate
                        b.onboard_list[p] -= b.alight_rate
                        if b.alight_list[p] < 0:
                            b.alight_list[p] = 0
                        is_stop = 1
                        b.is_serving = 1
                    if self.bus_stop_list[p].wait_num > 0:
                        is_stop = 1
                        b.serve_list[p] += self.bus_stop_list[p].board_rate
                        b.onboard_list[p] += self.bus_stop_list[p].board_rate
                        self.bus_stop_list[p].board()
                        # assign passgeners' destination
                        s = np.random.randint(1, self.bus_stop_num+1)
                        alight_index = (p + s) % self.bus_stop_num
                        b.alight_list[alight_index] += self.bus_stop_list[p].board_rate
                        b.is_serving = 1
                    if is_stop == 1:
                        b.stop()
                        break

            if b.is_serving == 0:
                b.move()

        # update state for each stop, including new arrival and wheter it is being served by a bus
        for p in range(self.bus_stop_num):
            if len(self.bus_list[-1].trajectory)>2:
                self.bus_stop_list[p].arrive()
            if p_flag[p] == 0:
                self.bus_stop_list[p].is_served = 0

        if len(self.bus_list[-1].trajectory) == 2:
            for p in range(self.bus_stop_num):
                self.bus_stop_list[p].wait_num = 30
        # quit if simulation is over
        if len(self.bus_list[0].trajectory)>self.sim_horizon:
            self.catching_time = 30 * len(self.bus_list[0].trajectory)
            print('Full-horizon simulation done!')
            return -1

        return 1

    def run(self):
        self.reset()
        while self.sim()>0:
            flag=1

        print('simulation done!')

    def reset(self,):
        self.start_time = datetime.datetime.now()
        self.update_time = datetime.datetime.now()
        self.check_time = datetime.datetime.now()
        self.stop_time = datetime.datetime.now()+  datetime.timedelta(seconds=self.sim_horizon)
        self.bus_list = []
        self.bus_stop_list = []

        for i in range(self.bus_num):
            b = bus(w=np.pi/(3.6*400)*30,capacity=72,radius=self.cooridor_radius, id=i,stop_nums=self.bus_stop_num,
                    color_='blue',emit_time=self.emit_time_list[i],dispatch_loc=np.pi*2/12*(self.bus_num-i-1)*2)
            self.bus_list.append(b)

        for i in range(self.bus_stop_num):

            bs = bus_stop(alpha=self.bus_stop_loc_list[i], radius=self.cooridor_radius, id=i,wait_num=0)
            self.bus_stop_list.append(bs)





