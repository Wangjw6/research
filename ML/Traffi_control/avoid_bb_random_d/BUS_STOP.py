import numpy as np

class bus_stop():
    '''
    id:               bus stop id
    arr_dis:          arrival distribution 0 for possion 1 for uniform
    schedule_hw:      set schedule arrival interval
    actual_bus_arr:   actual bus arrival time list
    bus_arr_interval: actual bus arrival interval list
    arr_rate:         arrival rate
    board_rate:       boarding rate
    tor_cap:          passenger tolerance on the maixmum loading number on bus
    wait_num:         number of passenger waiting on the bus stop
    corr_radius:      radius of bus corridor
    loc:              location of bus stop in angle
    scale:            scale factor in favor of visualization
    wait_time_sum:    total waiting time
    wait_num_sep:     accumulated waiting number between two consecutive arrival of buses
    '''
    def __init__(self,alpha,radius,id,arr_rate=1/2/30, board_rate=0.33, arr_dis=0, H=60*6,wait_num=0): # arr_dis : arrive distribution. 0 for uniform, 1 for possion
        self.id = id
        self.arr_dis = arr_dis
        self.schedule_hw = H # scheduled bus arrival headway
        self.actual_bus_arr=[]
        self.bus_arr_interval = []
        self.arr_rate =  arr_rate#min(abs(np.random.normal(loc=1/2/30,scale=0.1)),2/60 )# person / s
        # print(self.arr_rate)
        self.tor_cap = 120 # bus capacity
        self.board_rate = board_rate  # person / s
        self.wait_num = wait_num # 1 pixel for 10 person
        self.corr_radius = radius
        self.loc = alpha # determine stop location
        self.stop_x = self.corr_radius*np.cos(alpha)
        self.stop_y = self.corr_radius*np.sin(alpha)
        self.scale = 1. # scale waitline for visualization
        self.is_served = 0
        self.wait_time_sum = 0
        self.wait_num_sep = 0
        self.wait_num_all=0
    def board(self):
        # self.wait_num+=1 # come to board
        self.wait_num-=self.board_rate # finished boarding
        self.wait_num = max(self.wait_num,0.)

    def arrive(self):
        # self.wait_num+=1 # come to board
        if self.arr_dis==0:
            # k=np.random.poisson(self.arr_rate,1)[0]
            # self.wait_num+=k
            # self.wait_num_sep+=k
            # self.wait_num_all+=k
            k = np.random.poisson(self.arr_rate, 1)[0]
            self.wait_num_sep += self.wait_num*(1+1.)
            self.wait_num += k
            self.wait_num_all += k

        if self.arr_dis==1:
            self.wait_num+=self.arr_rate
            self.wait_num_sep += np.random.poisson(self.arr_rate, 1)[0]

    def wait_time_analysis(self):
        # self.wait_time_sum+=self.wait_num_sep*self.bus_arr_interval[-1]
        # self.wait_num_sep = 0

        self.wait_time_sum += self.wait_num_sep
        self.wait_num_sep = 0

