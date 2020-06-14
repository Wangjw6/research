import numpy as np

class bus():
    '''
    id:                bus id
    is_serving:        whehter bus is serving a stop 1:stop for service 0:on route -1:holding -2: not emit
    omega:             angle speed
    capacity:          bus capacity
    car_size:          car size for visualization
    track_radius:      radius of bus corridor
    color:             car color for visualization
    step:              simulation step increase with 1
    slack time:        time cost at each station
    emit_time:         time for departure
    is_emit:           emit flag
    serve_list:        record served  passgenger number for each stop of a bus through the simulation
    alight_list:       record numbers of passenger to alight in each stop in real time
    serve_level:       bus serve level
    dispatch_loc:      dispatch location
    loc:               bus current loc (reset when arriving at origin station in favor of visualization)
    travel_sum:        overall distance the bus has traveled
    arrival_schedule:  schedule arrval time on a stop
    arrival_bias:      bias between schedule arrival time and actual arrival time
    alight_rate:       alighting rate
    onboard_list:      record number of passenger on board boarding from each stop in real time
    slack_time:         slack time to force bus keep pace with the schedule <= hold time
    slack_time_sum:     sum of slack time
    hold_time:         holding time to force bus keep pace with the schedule
    hold_time_sum:     sum of holding time
    trajectory:        record bus trajectory
    hold_stop:         record at which station the bus is holding
    state:             state observation from the perspective of bus
    action:            control for each stop
    reward:            minimize mean of headway and variance of headway
    is_close           the bus is closed or not. 0:open 1:close
    ass_dispatch_loc:  assist locate the bus in trajectory construction
    hold_action:       record the where and when the holding function
    is_serving_rl:     flag for RL train sample collect
    special_state:     0 for normal ,1 for catching its leading bus
    serve_stop:        record which stop the bus is serving
    trip_record        record travel point for each od-pair trip
    trip_cost        record travel cost for each od-pair trip
    '''
    def __init__(self,w,capacity,radius,car_size=6,state_dim=6, action_dim=1,alight_rate=0.55  ,color_='blue',dispatch_loc=0,id=1,stop_nums=1,emit_time=0):
        self.id = id
        self.is_serving = 0
        self.omega = w
        self.capacity =capacity
        self.car_size = car_size
        self.track_radius = radius
        self.color = color_
        self.step = 0
        self.slack_time = 0
        self.slack_time_sum = 0
        self.hold_time = 0
        self.hold_time_sum = 0
        self.emit_time = emit_time
        self.is_emit = False
        self.serve_list = [0 for i in range(stop_nums)]
        self.alight_list = [0 for i in range(stop_nums)]
        self.serve_level = []
        self.dispatch_loc = dispatch_loc # starting station
        self.loc = dispatch_loc
        self.hold_stop = None
        self.hold_stop_temp=-2
        self.travel_sum = 0
        self.arrival_schedule = 0
        self.alight_rate = alight_rate
        self.arrival_bias = []
        self.onboard_list = [0 for i in range(stop_nums)]
        self.trajectory = []
        self.state = []
        self.action = [0 for i in range(action_dim)]
        self.reward = 0
        self.is_close = 0
        self.loc_set=[]
        self.ass_dispatch_loc=dispatch_loc
        self.hold_action=[]
        self.hold_action_w = []
        self.is_serving_rl = 0
        self.special_state = 0
        self.serve_stop = None
        self.stop_visit=[0 for i in range(12)]
        self.hold_time_list=[0 for i in range(stop_nums)]
        cx = self.track_radius*np.cos(dispatch_loc)
        cy = self.track_radius*np.sin(dispatch_loc)
        self.trip_record = [[[] for j in range(6) ] for i in range(12)] # 12 origin*6interval
        self.trip_cost = [[[] for j in range(6)  ] for i in range(12)]

    def move(self,):
        self.isserving = 0
        self.step += 1
        self.trajectory.append(self.travel_sum+self.dispatch_loc)
        self.loc_set.append(self.loc)
        self.loc = self.ass_dispatch_loc+self.omega*self.step # add random noise to bus speed
        self.travel_sum+=self.omega
        self.hold_action.append(0.)
        self.hold_action_w.append( 0.)
        if self.loc>=np.pi*2: # when vehicle arrive at the stop 1, loc returns to 0
            self.loc=0
            self.ass_dispatch_loc=0
            self.step=0
            self.pass_stop = []

    def stop(self):
        self.trajectory.append(self.travel_sum+self.dispatch_loc)

        self.isserving = 1
        self.slack_time_sum += 1.
        if self.is_serving==-1 and self.hold_stop!=self.hold_stop_temp and self.hold_time>30: # only calculate hold time if the bus is first held
            self.hold_action.append(self.loc)
            self.hold_action_w.append(self.hold_time)
            # print(self.hold_time)
            self.hold_time_sum += self.hold_time # 0.01=update step
            # print(self.hold_time)
            self.hold_stop_temp = self.hold_stop
            self.hold_time_list[self.hold_stop]+=self.hold_time
            self.is_serving_rl = -1
        else:
            if len(self.loc_set)>len(self.hold_action):
                self.hold_action.append(0.)
                self.hold_action_w.append(0)
        self.loc_set.append(self.loc)
        if self.loc>=np.pi*2:
            self.loc=0
            self.ass_dispatch_loc=0
            self.step = 0
            self.pass_stop = []


