#coding=utf-8

'''
"""
predicton.utils.entity
################################################
"""
 #obs_traj.append((feature_sequence[j].position.x,
                             #feature_sequence[j].position.y,
                             #feature_sequence[j].velocity_heading,
                             #feature_sequence[j].speed,
                             #feature_sequence[j].length,
                             #feature_sequence[j].width,
                             #feature_sequence[j].timestamp))
convert dict to
'''
class Data:
    def __init__(self):
        self._ego_x = ego_x
        self._ego_y = ego_y
        self._id = id
        self._tracking_time = tracking_time
        self._type = type
        self._length = length
        self._width = width
        self._height = height
    @property
    def position(self):
        return pos
    @property
    def accleration_vector(self):
        return accel_vec
    @property
    def Velocity(self):
        return vel
    @property
    def Lane(self):
        return lane
    @property
    def predicted_trajectory(self):
        return pre_traj
    def __getitem__(self, index : int):
        pass
    def __len(self):
        return len(self.data)
    def __str__(self):
        return f'"data":{self}'
    def __repr__(self):
        return self.__str__()     

class Lane_point:
    def __init__(self):
        self._heading = heading
        self._width = width
        self._relative_l = relative_l
        self._relative_s = relative_s
        self._angle_diff = angl_diff
        self._kappa = kappa
        self._scenario_type = scenario_type
    @property
    def position(self):
        return pos
    def __str__(self):
        return f'{self}'
    def __repr__(self):
        return self.__str__()
class Lane_segment:
    def __init__(self):
        self._lane_id = lane_id
        self._start_s = start_s
        self._end_s = end_s
        self._end_l = end_l
        self._total_length = total_length
        self._lane_turn_type = lane_turn_type
        self._adc_lane_point_idx = adc_point_idx
        self._adc_s = adc_s
    @property
    def lane_point(self):
        return lane_point
    def __str__(self):
        return f'{self}'
    def __repr__(self):
        return self.__str__()
class Lane_seq:
    def __init__(self):
        self._lane_seq_id = lane_seq_id
        self._label = label
        self._probablity = probablity
        self._vehicle_on_lane = vehicle_on_lane
        self.right_of_way = right_of_way
        self.time_to_lane_center = time_to_lane_center
        self.lane_s = lane_s
        self._lane_l = lane_l
        self._time_to_lane_edge = time_to_lane_edge
        self._adc_lane_segment_idx = adc_lane_segment_idx
        self._behavior_type = behavior_type
        self._lane_type = lane_type
        self._mlp_features = mlp_features
    @property
    def nearby_obstacles(self):
        return nearby_obstacles
    def __str__(self):
        return f'{self}'
    def __repr__(self):
        return self.__str__()
class Nearby_obstacles:
    def __init__(self):
        self._s = s
        self._l = l
        self._id = id
    @property
    def s(self):
        return self._s
    @property
    def l(self):
        return self._l
    @property
    def id(self):
        return self._id
    def __str__(self):
        return f'{self}'
    def __repr__(self):
        return self.__str__()
class Predicted_trajectory:
    def __init__(self,pre_traj):
        self.i = pre_traj['i']
        self.x = pre_traj['x']
        self.y = pre_traj['y']
        self.theta = pre_traj['lane_id']
        self.s = pre_traj['s']
        self.l = pre_traj['l']
        self.velocity = pre_traj['velocity']
        self.jerk = pre_traj['kappa']
        self.rel_time = pre_traj['relative_time']
    def __str__(self):
        return f'{self}'
    def __repr__(self):
        return self.__str__()
class Acceleration_vector:
    def __init__(self, vel):
        self.accel_vel_xx = accel_vec['x']
        self.vel_yy = accel_vec['y']
        self.vel_zz = accel_vec['z']  
    @property
    def x(self):
        return self.accel_vel_xx   
    @property
    def y(self):
        return self.accel_vel_yy    
    @property
    def z(self):
        return self.accel_vel_zz
    def __str__(self):
        return f'["x":{self.accel_vel_xx}, "y":{self.accel_vel_yy}, "z":{self.accel_vel_zz}]'
    def __repr__(self):
        return self.__str__()         
class Velocity_heading:
    def __init__(self,heading):
        self._heading = heading
        
    @property
    def velocity_heading(self):
        return self._heading
    def __str__(self):
        return f'"x":{self._heading}'
    def __repr__(self):
        return self.__str__()    
    
class Speed:
    def __init__(self, speed):
        self._speed = speed
    @property
    def speed(self):
        return self._speed
    def __str__(self):
        return f'"x":{self._speed}'
    def __repr__(self):
        return self.__str__()    
    
class Length:
    def __init__(self, length):
        self._length = length
    @property
    def length(self):
        return self._length
    def __str__(self):
        return f'"length":{self._length}'
    def __repr__(self):
        return self.__str__()    
    
class Width:
    def __init__(self,width):
        self._width = width
    @property
    def width(self):
        return self._width 
    def __str__(self):
        return f'"x":{self._width}'
    def __repr__(self):
        return self.__str__()    
    
    
class Length:
    def __init__(self, length):
        self._length = length
    @property
    def length(self):
        return self._length
    def __str__(self):
        return f'"length":{self._length}'
    def __repr__(self):
        return self.__str__() 
class Timestamp:
    def __init__(self, timestamp):
        self._timestamp  = timestamp
    @property
    def timestamp(self):
        return self._timestamp
    def __str__(self):
        return f'"x":{self._timestamp}'
    def __repr__(self):
        return self.__str__()    
    
class Velocity:
    def __init__(self, vel):
        self.vel_xx = vel['x']
        self.vel_yy = vel['y']
        self.vel_zz = vel['z']  
    @property
    def x(self):
        return self.vel_xx   
    @property
    def y(self):
        return self.vel_yy    
    @property
    def z(self):
        return self.vel_zz
    def __str__(self):
        return f'["x":{self.vel_xx}, "y":{self.vel_yy}, "z":{self.vel_zz}]'
    def __repr__(self):
        return self.__str__() 
class Position:
    def __init__(self, pos):
        self.xx = pos['x']
        self.yy = pos['y']
        self.zz = pos['z'] 
    @property
    def left_top(self):
        return self.xx, self.yy  
    @property
    def x(self):
        return self.xx   
    @property
    def y(self):
        return self.yy    
    @property
    def z(self):
        return self.zz
    def __str__(self):
        return f'["x":{self.xx}, "y":{self.yy}, "z":{self.zz}]'
    def __repr__(self):
        return self.__str__() 



#lane_point.p.x
#lane_point = [[1,2,3],[4,5,6]]
class Fea:
    
    def __init__(self, position,timestamp,width,speed,vel_heading,length):
        self._position = position
        self._timestamp = timestamp
        self._width = width
        self._speed = speed
        self._velocity_heading = vel_heading
        self._length = length
    @property
    def position(self):
        return self._position
    @property
    def velocity_heading(self):
        return self._velocity_heading
    @property
    def timestamp(self):
        return self._timestamp
    @property
    def width(self):
        return self._width
    @property
    def length(self):
        return self._length   
    @property
    def speed(self):
        return self._speed
       
    def __str__(self):
        return f'"pos":{self._pos},"timestamp":{self._timestamp},"width":{self._width},"speed":{self._speed},"velocity_heading":{self._velocity_heading}'
    def __repr__(self):
        return self.__str__()

#feature_seq= [{"heading":11,"width":1,"relative_l":23,"pos":{'x':2,'y':3,'z':4},"timestamp":2.22},{"timestamp":2.22,"heading":11,"width":1,"relative_l":23,"pos":{'x':2,'y':3,'z':4}},{"timestamp":2.22,"heading":11,"width":1,"relative_l":23,"pos":{'x':12,'y':13,'z':14}},{"timestamp":2.22,"heading":11,"width":1,"relative_l":23,"pos":{'x':2.1,'y':3.1,'z':4.1}}]
#fea_seq=[]
'''for fea_id, fea in enumerate(feature_seq):
    print(fea['pos'])
    pos = Pos(fea['pos'])
    timestamp = Timestamp(fea['timestamp'])
    width = Width(fea['width'])
    speed = Speed(fea['relative_l'])
    #pos = Pos(fea['pos'])
    #fea['pos']
    fea = Fea(pos,timestamp,width,speed,vel_heading)
    fea_seq.append(fea)
print(fea_seq)
print(fea_seq[0])'''