#coding=utf-8

import pandas as pd
import numpy as np
import math
import pickle
import os
import sys

from utils.apollo_prediction.data_pipelines.common.configure import parameters
from utils.common.Conversion import WorldCoordToObjCoord
from utils.common.Helper import mod2pi

OBSTACLE_FEATURE_SIZE    = 23 + 5 * 9  #障碍物特征长度  23  +  5*9  =68
INTERACTION_FEATURE_SIZE = 8           #交叉路口特征长度    8
SINGLE_LANE_FEATURE_SIZE = 4           #单车道特征长度    4
LANE_POINTS_SIZE         = 20          #车道点 长度       20
EPS = np.finfo(float).eps

def computeFeatures(df):
    #添加了5个feature: index, lane_feature_not_empty(true, false), lane_segment_not_empty(true,false), curr_lane_segments_not_empty(true,false), obs_history_gt1
    def calc_curr_lane_segments_num(x):
        curr_lane_segments = set()
        for lane_sequence in x['lane_graph']:
            if lane_sequence['vehicle_on_lane']:
                for lane_segment in lane_sequence['lane_segment']:
                    curr_lane_segments.add(lane_segment['lane_id'])
        return len(curr_lane_segments) > 0
    
    
    #此处主要是去重
    df['index'] = df[["id","timestamp"]].apply(lambda x: f"{str(int(x['id']))}@{x['timestamp']:.3f}",axis = 1)#增加index 将 index = id@timestamp
    print("原始data:",df.shape) #16*21
    df.drop_duplicates(subset=["id","timestamp"],inplace=True)# 去除重复值
    print("去除重复值后的data:",df.shape)
    df["lane_feature_not_empty"] = df["lane"].apply(lambda x: True if ("lane_feature" in x) and (x["lane_feature"] != None) else False) #增加"lane_feature_not_empty"
    df["lane_segment_not_empty"] = df["lane"].apply(lambda x: True if ("lane_graph" in x) 
                                                    and (x["lane_graph"])
                                                    and ("lane_segment" in x["lane_graph"][0])
                                                    and (x["lane_graph"][0]["lane_segment"]) else False) #增加"lane_segment_not_empty"
    df["curr_lane_segments_not_empty"] = df["lane"].apply(calc_curr_lane_segments_num)  #增加"curr_lane_segments_not_empty"
    id_cnt = df['id'].value_counts()
    df['obs_history_gt1'] = df['id'].apply(lambda x: True if x in id_cnt[id_cnt > 1].index else False)#是否至少有1帧历史数据
    df = df[df["lane_feature_not_empty"] & df["lane_segment_not_empty"] & 
            df["curr_lane_segments_not_empty"] & df['obs_history_gt1']]#都是为了去重，去空等操作
    
    print("去除lane_feature和lane_segment为空后的data:",df.shape)  
     
    if len(df) != 0: 
        #开始设置障碍物特征， 车道特征
        df_obs = SetObstacleFeatureValues(df)   #23+5*9=68       障碍物的   (16,68)
        df_lane = SetLaneFeatureValues(df)   #4*20 =80            (size_n,80)
        # print("df_lane shape:",df_lane.shape,"df_obs shape:",df_obs.shape)
        return df_obs.merge(df_lane, how="right", left_index=True, right_index = True)
    else:
        print('有效数据长度为0！！！')

def SetObstacleFeatureValues(df):
    '''
    23 +5*9  =68
    输入：1个obstacle的DataFrame
    '''  
    # print("df shape before obs feature:",df.shape) 21 + 8
    for col,v_key in zip(["theta", "lane_l", "dist_lb", "dist_rb", "lane_turn_type"],
                   ["angle_diff", "lane_l", "dist_to_left_boundary", "dist_to_right_boundary", "lane_turn_type"]):
        df[col] = df["lane"].apply(lambda x: x["lane_feature"][v_key]) #将lane_feature下的字典拿出来  有重复，去除重复添加4个
    
    #df.keys 为29个  theta:偏航角，
    res = [] 
    calc_base_cols = ["timestamp","theta", "lane_l", "dist_lb", "dist_rb", "lane_turn_type","speed"] #vel_history_y进行计算
    base_cols_map = {v:i for i,v in enumerate(calc_base_cols)}
    ##此处可以进行切片操作 
    max_item_list_len = 70
    seq_start = 0
    cur_res_index  = []
    for i in range(len(df[calc_base_cols])):
        if i - seq_start > max_item_list_len:
            seq_start += 1        
        w = df[seq_start:i+1][calc_base_cols]
        cur_res_index.append(slice(seq_start,i+1)) 
        res.append(df[cur_res_index][calc_base_cols])
    for index in cur_res_index:
        res.append(cur_res.index)
    for i in range(len(df[calc_base_cols])):
        if i - seq_start > max_item_list_len:
            seq_start += 1        
        w = df[seq_start:i+1][calc_base_cols]
        cur_res = np.array([w["timestamp"], w["theta"], w["lane_l"], w["dist_lb"],
                             w["dist_rb"], w["lane_turn_type"], w["speed"]])   
        start_timestamp = w["timestamp"].iloc[-1] - parameters['feature']['max_history_time'] #取当前点前7s内的数据
        cur_res = cur_res[:,cur_res[0,:] >= start_timestamp] #只取特定时间
        res.append(cur_res)
        #start_timestamp = w["timestamp"].iloc[-1] - parameters['feature']['max_history_time']
        
        pass
    #对数据进行切片处理，滑动最大窗口为7s(7*10=70个)  res为进行切片好的数据
    for w in df[calc_base_cols].rolling(2, min_periods=1):  #window=2  step=1,启动为第一行,至少一个
        if len(w) == 2:
            cur_res = np.append(res[-1][:],np.array([w["timestamp"], w["theta"], w["lane_l"], w["dist_lb"],
                                 w["dist_rb"], w["lane_turn_type"], w["speed"]])[:,-1].reshape(-1, 1), axis = 1) #拼接 (7,2)
            # cur_res = cur_res[:,-parameters['feature']['maximum_historical_frame_length']:]  #最多50帧
            # obs_feature_history_start_time最多8秒,但是max_history_time只是7秒的帧数，所以简化为一个条件（50帧的条件注释）
            start_timestamp = w["timestamp"].iloc[-1] - parameters['feature']['max_history_time'] #取当前点前7s内的数据
            cur_res = cur_res[:,cur_res[0,:] >= start_timestamp] #只取特定时间
            res.append(cur_res)
        elif len(w) == 1:#第一行   (7,1)
            res.append(np.array([w["timestamp"], w["theta"], w["lane_l"], w["dist_lb"],
                                 w["dist_rb"], w["lane_turn_type"], w["speed"]]))
    #'middle_res', 'hist_size'
    df['middle_res'] = [ele[:,::-1] for ele in res] #16个 逆序，   存放几个list，每个值为：len(calc_base_cols) * hist_size 的ndarray，
                                                    # 当前时刻放到最前面, 当前时刻最前,逆序排序
    df['hist_size'] = df['middle_res'].apply(lambda x: x.shape[1]) #历史数据长度
    #利用切片好的middle_re数据进行相关处理
    
    # Generate obstacle features生成障碍物的特征    23个(障碍物在车道上的变化情况)  curr_size=5(5帧)
    curr_size = 5
    df = generate_obs_feature(df, curr_size, base_cols_map)  #16*23
    
    #5*9=45      9个hist_res_cols   障碍物的历史位置，速度，加速度
    hist_res_cols = ['has_history','pos_history_x', 'pos_history_y','vel_history_x', 'vel_history_y','acc_history_x', 
                          'acc_history_y', 'vel_heading_history', 'vel_heading_changing_rate_history']  #9个  
    #是否有历史，位置(x,y)，速度(x,y)，加速度(x,y)，航向角(x,y)
    hist_res_map = {v:i for i,v in enumerate(hist_res_cols)}#h:0,p_x:1,p_y:2,v_x:3,v_y:4,a_x:5,a_y:6,v_h:7,v_h_r:8
    hist_size = parameters['feature']['cruise_historical_frame_length'] #取过去多少帧  最多为5帧  5
    hist_res = []
    #此处主要变化参考坐标系:以最新的障碍物位置为参考点，
    for w in df.rolling(5, min_periods=1):#window=5, step=1至少有一个数据1,2,3,4,5,5,5,5,5,5,5,5,5,5,5,5
        curr_hist_res = np.hstack((np.ones([hist_size, 1]), np.zeros([hist_size, len(hist_res_cols)-1])))#(5,9) 第一列都为1， has_history在第一个 为1
        prev_timestamp = w.iloc[-1]['timestamp'] #取最后一个为参考
        curr_timestamp = w.iloc[-1]['timestamp']
        obs_curr_pos = (w.iloc[-1]['position']['x'], w.iloc[-1]['position']['y']) #初始坐标
        obs_curr_heading = w.iloc[-1]['velocity_heading']
        
        for j in range(len(w)):
            if j > 0 and w.iloc[-(j+1)]['timestamp'] < (curr_timestamp - parameters['feature']['obs_feature_history_start_time']):
                break#j=1  在8s之前
            
            if j != 0 and curr_hist_res[j-1, hist_res_map['has_history']] == 0: #has_history
                curr_hist_res[j, hist_res_map['has_history']] = 0  #j != 0
                continue
            if w.iloc[-(j+1)]['position']:  #pos
                curr_hist_res[j, [hist_res_map['pos_history_x'], hist_res_map['pos_history_y']]] = WorldCoordToObjCoord(
                        (w.iloc[-(j+1)]["position"]['x'], w.iloc[-(j+1)]["position"]['y']), obs_curr_pos, obs_curr_heading)
            else:
                curr_hist_res[j, hist_res_map['has_history']] = 0
                    
            if w.iloc[-(j+1)]['velocity']:  #vec
                vel_end = WorldCoordToObjCoord(
                    ((w.iloc[-(j+1)]["velocity"]['x'], w.iloc[-(j+1)]["velocity"]['y'])), obs_curr_pos, obs_curr_heading)
                vel_begin = WorldCoordToObjCoord((0.0, 0.0), obs_curr_pos, obs_curr_heading)
                curr_hist_res[j, [hist_res_map['vel_history_x'], hist_res_map['vel_history_y']]] = [vel_end[0] - vel_begin[0],
                                                                                            vel_end[1] - vel_begin[1]]
            else:
                curr_hist_res[j, hist_res_map['has_history']] = 0   
            
            if w.iloc[-(j+1)]['acceleration_vector']:  #acc
                acc_end = WorldCoordToObjCoord(
                    (w.iloc[-(j+1)]["acceleration_vector"]['x'], w.iloc[-(j+1)]["acceleration_vector"]['y']), obs_curr_pos, obs_curr_heading)
                acc_begin = WorldCoordToObjCoord((0.0, 0.0), obs_curr_pos, obs_curr_heading)
                curr_hist_res[j, [hist_res_map['acc_history_x'], hist_res_map['acc_history_y']]] = [acc_end[0] - acc_begin[0],
                                                                                            acc_end[1] - acc_begin[1]]
            else:
                curr_hist_res[j, hist_res_map['has_history']] = 0        
                # curr_hist_res[j,:] = []
            
            if w.iloc[-(j+1)]['velocity_heading'] > -math.pi:    #heading 
                curr_hist_res[j, hist_res_map['vel_heading_history']] = mod2pi(w.iloc[-(j+1)]['velocity_heading'] - obs_curr_heading)
                if j != 0:
                    curr_hist_res[j, hist_res_map['vel_heading_changing_rate_history']] = (
                        curr_hist_res[j-1, hist_res_map['vel_heading_history']] - curr_hist_res[j, hist_res_map['vel_heading_history']])/(
                        EPS + w.iloc[-(j+1)]['timestamp'] - prev_timestamp)
                    prev_timestamp = w.iloc[-(j+1)]['timestamp']
            else:
                curr_hist_res[j, hist_res_map['has_history']] = 0        
        hist_res.append(curr_hist_res)
        
    df['middle_hist_res'] = hist_res  #(16,5,9)
    
    # print("data shape:",df.shape)
    hist_feature_cols  = [j + "-" + str(i) for i in  range(5) for j in hist_res_cols] #5*9=45
    df_obs = df['middle_hist_res'].apply(lambda x: x.reshape(-1)).apply(pd.Series, index = hist_feature_cols) #(16,45)
    df = pd.concat([df, df_obs],axis = 1,copy=False)
    print("增加history feature后data shape:",df.shape)
    
    feature_cols = ['index','theta_filtered','theta_mean','theta_filtered - theta_mean','angle_diff','angle_diff_rate',
                    'lane_l_filtered','lane_l_mean','lane_l_filtered - lane_l_mean','lane_l_diff','lane_l_diff_rate',
                    'speed_mean','acc','jerk','dist_lbs.front','dist_lb_rate','dist_lb_rate_curr','dist_rbs.front',
                    'dist_rb_rate','dist_rb_rate_curr','lane_turn_types.front 0','lane_turn_types.front 1',
                    'lane_turn_types.front 2','lane_turn_types.front 3'] + hist_feature_cols
    return df[feature_cols].set_index('index')  #only 23+5*9=68

def generate_obs_feature(df, curr_size, base_cols_map):
    def ComputeMean(arr,start,end):
        '''
        输入是一个1维array
        curr_size=5 :"theta"(angle_diff), "lane_l", "dist_lb", "dist_rb", "lane_turn_type"
        {'timestamp': 0, 'theta': 1, 'lane_l': 2, 'dist_lb': 3, 'dist_rb': 4, 'lane_turn_type': 5, 'speed': 6}
        数据是 7*n  (n不一样)
        output 23个
        '''
        end = min(end, len(arr))  #最多取5
        return np.mean(arr[start:end]) if end > start else 0
    #降序排列
    #theta_filtered, theta_mean,  'theta_filtered - theta_mean',  'lane_l_filtered', 'lane_l_mean','lane_l_filtered - lane_l_mean','speed_mean'
    df['theta_filtered'] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['theta'],:], 0, curr_size)) #theta过滤  最多为5帧最近的均值
    df['theta_mean'] = df[["middle_res","hist_size"]].apply(lambda x: ComputeMean(x["middle_res"][base_cols_map['theta'],:], 
                                                                                    0, x['hist_size']), axis = 1)  #有记录的求平均， 不过滤
    df['theta_filtered - theta_mean'] = df['theta_filtered'] - df['theta_mean']  #最近5帧均值减去有记录的均值的差值
    

    df['lane_l_filtered'] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['lane_l'],:], 0, curr_size)) #lane_l过滤 最多5帧的最近的均值
    df['lane_l_mean'] = df[["middle_res","hist_size"]].apply(lambda x: ComputeMean(x["middle_res"][base_cols_map['lane_l'],:], 
                                                                                    0, x['hist_size']), axis = 1)  #有记录的lane_l求平均
    df['lane_l_filtered - lane_l_mean'] = df['lane_l_filtered'] - df['lane_l_mean']  #差值
    df['speed_mean'] = df[["middle_res","hist_size"]].apply(lambda x: ComputeMean(x["middle_res"][base_cols_map['speed'],:],
                                                                                    0, x['hist_size']), axis = 1)  #求speed平均
    
    
    
    df["time_diff"] = df["middle_res"].apply(lambda x: x[base_cols_map["timestamp"],:][0] - x[base_cols_map["timestamp"],:][-1]) #时间差值
    
    #"dist_lbs.front", "dist_lb_rate"
    df["dist_lbs.front"] = df["middle_res"].apply(lambda x: x[base_cols_map["dist_lb"],:][0]) #距离到左边界的第一个值（最新的值）
    df["dist_lb_rate"] = df.apply(lambda x: 
                                    (x['middle_res'][base_cols_map['dist_lb'],:][0] - x['middle_res'][base_cols_map['dist_lb'],:][-1])/(
                                        x['middle_res'][base_cols_map['timestamp'],:][0] - x['middle_res'][base_cols_map['timestamp'],:][-1]
                                    ) if x["hist_size"] > 1 else 0, axis = 1)   #最后一个和第一个到左边界的距离的差值 除以时间差值 得到变化率
    df["dist_rbs.front"] = df["middle_res"].apply(lambda x: x[base_cols_map["dist_rb"],:][0]) #距离到右边界的第一个值
    df["dist_rb_rate"] = df.apply(lambda x: 
                                    (x['middle_res'][base_cols_map['dist_rb'],:][0] - x['middle_res'][base_cols_map['dist_rb'],:][-1])/(
                                        x['middle_res'][base_cols_map['timestamp'],:][0] - x['middle_res'][base_cols_map['timestamp'],:][-1]
                                    ) if x["hist_size"] > 1 else 0, axis = 1)   #最后一个和第一个到右边界的距离的差值 除以时间差值 得到变化率
    df["delta_t"] = df.apply(lambda x: x["time_diff"]/(x['hist_size'] - 1)  if x['hist_size'] > 1 else 0, axis = 1)  #时间差除以数据长度 的值
    df["angle_curr"] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['theta'],:], 0, curr_size)) #angle_curr=theta_filtered
    df["angle_prev"] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['theta'],:], curr_size, 2 * curr_size)) # 过滤，6-10之间平均值
    
    #'angle_diff'
    df['angle_diff'] = df.apply(lambda x: x["angle_curr"] - x["angle_prev"] if x['hist_size'] >= 2*curr_size else 0 , axis = 1) #过去5帧坐标均值减去过去10帧坐标均值求差值
    
    df["lane_l_curr"] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['lane_l'],:], 0, curr_size))  #lane_l过滤 最多为5个
    df["lane_l_prev"] = df["middle_res"].apply(lambda x: ComputeMean(x[base_cols_map['lane_l'],:], curr_size, 2 * curr_size)) #过滤，最多为10个
    
    #'lane_l_diff'
    df['lane_l_diff'] = df.apply(lambda x: x["lane_l_curr"] - x["lane_l_prev"] if x['hist_size'] >= 2*curr_size else 0, axis = 1)#差值
    
    #'angle_diff_rate', "lane_l_diff_rate"
    df["angle_diff_rate"] = df.apply(lambda x: x['angle_diff']/(x['delta_t']*curr_size) if x['delta_t'] > EPS else 0, axis=1)#过去5减去过去10之差除以帧时间间隔
    df["lane_l_diff_rate"] = df.apply(lambda x: x['lane_l_diff']/(x['delta_t']*curr_size) if x['delta_t'] > EPS else 0, axis=1)
#过去5减去过去10帧之差除以帧时间间隔
    df['speed_1st_recent'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['speed'],:], 0, curr_size)
                                                    if x["hist_size"] >= 3*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1) #1*5  5帧速度均值   
    df['speed_2nd_recent'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['speed'],:], curr_size, 2*curr_size)
                                                    if x["hist_size"] >= 3*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1)  #2*5  10帧均值
    df['speed_3rd_recent'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['speed'],:], 2*curr_size, 3*curr_size) 
                                                    if x["hist_size"] >= 3*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1)   #3*5  15帧均值
    #'acc', 'jerk'
    df['acc'] = df.apply(lambda x: (x['speed_1st_recent'] - x['speed_2nd_recent'])/(curr_size*x['delta_t'])
                                    if x["hist_size"] >= 3*curr_size and x['delta_t'] > EPS else 0,
                                    axis = 1) #加速度
    df['jerk'] = df.apply(lambda x: (x['speed_1st_recent'] - 2*x['speed_2nd_recent'] + x['speed_3rd_recent'])/
                                    (curr_size * curr_size * x['delta_t'] * x['delta_t'])
                                    if x["hist_size"] >= 3*curr_size and x['delta_t'] > EPS else 0,
                                    axis = 1)#加速度变化率
    
    df['dist_lb_curr'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['dist_lb'],:], 0, curr_size)
                                                    if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1) #帧大于等于10时候，求均值，否则为None ，5帧进行求均值
    df['dist_lb_prev'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['dist_lb'],:], curr_size, 2*curr_size)
                                                    if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1) #帧大于等于10时候，求均值，否则为None   10帧进行求均值
    #'dist_lb_rate_curr','dist_rb_curr','dist_rb_prev','dist_rb_rate_curr'
    df['dist_lb_rate_curr'] = df.apply(lambda x: (x['dist_lb_curr'] - x['dist_lb_prev'])/(curr_size * x['delta_t'])
                                                if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else 0,
                                                axis = 1)#之差除以帧时间间隔，不满足条件，值默认为0
    
    df['dist_rb_curr'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['dist_rb'],:], 0, curr_size)
                                                    if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1)
    df['dist_rb_prev'] = df.apply(lambda x: ComputeMean(x['middle_res'][base_cols_map['dist_rb'],:], curr_size, 2*curr_size)
                                                    if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else None,
                                                    axis = 1)
    df['dist_rb_rate_curr'] = df.apply(lambda x: (x['dist_rb_curr'] - x['dist_rb_prev'])/(curr_size * x['delta_t'])
                                                if x["hist_size"] >= 2*curr_size and x['delta_t'] > EPS else 0,
                                                axis = 1)
    
    #'lane_turn_types.front 0','lane_turn_types.front 1','lane_turn_types.front 2','lane_turn_types.front 3'
    df['lane_turn_types.front 0'] = df['lane_turn_type'].apply(lambda x: int(x == 0))  #no_turn
    df['lane_turn_types.front 1'] = df['lane_turn_type'].apply(lambda x: int(x == 1))  #left_turn
    df['lane_turn_types.front 2'] = df['lane_turn_type'].apply(lambda x: int(x == 2))  #right_turn
    df['lane_turn_types.front 3'] = df['lane_turn_type'].apply(lambda x: int(x == 3))  #u_turn
    
    return df

def SetLaneFeatureValues(df):
    #pos_x,pos_y, relative_heading(航向角), kappa(曲率)
    def calc_lane_point_feature(x):
        feature_values = []
        x_seq = x["lane_sequence"]
        for lane_seg in x_seq["lane_segment"]:
            if len(feature_values) >= SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE:  #单个车道点特征大小4()，车道点最大个数共20
                break
            for lane_point in lane_seg["lane_point"]:
                if len(feature_values) >= SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE:
                    break
                relative_s_l = WorldCoordToObjCoord(input_world_coord = (lane_point["position"]["x"],lane_point["position"]["y"]),
                                                    obj_world_coord = (x["position"]["x"], x["position"]["y"]),
                                                    obj_world_angle = x["velocity_heading"])#lane_point相对于当前障碍物坐标系的坐标
                
                relative_ang = mod2pi(lane_point['heading'] - x["velocity_heading"])#lane_point相对于当前障碍物坐标系的朝向

                feature_values.append(relative_s_l[1])
                feature_values.append(relative_s_l[0])
                feature_values.append(relative_ang)
                feature_values.append(lane_point['kappa'])#曲率
                #feature_values:pos_y, pos_x, ang, kappa
        if len(feature_values) < SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE:#80
            print(f"{int(x['id'])}@{x['timestamp']:.3f}:len of lane feature_values:{len(feature_values)}")
        
        # handle the dimension of feature_values is not enough        
        while (len(feature_values) >= 8) and (len(feature_values) < SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE):
            feature_values.append(2*feature_values[-4] - feature_values[-8]) #relative_l
            feature_values.append(2*feature_values[-4] - feature_values[-8]) #relative_s
            feature_values.append(feature_values[-4]) #relative_ang
            feature_values.append(0) #kappa
            
        return feature_values
    
    df = df[["id","timestamp",'index',"position","velocity_heading","lane"]].copy()#拿出6列的数据
    df["lane_sequence_len"] = df["lane"].apply(lambda x: len(x["lane_graph"]))
    df["lane_sequence"] = df["lane"].apply(lambda x: x["lane_graph"][0])#拿第0个lane_seq
    df_new = pd.DataFrame()#需要扩充的df
    for lane_seq_len in range(2,df["lane_sequence_len"].max()+1):
        for i in range(lane_seq_len-1):# i 表示增加i+1行
            tmp = df[df["lane_sequence_len"] == lane_seq_len].copy()
            tmp["lane_sequence"] = tmp["lane"].apply(lambda x: x["lane_graph"][i+1])#将第1个lane_seq给lane_seq
            df_new = df_new.append(tmp,ignore_index=True)
    df = pd.concat([df,df_new],axis=0)
    df = df.sort_values(by="index", kind= 'stable') #stable确保排序是稳定的
    
    df["lane_features"] = df[["id","timestamp","position","lane_sequence","velocity_heading"]].apply(lambda x: 
                                                                calc_lane_point_feature(x),axis=1) #lane_feature  20*4=80
    feature_cols = [j + "-" + str(i) for i in range(1,LANE_POINTS_SIZE+1) for j in ["lane_point_realtive_l",
                                            "lane_point_relative_s","lane_point_ang","lane_point_kappa"]]  #l:y, s:x
    print("data shape before drop insufficent lane_point feature:", df.shape)
    not_enough_lane_features = df['lane_features'].apply(lambda x:len(x) != (SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE))
    df.loc[not_enough_lane_features,'lane_features'] =  df.loc[not_enough_lane_features,'lane_features'].apply(
            lambda x:x.extend([None]*(SINGLE_LANE_FEATURE_SIZE * LANE_POINTS_SIZE - len(x))))
    # print("data shape before after insufficent lane_point feature:", df.shape)
    print("data shape:", df.shape)
    #lane_point_realtive_l ,lane_point_relative_s, lane_point_ang,lane_point_kappa
    df_feature = df['lane_features'].apply(pd.Series, index = feature_cols) 
    return pd.concat([df['index'], df_feature], axis = 1).set_index('index')



