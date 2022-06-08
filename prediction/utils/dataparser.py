#coding=utf-8
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns

import csv
import pickle
import json


#此处可以修改，以适合多种文件读取
def read_write_file(input_f,method='read'):
    f_ext=input_f.split(".")[-1]

    def pd_csv_read():
        if 'training_data2022-04-18-10-30-18' in input_f:
            data = pd.read_csv(input_f,sep="%",index_col = False,skiprows=[65702])#这行数据保存不完整，可能是bag的问题
        else:
            data = pd.read_csv(input_f,sep="%",index_col = False)  #read csv_file，分隔符为%
        data.rename(lambda x:x.strip(),axis=1,inplace=True)  #去除标题的空格   1column, 0index
        return data    
    def txt_read():
        with open(input_f, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line=line.strip()
                yield line
    def pkl_read():
        with open(input_f, mode='rb') as fin:
            data = pickle.load(fin)
            return data
    def csv_read():
        with open(input_f) as fin: #open(input_f, mode='rb')
            f = csv.reader(fin)
            for i in f:
                yield i       
    def json_read():
        with open(input_f, mode='r', encoding='utf-8') as fin:
            content=fin.read()#使用loads()方法，需要先读文件
            data=json.loads(content)    
            return data
    def txt_write(data):
        with open(input_f, mode='w', encoding='utf-8') as fo:
            fo.write(data)
    def pkl_write(data=[{'data':{'a':1,'b':2}}]):
        
        with open(input_f, mode='wb') as fin:
            pickle.dump(data,fin)
        
    def csv_write():
        with open(input_f, mode='rb') as fin:
            f = csv.reader(fin)
            for i in f:
                yield i       
    def json_write(content):
        with open(input_f, mode='w', encoding='utf-8') as fin:
            json.dump(content, fin)   
            print(f"dump json over")  
            #'csv':csv_read,
    route={'read':{'txt':txt_read,'pkl':pkl_read,'csv':pd_csv_read,'json':json_read},'write':{'txt':txt_write,'pkl':pkl_write,'csv':csv_write,'json':json_write}}
    #read_fn={'txt':txt_read,'pkl':pkl_read,'csv':csv_read,'json':json_read}
    #write_fn={'txt':txt_write,'pkl':pkl_write,'csv':csv_write,'json':json_write}
    #m={'read':read_fn,'write':write_fn}
    assert method in ('read','write'),"method is error"
    #r_w_fn=route[method][f_ext]    
    return route[method][f_ext] #从此处来选择要用哪个函数读取文件

#def read_file(file_path):
    #if 'training_data2022-04-18-10-30-18' in file_path:
        #data = pd.read_csv(file_path,sep="%",index_col = False,skiprows=[65702])#这行数据保存不完整，可能是bag的问题
    #else:
        #data = pd.read_csv(file_path,sep="%",index_col = False)  #read csv_file，分隔符为%
    #data.rename(lambda x:x.strip(),axis=1,inplace=True)  #去除标题的空格   1column, 0index
    #return data

def parse_point_data(point):  #x,y,z
    point = eval(point)#str to list
    return {"x": point[0],"y": point[1], "z": point[2]} #return dict

def parse_tracking_time(x):
    '''小于0.01s或大于4小时的数据认为异常'''
    # x = float(x)
    return 0 if (x < 0.01 or x > 3600*4) else x

def parse_pred_traj(data):
    data["predicted trajectory"] = data["predicted trajectory"].map(lambda x:eval(x))   #81*12
    return data  #共80个  (8*10) 80帧

def parse_lane(data):
    '''将嵌套深度为2维的list转换为需要的形式'''
    def helper(x):
        '''
        输入：x是个list，且长度大于1
        输出：将x[0]的元素作为key，x[1:]的元素作为value。如果x长度为n则输出长度为n-1的list，里面每个元素都是dict
        '''
        if isinstance(x,list) and len(x) > 1 and (not (len(x)==2 and x[1] == [None])):
            res = []
            for ele in x[1:]: #多个数据
                kv = {}
                for k,v in zip(x[0], ele):
                    kv[k] = v
                res.append(kv) #放在这里
            return res
        elif isinstance(x, list) and len(x) == 1:
            print("***空值***") 
            return []
        elif isinstance(x, list) and (len(x)==2 and x[1] == [None]):
            # print("***[None]***")
            return []
        
        #是为了将头和数据，转换为字典格式
    def parse_lane_feature(lane):
        lane = eval("{" + lane + "}") #str to dict 去引号
        for lanefeature in lane.keys(): #cur_lane_fea, nearby_lane_fea, lane_graph
            try:
                #current_lane_feature, nearby_lane_feature使得此处
                lane[lanefeature] = eval(lane[lanefeature]) #去引号
                lane[lanefeature] = helper(lane[lanefeature]) #都是将头去掉，转换为字典
                #lane_graph  lane_graph内部还有需要将头跟数据转换为字典
                if lanefeature == 'lane_graph' and lane[lanefeature]:#后一个条件是避免返回为空值的情况
                    for laneseq in lane[lanefeature]: #有4个，  list格式
                        # print("***",laneseq)
                        for laneseq_k in list(laneseq.keys()):#lanefeature为字典格式
                            if laneseq_k in ['nearby_obstacles',  'path_point'] and laneseq[laneseq_k]: #字典格式nearby_obs, path_point,并且值非0
                                laneseq[laneseq_k] = helper(laneseq[laneseq_k])
                            elif laneseq_k == "lane_segment" and laneseq[laneseq_k]:
                                laneseq[laneseq_k] = helper(laneseq[laneseq_k])
                                if laneseq[laneseq_k]:
                                    for lanesegment in laneseq[laneseq_k]: 
                                        for laneseg_k in list(lanesegment.keys()):
                                            
                                            if laneseg_k == "lane_point" and lanesegment[laneseg_k]: #lane_point还需要进行判断
                                                lanesegment[laneseg_k] = helper(lanesegment[laneseg_k])
                                                if lanesegment[laneseg_k]:
                                                    for lp in lanesegment[laneseg_k]:
                                                        for lanepoint_k in list(lp.keys()):
                                                            if lanepoint_k == "position" and lp[lanepoint_k]: #point position(x,y,x)
                                                                    lp[lanepoint_k] = {                                                       # {
                                                                        "x":lp[lanepoint_k][1][0],
                                                                        "y":lp[lanepoint_k][1][1],
                                                                        "z":lp[lanepoint_k][1][2]}
            except:
                print(lanefeature, "\n", lane[lanefeature],"\n",lane["lane_graph"])
                # print("...exception...")
                raise
        # print(lane)
        
        # 增加lane_feature
        min_heading_diff, min_absol_l = float('inf'), float('inf')  #如何判定
        for l in lane["current_lane_feature"]:  #从current_lane_featue中取heading_diff最小的为lane_feature
            if min_heading_diff > math.pi/4: #90°
            # if the angle diff is large, choose the min angle diff lane
                if l["angle_diff"] < min_heading_diff:
                    lane["lane_feature"] = l #将当前的l赋值给lane_feature
                    min_heading_diff = abs(l["angle_diff"])  #heading =angle航向角
                    min_absol_l = abs(l["lane_l"])#lane_l = min_absol_l
            elif min_absol_l > abs(l["lane_l"]):#// else choose the nearest lane
                lane["lane_feature"] = l
                min_absol_l = abs(l["lane_l"])    #min_absol_最小的绝对的l值
                
        # 解决没有lane_feature或lane_feature为空的情况  nearby_lane_feature没有用
        if (not (("lane_feature" in lane) and (lane["lane_feature"] != None))):
            min_heading_diff, min_absol_l = float('inf'), float('inf')
            for l in lane["nearby_lane_feature"]:
                if min_heading_diff > math.pi/4:
                # if the angle diff is large, choose the min angle diff lane
                    if l["angle_diff"] < min_heading_diff:
                        lane["lane_feature"] = l
                        min_heading_diff = abs(l["angle_diff"])
                        min_absol_l = abs(l["lane_l"])
                elif min_absol_l > abs(l["lane_l"]):#// else choose the nearest lane
                    lane["lane_feature"] = l
                    min_absol_l = abs(l["lane_l"])
        # print("lane_feature" in lane.keys())            
        return lane
    #调用的变量
    map_dic = {"Lane Graph":'lane_graph', 
               "Current Lane Feature":'current_lane_feature',
               "Nearby lane Feature":'nearby_lane_feature',
               "-nan":'None',"nan":"None"}#Current Lane Feature的lane_heading有这样的值
    #Lane_graph, cur_lane_fea, nearby_lane_fea,      -none, nan
    for k,v in map_dic.items():
        data['lane'] = data['lane'].str.replace(k, v)#取代原来的标签
    data["lane"] = data["lane"].map(lambda x: parse_lane_feature(x))  #input:lane
    return data

def parse_data(data):
    '''
   ['ego_x', 'ego_y', 'id',   
   'type', 'length','width', 'height', 
   'speed',  'acceleration', 'theta',
   'velocity_heading', 'is_still','priority',]  20个
   
   'timestamp',
   'position', 'velocity','acceleration_vector',
   'tracking time',
    'predicted trajectory', 
    'lane'
    
    ego_x,ego_y, id, type, length, width,height,speed, acceleration, theta, velocity_heading, is_still, priority,
    timestamp, position,velocity, acceleration_vector,tracking_time, predicted_trajectory, lane
    
    '''
    #parse_data
    data["timestamp"] = data["timestamp"]/1000 #ms转s      timestamp
    
    # 点坐标处理 pos, vec, acc_vec数据处理成字典 (x,y,z)
    point_cols = ["position","velocity","acceleration_vector"]
    for col in point_cols:
        data[col] = data[col].map(lambda x : parse_point_data(x))  #pos, vecl, acc_vec  to list
    
    # tracking time 处理
    data["tracking time"] = data["tracking time"].map(lambda x:parse_tracking_time(x)) #小于0.01s或大于4小时的数据认为异常置为0
    
    # predicted trajectory  data["predicted trajectory"]为字典头加数据  未用到
    data = parse_pred_traj(data)  #pred_traj
    
    # lane 处理 : current_lane_fea,  Nearby_lane_fea, lane_graph
    data = parse_lane(data)  #lane
    
    # 后处理
    # lane graph 为空
    print("以下lane graph为空：")
    print(data[data["lane"].apply(lambda x: x["lane_graph"] == None)])
    
    data = data[data["lane"].apply(lambda x: x["lane_graph"] != None)]   #lane_graph
    
    return data