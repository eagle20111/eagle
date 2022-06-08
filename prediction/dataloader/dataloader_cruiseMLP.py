#coding=utf-8
import torch
import os
import pickle
import pandas as pd
import numpy as np
import sys

import argparse
# sys.path.append("..")#根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from prediction.utils.dataparser import parse_data, read_write_file
from prediction.utils.compute_features_cruiseMLP import computeFeatures
from prediction.utils.apollo_prediction.data_pipelines.common.online_to_offline import LabelGenerator
from prediction.utils import init_seed
from prediction.utils.logger import init_logger
import logging
from logging import getLogger

# from utils.apollo_prediction.data_pipelines.data_preprocessing.features_labels_utils import CombineFeaturesAndLabels, MergeCombinedFeaturesAndLabels

#data_dir = "G:/wingide/AI2/crusieMLP/data"
#data_dir = "./data"
# data_dir = "D:/Code/project/data"  G:/wingide/AI2/crusieMLP/data
# model_dir = os.path.join(os.getcwd(),'Prediction_ML/model')
#model_dir = '../model'


def CombineFeaturesAndLabels(data_feature, data_label):
    id_obs_list = []
    obs_label_list = []
    for id_obs, feature_seq in data_label.items():
        for ind, obs_label in feature_seq.items():
            id_obs_list.append(id_obs)#16056
            obs_label_list.append(obs_label)#obstacle label  16056
    df_label = pd.DataFrame(obs_label_list, index=id_obs_list,columns=["y_label",
                                    'y_time_to_lane_center','y_time_to_lane_edge',"is_vehicle_on_lane"])
    if len(data_feature)== len(df_label) and all(data_feature.sort_index().index == 
                                                 df_label.sort_index().index):
        train_data = data_feature.merge(df_label, left_index=True, right_index=True)
        train_data.dropna(inplace=True) #去除缺失值
        return train_data
    else:
        if len(data_feature) != len(df_label):
            print(f"feature len:{len(data_feature)} != label len:{len(df_label)}")
            print("不一致的index：","\n",set(data_feature.index).symmetric_difference(set(df_label.index)))
        else:
            print(data_feature.sort_index().index[data_feature.sort_index().index != df_label.sort_index().index])
        raise
    
def get_train_data(data_dir, save_parsed_data = True, save_feature_file=False):
    
    #arg parse
    parser = argparse.ArgumentParser(description='crusieMLP data_preprocessing')
    parser.add_argument('--model', type=str, default='FCNN_CNN1D',help='FCNN_CNN1D|FullConn_NN')
    parser.add_argument('--seed', type=int, default= 1,help='seed') 
    parser.add_argument('--log_state', type=str, default='info',help='l') 
    parser.add_argument('--reproducibility', type=bool, default=True,help='reproducibility')  
    args = parser.parse_args()    
    
    #路径： 原始数据，解析后的数据，特征数据，标签数据，训练数据
    #use original_data to generate parsed_data,feature_data, label_data, train_data
    #path
    original_dirpath = os.path.join(data_dir,"original_data") #original
    parsed_dirpath = os.path.join(data_dir,"parsed_data")  #parsed
    feature_dirpath = os.path.join(data_dir,"feature_data") #feature
    label_dirpath = os.path.join(data_dir,"label_data")  #label
    train_data_dirpath = os.path.join(data_dir,"train_data") #train_data_dirpath
    
    config = {'model':'FCNN_CNN1D','state':'info'}
    #init seed
    init_seed(args.seed, args.reproducibility)
    
    #init_logger
    init_logger(config)
    logger = getLogger()
    
    logger.info(train_data_dirpath)
    #logger.info(train_data_dirpath)参数
    
    
    list_of_original_files = os.listdir(original_dirpath) #list_org_files_name  ...csv
    for file in list_of_original_files:
        full_file_path = os.path.join(original_dirpath, file) #get the path of file
        print("*"*50,os.path.split(full_file_path)[1],"*"*50)
        file_name =  file.split(".")[0]
        if not os.path.isfile(full_file_path):
            print("{} is not a valid file".format(full_file_path))
            continue
        
        # parse data
        save_name_parsed = file_name + ".h5"
        parsed_path = os.path.join(parsed_dirpath, save_name_parsed)#parsed_path
        if not os.path.exists(parsed_path):
            #read_write_file
            data = read_write_file(full_file_path)() #original data  pd_data, 返回pandas数据  4890*20
            #data = read_file(full_file_path) #original data  pd_data, 返回pandas数据  4890*20
            ##1. parse data  也就是将原数据进行简单处理，得到所需要的格式数据
            data = parse_data(data) #parse original data  4890
            if save_parsed_data:
                with open(parsed_path,"wb") as f:
                    pickle.dump(data.to_dict(orient='records'),f)
             
        else:
            with open(parsed_path,"rb") as f:
                data = pd.DataFrame.from_records(pickle.load(f)) 
    #上述都是为了将原数据进行解析预处理
                
                
                
        # generate features
        save_name_feature = file_name + ".h5"
        feature_path = os.path.join(feature_dirpath,save_name_feature)  #feature_path
        if not os.path.exists(feature_path):
            df_feature = pd.DataFrame()  #new df_feature新建一个数据特征pd
            for name,group in data.groupby('id'):  #按照obstacle id进行分组
                print("*"*30, f'id = {name}', "*"*30)
                
                ##compute Feature  148
                df_feature = pd.concat([df_feature, computeFeatures(group)])#df_feature
            print(f"{full_file_path} shape:",df_feature.shape) #(16056, 148)
            if save_feature_file:
                df_feature.to_hdf(feature_path, key = 'data')# df_fature save to feature_path file_name
        else:
            df_feature = pd.read_hdf(feature_path, key = 'data')#file_name
        
        # generate labels && combine features and labels
        save_name_train_data = file_name + ".cruise.h5"
        train_data_path = None 
        for root, dirs, files in os.walk(train_data_dirpath):
            if save_name_train_data in files:
                train_data_path = os.path.join(root,save_name_train_data)
                break
        
        if not train_data_path:
            # generate labels
            train_data_path = os.path.join(train_data_dirpath,save_name_train_data)
            print("Create Label {}".format(parsed_path))
            label_gen = LabelGenerator()#class init
            label_gen.LoadFeaturePBAndSaveLabelFiles(parsed_path)  #feature_dict
            label_dict = label_gen.LabelSingleLane() #4883个
            
            # combine features and labels
            train_data = CombineFeaturesAndLabels(df_feature, label_dict) #df_feature(16056, 148), label_dict:(4883,(4))其中标签4可能有多个值
            train_data.to_hdf(train_data_path, key = 'data',format='table', data_columns = True)#file_name(55189, 152)

