#coding=utf-8

import argparse

from prediction.dataloader.dataloader_cruiseMLP import get_train_data

if __name__ == "__main__":
    data_dir = "G:/wingide/AI2/prediction/data"
    #data_dir = "./data"
    # data_dir = "D:/Code/project/data"  G:/wingide/AI2/crusieMLP/data
    # model_dir = os.path.join(os.getcwd(),'Prediction_ML/model')
    model_dir = '../model'
    config = dict()
    parser = argparse.ArgumentParser(description='crusieMLP data_preprocessing')
    parser.add_argument('--data_dir', type=str, default=data_dir,help='training data (h5)')
    parser.add_argument('--model_dir', type=str, default=model_dir,help='training data (h5)')
    parser.add_argument('--model', type=str, default='FCNN_CNN1D',help='FCNN_CNN1D|FullConn_NN')
    parser.add_argument('--seed', type=str, default=model_dir,help='seed') 
    parser.add_argument('--log_state', type=str, default='info',help='l') 
    parser.add_argument('--reproducibility', type=str, default=model_dir,help='reproducibility')  
    args = parser.parse_args()
    
    
    get_train_data(data_dir, save_parsed_data = True, save_feature_file=True)    

