#coding=utf-8

"""
predection.config.configuator
"""

import re
import os
import sys
import yaml
import torch
from logging import getLogger

class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.
    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
   
    """    
    def __init(self, model=None,dataset=None):
        #config_dict is yaml
        self._init_parameters_category() #初始化参数类别
        self.yaml_loader = self._build_yaml_loader()#use yaml_loader load the yaml_file
        self.final_config_dict = self._load_config_files(config_file_list)
        
        #model_name, model_class,dataset_name
        
       
        self._init_device()  
    
    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Dataset'] = dataset_arguments
    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _load_config_files(self, file_list): #配置文件
        final_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    final_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return final_config_dict
    
    
    
    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")#final




    def __setitem__(self, key, value):# =
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):#[]
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):#in
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict
    
    def __str__(self):#print, str()
        args_info = ''
        for category in self.final_config_dict:
            args_info += category + ' Hyper Parameters: \n'
            args_info += '\n'.join([
                "{}={}".format(arg, value) for arg, value in self.final_config_dict.items()
                if arg in self.final_config_dict[category]
            ])
            args_info += '\n\n'
        return args_info
    ##############repr()：改变对象的字符串显示, %r默认调用__repr__()方法，如果是字符串会默认加上
    def __repr__(self):
        return self.__str__()
