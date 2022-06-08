#coding=utf-8

import datetime
import importlib
import os
import random
import yaml

import numpy as np
import torch

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str

def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn #初始化随机种子

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility再现性，复现性，是否要求
    """
    random.seed(seed) #每次一样
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


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
self.yaml_loader = self._build_yaml_loader()

def _update_internal_config_dict(self, file):
    with open(file, 'r', encoding='utf-8') as f:
        config_dict = yaml.load(f.read(), Loader=self.yaml_loader) #self.yaml_loader
        if config_dict is not None:
            self.internal_config_dict.update(config_dict)
    return config_dict

def _init_device(self):
    use_gpu = self.final_config_dict['use_gpu']  #True
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])  #self.final_config_dict['gpu_id'] = 0
    self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")#final