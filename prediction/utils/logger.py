#coding=utf-8

"""
prediction.utils.logger
###############################
"""

import logging
import os

from prediction.utils.utils import get_local_time, ensure_dir


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    LOGROOT = './log/'#指定位置
    dir_name = os.path.dirname(LOGROOT) #./log
    ensure_dir(dir_name)#没有路径创建路径

    logfilename = '{}-{}.log'.format(config['model'], get_local_time()) #logfilename文件名 BPR-时间.log

    logfilepath = os.path.join(LOGROOT, logfilename) #log文件路径

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt) #init

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"#state  asc
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath) #fh:FileHandler,文件输出路径
    fh.setLevel(level)#设置等级
    fh.setFormatter(fileformatter)#文件格式

    sh = logging.StreamHandler() #sh:StreamHandler，输出
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[fh, sh])