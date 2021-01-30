#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   utils  
@Time        :   2021/1/9 7:01 下午
@Author      :   Xuesong Chen
@Description :   
"""
import os
from datetime import datetime
from numpy import long
import time
# from constants import *

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    else:
        pass

def date2timestamp(date):
    if len(date) != 26:
        datetime_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        datetime_obj = datetime.strptime(date, "%Y-%m-%d %H-%M-%S.%f")
    local_timestamp = long(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)

    # obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return local_timestamp

def map_sat(x):
    if x > 2:
        return 1
    elif x < 2:
        return 0
    else:
        return -1

# def loop_template():
#     for username in user_name_list:
#         for task_id in FORMAL_TASK_ID_LIST:
#             for page_id in range(1, 7):
#                 page_uid = f'{task_id}_{page_id}'