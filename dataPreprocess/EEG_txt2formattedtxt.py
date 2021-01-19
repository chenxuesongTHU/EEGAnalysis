#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   EEG_txt2fif  
@Time        :   2021/1/2 2:11 下午
@Author      :   Xuesong Chen
@Description :
"""

# todo:  STEP 1
import time

from numpy import long

from common.constants import *
import pandas as pd
import pdb
from datetime import datetime
from tqdm import tqdm


def date2timestamp(date):
    if len(date) != 26:
        datetime_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        datetime_obj = datetime.strptime(date, "%Y-%m-%d %H-%M-%S.%f")
    local_timestamp = long(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)

    # obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return local_timestamp
# date2timestamp('2021-01-12 14-22-36.220278')

data_path = '../data/ExportedData'
# todo: 当有新被试时
username = user_name_list[16]
print(username)

if username in mobile_data_users:
    file = pd.read_csv(f'{data_path}/{username}.txt',
                       names=['timestamp', 'num',
                              '1channel_1point', '2channel_1point', '3channel_1point', '4channel_1point', '5channel_1point',
                              '1channel_2point', '2channel_2point', '3channel_2point', '4channel_2point', '5channel_2point',
                              '1channel_3point', '2channel_3point', '3channel_3point', '4channel_3point', '5channel_3point',
                              '1channel_4point', '2channel_4point', '3channel_4point', '4channel_4point', '5channel_4point',
                              '1channel_5point', '2channel_5point', '3channel_5point', '4channel_5point', '5channel_5point',
                              'oxygen', 'rate', 'x', 'y', 'z'],
                       # nrows=10,
                       )
else:
    file = pd.read_csv(f'{data_path}/{username}.txt',
                       names=['timestamp',
                              '1channel_1point', '2channel_1point', '3channel_1point', '4channel_1point', '5channel_1point',
                              '1channel_2point', '2channel_2point', '3channel_2point', '4channel_2point', '5channel_2point',
                              '1channel_3point', '2channel_3point', '3channel_3point', '4channel_3point', '5channel_3point',
                              '1channel_4point', '2channel_4point', '3channel_4point', '4channel_4point', '5channel_4point',
                              '1channel_5point', '2channel_5point', '3channel_5point', '4channel_5point', '5channel_5point',
                              'empty', 'oxygen', 'rate', 'x', 'y', 'z'],
                       # nrows=10,
                       )

time_stamp_list = []
channel_1_list = []
channel_2_list = []
channel_3_list = []
channel_4_list = []
channel_5_list = []
blood_oxygen_list = []
heart_rate_list = []
x_list = []
y_list = []
z_list = []

for row in tqdm(file.iterrows()):
    # key = ''
    for point_idx in range(1, 6):
        for channel_idx in range(1, 6):
            key = f'{channel_idx}channel_{point_idx}point'
            list_name = f'channel_{channel_idx}_list'
            eval(list_name).append(row[1][key])
        date = row[1]['timestamp']
        date = experiment_time[username] + date
        timestamp = date2timestamp(date)
        time_stamp_list.append(timestamp)
        blood_oxygen_list.append(row[1]['oxygen'])
        heart_rate_list.append(row[1]['rate'])
        x_list.append(row[1]['x'])
        y_list.append(row[1]['y'])
        z_list.append(row[1]['z'])

output_df = pd.DataFrame({
    'time': time_stamp_list,
    'Af8-O2': channel_1_list,
    'Fp2-O2': channel_2_list,
    'Fp1-O2': channel_3_list,
    'Af7-O2': channel_4_list,
    'O1-O2': channel_5_list,
    'blood_oxygen': blood_oxygen_list,
    'heart_rate': heart_rate_list,
    'x': x_list,
    'y': y_list,
    'z': z_list,
})

output_df.to_csv(f'../data/EEG/{username}.csv', index=False)
print(file)
print()