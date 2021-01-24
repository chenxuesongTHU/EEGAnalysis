#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        :   constants  
@Time        :   2020/12/30 11:57 下午
"""
import numpy as np
import pandas as pd
users = {
    '陈海天': '2016010106',
    '吴伟浩': '2019011220_',
    '吴婉瑄': '2019270058',
    '房晓宇': '2020211288',  # 80
    '王志红': '2021660037',  # 80
    '李兴航': '2020210925',
    '徐晓萌': '2020213873',
    '陈常越': '2019011019',
    '柯丽媚': '2020312519',  # 80  此位置开始引入十字
    '韩瑞瑞': '2016012769',
    '王子诰': '2016011219',  # 80
    '王珑霖': '2016012497',  # 80
    '邢泽明': '2020211099',  # 80
    '李子钰': '2020311319',  # 80
    '何剑宇': '2020310361',  # 80
    '李博达': '2017310560',
    # '吴越飏': '2018311513',
}


mobile_data_users = [
    '2016010106',
    '2019011220_',
    '2019270058',
    '2020211288',
]

experiment_time = {
    '2016010106': '2020-12-30 ',
    '2019011220_': '2021-01-06 ',
    '2019270058': '2021-01-07 ',
    '2020211288': '2021-01-07 ',
    '2021660037': '2021-01-12 ',
    '2020210925': '2021-01-12 ',
    # '2018311513': '2021-01-13 ',
    '2020213873': '2021-01-14 ',
    '2019011019': '2021-01-14 ',
    '2020312519': '2021-01-17 ',
    '2016012769': '2021-01-17 ',
    '2016011219': '2021-01-17 ',
    '2016012497': '2021-01-17 ',
    '2020211099': '2021-01-17 ',
    '2020311319': '2021-01-18 ',
    '2020310361': '2021-01-18 ',
    '2017310560': '2021-01-18 ',
}

user_name_list = list(users.values())

prj_path = '/Users/cxs/IR/EEG/EEGAnalysis'

FORMAL_TASK_ID_LIST = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24]

predictor_list =[
    'LR', 'SVM', 'KNN', 'RF', 'GBDT'
]



models_type = [
    'org_psd_features',
    'org_de_features',
    'org_psd_de_features',
    'eye_movement_AOI_area_gap=1s_duration=2s',  # e.g. 1-3s, 2-4s, ...
    'org_psd_de_features_with_uid',
    'psd_de_AOI_features',
    'max_AOI_psd_de_features',
    'min_AOI_psd_de_features',
    'whole_reading_time_psd_de_feature',
    'dwell_time',
    'eeg_dwell_time',
    'eye_movement_distance',
    'eye_movement_dwell_time', # distance
    'eeg_eye_movement_dwell_time',
    'AOI_dwell_time',
    'eeg_AOI_dwell_time',
]

feature_type = [
    'eye_movement_AOI_area_gap=1s_duration=2s_min_max_area_time_span',
]

FREQ_BANDS = {
    "delta": [0.5, 4],  # 1-3
    "theta": [4, 8],  # 4-7
    "alpha": [8, 13],  # 8-12
    "beta": [13, 31],  # 13-30
    "gamma": [31, 46]  # 31-45
}

# for username, user_stuId in users.items():
#     print(user_stuId, username, sep=',')
columns = ['0-2', '1-3', '2-4', '3-5']
