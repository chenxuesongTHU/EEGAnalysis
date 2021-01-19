#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   psd_features  
@Time        :   2021/1/2 7:44 下午
@Author      :   Xuesong Chen
@Description :   
"""

from common.reader.AnswerInfo import AnswerInfo
from common.constants import *
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy import stats

features_dic = defaultdict(lambda: defaultdict(lambda: []))

for username in user_name_list:
    for task_id in FORMAL_TASK_ID_LIST:
        for page_id in range(1, 7):

            answer_info = AnswerInfo(username, task_id, page_id)
            reading_start_timestamp = answer_info.reading_start_timestamp
            reading_end_timestamp = answer_info.reading_end_timestamp
            start_time = reading_start_timestamp
            end_time = reading_start_timestamp + 5000
            satisfaction = answer_info.get_satisfaction()
            df = pd.read_csv(f'../data/EEG/{username}.csv')
            df = df[(start_time <= df['time']) & (df['time'] <= end_time)]
            assert len(df) > 1200
            if satisfaction < 2:
                features_dic['blood_oxygen']['unsat'].append(np.mean(df['blood_oxygen']))
                features_dic['heart_rate']['unsat'].append(np.mean(df['heart_rate']))
            if satisfaction > 3:
                features_dic['blood_oxygen']['sat'].append(np.mean(df['blood_oxygen']))
                features_dic['heart_rate']['sat'].append(np.mean(df['heart_rate']))
            print()

    for type in ['blood_oxygen', 'heart_rate']:
        print(
            stats.ttest_ind(features_dic[type]['sat'], features_dic[type]['unsat'])
        )