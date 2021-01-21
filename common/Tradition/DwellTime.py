#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   DwellTime  
@Time        :   2021/1/21 3:26 下午
@Author      :   Xuesong Chen
@Description :   
"""
from common.reader.AnswerInfo import AnswerInfo
from common.constants import *

index_list = [f'{task_id}_{page_id}' for task_id in FORMAL_TASK_ID_LIST for page_id in range(1, 7)]
feature_name = models_type[9]

for username in user_name_list:
    dwell_time_feats_df = pd.DataFrame(
        columns=['satisfaction', 'dwellTime'],
        index=index_list,
    )
    for task_id in FORMAL_TASK_ID_LIST:
        for page_id in range(1, 7):
            answer_info = AnswerInfo(username, task_id, page_id)
            page_uid = f'{task_id}_{page_id}'

            satisfaction = answer_info.get_satisfaction()
            dwell_time = answer_info.get_reading_time()

            dwell_time_feats_df.at[page_uid, 'satisfaction'] = satisfaction
            dwell_time_feats_df.at[page_uid, 'dwellTime'] = dwell_time

    feat_path = f'{prj_path}/dataset/{username}/'
    dwell_time_feats_df.to_csv(f'{feat_path}/{feature_name}.csv')


