#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   dwell_time_dist_on_diff_relevance  
@Time        :   2021/3/4 11:21 下午
@Author      :   Xuesong Chen
@Description :   
"""

from common.constants import *
from collections import defaultdict
from scipy.stats import ttest_ind, ttest_rel


feature_type = models_type[9]

page_id_map = {
    '1': 'irrelevant',
    '2': 'irrelevant',
    '3': 'topic-relevant',
    '4': 'topic-relevant',
    '5': 'relevant',
    '6': 'relevant',
}

dwell_time_dist_dic = defaultdict(lambda: [])
for username in user_name_list:
    file = pd.read_csv(
        f'{prj_path}/dataset/{username}/{feature_type}.csv',
        index_col=0
    )
    for row in file.itertuples():
        print(row)
        _tmp_list = row.Index.split('_')
        task_id = _tmp_list[0]
        page_id = _tmp_list[1]
        dwell_time = row.dwellTime
        dwell_time_dist_dic[page_id_map[page_id]].append(dwell_time)
    print()

for page_id, _list in dwell_time_dist_dic.items():
    print(page_id, np.mean(_list))

print(ttest_ind(dwell_time_dist_dic['irrelevant'], dwell_time_dist_dic['topic-relevant']))
print(ttest_ind(dwell_time_dist_dic['irrelevant'], dwell_time_dist_dic['relevant']))
print(ttest_ind(dwell_time_dist_dic['relevant'], dwell_time_dist_dic['topic-relevant']))