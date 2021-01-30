#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   EEG_similarity  
@Time        :   2021/1/25 12:55 下午
@Author      :   Xuesong Chen
@Description :   
"""

from collections import defaultdict

import pandas as pd
import numpy as np

from common.constants import *
from common.utils import map_sat
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, ttest_rel


def get_pair_score(vec_lists, return_mean=True):
    if len(vec_lists) == 1:
        return None
    '''old method start'''
    # similarity_arr = cosine_similarity(vec_lists)
    # n = len(vec_lists)
    # _mean = (np.sum(similarity_arr) - n) / (n * (n-1))
    # return _mean
    '''old method start'''

    idx = 0
    dist_list = []
    for _list_a in vec_lists:
        idx += 1
        for _list_b in vec_lists[idx:]:
            _j = cosine_similarity([_list_a, _list_b])[0, 1]
            dist_list.append(_j)
    if return_mean:
        return np.mean(dist_list)
    else:
        return dist_list


def get_two_list_cos(vec_lists_a, vec_lists_b, return_mean=True):

    if len(vec_lists_a) == 0 or len(vec_lists_b) == 0:
        return None
    dist_list = []

    for _list_a in vec_lists_a:
        for _list_b in vec_lists_b:
            _j = cosine_similarity([_list_a, _list_b])[0, 1]
            dist_list.append(_j)

    if return_mean:
        return np.mean(dist_list)
    else:
        return dist_list


def same_IN_same_user(mediate_result):
    # sat VS sat and unsat VS unsat
    target = 'sat'
    # target = 'spam'
    target_val_dic = {
        1: [],
        0: [],
        2: [],   # sat VS unsat
    }
    for username, _dic in mediate_result.items():
        target_dic = _dic[target]
        for target_level, specific_level_dic in target_dic.items():
            for task_id, vec_list in specific_level_dic.items():
                cos_score = get_pair_score(vec_list)
                if cos_score != None:
                    target_val_dic[target_level].append(cos_score)

    # sat VS unsat
    for username, _dic in mediate_result.items():
        target_dic = _dic[target]
        for task_id in FORMAL_TASK_ID_LIST:
            unsat_psd_lists = target_dic[0][task_id]
            sat_psd_lists = target_dic[1][task_id]
            if len(unsat_psd_lists) < 1 or len(sat_psd_lists) < 1:
                continue
            cos_score = get_two_list_cos(unsat_psd_lists, sat_psd_lists)
            target_val_dic[2].append(cos_score)

    print(
        'same_IN_same_user', '\n',
        "sat=1:", np.mean(target_val_dic[1]).round(3), '\n',
        'sat=0:', np.mean(target_val_dic[0]).round(3), '\n',
        'sat VS unsat:', np.mean(target_val_dic[2]).round(3), '\n',
        'sat=1 and sat=0', ttest_ind(target_val_dic[1], target_val_dic[0])[1], '\n',
        'sat=1 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[1])[1], '\n',
        'sat=0 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[0])[1], '\n',
    )


def same_IN_cross_user(mediate_result):

    target = 'sat'
    target_val_dic = {
        1: [],
        0: [],
        2: [],   # sat VS unsat
    }

    for task_id in FORMAL_TASK_ID_LIST:

        diff_users_sat_eeg_list = []
        diff_users_unsat_eeg_list = []

        for username in user_name_list:
            diff_users_sat_eeg_list.append(
                mediate_result[username][target][1][task_id]
            )
            diff_users_unsat_eeg_list.append(
                mediate_result[username][target][0][task_id]
            )

        # SAT
        idx = 0
        for _list_a in diff_users_sat_eeg_list:
            idx += 1
            for _list_b in diff_users_sat_eeg_list[idx:]:
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[1].append(_j)

        # UNSAT
        idx = 0
        for _list_a in diff_users_unsat_eeg_list:
            idx += 1
            for _list_b in diff_users_unsat_eeg_list[idx:]:
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[0].append(_j)

        # SAT VS UNSAT
        for _idx_a, _list_a in enumerate(diff_users_sat_eeg_list):
            for _idx_b, _list_b in enumerate(diff_users_unsat_eeg_list):
                if _idx_a == _idx_b:
                    continue
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[2].append(_j)

    print(
        'same_IN_cross_user', '\n',
        "sat=1:", np.mean(target_val_dic[1]).round(3), '\n',
        'sat=0:', np.mean(target_val_dic[0]).round(3), '\n',
        'sat VS unsat:', np.mean(target_val_dic[2]).round(3), '\n',
        'sat=1 and sat=0', ttest_ind(target_val_dic[1], target_val_dic[0])[1], '\n',
        'sat=1 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[1])[1], '\n',
        'sat=0 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[0])[1], '\n',
    )


def cross_IN_same_user(mediate_result):

    target = 'sat'
    target_val_dic = {
        1: [],
        0: [],
        2: [],   # sat VS unsat
    }


    for username in user_name_list:

        diff_tasks_sat_eeg_list = []
        diff_tasks_unsat_eeg_list = []

        for task_id in FORMAL_TASK_ID_LIST:
            diff_tasks_sat_eeg_list.append(
                mediate_result[username][target][1][task_id]
            )
            diff_tasks_unsat_eeg_list.append(
                mediate_result[username][target][0][task_id]
            )

        # SAT
        idx = 0
        for _list_a in diff_tasks_sat_eeg_list:
            idx += 1
            for _list_b in diff_tasks_sat_eeg_list[idx:]:
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[1].append(_j)

        # UNSAT
        idx = 0
        for _list_a in diff_tasks_unsat_eeg_list:
            idx += 1
            for _list_b in diff_tasks_unsat_eeg_list[idx:]:
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[0].append(_j)

        # SAT VS UNSAT
        for _idx_a, _list_a in enumerate(diff_tasks_sat_eeg_list):
            for _idx_b, _list_b in enumerate(diff_tasks_unsat_eeg_list):
                if _idx_a == _idx_b:
                    continue
                _j = get_two_list_cos(_list_a, _list_b)
                if _j != None:
                    target_val_dic[2].append(_j)

    print(
        'cross_IN_same_user', '\n',
        "sat=1:", np.mean(target_val_dic[1]).round(3), '\n',
        'sat=0:', np.mean(target_val_dic[0]).round(3), '\n',
        'sat VS unsat:', np.mean(target_val_dic[2]).round(3), '\n',
        'sat=1 and sat=0', ttest_ind(target_val_dic[1], target_val_dic[0])[1], '\n',
        'sat=1 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[1])[1], '\n',
        'sat=0 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[0])[1], '\n',
    )



def cross_IN_cross_user(mediate_result):

    target = 'sat'
    target_val_dic = {
        1: [],
        0: [],
        2: [],   # sat VS unsat
    }

    diff_tasks_diff_users_sat_eeg_list = []
    diff_tasks_diff_users_unsat_eeg_list = []

    for task_id in FORMAL_TASK_ID_LIST:

        diff_users_sat_eeg_list = []
        diff_users_unsat_eeg_list = []

        for username in user_name_list:
            diff_users_sat_eeg_list.append(
                mediate_result[username][target][1][task_id]
            )
            diff_users_unsat_eeg_list.append(
                mediate_result[username][target][0][task_id]
            )

        diff_tasks_diff_users_sat_eeg_list.append(diff_users_sat_eeg_list)
        diff_tasks_diff_users_unsat_eeg_list.append(diff_users_unsat_eeg_list)

    # SAT
    idx = 0
    for _list_a in diff_tasks_diff_users_sat_eeg_list:
        idx += 1
        for _list_b in diff_tasks_diff_users_sat_eeg_list[idx:]:
            for _list_a_user_idx, user_a_level_eeg_list in enumerate(_list_a):
                for _list_b_user_idx, user_b_level_eeg_list in enumerate(_list_b):
                    if _list_b_user_idx !=  _list_a_user_idx:
                        _j = get_two_list_cos(user_a_level_eeg_list, user_b_level_eeg_list)
                        if _j != None:
                            target_val_dic[1].append(_j)

    # UNSAT
    idx = 0
    for _list_a in diff_tasks_diff_users_unsat_eeg_list:
        idx += 1
        for _list_b in diff_tasks_diff_users_unsat_eeg_list[idx:]:
            for _list_a_user_idx, user_a_level_eeg_list in enumerate(_list_a):
                for _list_b_user_idx, user_b_level_eeg_list in enumerate(_list_b):
                    if _list_b_user_idx !=  _list_a_user_idx:
                        _j = get_two_list_cos(user_a_level_eeg_list, user_b_level_eeg_list)
                        if _j != None:
                            target_val_dic[0].append(_j)

    # SAT VS UNSAT
    idx = 0
    for _list_a in diff_tasks_diff_users_sat_eeg_list:
        idx += 1
        for _list_b in diff_tasks_diff_users_unsat_eeg_list[idx:]:
            for _list_a_user_idx, user_a_level_eeg_list in enumerate(_list_a):
                for _list_b_user_idx, user_b_level_eeg_list in enumerate(_list_b):
                    if _list_b_user_idx !=  _list_a_user_idx:
                        _j = get_two_list_cos(user_a_level_eeg_list, user_b_level_eeg_list)
                        if _j != None:
                            target_val_dic[2].append(_j)

    print(
        'cross_IN_cross_user', '\n',
        "sat=1:", np.mean(target_val_dic[1]), '\n',
        'sat=0:', np.mean(target_val_dic[0]), '\n',
        'sat VS unsat:', np.mean(target_val_dic[2]), '\n',
        'sat=1 and sat=0', ttest_ind(target_val_dic[1], target_val_dic[0])[1], '\n',
        'sat=1 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[1])[1], '\n',
        'sat=0 and sat VS unsat', ttest_ind(target_val_dic[2], target_val_dic[0])[1], '\n',
    )



def loop_template():
    similarity_dic = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    feature_type = models_type[16]

    # 先取出指定类型的脑电psd向量
    mediate_result = defaultdict(
        # username -> sat | spam -> val -> task_id -> vector
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: []
                )
            )
        )
    )
    for username in user_name_list:
        file = pd.read_csv(f'{prj_path}/dataset/{username}/{feature_type}.csv', header=None, index_col=0)
        for row in file.itertuples():
            page_uid = row[0]
            satisfaction = row[1]
            eeg_vector = np.array(row[2:])

            page_id = int(page_uid.split('_')[1])
            task_id = int(page_uid.split('_')[0])
            binary_sat = map_sat(satisfaction)

            if page_id in [1, 2]:
                spam = 1
            elif page_id in [3, 4]:
                spam = 0
            else:
                spam = -1

            if binary_sat != -1:
                mediate_result[username]['sat'][binary_sat][task_id].append(eeg_vector)
            if spam != -1:
                mediate_result[username]['spam'][spam][task_id].append(eeg_vector)

    # same IN Same User
    same_IN_same_user(mediate_result)

    # Same IN cross user
    same_IN_cross_user(mediate_result)
    
    # cross IN Same User
    cross_IN_same_user(mediate_result)
    
    # cross IN cross User
    # cross_IN_cross_user(mediate_result)
    


if __name__ == '__main__':
    loop_template()
    pass
