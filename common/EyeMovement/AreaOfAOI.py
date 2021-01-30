#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   AreaOfAOI  
@Time        :   2021/1/20 12:55 下午
@Author      :   Xuesong Chen
@Description :   
"""
from sklearn import preprocessing
from scipy.stats import ttest_ind

from common.constants import *


def compare_area_of_AOI():
    # 用户在看满意/不满意网页时的AOI面积
    all_users_sat_df = np.empty([0, 4], dtype=float)
    all_users_unsat_df = np.empty([0, 4], dtype=float)
    columns = ['0-2', '1-3', '2-4', '3-5']
    # all_users_unsat_df = pd.DataFrame(
    #     columns=['0-2', '1-3', '2-4', '3-5']
    # )
    file_name = models_type[3]  # 3, 11
    for username in user_name_list:
        df = pd.read_csv(f'{prj_path}/dataset/{username}/{file_name}.csv', index_col=0)
        sat_df = df.loc[(df['satisfaction'] == 4) | (df['satisfaction'] == 3)]
        unsat_df = df.loc[(df['satisfaction'] == 0) | (df['satisfaction'] == 1)]
        sat_df.drop(['satisfaction'], axis=1, inplace=True)
        unsat_df.drop(['satisfaction'], axis=1, inplace=True)
        sat_df = preprocessing.normalize(sat_df, norm='l1', axis=1)
        unsat_df = preprocessing.normalize(unsat_df, norm='l1', axis=1)
        all_users_sat_df = np.concatenate((all_users_sat_df, sat_df), axis=0)
        all_users_unsat_df = np.concatenate((all_users_unsat_df, unsat_df), axis=0)

    all_users_sat_df_diff_time_span_list = all_users_sat_df.mean(axis=0)
    all_users_unsat_df_diff_time_span_list = all_users_unsat_df.mean(axis=0)
    all_users_sat_df_diff_time_span_list = [round(_, 3) for _ in all_users_sat_df_diff_time_span_list]
    all_users_unsat_df_diff_time_span_list = [round(_, 3) for _ in all_users_unsat_df_diff_time_span_list]

    print(
        'sat:', all_users_sat_df_diff_time_span_list, '\n',
        'unsat:', all_users_unsat_df_diff_time_span_list,
    )

    for idx, column in zip(range(0, 4), columns):
        print(ttest_ind(all_users_sat_df[:, idx], all_users_unsat_df[:, idx]))


def get_max_min_area_time_span():
    file_name = models_type[3]  # 3, 11
    for username in user_name_list:
        df = pd.read_csv(f'{prj_path}/dataset/{username}/{file_name}.csv', index_col=0)
        df.drop(['satisfaction'], axis=1, inplace=True)
        max_area_span_df = df.idxmax(axis=1).to_frame('max')
        min_area_span_df = df.idxmin(axis=1).to_frame('min')
        result_df = pd.concat([max_area_span_df, min_area_span_df], axis=1)
        result_df.to_csv(f'{prj_path}/dataset/{username}/{file_name}_min_max_distance_time_span.csv')
        # print()


if __name__ == '__main__':
    compare_area_of_AOI()
    # get_max_min_area_time_span()
