#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   EyeMovement  
@Time        :   2021/1/19 10:07 下午
@Author      :   Xuesong Chen
@Description :   
"""

from common.constants import *
from common.reader.AnswerInfo import AnswerInfo
from common.utils import *


class EyeMovement:

    def __init__(self, username):
        self.df = pd.read_csv(f'{prj_path}/data/EyeMovement/{username}.csv')
        self.username = username

    def get_eye_movement(self, task_id, page_id, start_time_bias=0, duration=5):
        '''
        :param task_id:
        :param page_id:
        :param start_time_bias: 开始时间的偏移，单位为s
        :param duration: 持续时长，单位为s
        :return: (x, y, duration) dataframe形式 or tuple方式返回fixation序列
        '''
        start_time_bias = 1000 * start_time_bias
        duration = 1000 * duration
        answer_info = AnswerInfo(self.username, task_id, page_id)
        start_time = answer_info.reading_start_timestamp + start_time_bias
        end_time = start_time + duration
        page_eye_movement_df = self.df[(start_time <= self.df['TimeStamp']) & (self.df['TimeStamp'] <= end_time)]
        page_eye_movement_df.drop(['TimeStamp', 'FixationIndex'], axis=1, inplace=True)

        # 把不同时长的fixation看作是相同权重的
        # if comment: 时间长的fixation权重高，在计算面积时，不会轻易去掉该点所占的位置
        # page_eye_movement_df.drop_duplicates(inplace=True)

        # fixation_list = []
        # for row in page_eye_movement_df.itertuples():
        #     fixation_list.append(
        #         (row.FixationPointX, row.FixationPointY, row.GazeEventDuration)
        #     )
        # return fixation_list
        return page_eye_movement_df

    def get_area(self, task_id, page_id, start_time_bias=0, duration=5000, use_area_ratio=0.8):
        page_eye_movement_df = self.get_eye_movement(task_id, page_id, start_time_bias, duration)

        sorted_x = sorted(page_eye_movement_df['FixationPointX'])
        sorted_y = sorted(page_eye_movement_df['FixationPointY'])

        if len(sorted_x) == 0:  # 如果当前阶段没有fixation，则area of AOI为0
            return 0

        assert sorted_x[-1] <= 1920 and sorted_y[-1] <= 1080

        n_fixation_points = len(sorted_x) - 1  # 转为index，可取得范围为[0, n_fixation_points-1]
        low_threshold = round((1 - use_area_ratio) * n_fixation_points)
        high_threshold = round(use_area_ratio * n_fixation_points)

        left = sorted_x[low_threshold]
        right = sorted_x[high_threshold]
        top = sorted_y[low_threshold]
        bottom = sorted_y[high_threshold]
        area = (right - left) * (bottom - top)

        return area


if __name__ == '__main__':
    duration = 2
    start_time_gap = 1
    feature_name = models_type[3]
    time_span_list = [f'{start_idx}-{start_idx + duration}' for start_idx in range(0, 4)]
    index_list = [f'{task_id}_{page_id}' for task_id in FORMAL_TASK_ID_LIST for page_id in range(1, 7)]
    for username in user_name_list:
        eye_movement_feats_df = pd.DataFrame(
            columns=['satisfaction'] + time_span_list,
            index=index_list,
        )
        eye_movement = EyeMovement(username)
        for task_id in FORMAL_TASK_ID_LIST:
            for page_id in range(1, 7):
                answer_info = AnswerInfo(username, task_id, page_id)
                page_uid = f'{task_id}_{page_id}'
                eye_movement_feats_df.at[page_uid, 'satisfaction'] = answer_info.satisfaction
                for start_time in range(0, 4):
                    time_span = f'{start_time}-{start_time + duration}'
                    area = eye_movement.get_area(
                        task_id, page_id, start_time_bias=start_time, duration=duration
                    )
                    eye_movement_feats_df.at[page_uid, time_span] = area

        feat_path = f'{prj_path}/dataset/{username}/'
        eye_movement_feats_df.to_csv(f'{feat_path}/{feature_name}.csv')
