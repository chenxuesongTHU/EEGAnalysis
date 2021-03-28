#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   PageEEG  
@Time        :   2021/1/3 10:06 下午
@Author      :   Xuesong Chen
@Description :   
"""

import mne
from mne.io import RawArray

from common.constants import *
from common.reader.AnswerInfo import AnswerInfo


class PageEEG:

    def __init__(self, username, task_id, page_id, use_symmetry=False, time_bias=0):
        '''
        :param username:
        :param task_id:
        :param page_id:
        :param time_bias: 时间单位是ms
                          负数代表提前多长时间，正数代表延后多长时间
                          由于电脑时间戳偏后，因此该值应为负数
        '''
        answer_info = AnswerInfo(username, task_id, page_id)
        self.start_time = answer_info.reading_start_timestamp + time_bias
        self.end_time = answer_info.reading_end_timestamp + time_bias
        self.satisfaction = answer_info.get_satisfaction()
        self.df = pd.read_csv(f'{prj_path}/data/EEG/{username}.csv')
        # 是否使用对称电极
        self.use_symmetry = use_symmetry
        self.__init_info()
        if use_symmetry:
            self.__add_symmetrical_position()

    def __init_info(self):
        sampling_freq = 250
        ch_names = ['Time',
                    'Af8-O2', 'Fp2-O2', 'Fp1-O2', 'Af7-O2', 'O1-O2',
                    'blood_oxygen', 'heart_rate',
                    'x', 'y', 'z'
                    ]

        '''
            misc: 代表时间
            ref_meg: 代表陀螺仪数据
        '''
        ch_types = ['misc'] + ['eeg'] * 5 + ['bio'] * 2 + ['ref_meg'] * 3

        if self.use_symmetry:
            ch_names.extend([
                'Af8-Af7', 'Fp2-Fp1'
            ])
            ch_types += ['eeg'] * 2

        self.info = mne.create_info(ch_names=ch_names,
                                    ch_types=ch_types,
                                    sfreq=sampling_freq)

    def __add_symmetrical_position(self):
        self.df['Af8-Af7'] = self.df['Af8-O2'] - self.df['Af7-O2']
        self.df['Fp2-Fp1'] = self.df['Fp2-O2'] - self.df['Fp1-O2']

    def get_satisfaction(self):
        return self.satisfaction

    def get_start_reading_EEG(self, duration=5000):
        n_samples = int(duration / 1000 * 250)
        df = self.df[(self.start_time <= self.df['time']) & (self.df['time'] <= self.end_time)][:n_samples]
        if len(df) == 0:
            return 'NoRecord'
        raw_eeg_file = RawArray(df.T, self.info)
        # raw_eeg_file.info['meas_date'] = datetime.fromtimestamp(df['time'].iloc[0] / 1000)

        return raw_eeg_file

    def get_end_reading_EEG(self, duration=5000):
        n_samples = int(duration / 1000 * 250)
        df = self.df[(self.df['time'] <= self.end_time)][-n_samples:]
        if len(df) == 0:
            return 'NoRecord'
        raw_eeg_file = RawArray(df.T, self.info)
        # raw_eeg_file.info['meas_date'] = datetime.fromtimestamp(df['time'].iloc[0] / 1000)

        return raw_eeg_file

    def get_reading_EEG(self):
        df = self.df[(self.start_time <= self.df['time']) & (self.df['time'] <= self.end_time)]
        if len(df) == 0:
            return 'NoRecord'
        raw_eeg_file = RawArray(df.T, self.info)
        return raw_eeg_file

    def get_fixation_EEG(self):

        pass

    def get_saccade_EEG(self):
        pass


if __name__ == '__main__':
    page_eeg = PageEEG('2019270058', 1, 6)
    page_eeg.get_start_reading_EEG()
    page_eeg.get_end_reading_EEG()
