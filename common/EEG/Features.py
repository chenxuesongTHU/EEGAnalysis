#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   features
@Time        :   2020/12/30 11:57 下午
@Author      :   Xuesong Chen
@Description :   
"""
# todo:  STEP 2
import math

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.time_frequency import psd_welch

from common.constants import *
from common.reader.PageEEG import PageEEG
from common.utils import *

# FREQ_BANDS = {"delta": [0.5, 4.5],
#               "theta": [4.5, 8.5],
#               "alpha": [8.5, 11.5],
#               "sigma": [11.5, 15.5],
#               "beta": [15.5, 30]}

# 注释中左闭右闭
FREQ_BANDS = {
    "delta": [0.5, 4],   # 1-3
    "theta": [4, 8],     # 4-7
    "alpha": [8, 13],    # 8-12
    "beta": [13, 31],    # 13-30
    "gamma": [31, 46]    # 31-45
}
tmp = {
"delta": [0.5, 4],  "theta": [4, 8],  "alpha": [8, 13], "beta": [13, 31], "gamma": [31, 46]
}
# iter_freqs = [
#     ('Theta', 4, 7),
#     ('Alpha', 8, 12),
#     ('Beta', 13, 25),
#     ('Gamma', 30, 45)
# ]

def plot_sample_channel(raw):
    # 滤波效果可视化
    _data = list(np.squeeze(raw.copy().pick_channels(['Af8-O2'])._data))
    plt.plot(_data)
    plt.show()


class Features:

    def __init__(self, raw):
        '''
        :param raw: mne包中的raw
        '''
        self.raw = raw
        # self.ICA_transformor()
        # raw.plot_psd(tmax=np.inf, fmax=250, average=True)
        # self.__resample()
        # self.raw = raw.filter(l_freq=0.5, h_freq=30)

    def __resample(self, sfreq=125):
        self.raw = self.raw.resample(sfreq=sfreq)

    def filter(self, filters):
        '''
        :param filters: [Savitzky-Golay, fir]
        :return:
        '''
        # plot_sample_channel(self.raw)

        if 'Savitzky-Golay' in filters:
            plt.subplots()
            self.raw = self.raw.savgol_filter(h_freq=45)
            # plot_sample_channel(self.raw)

        if 'fir' in filters:
            plt.subplots()
            self.raw = self.raw.filter(l_freq=0.5, h_freq=45)
            # plot_sample_channel(self.raw)

    def ICA_transformor(self):

        ica = ICA(n_components=5, random_state=97)
        ica.fit(self.raw)
        array_for_raw = ica.get_sources(self.raw).get_data()

        info = mne.create_info(ch_names=['ic1', 'ic2', 'ic3', 'ic4', 'ic5'],
                               ch_types=['eeg'] * 5,
                               sfreq=250)

        raw_eeg_file = RawArray(array_for_raw, info)

        self.raw = raw_eeg_file

        # ica.plot_sources(self.raw, show_scrollbars=False)


    def get_psd(self):
        psds, freqs = psd_welch(self.raw, picks='eeg', fmin=0.5, fmax=45.)
        psds /= np.sum(psds, axis=-1, keepdims=True)
        PSD_feat_list = [] # PSD features
        # DE_feat_list = []

        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            # des_band = np.log2(100 * psds_band)
            PSD_feat_list.append(psds_band.reshape(len(psds), -1))
            # DE_feat_list.append(des_band.reshape(len(psds), -1))

        return np.concatenate(PSD_feat_list, axis=1)
               # np.concatenate(DE_feat_list, axis=1)


    def get_de(self):
        psds, freqs = psd_welch(self.raw, picks='eeg', fmin=0.5, fmax=45.)
        psds /= np.sum(psds, axis=-1, keepdims=True)
        # PSD_feat_list = [] # PSD features
        DE_feat_list = []

        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            des_band = np.log2(100 * psds_band)
            # PSD_feat_list.append(psds_band.reshape(len(psds), -1))
            DE_feat_list.append(des_band.reshape(len(psds), -1))

        return np.concatenate(DE_feat_list, axis=1)


    def get_psd_de(self):
        psds, freqs = psd_welch(self.raw, picks='eeg', fmin=0.5, fmax=45.)
        psds /= np.sum(psds, axis=-1, keepdims=True)
        PSD_feat_list = [] # PSD features
        DE_feat_list = []

        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            des_band = np.log2(100 * psds_band)
            PSD_feat_list.append(psds_band.reshape(len(psds), -1))
            DE_feat_list.append(des_band.reshape(len(psds), -1))

        PSD_feat_list.extend(DE_feat_list)    # psd feature + de feature
        return np.concatenate(PSD_feat_list, axis=1)


    def get_STFT_psd(self, windows=3, step=1):
        '''

        :param windows: 窗口大小，单位为秒
        :param step: 滑动长度，单位为秒
        :return: 变换成1维度向量的格式是 window 1; window 2; window 3
                 其中每个window中的信息与get_psd保持一致
        '''
        last_time = round(self.raw._last_time)
        n_steps = int((last_time - windows) / step) + 1
        psd_list = []
        for step_idx in range(n_steps):
            window_raw = self.raw.copy().crop(tmin=0, tmax=3)
            psd_list.append(get_psd_by_raw(window_raw))

        return np.array(psd_list)



def get_psd_by_raw(raw):
    psds, freqs = psd_welch(raw, picks='eeg', fmin=0.5, fmax=45.)
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


def draw_whole_sat_and_unsat(idx):
    time_bias = 0
    sat_raw = None
    unstat_raw = None
    duration = 5000

    for username in user_name_list[idx:]:
    # username = user_name_list[idx:]
        sat_raw = None
        unstat_raw = None

        for task_id in FORMAL_TASK_ID_LIST:
            # task_id = 22
            for page_id in range(1, 7):
                # 存在bug的数据
                if username == '2019270058' and task_id == 22 and page_id == 5:
                    continue
                page_eeg = PageEEG(username, task_id, page_id, time_bias=time_bias)
                page_eeg_raw = page_eeg.get_start_reading_EEG(duration=duration)
                if page_eeg_raw == 'NoRecord':
                    continue
                sat_level = page_eeg.get_satisfaction()
                if sat_level > 2:
                    if sat_raw == None:
                        sat_raw = page_eeg_raw.copy()
                    else:
                        sat_raw.append(page_eeg_raw)
                if sat_level < 2:
                    if unstat_raw == None:
                        unstat_raw = page_eeg_raw.copy()
                    else:
                        unstat_raw.append(page_eeg_raw)

        fig, ax = plt.subplots()
        sat_raw.plot_psd(area_mode=None, fmin=0.1, fmax=45., show=False, color='red',
                         ax=ax,
                         picks=['eeg'], average=True, spatial_colors=False)

        unstat_raw.plot_psd(area_mode=None, fmin=0.1, fmax=45., show=False, color='black',
                            ax=ax,
                            picks=['eeg'], average=True, spatial_colors=False)
        ax.legend(ax.lines[2::3], ['sat', 'unsat'])
        plt.title(f'PSD of {username}')
        plt.savefig(f'pre_result/PSD_{username}_time_bias={time_bias}_duration={duration}.pdf', format='pdf')


def draw_single_page_sat_and_unsat():
    sat_raw = None
    unstat_raw = None
    fig, ax = plt.subplots()

    for username in user_name_list:
        for task_id in FORMAL_TASK_ID_LIST:
            for page_id in range(1, 7):
                page_eeg = PageEEG(username, task_id, page_id)
                page_eeg_raw = page_eeg.get_start_reading_EEG()
                sat_level = page_eeg.get_satisfaction()
                if sat_level > 2:
                    if sat_raw == None:
                        sat_raw = page_eeg_raw.copy()
                    else:
                        sat_raw.append(page_eeg_raw)
                    page_eeg_raw.plot_psd(area_mode=None, fmin=0.1, fmax=20., show=False, color='red',
                                          ax=ax,
                                          picks=['eeg'], average=True, spatial_colors=False)
                if sat_level < 2:
                    if unstat_raw == None:
                        unstat_raw = page_eeg_raw.copy()
                    else:
                        unstat_raw.append(page_eeg_raw)
                    page_eeg_raw.plot_psd(area_mode=None, fmin=0.1, fmax=20., show=False, color='black',
                                          ax=ax,
                                          picks=['eeg'], average=True, spatial_colors=False)

    # ax.legend(ax.lines[2::3], ['sat', 'unsat'])
    plt.title('PSD of EEG')
    plt.savefig(f'PSD_{user_name_list[0]}_single_page.pdf', format='pdf')


def test_features():

    page_eeg = PageEEG('2019270058', 5, 3)
    page_eeg_raw = page_eeg.get_start_reading_EEG()
    feats = Features(page_eeg_raw)
    psd_feats = feats.get_psd_de()


def output_users_EEG_feats():
    '''
    该函数用来导出所有用户浏览landing page时的EEG特征
    :return:
    '''
    # todo: username, username_list, models_type
    # model_type_idx = 2
    user_idx = 5
    phrase = 'whole'
    # phrase = 'start5s'

    for user_idx in range(2, 17):

        username = user_name_list[user_idx]
        sat_raw = None
        unstat_raw = None
        for model_type_idx in range(8, 9):
            file_name = models_type[model_type_idx]
            path = f'{prj_path}/dataset/{username}'
            mkdir(path)
            output_file = open(f'{path}/{file_name}.csv', 'w')
            # channel 1的波段alpha, beta ...  channel 2的波段alpha, beta ...
            # for username in user_name_list[1:2]:
            for task_id in FORMAL_TASK_ID_LIST:
                for page_id in range(1, 7):
                    page_eeg = PageEEG(username, task_id, page_id)
                    if phrase == 'start5s':
                        page_eeg_raw = page_eeg.get_start_reading_EEG()
                    if phrase == 'whole':
                        page_eeg_raw = page_eeg.get_reading_EEG()

                    if page_eeg_raw == 'NoRecord':
                        continue
                    sat_level = page_eeg.get_satisfaction()
                    feats = Features(page_eeg_raw)

                    # feats.filter(['Savitzky-Golay'])
                    # feats.filter(['fir'])
                    # feats.filter(['Savitzky-Golay', 'fir'])

                    if model_type_idx == 0:
                        psd_feats = feats.get_psd()
                    elif model_type_idx == 1:
                        psd_feats = feats.get_de()
                    elif model_type_idx in [2, 4, 8]:
                        psd_feats = feats.get_psd_de()
                    else:
                        pass
                    # psd_feats = feats.get_de()
                    # psd_feats = feats.get_STFT_psd()
                    page_uid = f'{task_id}_{page_id}'
                    output_feats = list(np.squeeze(psd_feats.reshape(1, -1)))
                    output_feats = [str(i) for i in output_feats]
                    output_file.write(page_uid + ',' + str(sat_level) + ',' + ','.join(output_feats) + '\n')

            output_file.close()


def output_EEG_feats_of_min_max_area():
    '''
    导出用户的AOI面积最大最小时段内的EEG特征
    :return:
    '''
    for user_idx in range(0, 17):

        username = user_name_list[user_idx]
        path = f'{prj_path}/dataset/{username}'
        time_span_src_file = pd.read_csv(f'{path}/{feature_type[0]}.csv', index_col=0)

        for model_type_idx in range(6, 8):
            file_name = models_type[model_type_idx]
            mkdir(path)
            output_file = open(f'{path}/{file_name}.csv', 'w')
            # channel 1的波段alpha, beta ...  channel 2的波段alpha, beta ...
            # for username in user_name_list[1:2]:
            for task_id in FORMAL_TASK_ID_LIST:
                for page_id in range(1, 7):
                    # 获取开始时间
                    page_uid = f'{task_id}_{page_id}'
                    col_name = file_name[:3]
                    time_span = time_span_src_file.at[page_uid, col_name]
                    start_time = int(time_span[0]) * 1000
                    duration = 2000
                    page_eeg = PageEEG(username, task_id, page_id, time_bias=start_time)
                    page_eeg_raw = page_eeg.get_start_reading_EEG(duration)

                    if page_eeg_raw == 'NoRecord':
                        continue
                    sat_level = page_eeg.get_satisfaction()
                    feats = Features(page_eeg_raw)

                    # feats.filter(['Savitzky-Golay'])
                    # feats.filter(['fir'])
                    # feats.filter(['Savitzky-Golay', 'fir'])

                    if model_type_idx == 0:
                        psd_feats = feats.get_psd()
                    elif model_type_idx == 1:
                        psd_feats = feats.get_de()
                    elif model_type_idx in [2, 4, 6, 7]:
                        psd_feats = feats.get_psd_de()
                    else:
                        pass
                    # psd_feats = feats.get_de()
                    # psd_feats = feats.get_STFT_psd()
                    page_uid = f'{task_id}_{page_id}'
                    output_feats = list(np.squeeze(psd_feats.reshape(1, -1)))
                    output_feats = [str(i) for i in output_feats]
                    output_file.write(page_uid + ',' + str(sat_level) + ',' + ','.join(output_feats) + '\n')

            output_file.close()


if __name__ == '__main__':

    # draw_whole_sat_and_unsat(14)
    # test_features()
    # output_users_EEG_feats()
    # output_EEG_feats_of_min_max_area()
    output_users_EEG_feats()
