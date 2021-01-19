#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   load_formatted_txt  
@Time        :   2021/1/2 7:14 下午
@Author      :   Xuesong Chen
@Description :   
"""

from mne.io import RawArray
import pandas as pd
from common.constants import *
import mne
from datetime import datetime


username = user_name_list[0]
df = pd.read_csv(f'../data/EEG/{username}.csv')

n_channel = 11
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
info = mne.create_info(ch_names=ch_names,
                       ch_types=ch_types,
                       sfreq=sampling_freq)
print(info)
raw_eeg_file = RawArray(df.T, info)
raw_eeg_file.info['meas_date'] = datetime.fromtimestamp(df['time'][0]/1000)

# Annotating EEG data


raw_eeg_file.plot(show_scrollbars=False, show_scalebars=False)