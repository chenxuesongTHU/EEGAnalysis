#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   exported_useful_eyeMovement_data.py  
@Time        :   2021/1/19 6:20 下午
@Author      :   Xuesong Chen
@Description :
"""

from tqdm import tqdm

from common.constants import *
from common.utils import date2timestamp

# username = user_name_list[0]
for username in user_name_list[1:]:
    output_file = open(f'{prj_path}/data/EyeMovement/{username}.csv', 'w')
    file = pd.read_excel(f'{prj_path}/data/ExportedData/{username}.xlsx',
                         usecols=['LocalTimeStamp', 'FixationIndex', 'GazeEventDuration', 'FixationPointX (MCSpx)',
                                  'FixationPointY (MCSpx)'],
                         # nrows=100,
                         )
    # extracted_column = file[
    #     ['LocalTimeStamp', 'FixationIndex', 'GazeEventDuration', 'FixationPointX (MCSpx)',
    #      'FixationPointY (MCSpx)']]
    date = experiment_time[username]
    print('TimeStamp', 'FixationIndex', 'FixationPointX', 'FixationPointY', 'GazeEventDuration',
          sep=',',
          file=output_file)
    for index in tqdm(range(file.shape[0])):

        row = file.iloc[index]
        fixation_index = row['FixationIndex']
        fixationPointX = row['FixationPointX (MCSpx)']
        fixationPointY = row['FixationPointY (MCSpx)']
        gazeEventDuration = row['GazeEventDuration']
        localTimeStamp = row['LocalTimeStamp']
        if type(localTimeStamp) is not str:
            continue
        date_time = date + localTimeStamp
        timestamp = date2timestamp(date_time)
        if np.isnan(fixation_index) or np.isnan(fixationPointX) or np.isnan(fixationPointY):
            continue
        print(timestamp, fixation_index, fixationPointX, fixationPointY, gazeEventDuration,
              sep=',', file=output_file)

        # if index > 100:
        #     break

    output_file.close()
