#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   TaskInfo  
@Time        :   2021/1/19 12:50 下午
@Author      :   Xuesong Chen
@Description :   
"""

import sqlite3
from collections import defaultdict

from common.constants import *


class TaskInfo:

    def __init__(self, username, task_id):
        self.conn = sqlite3.connect(f'{prj_path}/data/ExportedData/db.sqlite3')
        c = self.conn.cursor()
        base_sql = "SELECT * FROM anno_task_info WHERE username = '%s'" % username
        sql = base_sql + " AND task_id = %s" % (task_id)
        obj = c.execute(sql)

        record_list = obj.fetchall()

        assert len(record_list) == 1

        record = record_list[0]
        id = record[0]
        username = record[1]
        is_formal_task = record[2]
        task_id = record[3]
        self.familiarity = record[4]
        self.difficulty = record[5]
        self.start_timestamp = record[6]
        self.end_timestamp = record[7]

    def get_familiarity(self):
        return self.familiarity

    def get_difficulty(self):
        return self.difficulty


if __name__ == '__main__':

    familiarity_dist_dic = defaultdict(lambda: 0)
    difficulty_dist_dic = defaultdict(lambda: 0)
    for username in user_name_list:
        for task_id in FORMAL_TASK_ID_LIST:
            task_info = TaskInfo(username, task_id)
            familiarity = task_info.get_familiarity()
            difficulty = task_info.get_difficulty()
            familiarity_dist_dic[familiarity] += 1
            difficulty_dist_dic[difficulty] += 1

    print(
        'familiarity', familiarity_dist_dic, '\n',
        'difficulty', difficulty_dist_dic, '\n'
    )
