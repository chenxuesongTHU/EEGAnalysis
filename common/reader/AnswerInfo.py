#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@File        :   AnswerInfo
@Time        :   2020/12/31 10:33 上午
@Author      :   Xuesong Chen
@Description :   
"""

import sqlite3
from common.constants import prj_path

class AnswerInfo:

    def __init__(self, user_name, task_id, page_id):
        self.conn = sqlite3.connect(f'{prj_path}/data/ExportedData/db.sqlite3')
        c = self.conn.cursor()
        base_sql = "SELECT * FROM anno_page_satisfaction WHERE username = '%s'" % user_name
        sql = base_sql + " AND task_id = %s AND page_id = %s"%(task_id, page_id)
        obj = c.execute(sql)

        record_list = obj.fetchall()

        assert len(record_list) == 1

        record = record_list[0]
        id = record[0]
        username = record[1]
        task_id = record[2]
        page_id = record[3]
        self.annotation_start_timestamp = record[4]
        self.annotation_end_timestamp = record[5]
        self.satisfaction = record[6]
        self.page_height = record[7]
        self.reading_end_timestamp = record[8]
        self.reading_start_timestamp = record[9]


    def __del__(self):
        self.conn.close()


    def get_reading_start_timestamp(self):
        return self.reading_start_timestamp


    def get_reading_end_timestamp(self):
        return self.reading_end_timestamp

    def get_reading_time(self):
        return self.reading_end_timestamp - self.reading_start_timestamp

    def get_satisfaction(self):
        return self.satisfaction


if __name__ == '__main__':

    answer_info = AnswerInfo('2016010106', 2, 2)
    print(
        answer_info.get_reading_start_timestamp(),
        answer_info.get_reading_end_timestamp(),
        answer_info.get_satisfaction()
    )
