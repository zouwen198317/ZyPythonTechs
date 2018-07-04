#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  :   zzg
# @Contact :   xfgczzg@163.com
# @Software:   PyCharm
# @File    :   sample_04_pandas2.py
# @Date    :   2018/7/4
# @Desc    :

import common_header

import pandas as pd

if __name__ == '__main__':
    print("文件操作:")
    # t_xls_1 = pd.read_excel("./data/test.xlsx", sheetname=0, encording="gb2312")
    t_xls_1 = pd.read_excel("./data/test.xlsx", sheetname='sheet1')
    print(t_xls_1)
    print('----------------------------')
