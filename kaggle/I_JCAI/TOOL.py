# -*- coding:utf-8 -*-
"""
Created on 18-12-13 下午2:11
@Author:Johnson
@Email:593956670@qq.com 
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from textwrap import wrap
import xgboost as xgb
import copy
import datetime

def Datestr2DGap(ser,start_date):
    """
    :param ser:
    :param start_date:
    :return:ralatetive date from the start_date
    """
    temp = pd.to_datetime(ser)-pd.to_datetime(start_date)
    ser_DGap = [(lambda x:(x.days())) for x in temp]
    return ser_DGap

def Datestr2Dofw(ser):
    """
    :param ser:
    :return:return day of week
    """
    ser_DofW = pd.to_datetime(ser).dt.dayofweek
    return ser_DofW

def Const_Datestr(start_date,date_len):
    return [str((datetime.datetime.strptime(start_date,'%Y-%m-%d')+datetime.timedelta(days=x)).date())for x in range(date_len)]


def Const_Datestr2(start_date,end_date):
    day_N = (datetime.datetime.strptime(end_date,'%Y-%m-%d')-datetime.datetime.strptime(start_date,'%Y-%m-%d')).days+1
    return day_N

def Const_Datestr3(prefix,start_date,end_date):
    day_N = (datetime.datetime.strptime(end_date, '%Y-%m-%d') - datetime.datetime.strptime(start_date,'%Y-%m-%d')).days + 1
    return_string = [(prefix+str((datetime.datetime.strptime(start_date,'%Y-%m-%d')+datetime.timedelta(days=x)))) for x in range(day_N)]

def Datedate2Datestr(ser):
    return [(lambda x:str(x.date())) for x in ser]

def str2date(tt):
    return datetime.datetime.strptime(tt,'%Y-%m-%d').date()

def loss_round(result):
    result_int = np.floor(result)
    if result**2 <= result_int**2+result_int:
        result_back = np.floor(result)
    else:
        result_back = np.ceil(result)
    return result_back


