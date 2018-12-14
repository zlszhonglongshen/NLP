# -*- coding:utf-8 -*-
"""
Created on 18-12-13 下午4:10
@Author:Johnson
@Email:593956670@qq.com 
"""
import numpy as np
import pandas as pd
import math
import sys
sys.path.append('../Tools')

def ssd(temp,velo,humi):
    score = (1.818*temp+18.18) * (0.88+0.002*humi) + 1.0*(temp -32)/(45-temp) - 3.2*velo  + 18.2
    return score

weather_raw = pd.read_csv('weather_raw.csv',encoding='gbk',low_memory=False)


def apm2decimal(ser):
    tt = ser.replace(' ',':').split(':')
    tt[0] = np.int(tt[0])%12
    if tt[2]=='AM':
        return np.float(tt[0])+np.float(tt[1])/60.
    if tt[2]=='PM':
        return np.float(tt[0])+np.float(tt[2])/60.+12.

def Eventclean(ser):
    try:
        if math.isnan(ser):
            return 'None'
    except:
        tt = ser.replace('\n','r').replace('\t','r').split('\r')
        tt2 = ''.join(tt)
        return tt2

weather_raw = weather_raw[['DATE','Time','Temp','visibility','Wind_speed']]

weather_raw['Time'] = [(lambda x:apm2decimal(x))(x) for x in weather_raw['Time']]
weather_raw['Event'] = [(lambda x:Eventclean(x))(x) for x in weather_raw['Event']]
weather_raw['visibility'] = weather_raw['visibility'].replace('-',np.nan).fillna(method='ffill')
weather_raw['visibility'] = pd.to_numeric(weather_raw['visibility'],errors='ignore')

weather_raw['Temp'] = weather_raw['Temp'].replae('-',0.0)
weather_raw['Temp'] = pd.to_numeric(weather_raw['Temp'],errors='ignore')

weather_raw.loc[weather_raw['wind_speed']=='calm','wind_speed'] = 0.0
weather_raw['Wind_speed'] = weather_raw['Wind_speed'].replace('-','3.6')
weather_raw['Wind_speed'] = pd.to_numeric(weather_raw['Wind_speed'], errors='ignore')
weather_raw['Wind_speed'] = weather_raw['Wind_speed']/3.6

weather_raw['Humidity'] = weather_raw['Humidity'].replace('N/A%','5%')
weather_raw.loc[ weather_raw['Humidity'] == '%','Humidity']= '5%'
weather_raw['Humidity'] = [(lambda x: (np.int(x.split('%')[0]) ) ) (x) for x in weather_raw['Humidity']]

weather_raw['SSD'] = ssd(weather_raw['Temp'] ,weather_raw['Wind_speed'],weather_raw['Humidity'])

weather_raw.loc[ weather_raw['Condition'] == 'Unknown','Condition']= np.nan
weather_raw['Condition'] = weather_raw['Condition'].fillna(method='ffill')


WEATHER_CON_LEVEL = pd.read_csv('WEATHER_CON_LEVEL.csv')
WEATHER_raw = pd.merge(weather_raw, WEATHER_CON_LEVEL, on = 'Condition', how = 'left')
WEATHER_raw[['RAIN_IND','CLEAR_IND']] = WEATHER_raw[['RAIN_IND','CLEAR_IND']].fillna(0.0)


WEATHER_raw = WEATHER_raw[['DATE','Time','CITY_EN','SSD','RAIN_IND','CLEAR_IND']]


time1 = WEATHER_raw[((WEATHER_raw['Time']<=18.5) & ((WEATHER_raw['Time']>=11)) )]
#
time1_group = time1.groupby(['CITY_EN','DATE'],as_index = False).mean()
#
time1_group['SSD_C'] = np.abs(time1_group['SSD']-60) - np.abs(time1_group['SSD'].shift(1) -60)


time1_group = time1_group[((time1_group['DATE']<='2016-11-20') &(time1_group['DATE']>='2015-06-26')) ]


time1_group = time1_group.rename(columns = {'SSD':'RC','SSD_C':'RE','RAIN_IND':'RG','CLEAR_IND':'RI'})
#
time1_group = time1_group[['CITY_EN','DATE','RC','RE','RG','RI']]
time1_group.to_csv('WEATHER_FEATURES.csv',index = False)