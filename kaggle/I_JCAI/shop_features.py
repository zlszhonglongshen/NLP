# -*- coding:utf-8 -*-
"""
Created on 18-12-13 下午4:53
@Author:Johnson
@Email:593956670@qq.com 
"""
import copy
import sys
import pandas as pd
import numpy as np
sys.path.append('../Tools')
import datetime

shop_info_en = pd.read_csv('./shop_info_en.csv')

#category infomation
category = pd.read_csv('../shop_cat.csv')
one_hot = pd.get_dummies(category['CAT'])
category = category.join(one_hot)
shop_sj = map(lambda x:'SJ'+str(x).zfill(2),np.arange(15))

category.columns = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN','Num', 'CAT'] + shop_sj
shop_info_en = pd.merge(shop_info_en,category,on = ['SHOP_CA1_EN','SHOP_CA2_EN','SHOP_CA3_EN'],how = 'left')

shop_info_en['shop_sco'] = shop_info_en['shop_sco'].fillna(shop_info_en['shop_sco'].mean())
shop_info_en['shop_sco'] = shop_info_en['shop_sco'].fillna(shop_info_en['shop_sco'].mean())
shop_info_en = shop_info_en[['SHOP_ID','CITY_EN','SHOP_PAY','SHOP_SCO','SHOP_COM','SHOP_LEV','SHOP_LOC_N'] + shop_sj]
shop_sd = map(lambda x:'sd'+str(x).zfill(2),np.arange(5))
shop_info_en.columns = ['shop_id','city_en']+shop_sd+shop_sj

paynw = pd.read_csv('./date_new/user_pay_pay_new.csv')
vienw = pd.read_csv('./data_new/user_view_new.csv')

holi = pd.read_csv("./holi.csv")
holi['date'] = [(lambda x:str(datetime.datetime.strptime(str(x),'%Y%m%d').date()))(x) for i in holi['date']]

#%% calculate top hours SE SF
top_n = 1
shop_id = []
shop_hour_head = []
shop_pct_head = []
for shop_ind in range(1,2001):
    tt = paynw[paynw['shop_id']==shop_ind]
    tt2 = tt.groupby('hour',as_index=False).sum()
    tt3 = tt2.sort_values('num_post',ascending=False,inplace=False)
    tt4 = tt3.head(top_n)['hour'].values
    tt5 = tt3.head(top_n)['num_post'].values/tt3['num_post'].sum()
    shop_id.append(shop_ind)
    shop_hour_head.append(tt4)
    shop_pct_head.append(tt5)

shop_id_df = pd.DataFrame(shop_id)
shop_hour_head_df = pd.DataFrame(shop_hour_head)
shop_pct_head_df = pd.DataFrame(shop_pct_head)

sell_info = pd.concat([shop_id_df,shop_hour_head_df,shop_pct_head_df],axis=1)
shop_se = [(lambda x:('se'+str(x).zfill(2)))(x) for x in range(top_n)]
shop_sf = [(lambda x:('sf'+str(x).zfill(2)))(x) for x in range(top_n)]
sell_info.columns = ['shop_id']+shop_se+shop_sf



#calculate top hours
shop_id = []
shop_open = []
shop_close = []
shop_last = []
shop_mean = []
for shop_ind in range(1,2001):
    tt = paynw[paynw['shop_id']==shop_ind]
    tt2 = tt.groupby('date',as_index=False).min().mean()
    tt3 = tt.groupby('date',as_index=False).max().mean()
    tt['mean'] = tt['num_post']*tt['hour']
    shop_id.append(shop_ind)
    shop_open.append(tt2.hour)
    shop_close.append(tt3.hour)
    shop_last.append(tt3.hour-tt2.hour)
    shop_mean.append(tt['mean'].sum()/tt['num_post'].sum())
shop_id_df = pd.DataFrame(shop_id)
shop_open_df = pd.DataFrame(shop_open)
shop_close_df = pd.DataFrame(shop_close)
shop_last_df = pd.DataFrame(shop_last)
shop_mean_df = pd.DataFrame(shop_mean)
hour_info = pd.concat([],axis=1)
shop_sg = map(lambda x:'SG'+str(s).zfill(2),np.arange(4))
hour_info.columns = ['shop_id']+shop_sg
#need to fillna 0

