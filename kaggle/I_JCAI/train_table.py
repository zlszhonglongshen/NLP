# -*- coding:utf-8 -*-
"""
Created on 18-12-14 下午2:26
@Author:Johnson
@Email:593956670@qq.com 
"""
import pandas as pd
import numpy as np
import datetime
import copy

import sys
holi = pd.read_csv('./holi.csv')
holi = holi.set_index(['DATE'],drop=True)
holi_tab = holi.transpose()
holi_tab.columns = [str((datetime.datetime.strptime('20150626','%Y%m%d')+datetime.timedelta(days=x)).date())for x in range(holi_tab.shape[1])]

#readin shop data
