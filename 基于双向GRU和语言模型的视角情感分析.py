#coding:utf-8
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import time
import os

print('read data...')
train_data = pd.read_csv('Train.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
test_data = pd.read_csv('Test.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
train_label = pd.read_csv('Label.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
addition_data = pd.read_csv('addition_data.csv', header=None, encoding='utf-8')[0]
train_data.dropna(inplace=True) # drop some empty sentences
neg_data = pd.read_excel('neg.xls', header=None)[0]
pos_data = pd.read_excel('pos.xls', header=None)[0]


