#coding=GBK
'''
Created on 2017��4��5��

@author: Scorpio.Lu
'''
import collections
import re  
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
'''
��ȡԴ�ļ�,��תΪlist���
@param filename:�ļ���
@return: list of words
'''
def read_file(filename):
    f = open(filename,'r',encoding='gbk')
    file_read = f.readlines()
    words_ = re.sub(""," ",file_read).lower() #����ƥ�䣬ֻ���µ��ʣ��Ҵ�д��Сд
    words = list(words_.split()) #length of words
    return words

words = read_file('../')
vocabulary_size = 2000 #Ԥ����Ƶ�����ʿ�ĳ���
count = [['UNK',-1]] #��ʼ������Ƶ��ͳ�Ƽ���

'''
1����words�г��ֹ��ĵ�����Ƶ��ͳ�ƣ�ȡtop 1999Ƶ���ĵ��ʷ���dictionary�У��Ա���ٲ�ѯ
2�����Ȿ�鵥�ʿ���룬������top 1999֮��ĵ��ʣ�ͳһ���䡰UNK",���Ϊ0����ͳ����Щ���ʵ�����
3:return �Ȿ��ı���data,ÿ�����ʵ�Ƶ��ͳ��count���ʻ��dictionary�Լ���ת��ʽreverse_dictionary
'''

def build_dataset(words):
    counter = collections.Counter(words).most_common(vocabulary_size-1)
    count.extend(counter)
    #�dictionary
    dictionary = {}
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = []
    #ȫ������תΪ���
    #���ж���������Ƿ������dictionary ����� ��תΪ��� ������� ��תΪ���0
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count+=1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary = build_dataset(words)
del words #ɾ��ԭʼ�����б���Լ�ڴ�

data_index = 0

'''
����skip-gramģʽ
����Word2vecѵ������
@:param batch_size:ÿ������ѵ����������
@:param num_skips:Ϊÿ���������ɶ�������������������2����batch_size��Ϣ��num_skips������������������ȷ����һ��Ŀ��ʻ����ɵ�������ͬһ������
'''

def generate_bac
