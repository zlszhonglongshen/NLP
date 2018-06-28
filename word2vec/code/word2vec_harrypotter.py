#coding=GBK
'''
Created on 2017年4月5日

@author: Scorpio.Lu
'''
import collections
import re  
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
'''
读取源文件,并转为list输出
@param filename:文件名
@return: list of words
'''
def read_file(filename):
    f = open(filename,'r',encoding='gbk')
    file_read = f.readlines()
    words_ = re.sub(""," ",file_read).lower() #正则匹配，只留下单词，且大写改小写
    words = list(words_.split()) #length of words
    return words

words = read_file('../')
vocabulary_size = 2000 #预定义频繁单词库的长度
count = [['UNK',-1]] #初始化单词频数统计集合

'''
1：给words中出现过的单词做频数统计，取top 1999频数的单词放入dictionary中，以便快速查询
2：给这本书单词库编码，出现在top 1999之外的单词，统一另其“UNK",编号为0，并统计这些单词的数量
3:return 这本书的编码data,每个单词的频数统计count，词汇表dictionary以及反转形式reverse_dictionary
'''

def build_dataset(words):
    counter = collections.Counter(words).most_common(vocabulary_size-1)
    count.extend(counter)
    #搭建dictionary
    dictionary = {}
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = []
    #全部单词转为编号
    #先判断这个单词是否出现在dictionary 如果是 就转为编号 如果不是 则转为编号0
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
del words #删除原始单词列表，节约内存

data_index = 0

'''
采用skip-gram模式
生成Word2vec训练样本
@:param batch_size:每个批次训练多少样本
@:param num_skips:为每个单词生成多少样本，本次试验是2个，batch_size鼻息是num_skips的整数倍，这样可以确保由一个目标词汇生成的样本在同一批次中
'''

def generate_bac
