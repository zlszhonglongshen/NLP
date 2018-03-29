# coding=utf-8
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
import re

d = pd.read_json('data.json') #训练数据已经被预处理成为标准的json格式
d.index = range(len(d))
word_size = 128 #词向量维度
maxlen = 80 #句子截断长度

'''
修改分词函数，主要是：
1、英文和数字部分不分词，直接返回；
2、双书名号里边的内容不分词；
3、双引号里边如果是十字以内的内容不分词；
4、超出范围内的字符全部替换为空格；
5、分词使用结巴分词，并关闭新词发现功能。
'''
not_cuts = re.compile(u'([\da-zA-Z \.]+)|《(.*?)》|“(.{1,10})”')
re_replace = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z《》\(\)（）“”·\.]')

def mycut(s):
    result = []
    j = 0
    s = re_replace.sub(' ', s)
    for i in not_cuts.finditer(s):
        result.extend(jieba.lcut(s[j:i.start()], HMM=False))
        if s[i.start()] in [u'《', u'“']:
            result.extend([s[i.start()], s[i.start()+1:i.end()-1], s[i.end()-1]])
        else:
            result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(jieba.lcut(s[j:], HMM=False))
    return result

d['words'] = d['content'].apply(mycut) #分词

def label(k): #将输出结果转换为标签序列
    s = d['words'][k]
    r = ['0']*len(s)
    for i in range(len(s)):
        for j in d['core_entity'][k]:
            if s[i] in j:
                r[i] = '1'
                break
    s = ''.join(r)
    r = [0]*len(s)
    for i in re.finditer('1+', s):
        if i.end() - i.start() > 1:
            r[i.start()] = 2
            r[i.end()-1] = 4
            for j in range(i.start()+1, i.end()-1):
                r[j] = 3
        else:
            r[i.start()] = 1
    return r

d['label'] = map(label,tqdm(iter(d.index)))

#随机打乱数据
idx = range(len(d))
d.index = idx
np.random.shuffle(idx)
d = d.loc[idx]
d.index = range(len(d))

#读入测试数据并进行分词
dd = open('opendata_20w').read().decode('utf-8').split('\n')
dd = pd.DataFrame([dd]).T
dd.columns = ['content']
dd = dd[:-1]
print (u'测试语料分词中......')
dd['words'] = dd['content'].apply(mycut)


'''
用gensim来训练Word2vec
1：联合训练预料和测试预料一起训练
2：经过测试用skip gram效果会好些
'''
import gensim,logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word2vec = gensim.models.Word2Vec(dd['words'].append(d['words']),
                                  min_count=1,
                                  size=word_size,
                                  workers=20,
                                  iter=20,
                                  window=8,
                                  negative=8,
                                  sg=1)

# word2vec.save('word2vec_words_final.model')
word2vec.init_sims(replace=True) #预先归一化，使得词向量不受尺度影响

print (u'正在进行第一次训练......')
'''
用最新版本的Keras训练模型，使用GPU加速（我的是GTX 960）
其中Bidirectional函数目前要在github版本才有
'''
from keras.layers import Dense, LSTM, Lambda, TimeDistributed, Input, Masking, Bidirectional
from keras.models import Model
from keras.utils import np_utils
from keras.regularizers import l1 #通过L1正则项，使得输出更加稀疏

sequence = Input(shape=(maxlen, word_size))
mask = Masking(mask_value=0.)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(mask)
blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(blstm)
output = TimeDistributed(Dense(5, activation='softmax', activity_regularizer=l1(0.01)))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



