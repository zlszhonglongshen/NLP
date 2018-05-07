# coding=utf-8
# from DB import DB
import re
import jieba
import jieba.analyse
import codecs
import nltk
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def wordsCluster(textUrl, fencijieguo='fencijieguo.txt', vectorSize=100, classCount=10):
    '''
    textUrl:输入文本的本地路径，
    fencijieguo：分词结果存储到本地路径，
    vectorSize：词向量大小，
    classCount：分类大小


    '''

    # 读取文本
    textstr = ''

    for line in open(textUrl):
        textstr += line
    # 使用jieba分词

    # 分词结果放入到的文件路径
    outfenci = codecs.open(fencijieguo, "a+", 'utf-8')
    tempList = re.split(u'[。？！?!]', textstr)
    for row in tempList:
        if row != None and row != '':
            # 分词结果放入到文件中
            readline = ' '.join(list(jieba.cut(row, cut_all=False))) + '\n'
            outfenci.write(readline)
    outfenci.close()

    # word2vec向量化
    model = Word2Vec(LineSentence(fencijieguo), size=vectorSize, window=5, min_count=3, workers=4)

    # 获取model里面的所有关键词
    keys = model.wv.vocab.keys()

    # 获取词对于的词向量
    wordvector = []
    for key in keys:
        wordvector.append(model[key])

    # 分类
    clf = KMeans(n_clusters=classCount)
    s = clf.fit(wordvector)
    print(s)
    # 获取到所有词向量所属类别
    labels = clf.labels_

    # 把是一类的放入到一个集合
    classCollects = {}
    for i in range(len(keys)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(keys[i])
        else:
            classCollects[labels[i]] = [keys[i]]

    return classCollects