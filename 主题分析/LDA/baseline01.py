# coding=utf-8
#使用python进行简单文本类数据分析，包括：
# 1：分词
# 2：生成语料库，tf-idf加权
# 3：lda主题提取模型
# 4：词向量化Word2vec

import pymysql
import pandas as pd
import pandas.io.sql as sql
import jieba
import nltk
import jieba.posseg as pseg
from gensim import corpora,models,similarities
import re
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INGO)

if __name__=='__main__':
    #用户词典导入
    jieba.load_userdict("f:/userdict.txt")
    #1.读取数据
    conn = pymysql.connect(host='', port=3306, charset='utf8', user='', passwd='', db='')
    df = sql.read_sql('select * from test',conn)
    conn.close()
    cont = df['commont']
    #2.简单的过滤某些特定词
    pattern = r'标签|心得'
    regx = re.compile(pattern)
    r = lambda x:regx.sub('',x)
    filtercont = cont.map(r)
    #分词+选词
    nwordall = []
    for t in cont:
        words = pseg.cut(t)
        nword = ['']
        for w in words:
            if ((w.flag == 'n' or w.flag == 'v' or w.flag == 'a') and len(w.word) > 1):
                nword.append(w.word)
        nwordall.append(nword)

    #3.选择后的词生成词典
    dictionary = corpora.Dictionary(nwordall)
    # print dictionary.token2id
    # 生成语料库
    corpus = [dictionary.doc2bow(text) for text in nwordall]
    # tfidf加权
    tfidf = models.TfidfModel(corpus)
    # print tfidf.dfsx
    # print tfidf.idf
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #      print doc
    #4.主题模型lda,可用于降维
    #lda流式数据建模计算，每块10000条数据记录，提取50个主题
    lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=50,update_every=1, chunksize=10000, passes=1)
    for i in range(0,3):
        print(lda.print_topics(i)[0])
    # lda全部数据建模，提取100个主题
    # lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100, update_every=0, passes=20)
    # 利用原模型预测新文本主题
    # doc_lda = lda[corpus_tfidf]

    #5.word2vec词向量化，可用于比较词相似度，寻找对应关系，词聚类
    # sentences = models.word2vec.LineSentence(nwordall)
    #size为词向量为维度数，window窗口范围，min_count频数小于5的词忽略，workers是线程数
    model = models.word2vec.Word2Vec(nwordall,size=100,window=5,min_count=5,workers=4)
    # model.save("F:\word2vecmodels") 建模速度慢，建议保存，后续直接调用
    # model = models.word2vec.Word2Vec.load("F:\word2vecmodels")
    # 向量表示
    sim = model.most_similar(positive=[u'洗衣', u'方便'])
    #相近词
    for s in sim:
        print(s[0],s[1])

