#!/usr/bin/python_人脸属性相关
"""
https://flystarhe.github.io/2016/09/04/word2vec-test/
"""
#coding:utf-8
import os,sys,codecs
import gensim,logging,multiprocessing
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ("Usage: python_人脸属性相关 script.py infile outfile")
        sys.exit()
    infile, outfile = sys.argv[1:3]
    model = gensim.models.Word2Vec(gensim.models.word2vec.LineSentence(infile), size=400, window=5, min_count=5, sg=0,
                                   workers=multiprocessing.cpu_count())
    model.save(outfile)
    model.save_word2vec_format(outfile + '.vector', binary=False)



    """
>>> from gensim.models import Word2Vec
>>> model = Word2Vec.load_word2vec_format("zh.wiki.model.vector", binary=False)
>>> model[u"男人"] #词向量
array([  3.70501429e-01,  -2.38224363e+00,  -1.20320223e-01,  ..
>>> model.similarity(u"男人", u"女人")
0.8284998105297946
>>> print model.doesnt_match(u"早餐 晚餐 午餐 中心".split())
中心
>>> words = model.most_similar(u"男人")
>>> for word in words:
...     print word[0], word[1]
...
    """


    """


    """