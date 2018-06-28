#coding=utf-8
import os
import numpy as np
os.chdir(r"F:\sentiment analysis\data")

class BestWords(object):
    ## generate best words to consider bigger KaFang value, better the word
    def __init__(self,words,posLen,wordNum=3000): ## posWords and negWords are Series
        self.posWords = words[0:posLen] # if ['word'] return df else series
        self.negWords = words[posLen:]
        self.wordNum = wordNum
        self.totalDict = {}
        self.posDict = {}
        self.negDict = {}

    def _add2worddict(self):

        def loop(x,dict1,dict2):
            for w in x:
                dict1[w] = dict1.get(w,0)+1
                dict2[w] = dict2.get(w,0)+1

        self.posWords.map(lambda x:loop(x,self.posDict,self.totalDict))
        self.negWords.map(lambda x:loop(x,self.negDict,self.totalDict))


    @staticmethod
    def calScore(n_ii, n_ix, n_xi, n_xx): #以卡方值来说明一个词是否可区别pos或neg
        n_ii = n_ii # 该word的postive次数
        n_io = n_xi - n_ii #剩余词语的postive次数
        n_oi = n_ix - n_ii #该词的negtive次数
        n_oo = n_xx - n_ii - n_oi - n_io #剩余词语的negtive次数
        return n_xx * (float((n_ii*n_oo - n_io*n_oi)**2)/((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo)))

    def _bestWords(self,delStopWords=False,inSentimentWords=True):
        pos_word_num = len(self.posDict)
        total_word_num = len(self.totalDict)
        score = {word:BestWords.calScore(self.posDict.get(word,0),freq,pos_word_num,total_word_num) for word,freq in self.totalDict.items()}
        sortedScore = sorted(score.items(),key=lambda x:x[1],reverse=True) ## dict.iterms() is a list with tuple element
        if delStopWords:
            stopWords = self._readStopWords()
            best_words = [w for w,s in sortedScore[:self.wordNum] if w not in stopWords]
        else:
            best_words = [w for w,s in sortedScore[:self.wordNum]]
        if inSentimentWords:
            sentiWords = np.load("sentimentWords.npy")
            best_words = [w for w,s in sortedScore if w in sentiWords][:self.wordNum]
        self.best_words = best_words

    def genBestWords(self):
        self._add2worddict()
        self._bestWords()
        return self.best_words

    def _readStopWords(self):
        pass


if __name__=="__main__":
    pass
    # import pandas as pd
    # import jieba
    # dataPath = r"F:\sentiment analysis\data"
    # neg=pd.read_excel(dataPath+r'\neg.xls',header=None,index=None)
    # pos=pd.read_excel(dataPath+r'\pos.xls',header=None,index=None)
    # pos['mark']=1
    # neg['mark']=0
    # pn=pd.concat([pos,neg],ignore_index=True)
    # cw = lambda x: list(jieba.cut(x))
    # pn['words'] = pn[0].apply(cw)
    # bw = BestWords(pn['words'],len(pos))
    # bw._add2worddict()
    # bw._bestWords()
    # print bw.best_words


