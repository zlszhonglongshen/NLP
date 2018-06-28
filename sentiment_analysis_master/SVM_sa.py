#coding=utf-8
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
import os
from pandas import Series,DataFrame
from sklearn.preprocessing import scale
from .BestWords import BestWords
import argparse

""" two feature vector:
"""
# path = r"F:\sentiment analysis\data"
# modelPath = path+r"\model"

class SVM_SA(object):
    def __init__(self,dataPath=None,modelPath=None,vectorFun=1,test_ratio=0.3,word_dim=300):
        # dataPath=None,modelPath=None,maxLen=50,test_ratio=0.3,output_dim=256,lstm_dim=128,drop_ratio=0.5,batch_size=32,epoch=10
        self.vectorFun = vectorFun # 1:w2v, 2:word count
        self.test_ratio = test_ratio
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.model = None
        self.word_dim = word_dim
        self.best_words = None

    def loadRawData(self): # load and cut
        neg=pd.read_excel(self.dataPath+r'neg.xls',header=None,index=None)
        pos=pd.read_excel(self.dataPath+r'pos.xls',header=None,index=None)
        pos['mark']=1
        neg['mark']=0
        pn=pd.concat([pos,neg],ignore_index=True)
        cw = lambda x: list(jieba.cut(x))
        pn['words'] = pn[0].apply(cw)
        self.pn = pn
        pn.to_csv(self.dataPath+r'pn.csv',encoding="utf-8")
        ## generate best words here
        self.best_words = BestWords(pn['words'],len(pos)).genBestWords()
        np.save(self.modelPath+r'best_words.npy',self.best_words)
        # self.best_words = np.load(self.modelPath+r"best_words.npy")
        # print "Loading data success"


    def genVector(self,min_count=10):
        if self.vectorFun==1:
            self._genW2Vector(min_count)
        elif self.vectorFun==2:
            self._genCountVector()
        else:
            # print "can only choose 1(w2v) or 2(word count) 2 funs, default w2v will be executed"
            self._genW2Vector()
        # print "Generate vectors successfully"

    def _genW2Vector(self,min_count=10):
        self._genW2VModel(min_count)
        self.pn['vector'] = self.pn['words'].map(lambda x: self._genAvgVector4Words(x))

    def _genAvgVector4Words(self,wordsList): ## 对句子的每个词生成平均的词向量
        vec = np.zeros(self.word_dim)
        count = 0.
        for w in wordsList:
            count += 1
            try:
                vec += self.w2vModel[w]
            except KeyError:
                continue
        if count!=0:
            vec /= count
        return vec

    def _genW2VModel(self,min_count=10):
        w2v = Word2Vec(size=self.word_dim,min_count=min_count)
        w2v.build_vocab(np.array(self.pn['words']))
        w2v.train(np.array(self.pn['words']))
        self.w2vModel = w2v
        print("w2v model was trained and saved")
        w2v.save(self.modelPath+r"w2v.pkl") #词向量模型

    def _genCountVector(self):
        self.pn['vector'] = self.pn['words'].map(lambda x:[x.count(w) for w in self.best_words])


    def train(self,min_count=10):
        self.loadRawData()
        self.genVector(min_count=min_count)
        x_train,x_test,y_train,y_test=train_test_split(self.pn['vector'],self.pn['mark'],test_size=self.test_ratio)
        model = SVC(kernel='rbf',verbose=True,probability=True)
        # print "SVM model train begin===================="
        x_train = np.vstack(x_train)
        x_test = np.vstack(x_test)
        model.fit(x_train,np.array(y_train)) ## x_train should be two dimensional array!!!!
        # print "SVM model train finished=================="
        self.model = model
        # print "Predicted Acc: "+str(model.score(x_test,y_test))
        joblib.dump(model,self.modelPath+r"svmModel"+str(self.vectorFun)+r".pkl")



    def predictInput(self,s):
        if self.model is None:
            self.loadModel()
        tag = self._predictInput2(s) if self.vectorFun==2 else self._predictInput1(s)
        res = "postive" if tag>=0.5 else "negtive"
        # print "Predicted sentiment is: "+res+" score = "+str(tag)
        return res+" score = "+str(tag)

    def loadModel(self):
        # print 'Try to load model......'
        try:
            self.model = joblib.load(self.modelPath+r"\svmModel"+str(self.vectorFun)+".pkl")
            if self.vectorFun == 2:
                self.best_words = np.load(self.modelPath+r"\best_words.npy")
            else:
                self.w2vModel = Word2Vec.load(self.modelPath+r"w2v.pkl")
            # print "Load model success"
        except:
            if self.dataPath is not None:
                try:
                    # print 'Model is unavaliable in the model path, try to train a new model with default params......'
                    self.train()
                except:
                    "Sth. is wrong"

    def _predictInput2(self,s): ## 78%
        words = list(jieba.cut(s))
        vec = [words.count(w) for w in self.best_words]
        return self.model.predict_proba(vec)[0,0]

    def _predictInput1(self,s): ## 77%
        words = list(jieba.cut(s))
        w2v = self._genAvgVector4Words(words)
        return self.model.predict(w2v)[0,0]


if __name__=="__main__":
    s = "这个书包质量不错，但是性价比很低"
    parser = argparse.ArgumentParser()
    parser.add_argument('-fun',dest="vectorFun",nargs='?',help="define the epoch num for the training process",type=int,default=1)
    parser.add_argument('-del',dest="delete",nargs='?',help="specify whether to delete trained model",type=bool,default=False)
    parser.add_argument('-path1',dest="dataPath",nargs='?',type=str,default="/home/pansl/sa/data/data2/",help="the path to load data")
    parser.add_argument('-path2',dest="modelPath",nargs='?',type=str,default="/home/pansl/sa/data/data2/model/",help="the path to save the model")
    parser.add_argument('-s',dest="input",nargs='?',type=str,default=s,help="the sentence you want to check")
    args = parser.parse_args()
    svm =  SVM_SA(dataPath=args.dataPath,modelPath=args.modelPath,vectorFun=args.vectorFun)
    # svm =  SVM_SA(dataPath=r"F:\sentiment analysis\data",modelPath=r"F:\sentiment analysis\data\model",vectorFun=2)
    svm.predictInput(s)


