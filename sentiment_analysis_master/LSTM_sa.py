#coding=utf-8
import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.models import model_from_yaml
import argparse
import yaml
import os
from pandas import Series,DataFrame

class LSTM_SA(object):
    def __init__(self,dataPath=None,modelPath=None,maxLen=50,test_ratio=0.3,output_dim=256,lstm_dim=128,drop_ratio=0.5,batch_size=32,epoch=10):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.model = None
        self.maxLen = maxLen # should be consistent with model's maxLen
        self.test_ratio = test_ratio
        self.output_dim = output_dim
        self.lstm_dim = lstm_dim
        self.drop_ratio = drop_ratio
        self.batch_size = batch_size
        self.epoch = epoch
        self.w2idx = None

    def loadRawData(self): # load and cut
        neg=pd.read_excel(self.dataPath+'neg.xls',header=None,index=None)
        pos=pd.read_excel(self.dataPath+'pos.xls',header=None,index=None)
        pos['mark']=1
        neg['mark']=0
        pn=pd.concat([pos,neg],ignore_index=True)
        cw = lambda x: list(jieba.cut(x))
        pn['words'] = pn[0].apply(cw)
        self.pn = pn
        # print "Loading data success"

    def genW2Idx(self):
        words = self.pn['words']
        w = []
        for i in words:
            w.extend(i)
        w = np.unique(w)
        id = range(1,len(w)+1)
        self.w2idx = dict(zip(w,id))
        np.save(self.modelPath+'w2idx.npy', self.w2idx)
        # print "save w2idx dict success"

    def _w2idx(self,wordList):
        return [self.w2idx.get(w,0) for w in wordList]


    def _genTrainTestData(self):
        from sklearn.cross_validation import train_test_split
        self.pn['sent'] = self.pn['words'].apply(lambda x:self._w2idx(x)) # convert words to word index list(LSTM inpput)
        self.pn['sent'] = list(sequence.pad_sequences(self.pn['sent'],maxlen=self.maxLen)) # pad to same maxLen
        x_train,x_test,y_train,y_test=train_test_split(self.pn['sent'],self.pn['mark'],test_size=self.test_ratio)
        x_train = np.vstack(x_train) ## all the input array dimensions should be the same: sentence pad was done
        x_test = np.vstack(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print( "split data success")
        return x_train,x_test,y_train,y_test

    def defineLSTM(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.w2idx)+1, output_dim=self.output_dim, input_length=self.maxLen)) #input_dim: 输入的词的维度， output_dim:词向量维度，input_length 每个句子的词长度
        model.add(LSTM(self.lstm_dim)) # try using a GRU instead, for fun
        model.add(Dropout(self.drop_ratio))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print( "define LSTM success")
        return model

    def train(self,test=True):
        self.loadRawData()
        self.genW2Idx()
        x_train,x_test,y_train,y_test = self._genTrainTestData()
        model = self.defineLSTM()
        model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
        model.fit(x_train, y_train, batch_size=self.batch_size, nb_epoch=self.epoch) #训练时间为若干个小时
        self.model = model
        if test:
            classes = model.predict_classes(x_test)
            acc = np_utils.accuracy(classes, y_test)
            print('Test accuracy:', acc)
        yaml_string = model.to_yaml()
        with open(self.modelPath+'lstm.yml', 'w') as outfile:
            outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
        model.save_weights(self.modelPath+'lstm.h5')
        print("Model is saved")

    def loadModel(self):
        print ('Try to load model......')
        try:
            with open(self.modelPath+'lstm.yml', 'r') as f:
                yaml_string = yaml.load(f)
            model = model_from_yaml(yaml_string)
            print ('Loading weights......')
            model.load_weights(self.modelPath+'lstm.h5')
            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            self.model = model
            self.w2idx = np.load(self.modelPath+'w2idx.npy').item()
        except:
            if self.dataPath is not None:
                try:
                    print ('Model is unavaliable in the model path, try to train a new model with default params......')
                    self.train()
                except:
                    "Sth. is wrong"

    def predictInput(self,s):
        if self.model is None:
            self.loadModel()
        if self.model is not None:
            ws = list(jieba.cut(s))
            idx = [self.w2idx.get(w,0) for w in ws]
            idx = np.array(idx).reshape((1,-1))
            cutIdx = list(sequence.pad_sequences(idx,maxlen=self.maxLen))
            tag = self.model.predict(np.vstack(cutIdx))[0][0]
            res = "postive" if tag>=0.5 else "negtive"
            # print "Predicted sentiment is: "+res+" score = "+str(tag)
            return res+" score = "+str(tag)
        else:
            print ("Training is failed")

if __name__=="__main__":
    s = "这个手机很差，性价比低"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',dest="epoch",nargs='?',help="define the epoch num for the training process",type=int,default=4)
    parser.add_argument('-bs',dest="batch_size",nargs='?',help="define the batch size",type=int,default=32)
    parser.add_argument('-del',dest="delete",nargs='?',help="specify whether to delete trained model",type=bool,default=False)
    parser.add_argument('-path1',dest="dataPath",nargs='?',type=str,default="/home/pansl/sa/data/data2/",help="the path to load data")
    parser.add_argument('-path2',dest="modelPath",nargs='?',type=str,default="/home/pansl/sa/data/data2/model/",help="the path to save the model")
    parser.add_argument('-s',dest="input",nargs='?',type=str,default=s,help="the sentence you want to check")
    args = parser.parse_args()

    if args.delete: # to retrain the model,del it first
        os.system("rm -rf "+args.modelPath)
        os.system("mkdir "+args.modelPath)
    lstm =  LSTM_SA(dataPath=args.dataPath,modelPath=args.modelPath,epoch=args.epoch,batch_size=args.batch_size)
    lstm.predictInput(args.input)








