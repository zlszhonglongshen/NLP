# coding=utf-8
import os
import re
import codecs
import jieba
os.chdir(r"F:\sentiment analysis\data\dictData")

class Dict_sa(object):

    def __init__(self,dataPath="/home/pansl/sa/data/data2/dictData/"): ## dataPath=""
        # load dicts
        self.adverbDict = Dict_sa.loadDict("adverb_dict.txt",dataPath)
        self.conjunctionDict = Dict_sa.loadDict("conjunction_dict.txt",dataPath)
        self.denialDict = Dict_sa.loadDict("denial_dict.txt",dataPath)
        self.negativeDict = Dict_sa.loadDict("negative_dict.txt",dataPath)
        self.positiveDict = Dict_sa.loadDict("positive_dict.txt",dataPath)
        self.puctuationDict = Dict_sa.loadDict("punctuation_dict.txt",dataPath)

    def analyseSentence(self,sentence):
        score = 0
        clauses = Dict_sa.splitSentence(sentence)
        for clau in clauses:
            score += self.analyseClause(clau)
        if 0.5>score>-0.5:
            tag = "neutral"
        elif score>0.5:
            tag = "positive"
        else:
            tag = "negtive"
        return tag + " score = "+ str(score)



    def analyseClause(self,clause):
        segs = list(jieba.cut(clause))
        score = 0
        conjLs, denialLs, puncts, pos, neg = {},{},{},{},{}
        lastSentimentW = 0
        for i,w in enumerate(segs): ## 对于情感词 直接加和 连词、否定词、标点作记录
            if w in self.conjunctionDict:
                conjLs[w] = self.conjunctionDict.get(w)
            elif w in self.puctuationDict:
                puncts[w] = self.puctuationDict.get(w)
            elif w in self.negativeDict:
                info = self.analyseSentimentW(w,segs[lastSentimentW:i],"neg")
                lastSentimentW = i
                score += info['score']
            elif w in self.positiveDict:
                info = self.analyseSentimentW(w,segs[lastSentimentW:i],"pos")
                lastSentimentW = i
                score += info['score']
            else:
                continue
        for conj,s in conjLs.items():
            score *= s
        for punc,s in puncts.items():
            score *= s
        return score


    def analyseSentimentW(self,w,segs,tag):
        adverbLs = [] # w:[score,pos]
        denialLs = []
        score = -self.negativeDict.get(w) if tag=="neg" else self.positiveDict.get(w)
        centerW = [w,score]
        w = 1
        for i,w in enumerate(segs):
            if w in self.denialDict:
                denialLs.append([w,self.denialDict.get(w),i])
            elif w in self.adverbDict:
                adverbLs.append([w,self.adverbDict.get(w),i])
            else:
                continue
        wDen = -1 if len(denialLs)%2==1 else 1
        wAdv = 1
        for i in adverbLs:
            wAdv *= i[1]
        w = 1
        if len(adverbLs)==1 and len(denialLs)==1 and adverbLs[0][2]>denialLs[0][2]: #很不 不很
            w = 0.5
        score = score*wDen*wAdv*w
        return {"score":score,"cen":centerW,"adverbLs":adverbLs,"denialLs":denialLs}


    @staticmethod
    def loadDict(file,dataPath):
        file = dataPath+file
        saDict = {}
        with codecs.open(file,encoding="utf-8") as f:
            for line in f.readlines():
                pat = re.compile(u'\s+')
                res = pat.split(line.strip())
                if len(res)==2:
                    saDict[res[0]] = float(res[1])
        return saDict


    @staticmethod
    def splitSentence(sentence):
        pattern = re.compile(u"[，、。%！；？?,!～~.…]+")
        split_clauses = pattern.split(sentence.strip())
        punctuations = pattern.findall(sentence.strip())
        try:
            split_clauses.remove("")
        except ValueError:
            pass
        punctuations.append("")
        clauses = [''.join(x) for x in zip(split_clauses, punctuations)]
        return clauses




if __name__=="__main__":
    d = Dict_sa()
    os.chdir(r"F:\sentiment analysis\data")
    import pandas as pd
    # neg = pd.read_excel("neg.xls",header=None,index=None)
    # neg = neg.iloc[:,:]
    # # s = u'这本书观点不错，但举的论据似乎不是很好。比如，女人的外貌很重要--其论据就是某某整容之后交了多帅的男友等等。。。不是特别喜欢'
    # # print repr(neg[0][0])
    # neg["res"] = neg[0].map(lambda x:d.analyseSentence(x))
    # neg.to_csv("1.csv",encoding="utf-8")
    # print float(sum(neg["res"]<0))/len(neg)
    #
    # pos = pd.read_excel("pos.xls",header=None,index=None)
    # pos = pos.iloc[:,:]
    # pos["res"] = pos[0].map(lambda x:d.analyseSentence(x))
    # pos.to_csv("2.csv",encoding="utf-8")
    # print float(sum(pos["res"]>0))/len(pos)
    s1 = u"满怀着期待购买了毕淑敏的《女心理师》，阅读之前对这部作品充满了期待，因为我已读过不少毕淑敏的作品。非常喜欢她用平实而又富有哲理的简洁的语言讲述的一个又一个故事，特别是08年读过的《女工》，连续读了很多次，非常喜欢。说句心里话，读完《女心理师》感觉有些失望。虽然我是一个学理科的人，但非常喜欢阅读。这部作品的主人公贺顿是一个由于自身的出身卑微，但又聪明、有一定追求的人。在不断的努力下获取了心理咨询师的资格，为了开办诊所，嫁给一个内心厚道但根本没有感情的男人。"
    s = u'我不知道，这样的女主人公带给读者的是什么？同时，作品中的几个故事远远脱离日常生活，看着就感觉作者纯粹是为了写作而写作，毫无价值，除了有几句话让我留有一些印象外，其余的看完书后就忘记了。真是失望！'
    # print repr(pos[0][0])
    s = u"我觉得你挺好的"
    print( d.analyseSentence(s))


# print "/".join(list(jieba.cut(s)))


