# -*- coding:utf-8 -*-
"""
Created on 18-12-14 下午4:04
@Author:Johnson
@Email:593956670@qq.com 
"""
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
from sklearn2pmml import PMMLPipeline,sklearn2pmml

data = np.random.rand(10,3)
# print(data)

estimator = KMeans(n_clusters=2,init='k-means++',n_jobs=1)

km_pipeline = PMMLPipeline([("KM",estimator)])

km_pipeline.fit(data)

sklearn2pmml(km_pipeline,"KM.pmml")




