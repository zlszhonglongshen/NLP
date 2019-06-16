# -*- coding: utf-8 -*-
"""
Created on 2019/1/8 11:05
@Author: Johnson
@Email:593956670@qq.com
@File: LR_PMML.py
"""
import pandas
import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()

# 创建带有特征名称的 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline

pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], [ContinuousDomain(), Imputer()])])),
	("pca", PCA(n_components = 3)),
	("selector", SelectKBest(k = 2)),
	("classifier", LogisticRegression())
])
pipeline.fit(iris_df, iris.target)

from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "LogisticRegressionIris.pmml", with_repr = True)