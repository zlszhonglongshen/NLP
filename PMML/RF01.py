# -*- coding: utf-8 -*-
"""
Created on 2018/12/14 18:07
@Author: Johnson
@Email:593956670@qq.com
@Software: PyCharm
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import PMMLPipeline, sklearn2pmml
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn_pandas import DataFrameMapper


iris = load_iris()

# 创建带有特征名称的 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head())

class DataPrepare1(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x["new1"] = x["sepal length (cm)"] + x["sepal width (cm)"]
        # new1 = x["sepal length (cm)"] + x["sepal width (cm)"]
        # print("new1")
        # print(np.c_[x,new1])
        return x


class DataPrepare2(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x["new2"] = 2 * x["sepal length (cm)"]
        # new2 = 2 * x["sepal length (cm)"]
        # x = x.drop("Species", axis=1)
        print(x.head())
        return x

default_mapper = DataFrameMapper([(i, None) for i in iris.feature_names + ['Species']])
# 创建模型管道
iris_pipeline = PMMLPipeline([
 ("new1", DataPrepare1()),
 ("new2", DataPrepare2()),
 ("classifier", RandomForestClassifier())
])

# 训练模型
iris_pipeline.fit_transform(iris_df, iris.target)

print(iris_pipeline)

# 导出模型到 RandomForestClassifier_Iris.pmml 文件
sklearn2pmml(iris_pipeline, "RF.pmml")
print("程序运行完成!!!")