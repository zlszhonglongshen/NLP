# -*- coding: utf-8 -*-
"""
Created on 2018/12/14 18:07
@Author: Johnson
@Email:593956670@qq.com
@Software: PyCharm
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import PMMLPipeline, sklearn2pmml
import pandas as pd


iris = load_iris()

# 创建带有特征名称的 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 创建模型管道
iris_pipeline = PMMLPipeline([
 ("classifier", RandomForestClassifier())
])

# 训练模型
iris_pipeline.fit(iris_df, iris.target)

# 导出模型到 RandomForestClassifier_Iris.pmml 文件
sklearn2pmml(iris_pipeline, "RandomForestClassifier_Iris.pmml")