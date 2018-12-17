# -*- coding: utf-8 -*-
"""
Created on 2018/12/17 17:49
@Author: Johnson
@Email:593956670@qq.com
@Software: PyCharm
"""
import pandas as pd
from  xgboost.sklearn import XGBClassifier
from sklearn2pmml import PMMLPipeline
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris

iris = load_iris()
# 创建带有特征名称的 DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
#nthread=4,# cpu 线程数 默认最大
learning_rate= 0.3, # 如同学习率
min_child_weight=1,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
max_depth=6, # 构建树的深度，越大越容易过拟合
gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
n_estimators=100, #树的个数
seed=1000)



pipeline = PMMLPipeline([("classifier", clf)])
pipeline.fit(iris_df, iris.target)
sklearn2pmml(pipeline,"xgboost.pmml",with_repr = True)
