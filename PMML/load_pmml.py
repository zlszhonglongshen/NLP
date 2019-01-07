# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 18:30:11 2018

@author: johnson.zhong
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn_pmml_model.tree import PMMLTreeClassifier

# Prepare data
iris = load_iris()

# We only take the two corresponding features
X = pd.DataFrame(iris.data)
X.columns = np.array(iris.feature_names)
y = pd.Series(np.array(iris.target_names)[iris.target])
y.name = "Class"

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=123)

df = pd.concat([Xte, yte], axis=1)

clf = PMMLTreeClassifier(pmml="C:\\Users\\johnson.zhong\\Desktop\\sklearn-pmml-model-master\\models\\RandomForestClassifier_Iris.pmml")
print(clf.predict(Xte))
print(clf.score(Xte, yte))