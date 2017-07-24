from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pprint import pprint

"""
Iris Data Set: 

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa      = 0
-- Iris Versicolour = 1
-- Iris Virginica   = 2


Logistic regression

"""


def showFiled(predictionValue):
      df = pd.DataFrame({"predictionValue": predictionValue})
      print(df)
      df["category"] = df["predictionValue"].astype("category")
      print('categories >>>\n', df["category"].cat.categories.size)
      df["category"].cat.categories = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"][
                                      :df["category"].cat.categories.size]
      df["category"] = df["category"].cat.set_categories(["Iris Setosa", "Iris Versicolour", "Iris Virginica"])
      print(df)
      print("Data is belong to for categories >> \n", df["category"])

iris = datasets.load_iris()

print(type(iris))
print(iris.data, iris.target,
      iris.target_names,
      iris.DESCR,
      iris.feature_names)

print('\niris.target >>\n', iris.target)
print('\niris.target_names >>\n', iris.target_names)
print('\niris.data.shape >>\n', iris.data.shape)

X= iris.data
y = iris.target

print('\nX.shape >>', X.shape)
print('\ny.shape >>', y.shape)
# k= 1
logisticRegressionEstimator = LogisticRegression()


print(logisticRegressionEstimator)

# trainnig the data
logisticRegressionEstimator.fit(X,y)


#predict
predictionValue = logisticRegressionEstimator.predict([[3,5,4,2],[5,4,3,2]])
print('\npredictionValue > ', predictionValue)
showFiled(predictionValue )






