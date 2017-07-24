from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

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


k-neighbours-algorithm  : KNNN

fit(X, y) 	                                    Fit the model using X as training data and y as target values
get_params([deep]) 	                            Get parameters for this estimator.
kneighbors([X, n_neighbors, return_distance]) 	Finds the K-neighbors of a point.
kneighbors_graph([X, n_neighbors, mode]) 	    Computes the (weighted) graph of k-Neighbors for points in X
predict(X) 	                                    Predict the class labels for the provided data
predict_proba(X) 	                            Return probability estimates for the test data X.
score(X, y[, sample_weight]) 	                Returns the mean accuracy on the given test data and labels.
set_params(\*\*params) 	                        Set the parameters of this estimator.

"""

def showFiled(predictionValue):
      df = pd.DataFrame({"predictionValue": predictionValue})
      print(df)
      df["category"] = df["predictionValue"].astype("category")
      print('categories >>>\n', df["category"].cat.categories.size)
      df["category"].cat.categories = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"][:df["category"].cat.categories.size]
      df["category"] = df["category"].cat.set_categories(["Iris Setosa", "Iris Versicolour", "Iris Virginica"])
      print(df)
      print("Data is belong to for categories >> \n",df["category"])



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
knnEstimator = KNeighborsClassifier(n_neighbors=1)


print(knnEstimator)

# trainnig the data
knnEstimator.fit(X,y)


#predict
inputDataSet = [[3,5,4,2],[5,4,3,2]]
predictionValue = knnEstimator.predict(inputDataSet)
print('\npredictionValue > ', predictionValue , inputDataSet)
showFiled(predictionValue )

# tune the model
knnEstimator = KNeighborsClassifier(n_neighbors=5)
knnEstimator.fit(X,y)
#predict
inputDataSet = [[3,5,4,2],[5,4,3,2]]
predictionValue = knnEstimator.predict(inputDataSet)
print('\npredictionValue for k=5 > ', predictionValue)

showFiled(predictionValue )


