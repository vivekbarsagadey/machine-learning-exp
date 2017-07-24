from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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

def testDataEvalution(logisticRegressionEstimator, testData , actualData =[]):
      # predict
      predictionValue = logisticRegressionEstimator.predict(testData)
      print('\nactualData > ', actualData)
      print('\npredictionValue > ', predictionValue)
      showFiled(predictionValue)
      diff_arr = np.equal(actualData, predictionValue)
      print('diff_arr >> ',diff_arr)
      correct_answers = np.sum(diff_arr)
      percent_diff = correct_answers / actualData.__len__() * 100
      print("Percentage Match is: ", percent_diff)

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
testDataEvalution(logisticRegressionEstimator ,[[3,5,4,2],[5,4,3,2]],[0,0] )


logisticRegressionEstimator.fit(X,y)
testDataEvalution(logisticRegressionEstimator ,X , iris.target)

"""
Training Accuracy where train data and test data are same for calculating accuracy_score
"""


# metrics will use to check accuracy, Metrics to assess performance on classification task given class prediction
logisticRegressionEstimator = LogisticRegression()
logisticRegressionEstimator.fit(X,y)
predictionValue = logisticRegressionEstimator.predict(X)
print ("Metrics Accuracy Score for Logistic Regression Estimator  : " , metrics.accuracy_score(y,predictionValue))


# Train/ Test split
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

logisticRegressionEstimator = LogisticRegression()
logisticRegressionEstimator.fit(X_train,y_train)
predictionValue = logisticRegressionEstimator.predict(X_test)
print ("Metrics Accuracy Score for Logistic Regression Estimator in Train/ Test split method : " , metrics.accuracy_score(y_test,predictionValue))

