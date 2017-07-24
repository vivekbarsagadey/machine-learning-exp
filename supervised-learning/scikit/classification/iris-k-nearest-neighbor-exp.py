from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
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

def testDataEvalution(knnEstimator , testData , actualData =[] , k = 5):
      predictionValue = knnEstimator.predict(testData)
      print('\npredictionValue for k=5 > ', predictionValue)
      showFiled(predictionValue)
      diff_arr = np.equal(actualData, predictionValue)
      #print('diff_arr >> ', diff_arr)
      correct_answers = np.sum(diff_arr)
      percent_diff = correct_answers / actualData.__len__() * 100
      print("The k is : " , k)
      print("Percentage Match " , percent_diff )

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

"""
Training Accuracy where train data and test data are same for calculating accuracy_score
"""

# Evaluation of complete data set
knnEstimator = KNeighborsClassifier(n_neighbors=1)
knnEstimator.fit(X,y)
testDataEvalution(knnEstimator, X, y , 1)


knnEstimator = KNeighborsClassifier(n_neighbors=5)
knnEstimator.fit(X,y)
testDataEvalution(knnEstimator, X, y, 5)



knnEstimator = KNeighborsClassifier(n_neighbors=10)
knnEstimator.fit(X,y)
testDataEvalution(knnEstimator, X, y, 10)

# metrics will use to check accuracy, Metrics to assess performance on classification task given class prediction
knnEstimator = KNeighborsClassifier(n_neighbors=1)
knnEstimator.fit(X,y)
predictionValue = knnEstimator.predict(X)
print (" Metrics Accuracy Score for k 1  : " , metrics.accuracy_score(y,predictionValue))

knnEstimator = KNeighborsClassifier(n_neighbors=5)
knnEstimator.fit(X,y)
predictionValue = knnEstimator.predict(X)
print (" Metrics Accuracy Score for k 5  : " , metrics.accuracy_score(y,predictionValue))

knnEstimator = KNeighborsClassifier(n_neighbors=10)
knnEstimator.fit(X,y)
predictionValue = knnEstimator.predict(X)
print (" Metrics Accuracy Score for k 10  : " , metrics.accuracy_score(y,predictionValue))


# Train/ Test split
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

knnEstimator = KNeighborsClassifier(n_neighbors=10)
knnEstimator.fit(X_train,y_train)
predictionValue = knnEstimator.predict(X_test)
print ("Metrics Accuracy Score for K Neighbors Classifier in Train/ Test split method : " , metrics.accuracy_score(y_test,predictionValue))

# lets ckeck with 25 value of k from 1 to 25
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


k_range = range(1,51)
score = [];

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

for k in k_range:
      knnEstimator = KNeighborsClassifier(n_neighbors=k)
      knnEstimator.fit(X_train, y_train)
      predictionValue = knnEstimator.predict(X_test)
      #print("Metrics Accuracy Score for K Neighbors Classifier in Train/ Test split method : ", metrics.accuracy_score(y_test, predictionValue))
      score.append( metrics.accuracy_score(y_test, predictionValue))


print(score)
plt.plot(k_range,score)
plt.xlabel("K (K Neighbors Classifier)")
plt.ylabel("Test Accuracy")
plt.show()
