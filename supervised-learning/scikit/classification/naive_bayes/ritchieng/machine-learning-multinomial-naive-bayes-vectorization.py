"""



1> Model building in scikit-learn (refresher)
2> Representing text as numerical data
3> Reading a text-based dataset into pandas
4> Vectorizing our dataset
5> Building and evaluating a model
6> Comparing models
7> Examining a model for further insight
8> Practicing this workflow on another dataset
9> Tuning the vectorizer (discussion)
"""



"""
1. Model building in scikit-learn (refresher)
"""

from sklearn.datasets import load_iris
iris = load_iris();


# store the feature matrix (X) and response vector (y)

# uppercase X because it's an m x n matrix
X = iris.data

# lowercase y because it's a m x 1 vector
y = iris.target

# check the shapes of X and y
print('X dimensionality', X.shape)
print('y dimensionality', y.shape)

# examine the first 5 rows of the feature matrix (including the feature names)
import pandas as pd
data = pd.DataFrame(X, columns=iris.feature_names)
print (data.head())

## 4 STEP MODELLING

# 1. import the class
from sklearn.neighbors import KNeighborsClassifier

# 2. instantiate the model (with the default parameters)
knn = KNeighborsClassifier()

# 3. fit the model with data (occurs in-place)
knn.fit(X, y)

out = knn.predict([[3, 5, 4, 2]])
