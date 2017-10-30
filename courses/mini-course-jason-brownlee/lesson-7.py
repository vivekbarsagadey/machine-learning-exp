"""
Algorithm Evaluation with Re-sampling method
We will learn re sample data by using scikit lear
"""

import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.preprocessing import StandardScaler


"""
load data from csv
"""

url = "https://goo.gl/vhm1eU"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pd.read_csv(url,names=names)

print('Data frame head: \n{}'.format(data.head(5)))
print('Data frame describe: \n{}'.format(data.describe()))

"""
Perfrom StandardScaler
"""

array = data.values
"""Prepare X, first 8 columns and all the row : syntax  array[ row, columns ]  """
X = array[:,0:8]
y = array[:,8]
print('X: \n{}'.format(X[0:5,:]))
print('Y: \n{}'.format(y[0:5]))



"""
Split train and test data set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(">>>>>>>>>>>>>>>>>>>>> Split from train_test_split >>>>>>>> \n")
print('X_train: \n{}'.format(X_train))
print('X_test: \n{}'.format(X_test))
print('y_train: \n{}'.format(y_train))
print('y_test: \n{}'.format(y_test))


from sklearn.model_selection import KFold # import KFold

"""
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
"""

kf = KFold(n_splits=2) # Define the split - into 2 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

print(">>>>>>>>>>>>>>>>>>>>>  Split from KFold >>>>>>>> \n")
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


print('X_train: \n{}'.format(X_train))
print('X_test: \n{}'.format(X_test))
print('y_train: \n{}'.format(y_train))
print('y_test: \n{}'.format(y_test))


"""
Test with simple logistic regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
result = cross_val_score(model,X,y, cv=kf)
print('Accuracy: %.3f%% (%.3f%%)' % (result.mean()*100.0,result.std()*100.0))

