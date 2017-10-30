"""
Spot Check Algo (testing with k-Nearest Neighbors)
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

url = "https://goo.gl/sXleFv"
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data=pd.read_csv(url,names=names,delim_whitespace=True)

print('Data frame head: \n{}'.format(data.head(5)))
print('Data frame describe: \n{}'.format(data.describe()))

"""
Perfrom StandardScaler
"""

array = data.values
"""Prepare X, first 8 columns and all the row : syntax  array[ row, columns ]  """
X = array[:,0:13]
y = array[:,13]




from sklearn.model_selection import KFold # import KFold

"""
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
"""

kf = KFold(n_splits=10,random_state=7)
print(">>>>>>>>>>>>>>>>>>>>>  Split from KFold >>>>>>>> \n")
print(kf)


"""
Test with simple K Neighbors regression
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

model = KNeighborsRegressor()

"""
neg_mean_squared_error scoring
"""
scoring = "neg_mean_squared_error"
result = cross_val_score(model,X,y, cv=kf , scoring=scoring)
print('Accuracy: %.3f%% ' % (result.mean()))

