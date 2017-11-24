"""

Improve Accuracy with combine the prediction from multiple models

1> ensemble with random forest and Extra trees algorithm
2> boosting ensembles with Gradient Booting machine and AdaBoost Algorithm
3> Ensembles using by combining the prediction from multiple models together.

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

array = data.values
"""Prepare X, first 8 columns and all the row : syntax  array[ row, columns ]  """
X = array[:,0:8]
y = array[:,8]


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

num_trees =100
max_features = 3
kFold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
result = cross_val_score(model , X , y , cv=kFold)
print('RandomForestClassifier Accuracy: %.3f%% (%.3f%%)' % ( result.mean()*100.0, result.std()*100.0))



from sklearn.ensemble import ExtraTreesClassifier

num_trees =100
max_features = 3
kFold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=num_trees,max_features=max_features)
result = cross_val_score(model , X , y , cv=kFold)
print('ExtraTreesClassifier Accuracy: %.3f%% (%.3f%%)' % ( result.mean()*100.0, result.std()*100.0))



from sklearn.tree import DecisionTreeClassifier

max_features = 3
kFold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier(max_features=max_features)
result = cross_val_score(model , X , y , cv=kFold)
print('DecisionTreeClassifier Accuracy: %.3f%% (%.3f%%)' % ( result.mean()*100.0, result.std()*100.0))

