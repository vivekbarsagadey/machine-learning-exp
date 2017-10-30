"""
Algorithm Evaluation Metrics
Evaluate different algorithm performance metrics in scikit learn kit
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




from sklearn.model_selection import KFold # import KFold

"""
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
"""

kf = KFold(n_splits=10,random_state=7)
print(">>>>>>>>>>>>>>>>>>>>>  Split from KFold >>>>>>>> \n")
print(kf)


"""
Test with simple logistic regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression()

"""
Log loss metric
"""
scoring = "neg_log_loss"
result = cross_val_score(model,X,y, cv=kf , scoring=scoring)
print('Accuracy: %.3f%% (%.3f%%)' % (result.mean()*100.0,result.std()*100.0))

