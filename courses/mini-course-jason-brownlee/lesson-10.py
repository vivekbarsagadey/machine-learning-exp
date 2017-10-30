"""
Model comparison and selection
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


models = []
models.append(('LR' , LogisticRegression()))
models.append(('LDA' , LinearDiscriminantAnalysis()))


results = []
names = []
scoring = "accuracy"

for name, model in models:
    kf = KFold(n_splits=10, random_state=7)
    result = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    results.append(result)
    names.append(name)
    print('Name : %s:  Accuracy: %.3f%% (%.3f%%)' % (name, result.mean()*100.0, result.std()*100.0))



