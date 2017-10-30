"""
Improve Accuracy with Algorithm tuning

1> Tune the parameters of an algorithm using a grid search
2> Tune the parameters of an algorithm using a random search

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

alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, y)

print('Best Score: {}'.format(grid.best_score_))
print('Best Estimator: {}'.format(grid.best_estimator_.alpha))


