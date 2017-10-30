"""
Prepare for modeling by pre processing data
"""

import sys
import scipy
import numpy
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
Y = array[:,8]
print('X: \n{}'.format(X[0:5,:]))
#print('Y: \n{}'.format(Y))

scalerX = StandardScaler().fit(X)
re_scaler_X = scalerX.transform(X)
numpy.set_printoptions(precision=3)
print('re_scaler_X: \n{}'.format(re_scaler_X[0:5,:]))





