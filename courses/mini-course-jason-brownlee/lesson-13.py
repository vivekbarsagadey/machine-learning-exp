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


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(">>>>>>>>>>>>>>>>>>>>> Split from train_test_split >>>>>>>> \n")
print('X_train: \n{}'.format(X_train))
print('X_test: \n{}'.format(X_test))
print('y_train: \n{}'.format(y_train))
print('y_test: \n{}'.format(y_test))


model = LogisticRegression()
model.fit(X_train,y_train)

"""
Save model in to disk
"""
import pickle

filename = './model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


"""
Read model from strose and test the result
"""
loaded_model = pickle.load(open(filename,'rb'))
result = loaded_model.score(X_test,y_test)
print('result: \n{}'.format(result))





