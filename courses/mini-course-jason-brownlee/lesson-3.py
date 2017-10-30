"""
Load data from CSV
"""

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


"""
load data from csv
"""

url = "https://goo.gl/vhm1eU"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pandas.read_csv(url,names=names)

print('Data frame : \n{}'.format(data))

print('Data frame shape: {}'.format(data.shape))
print('Data frame dimension: {}'.format(data.ndim))
print('Data frame head: \n{}'.format(data.head(5)))




