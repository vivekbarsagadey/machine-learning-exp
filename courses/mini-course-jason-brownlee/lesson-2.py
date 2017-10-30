"""
Get Around in python, numpy, matplotlib and pandas
"""

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn


"""
list and flow in python
"""

myarray = numpy.array([[1,2,3],[4,5,6]])
rownames = ['a','b']
colnames = ['one','two','three']

mydataframe = pandas.DataFrame(myarray,index=rownames,columns=colnames)

print('Data frame : \n{}'.format(mydataframe))

print('Data frame shape: {}'.format(mydataframe.shape))
print('Data frame dimension: {}'.format(mydataframe.ndim))
print('Data frame head: \n{}'.format(mydataframe.head(5)))




