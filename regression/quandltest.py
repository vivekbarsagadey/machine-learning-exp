'''

Wiki EOD Stock Prices regression testing from quandl
'''


import pandas as pd
import quandl , math
import numpy as np
from scipy import sparse
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df =  quandl.get("WIKI/GOOGL")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.00
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.00


df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

print(df.head())

forcast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)

forcast_out  = int(math.ceil(0.01*len(df)))
print(forcast_out)

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)


x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

x = preprocessing.scale(x)
y = np.array(df['label'])


x_train , x_test , y_train , y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = LinearRegression()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print('accuracy' , accuracy)







