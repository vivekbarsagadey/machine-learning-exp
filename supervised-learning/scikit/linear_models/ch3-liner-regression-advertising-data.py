import pandas as pd

"""

data set will be TV  Radio  Newspaper  Sales
where features are [TV , Radio , Newspaper]
and response will be [Sales]
"""

data = pd.read_csv("../../../data/Advertising.csv",index_col=0)
print(data.head())
print(data.tail())

import matplotlib.pyplot as plt
import seaborn as sns

feature_col = ['TV' , 'Radio' , 'Newspaper']
response_col = ['Sales']

sns.pairplot(data,x_vars=feature_col,y_vars= response_col)

plt.tight_layout()
#plt.show()

#bit more clear data set
sns.pairplot(data,x_vars=feature_col,y_vars= response_col,size=7,aspect=0.7)
#plt.show()


#show leanear line for general distrubtion
sns.pairplot(data,x_vars=feature_col,y_vars= response_col,size=7,aspect=0.7 , kind="reg")
plt.savefig('./report/Advertising.png', dpi=300)
#plt.show()



"""

y= b0 + b1*x1 + b2*x2 + b3*x3 .....

b0 = is the intercept
b1 is the coefficient for x1 (first feature)
bn is the coefficient for xn (nth feature)

Current case
y = b0 + (b1 * TV) + (b2 * Radio) + (b3 * Newspaper)

b value are called the model coefficient

Lets think about least squered

"""

# define data

X = data[feature_col]
# X is equal to data[['TV' , 'Radio' , 'Newspaper']]
print('X is >>>' , X.head())
print('X shape >>>' , X.shape)
print('X type >>>' , type(X))

y = data['Sales']
# y is equal to data[['Sales']]
print('y is >>>' , y.head())
print('y shape is >>>' , y.shape)
print('y type is >>>' , type(y))


# Train/ Test split
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
print('X_train.shape' , X_train.shape)
print('X_test.shape' , X_test.shape)
print('y_train.shape ', y_train.shape)
print( 'y_test.shape',  y_test.shape)


"""
Build modal 

"""
from sklearn.linear_model import LinearRegression

linearRegressionModal = LinearRegression()

linearRegressionModal.fit(X_train, y_train)
print('coefficient >>> ' , linearRegressionModal.coef_)
print('intercept >>> ' , linearRegressionModal.intercept_)

"""
Exp:
coefficient >>>  [ 0.04656457  0.17915812  0.00345046] ==> ['TV' , 'Radio' , 'Newspaper'] ==>[b1  b2  b3]
intercept >>>  2.87696662232  ==> [b0]

zip for show all  the relation
"""
zip_data = zip(feature_col,linearRegressionModal.coef_)

print(zip_data)


y_pred = linearRegressionModal.predict(X_test)
print("y_pred >> ",y_pred)
print( 'y_test.shape',  y_test.shape)
print("y_pred.shape >> ",y_pred.shape)

"""
Calculate error by using Metrics

MAE  : 
MSE  :
RMSE :

"""

import numpy as np
from sklearn import metrics


mae_value = metrics.mean_absolute_error(y_test,y_pred)
print("MAE (mean_absolute_error)  : " , mae_value)

mse_value = metrics.mean_squared_error(y_test,y_pred)
print("MSE (mean_squared_error)  : " , mse_value)

rmse_value = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print(" RMSE (mean_squared_error with sqrt)  : " , rmse_value)


"""
Remove Newspaper with our feature set
"""


X = data[['TV' , 'Radio']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
linearRegressionModal = LinearRegression()
linearRegressionModal.fit(X_train, y_train)
y_pred = linearRegressionModal.predict(X_test)
new_rmse_value = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print("New RMSE (mean_squared_error with sqrt)  : " , new_rmse_value)
print("Error should be reduce :", " ::::::::::: Old rmse", rmse_value , " ::::::::  New rmse ",new_rmse_value)

