import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv('housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print("--------------- normal -------------")
# print(df.head())
print(df["CRIM"].head())

print("--------------- trans -------------")
print(df.T.head())
print("--------------- feature and data -------------")

feature_col = ['RM','LSTAT']
response_col = ['MEDV']

print("--------------- normal -------------")
print(df["MEDV"].head())
print(df["LSTAT"].head())
print(df["RM"].head())




print("--------------- X -------------")
X = df[feature_col]
# X is equal to data[['MEDV' ]]
print('X is >>>', X.head())
print('X shape >>>', X.shape)

print("--------------- Y -------------")
y = df['MEDV']
# X is equal to data[['MEDV' ]]
print('Y is >>>', y.head())
print('Y shape >>>', y.shape)


print("--------------- tran and test split -------------")
# Train/ Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#print(" X_train, X_test, y_train, y_test  >>>> \n ",X_train, X_test, y_train, y_test)

def analisysBySk(X_train, X_test, y_train, y_test, feature_col):
    from sklearn.linear_model import LinearRegression
    linearRegressionModal = LinearRegression()
    for i in list(range(1000)):
        linearRegressionModal.fit(X_train, y_train)

    print('coefficient >>> ', linearRegressionModal.coef_)
    print('intercept >>> ', linearRegressionModal.intercept_)
    zip_data = zip(feature_col, linearRegressionModal.coef_)
    y_pred = linearRegressionModal.predict(X_test)
    print("y_pred>>>>>>>>>", y_pred)
    datalist = list(zip(X_test.values, y_test.values, y_pred))
    for fei, rei, repei in datalist:
        print("data analisys >>>>  ", " X (MEDV): ", fei[0], " Y (RM): ", rei, " Y (RM): ", repei, " Total % loos" ,((rei - repei)/rei)*100)


def analisysByTensor(X_train, X_test, y_train, y_test, feature_col):
    print(" ---------------- analisysByTensor ------------ ")
    print("Feature 1 >>> ",X_train["RM"].values)
    print("Feature 2 >>> ", X_train["LSTAT"].values)

    x1 = tf.placeholder(tf.float32, [None, 1], name="x1")
    x2 = tf.placeholder(tf.float32, [None, 1], name="x1")
    W1 = tf.Variable(tf.zeros([1, 1],dtype=tf.float32), name="W1")
    W2 = tf.Variable(tf.zeros([1, 1],dtype=tf.float32), name="W2")
    y_data = y_train.values

    fitval = {x1 :X_train["RM"].values , x2 : X_train["LSTAT"].values }

    def get_model():
        product_1 = X_train["RM"].values * W1
        product_2 = X_train["LSTAT"].values * W2
        return product_1 + product_2

    y = get_model()

    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    # Before starting, initialize the variables
    init = tf.global_variables_initializer()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    # Fit the line.
    for step in range(2001):
        sess.run(train)
        if step % 200 == 0:
            print(step, sess.run([W1,W2]))




#analisysBySk(X_train, X_test, y_train, y_test, feature_col)

analisysByTensor(X_train, X_test, y_train, y_test, feature_col)