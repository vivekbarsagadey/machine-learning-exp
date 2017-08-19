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

feature_col = ['RM']
response_col = ['MEDV']

print("--------------- normal -------------")
print(df["MEDV"].head())
print(df["RM"].head())

X = df[feature_col]
# X is equal to data[['MEDV' ]]
print('X is >>>', X.head())
print('X shape >>>', X.shape)

y = df['MEDV']

print("--------------- tran and test split -------------")
# Train/ Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


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
        print("data analisys >>>>  ", " X (RM): ", fei[0], " Y (MEDV): ", rei, " Y (MEDV) for y_pred: ", repei , " Total % loos" ,((rei - repei)/rei)*100)



def analisysByTensor(X_train, X_test, y_train, y_test, feature_col):
    print(" ---------------- analisysByTensor ------------ ")
    # Set wrong model weights
    W = tf.Variable(tf.random_normal([1]), name='weight')
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Linear model
    hypothesis = X * W
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    # Minimize: Gradient Descent Magic
    learning_rate = 0.001
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    update = W.assign(descent)
    # Launch the graph in a session.
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(update, feed_dict={X: X_train, Y: y_train})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: X_train, Y: y_train}), sess.run(W))

    # need to understand
    testing_cost = sess.run(cost, feed_dict={X: X_test, Y: y_test})

    print("testing_cost", testing_cost)


def analisysByTensorGradientDescentOptimizer(X_train, X_test, y_train, y_test, feature_col):
    print(" ---------------- analisysByTensorGradientDescentOptimizer ------------ ")
    import tensorflow as tf
    # create random data
    x_data = X_train.values
    y_data = y_train.values

    # Find values for W that compute y_data = W * x_data
    W = tf.Variable(tf.random_normal([1]), name='weight')
    print(" ---------------- creating modal ------------ ")

    def get_model(b=0):
        return W * x_data + b

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
            print(step, sess.run(W))

    # apply test data





#analisysBySk(X_train, X_test, y_train, y_test, feature_col)

#analisysByTensor(X_train, X_test, y_train, y_test, feature_col)

analisysByTensorGradientDescentOptimizer(X_train, X_test, y_train, y_test, feature_col)
