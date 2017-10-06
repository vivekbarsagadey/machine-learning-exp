"""

Multi-Classification Problem Examples (MLR):

    Given fruit features like color, size, taste, weight, shape. Predicting the fruit type.
    By analyzing the skin, predicting the different skin disease.

Irises dataset for classification

This famous classification dataset first time used in Fisher’s classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems. Iris dataset is having 4 features of iris flower and one target class.

The 4 features are

    SepalLengthCm
    SepalWidthCm
    PetalLengthCm
    PetalWidthCm

The target class

The flower species type is the target class and it having 3 types

    Setosa
    Versicolor
    Virginica


"""

# Required Packages
from sklearn import datasets  # To Get iris dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt  # To visuvalizing the data

# import iris data to model Svm classifier
iris = datasets.load_iris()

print("Iris data set Description :: ", iris['DESCR'])
print("Iris feature data :: ", iris['data'])
print("Iris target :: ", iris['target'])


def visuvalize_sepal_data():
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Sepal Width & Length')
    plt.show()


#visuvalize_sepal_data()



"""
Standardize Features
"""
X = iris.data
y = iris.target
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')


# Train model
model = clf.fit(X_std, y)

# Create new observation
new_observation = [[.5, .5, .5, .5]]
#new_observation = [[5.9 , 3.,   5.1 , 1.8]]

# Predict class
output = model.predict(new_observation)

print(" Output :: {}".format(output))


# View predicted probabilities
print(" View predicted probabilities :: {}".format(model.predict_proba(new_observation)))


