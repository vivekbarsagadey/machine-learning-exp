
"""


C-Support Vector Classification.

kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable

Default value
C=1.0,
kernel=’rbf’,
degree=3,
gamma=’auto’,
coef0=0.0,
shrinking=True,
probability=False,
tol=0.001,
cache_size=200,
class_weight=None,
verbose=False,
max_iter=-1,
decision_function_shape=’ovr’,
random_state=None

"""

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load the diabetes dataset

"""
Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements 
were obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of 
disease progression one year after baseline.
"""

diabetes = datasets.load_diabetes()

print('\n ------------------ data (X) ------------------------\n ', diabetes.data)
print('\n ------------------- target (Y) ------------ \n ', diabetes.target)

# create data
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

clf = SVC()
# Train the model using the training sets
clf.fit(diabetes_X_train, diabetes_y_train)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, clf.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#print(clf.predict([[-0.8, -1]]))

#print('Predict value:' , clf.predict([[-0.8, -1]]))