

"""
In statistics, an estimator is a rule for calculating an estimate of a given quantity based on observed data: thus the rule (the estimator), the quantity of interest (the estimand) and its result (the estimate) are distinguished.
There are point and interval estimators.

In scikit-learn, an estimator for classification is a Python object that implements the methods fit(X, y) and predict(T).
"""

from sklearn import svm
from sklearn import datasets


"""
Load data from digit data set
"""

digits = datasets.load_digits()


svc_estimator = svm.SVC(gamma=0.001, C=100.)
print(svc_estimator)

# fit data from digits data set for training data
svc_estimator.fit(digits.data[:-1], digits.target[:-1])

# predict data from actual data set
svc_predict = svc_estimator.predict(digits.data[-1:])

print("svc_predict" ,svc_predict)