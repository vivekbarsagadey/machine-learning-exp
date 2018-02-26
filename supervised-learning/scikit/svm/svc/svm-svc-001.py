
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

import numpy as np
from sklearn.svm import SVC


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])



clf = SVC()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))

print('Predict value:' , clf.predict([[-0.8, -1]]))