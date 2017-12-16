

"""
Gaussian naive bayes, bayesian learning, and bayesian networks

(Prior probability)(Test evidence) --> (Posterior probability)

"""


from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# Create features' DataFrame and response Series
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)


# Instantiate: create object
gnb = GaussianNB()

# Fit
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("acc", acc)