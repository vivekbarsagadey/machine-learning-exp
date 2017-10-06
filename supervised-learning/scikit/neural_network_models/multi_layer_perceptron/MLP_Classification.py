"""

Multinomial Logistic Regression Workflow/ Stages:

    Inputs
    Linear model
    Logits
    Softmax Function
    Cross Entropy
    One-Hot-Encoding


"""

from sklearn.neural_network import MLPClassifier


X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)


clf.fit(X, y)
pred = clf.predict([[2., 2.], [-1., -2.]])

print('perd\n' , pred)