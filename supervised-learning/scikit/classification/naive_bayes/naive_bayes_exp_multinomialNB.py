
#  ==== Probabilistic model (MultinomialNB ) ====
#  posterior = ( prior * likelihood) / evidence

#   p(x/y) = p(y/x)*p(x) / p(y)
#  posterior = p(x/y)
#  prior = p(y/x)
#  likelihood = p(x)
#  evidence = p(y)

"""
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word
counts for text classification). The multinomial distribution normally requires integer feature counts.
However, in practice, fractional counts such as tf-idf may also work.
"""


from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd


X = np.random.randint(5, size=(6, 100))

y = np.array([1, 2, 3, 4, 5, 6])

print("X" , X)
print("y" , y)

#Create a Gaussian Classifier
model = MultinomialNB()

# Train the model using the training sets
model.fit(X, y.ravel())

#Predict Output
predicted= model.predict(X[2:3])
print("predicted >>>" , predicted)

