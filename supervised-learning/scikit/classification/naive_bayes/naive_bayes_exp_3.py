
#  ==== Probabilistic model ====
#  posterior = ( prior * likelihood) / evidence

#   p(x/y) = p(y/x)*p(x) / p(y)
#  posterior = p(x/y)
#  prior = p(y/x)
#  likelihood = p(x)
#  evidence = p(y)


from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd


simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']


# 1. import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer

# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()

# 3. fit
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)

print("vocabulary >>> " ,vect.get_feature_names());


# 4. transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)

print("simple_train_dtm >>> ",simple_train_dtm.toarray())


# examine the vocabulary and document-term matrix together
# pd.DataFrame(matrix, columns=columns)
print (pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names()))

# check the type of the document-term matrix
type(simple_train_dtm)


# examine the sparse matrix contents
# left: coordinates of non-zero values
# right: values at that point
# CountVectorizer() will output a sparse matrix
print('sparse matrix')
print(simple_train_dtm)

print('dense matrix')
print(simple_train_dtm.toarray())


# example text for model testing
simple_test = ['Please don\'t call me']
# 4. transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())


nb = MultinomialNB()

nb.fit(simple_train_dtm, simple_test_dtm)