"""

The multinomial Naive Bayes classifier is suitable for classification with discrete features
(e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts


> Representing text as numerical data
> Vectorizing our dataset
> Building and evaluating a model
"""
import pandas as pd

# read file into pandas using a relative path
path = 'data/supervised/NaiveBayes/sms.tsv'
features = ['label', 'message']
sms = pd.read_table(path, header=None, names=features)

print(" shape ",sms.shape)
print("sms \n" , sms.head())

# examine the class distribution
print(sms.label.value_counts())

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1});

print("Updated data \n",sms.head());

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

# split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




"""
Vectorizing our dataset
"""

from sklearn.feature_extraction.text import CountVectorizer
# 2. instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X_train)


# 4. transform training data
X_train_dtm = vect.transform(X_train)

# equivalently: combine fit and transform into a single step
# this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)

print("X_train_dtm >>> ",X_train_dtm)

# 4. transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)

print("X_test_dtm >>> ",X_test_dtm)


# 1. import
from sklearn.naive_bayes import MultinomialNB

# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

# 4. make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
print("accuracy of class predictions >>> ",metrics.accuracy_score(y_test, y_pred_class))


