"""

The multinomial Naive Bayes classifier is suitable for classification with discrete features
(e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts

1> Model building in scikit-learn (refresher)
2> Representing text as numerical data
3> Reading a text-based dataset into pandas
4> Vectorizing our dataset
5> Building and evaluating a model
6> Comparing models
7> Examining a model for further insight
8> Practicing this workflow on another dataset
9> Tuning the vectorizer (discussion)
"""



"""
1. Model building in scikit-learn (refresher)
"""

from sklearn.datasets import load_iris
import pandas as pd


def model_building_in_scikit_learn_refresher():

    iris = load_iris();


    # store the feature matrix (X) and response vector (y)

    # uppercase X because it's an m x n matrix
    X = iris.data

    # lowercase y because it's a m x 1 vector
    y = iris.target

    # check the shapes of X and y
    print('X dimensionality', X.shape)
    print('y dimensionality', y.shape)

    # examine the first 5 rows of the feature matrix (including the feature names)

    data = pd.DataFrame(X, columns=iris.feature_names)
    print (data.head())

    ## 4 STEP MODELLING

    # 1. import the class
    from sklearn.neighbors import KNeighborsClassifier

    # 2. instantiate the model (with the default parameters)
    knn = KNeighborsClassifier()

    # 3. fit the model with data (occurs in-place)
    knn.fit(X, y)

    out = knn.predict([[3, 5, 4, 2]])


#model_building_in_scikit_learn_refresher();

"""
2. Representing text as numerical data

expect numerical feature vectors with a fixed size

"""

def representing_text_as_numerical_data():
    # example text for model training (SMS messages)
    simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

    # Steps for Vectorization

    # 1. import and instantiate CountVectorizer (with the default parameters)
    from sklearn.feature_extraction.text import CountVectorizer

    # 2. instantiate CountVectorizer (vectorizer)
    vect = CountVectorizer()

    # 3. fit
    # learn the 'vocabulary' of the training data (occurs in-place)
    vect.fit(simple_train)
    """
    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)
    
    It took out "a" due to the token_pattern (regex shown above)
    lower_case=True made all lowercase
    Alphabetical order
    No duplicate words
    
    """


    # examine the fitted vocabulary
    print(vect.get_feature_names())


    # 4. transform training data into a 'document-term matrix' for sparse matrix
    simple_train_dtm = vect.transform(simple_train)
    print("simple_train_dtm sparse matrix >>\n",simple_train_dtm)

    # convert sparse matrix to a dense matrix
    print("simple_train_dtm dense matrix >>\n",simple_train_dtm.toarray())


    """
    
    sparse matrix
    
    only store non-zero values
    if you have 0's, it'll only store the coordinates of the 0's
    dense matrix
    
    seeing zero's and storing them
    if you have 1000 x 1000 of 0's, you'll store all
    
    """




    # examine the vocabulary and document-term matrix together
    # pd.DataFrame(matrix, columns=columns)
    simple_train_dtm_dataframe = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
    print("simple_train_dtm_dataframe \n" , simple_train_dtm_dataframe)


    # check the type of the document-term matrix
    print("type of simple_train_dtm " , type(simple_train_dtm))


    # example text for model testing
    simple_test = ['Please don\'t call me']

    # 4. transform testing data into a document-term matrix (using existing vocabulary)
    simple_test_dtm = vect.transform(simple_test)
    print( "simple_test_dtm  document-term matrix \n", simple_test_dtm.toarray())

    # It dropped the word "don't", why are we ok with the fact that the word "don't" drops?


    simple_test_dtm_dataframe = pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())

    print(" simple_test_dtm_dataframe   >>>>>>>>>>>>> \n" , simple_test_dtm_dataframe)

    """
    Summary:

    vect.fit(train) learns the vocabulary of the training data
    vect.transform(train) uses the fitted vocabulary to build a document-term matrix from the training data
    vect.transform(test) uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)

    """

#representing_text_as_numerical_data();


"""
Reading a text-based dataset into pandas

"""

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


"""
Building and evaluating a model

"""

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


# examine class distribution
print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
print('Manual null accuracy:',(1208 / (1208 + 185)))