# Loading the library with the iris dataset
from sklearn.datasets import load_iris
# Loading scikit’s random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Loading pandas
import pandas as pd
# Loading numpy
import numpy as np
# Setting random seed
np.random.seed(0)


# Creating an object called iris with the iris data
iris = load_iris()
# Creating a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Viewing the top 5 rows
print(df.head())

# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target,iris .target_names)

# Viewing the top 5 rows
print(df.head())

# Creating Test and Train Data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# View the top 5 rows
print(df.head())

# Creating dataframes with test rows and training rows
train, test = df[df['is_train']== True],df[df['is_train'] == False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:',len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature columns names
features = df.columns[:4]
# View features
print(features)

# Converting each species naine into digits
y = pd.factorize(train['species'])[0]
# Viewing target
print(y)

# Creating a random forest Classifier.
clf = RandomForestClassifier(n_jobs=2, random_state=0)
# Training the classifier
clf.fit(train[features], y)

# Applying the trained Classifier to the test
clf.predict(test[features])


# Viewing the predicted probabilities of the first 10 observations
predict_proba = clf.predict_proba(test[features])[0:10]
print(predict_proba)


# mapping names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# View the PREDICTED species for the first five observations
print(preds[0:5])

# Viewing the ACTUAL species for the first five observations
print(test['species'].head())

# Creating confusion matrix
confusion_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'],colnames=['Predicted Species'])
print(confusion_matrix)