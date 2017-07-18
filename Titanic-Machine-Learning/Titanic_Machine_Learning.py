import numpy as np
import pandas as pd

#The Machine learning alogorithm
from sklearn.ensemble import RandomForestClassifier

# Test train split
from sklearn.cross_validation import train_test_split

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

# Used to write our model to a file
from sklearn.externals import joblib


data = pd.read_csv("data/titanic_train.csv")
print(data.head())
print('columns >>>>>>>>>>>>>>>>>>>>>\n ',data.columns)
print('columns >>>>>>>>>>>>>>>>>>>>>\n ',data.describe())
median_age = data['age'].median()
print("Median age is {}".format(median_age))


print(data['age'].head())
data['age'].fillna(median_age, inplace = True)
print(data['age'].head())

data_inputs = data[["pclass", "age", "sex"]]
print(data_inputs.head())

expected_output = data[["survived"]]
print(expected_output.head())

data_inputs["pclass"].replace("3rd", 3, inplace = True)
data_inputs["pclass"].replace("2nd", 2, inplace = True)
data_inputs["pclass"].replace("1st", 1, inplace = True)
print(data_inputs.head())

data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
print(data_inputs.head())

inputs_train, inputs_test, expected_output_train, expected_output_test   = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print(inputs_train.head())
print(expected_output_train.head())

rf = RandomForestClassifier (n_estimators=100)
rf.fit(inputs_train, expected_output_train)

accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))

#joblib.dump(rf, "data/out/titanic_model1", compress=9)



#rf = joblib.load("data/out/titanic_model1")

testData = pd.read_csv("data/titanic_test.csv")
testData['age'].fillna(median_age, inplace = True)
testData_inputs = data[["pclass", "age", "sex"]]
#expected_output = data[["survived"]]
testData_inputs["pclass"].replace("3rd", 3, inplace = True)
testData_inputs["pclass"].replace("2nd", 2, inplace = True)
testData_inputs["pclass"].replace("1st", 1, inplace = True)
testData_inputs["sex"] = np.where(testData_inputs["sex"] == "female", 0, 1)
pred = rf.predict(testData_inputs)
print("predict is ",pred)


def find_err(pred):
    titanic_data = np.loadtxt("data/titanic_results.txt", dtype="int32")
    diff_arr = np.equal(titanic_data, pred[:titanic_data.__len__()])
    correct_answers = np.sum(diff_arr)
    percent_diff = correct_answers / titanic_data.__len__() * 100
    print("Titanic: Percentage Match is: ", percent_diff)


find_err(pred)

