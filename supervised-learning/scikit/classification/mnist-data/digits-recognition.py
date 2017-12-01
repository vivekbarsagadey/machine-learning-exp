"""

Digit Recognizer
Learn computer vision fundamentals with the famous MNIST data
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
"""


import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("supervised-learning/scikit/classification/mnist-data/dataset/train.csv").as_matrix()
testdata = pd.read_csv("supervised-learning/scikit/classification/mnist-data/dataset/test.csv").as_matrix()
clf = DecisionTreeClassifier()
x = data[0:, 1: ]
y = data[0:, 0 ]
clf.fit(x ,y)

def predictModal(xTest) :
    prd = clf.predict([xTest])
    print("New Predict value is >>>>>>>>>>>>>>>>>>>>>> : " ,prd[0] )
    xTest.shape = (28,28)
    pt.imshow(255-xTest , cmap = "gray");
    pt.show()



predictModal(testdata[8])
predictModal(testdata[5])
predictModal(testdata[77])

