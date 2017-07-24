from sklearn import tree

#hight , weigth and foot size
X = [[180,90,60],[175,90,61],[177,89,60],[154,75,55],[154,55,45],[149,52,42],[175,65,55]]
Y = ['M','M','M','M','F','F','F']

df = tree.DecisionTreeClassifier();
df.fit(X,Y)
pr = df.predict([130,30,45])
print(pr)


