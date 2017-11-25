
#  ==== Probabilistic model (Gaussian / Multinomial naive Bayes ) ====
#  posterior = ( prior * likelihood) / evidence

#   p(x/y) = p(y/x)*p(x) / p(y)
#  posterior = p(x/y)
#  prior = p(y/x)
#  likelihood = p(x)
#  evidence = p(y)


from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#assigning predictor and target variables
# the data set is ["height" 	"weight" "foot size"] and class [Sex]

df = pd.DataFrame(data=np.array([
[6,180,12 , 0],
[5.92,190,11 , 0],
[5.58,170,12 , 0],
[5.92,165,10 , 0],
[5,100,6 , 1],
[5.5,150,8, 1],
[5.42,130,7, 1],
[5.75,150,9 ,1 ]]
),columns=['height','weight','foot size' , 'gender'])

print(df)
print(df.dtypes)

print('\n ------------ index -------------------\n ',df.index)
print('\n ------------ columns -------------------\n ',df.columns)
print('\n ------------ values -------------------\n ',df.values)


x= df[['height','weight','foot size' ]].values
print("x" , x)
y= df[['gender']].values

print("y" , y)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(x, y)

#Predict Output
predicted= model.predict([[6,130,8]])
print("predicted >>>" , predicted)

#Output: ([1])