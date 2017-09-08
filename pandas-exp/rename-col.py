import pandas as pd

ufo = pd.read_csv("../data/ufo.csv")
print(ufo.columns)
print(ufo.head(5))

ufo.rename(columns={"Reported Shape":"Reported_Shape" , "Reported State" :"Reported_State"} , inplace=True)
print(ufo.columns)

new_col = ['city', 'rep' , 'shape' , 'state', 'time']

ufo.columns = new_col

print(ufo.columns)
print(ufo.head(5))

new_col = ['Our city', 'Reported' , 'shape' , 'state', 'time']
ufo = pd.read_csv("../data/ufo.csv" , names=new_col , header=0)
print(ufo.columns)
print(ufo.head(5))

ufo.columns = ufo.columns.str.replace(" ","_")
print(ufo.columns)
print(ufo.head(5))
print(ufo.shape)

ufo.drop('Reported',axis=1,inplace=True)
print(ufo.columns)
print(ufo.head(5))
print(ufo.shape)

ufo.drop(['Our_city','state'],axis=1,inplace=True)
print(ufo.columns)
print(ufo.head(5))
print(ufo.shape)


ufo.drop([0,1],axis=0,inplace=True)
print(ufo.columns)
print(ufo.head(5))
print(ufo.shape)


