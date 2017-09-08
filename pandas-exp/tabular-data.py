import pandas as pd

#http://bit.Ly/chiporders
"""
Table data
"""
orders = pd.read_table("../data/chipotle.tsv")


user_col = ['user_id','age','gender','oct','zip_code']
movieusers = pd.read_table("http://bit.Ly/movieusers" , delimiter="|" , header=None , names=user_col)
print(movieusers.tail(5))



