import pandas as pd

movies = pd.read_csv("../data/imdb_1000.csv")

print(movies.head())
print(movies.shape)

booleans = []

for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)



"""
Convert normal array to panda series
"""

is_long = pd.Series(booleans)
print(is_long.head())
print(is_long.shape)
print(movies[is_long])

condition = movies.duration >= 200
print(movies[condition])

print(movies[is_long].shape)
print(movies[condition].shape)

print(movies[condition].genre)

mult_condition = (movies.duration >= 200) & (movies.genre =="Drama")
print(" Drama and duration >= 200 data set \n",movies[ mult_condition].genre)

"""
loc method for getting row and col
"""
condition = movies.duration >= 200
print("loc  >>> \n" , movies.loc[condition,'genre'])
print("loc  >>> \n" , movies.loc[condition,['genre' ,'title']])

mult_condition = (movies.duration >= 200) & (movies.genre =="Drama")
print("loc  >>> \n" , movies.loc[mult_condition,['genre' ,'title']])

"""
Use isin method for spec column 
"""

print(movies[movies.genre.isin(['Drama','Action'])])
print(movies.loc[movies.genre.isin(['Drama','Action']),['genre' ,'title']])
