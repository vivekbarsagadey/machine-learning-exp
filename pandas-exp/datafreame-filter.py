import pandas as pd

movies = pd.read_csv("../data/imdb_1000.csv")

print(movies.head())
print(movies.shape)

booleans = []

for length in movies.duration:
    if length > 100:
        booleans.append(True)
    else:
        booleans.append(False)

print(booleans.shape)

"""
Convert normal array to panda series
"""

is_long = pd.Series(booleans)
print(is_long.head())


