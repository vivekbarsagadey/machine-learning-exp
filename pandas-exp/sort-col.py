import pandas as pd

movies = pd.read_csv("../data/imdb_1000.csv")
print(movies.columns)
print(movies.head(5))

print(movies.title)
print(movies['title'].sort_values())


print(movies.sort_values('title'))


print(movies.sort_values(['content_rating','title']))
