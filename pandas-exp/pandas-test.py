import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns:
dates = pd.date_range('20170713',periods=10)
print(dates)

#Creating a DataFrame by passing a dict of objects that can be converted to series-like.
df = pd.DataFrame({ 'A' : 1.,'B' : pd.Timestamp('20170701'),
                     'C' : pd.Series(1,index=list(range(5)),dtype='float32'),
                     'D' : np.array([3] * 5,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train","train"]),
                     'F' : 'foo' })
print(df)
print(df.dtypes)

print('\n ------------ head -------------------\n ',df.head(3))
print('\n ------------ tail -------------------\n ',df.tail(2))
print('\n ------------ index -------------------\n ',df.index)
print('\n ------------ columns -------------------\n ',df.columns)
print('\n ------------ values -------------------\n ',df.values)
print('\n ------------ describe -------------------\n ',df.describe())
print('\n ------------ Transposing  -------------------\n ',df.T)
print('\n ------------ sort_index -------------------\n ',df.sort_index(axis=1, ascending=False))
print('\n ------------ sort_by value -------------------\n ',df.sort_values(by='E'))

print('\n ------------ df value -------------------\n ',df['A'])
print('\n ------------ show range -------------------\n ',df[0:3])
print('\n ------------ show range -------------------\n ',df[2:])
print('\n ------------ show range -------------------\n ',df[0:-1])

#Selection by Label
print(df.loc[:,['A','E']])

#Boolean Indexing
print(df[df.A > 0])


print(df.mean())

#Merge
df = pd.DataFrame(np.random.randn(10, 4),columns=['A','B','C','D'])
print("DataFrame ---- \n"  , df)
pieces = [df[:3], df[3:7], df[7:]]
print('Total price \n ',pd.concat(pieces))


df = pd.DataFrame({
    'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
    'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
    'C' : np.random.randn(8),
    'D' : np.random.randn(8)})

print(df)
print(df.groupby('A').sum())
print(df.groupby(['A','B']).sum())


tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
print('tuples  ::::::: \n',tuples)

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print(df)

stacked = df.stack()
print('stacked >>>>>>>> \n',stacked)


#Pivot Tables.

df = pd.DataFrame({
    'A' : ['one', 'one', 'two', 'three'] * 3,
    'B' : ['A', 'B', 'C'] * 4,
    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D' : np.random.randn(12),
    'E' : np.random.randn(12)})
print(df)
print("pivot_table >>>>>>>>>>>>\n")
print( pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

#Categoricals
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
print(df)
df["grade"] = df["raw_grade"].astype("category")
print('grade >>>\n',df["grade"])
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print('grade categories>>>\n',df)
print(df.sort_values(by="grade"))
print(df.groupby("grade").size())