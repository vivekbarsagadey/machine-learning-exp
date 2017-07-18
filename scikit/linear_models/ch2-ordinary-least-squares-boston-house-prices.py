import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.linearmodels as snslm

from sklearn import datasets, linear_model


def get_frame(dataset, target_name):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df[target_name] = dataset.target
    return df

def print_structure(dataset, file):
    print('Analyzing dataset structure')
    print('Number of instances:', dataset.data.shape[0], file=file)
    print('Number of attributes:', dataset.data.shape[1], file=file)
    print('Attribute names:', ', '.join(dataset.feature_names), file=file)


def summarize_distributions(df, file):
    print('Attribute distribution summary:', file=file)
    # pd.set_option('display.width', 200)
    desc = df.describe().T
    desc['mode'] = df.mode().ix[0]
    print(desc, file=file)
    # print(df.describe().T[['count','mean','std','min','50%','max']], file=file)

    missing_counts = pd.isnull(df).sum()
    if missing_counts.any():
        print('Missing values:', file=file)
        print(missing_counts, file=file)
    else:
        print('Missing values: NONE', file=file)

def print_correlations(df, file):
    print('Analyzing attribute pairwise correlations')
    print("Pearson's correlation:", file=file)
    pearson = df.corr(method='pearson')
    print(pearson, file=file)
    print("Spearman's correlation:", file=file)
    spearman = df.corr(method='spearman')
    print(spearman, file=file)

    def predictivity(correlations):
        corrs_with_target = correlations.ix[-1][:-1]
        return corrs_with_target[abs(corrs_with_target).argsort()[::-1]]

    print('Attribute-target correlations (Pearson):', file=file)
    print(predictivity(pearson), file=file)
    print('Attribute-target correlations (Spearman):', file=file)
    print(predictivity(spearman), file=file)

    print('Important attribute correlations (Pearson):', file=file)
    attrs = pearson.iloc[:-1, :-1]  # all except target
    # only important correlations and not auto-correlations
    threshold = 0.5
    important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
        .unstack().dropna().to_dict()
    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) \
                  for key in important_corrs])), columns=['attribute pair', 'correlation'])
    unique_important_corrs = unique_important_corrs.ix[
        abs(unique_important_corrs['correlation']).argsort()[::-1]]
    print(unique_important_corrs, file=file)

# Load the boston_datasets dataset

def attribute_correlations(df, img_file='attr_correlations.png'):
    print('Plotting attribute pairwise correlations')
    # custom figure size (in inches) to cotrol the relative font size
    fig, ax = plt.subplots(figsize=(10, 10))
    # nice custom red-blue diverging colormap with white center
    cmap = sns.diverging_palette(250, 10, n=3, as_cmap=True)
    # Correlation plot
    # - attribute names on diagonal
    # - color-coded correlation value in lower triangle
    # - values and significance in the upper triangle
    # - color bar
    # If there a lot of attributes we can disable the annotations:
    # annot=False, sig_stars=False, diag_names=False
    snslm.corrplot(df, ax=ax, cmap=cmap)
    # remove white borders
    fig.tight_layout()
    fig.savefig(img_file)
    plt.close(fig)

"""

    CRIM per capita crime rate by town
    ZN proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS proportion of non-retail business acres per town
    CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX nitric oxides concentration (parts per 10 million)
    RM average number of rooms per dwelling
    AGE proportion of owner-occupied units built prior to 1940
    DIS weighted distances to five Boston employment centres
    RAD index of accessibility to radial highways
    TAX full-value property-tax rate per $10,000
    PTRATIO pupil-teacher ratio by town
    B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT % lower status of the population
    MEDV Median value of owner-occupied homes in $1000’s

"""

boston_datasets = datasets.load_boston()

print('\n ------------------ feature names ------------------------\n '  , boston_datasets.feature_names)
print('\n ------------------ data (X) ------------------------\n ' , boston_datasets.data)
print('\n ------------------- target (Y) ------------ \n ' , boston_datasets.target)

data = get_frame(boston_datasets, target_name='MEDV')
print('\n ------------------- data frames ------------ \n ' , data)
report_file = open("load_boston_report.txt", 'w')
print_structure(boston_datasets,file = report_file)
summarize_distributions(data,report_file)
print_correlations(data,report_file )
attribute_correlations(data)

print("data", data)



# Create linear regression object
regr = linear_model.LinearRegression()

regr.fit(boston_datasets.data[:-1], boston_datasets.target[:-1])
print("Data ", boston_datasets.target[-1:] , "target : " ,list(boston_datasets.data[-1:]))
print("Data Prd : " ,regr.predict(boston_datasets.data[-1:]))