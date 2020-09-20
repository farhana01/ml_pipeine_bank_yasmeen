# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:41 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

test_data= pd.read_csv("bank_test.csv")
print(test_data.head())

print("===============describe================")
print(test_data.describe())

print("================info===================")
print(test_data.info())

# null values
print(test_data.isnull())
#null values per column
print(test_data.isnull().sum())
# total null values
print(test_data.isnull().sum().sum())

#from numpy import nan
# test_data = test_data.replace(0, nan)

test_data.drop(test_data.columns[0], axis=1, inplace= True)
test_data.info()

# # treat missing value with sklearn
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=nan, strategy='mean')
# imputer.fit(test_data[:,2:3])
# test_data[:,:-1] = imputer.transform(test_data[:, :-1])


test_data['balance'].fillna(test_data['balance'].mean(),inplace=True)
print(test_data['balance'].isnull().sum())

# fill column with mode
test_data['education'].fillna(test_data['education'].mode().iloc[0], inplace=True)
print(test_data['education'].isnull().sum())
test_data['month'].fillna(test_data['month'].mode().iloc[0], inplace=True)
print(test_data['month'].isnull().sum())
print(test_data.isnull().sum())


#vizualizing outliers
sns.set(style="whitegrid")
ax=sns.boxplot(data=train_data["age"])

#remove outliers
from scipy import stats
thresh=3
train_data=train_data[np.abs(stats.zscore(train_data["age"]))<thresh]
#vizualizing outliers
sns.set(style="whitegrid")
ax=sns.boxplot(data=train_data["age"])

train_data=train_data[np.abs(stats.zscore(train_data["balance"]))<thresh]
train_data=train_data[np.abs(stats.zscore(train_data["day"]))<thresh]
train_data=train_data[np.abs(stats.zscore(train_data["duration"]))<thresh]
train_data=train_data[np.abs(stats.zscore(train_data["campaign"]))<thresh]
train_data=train_data[np.abs(stats.zscore(train_data["pdays"]))<thresh]
train_data=train_data[np.abs(stats.zscore(train_data["previous"]))<thresh]
