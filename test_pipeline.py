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
ax=sns.boxplot(data=test_data["age"])

#remove outliers
from scipy import stats
thresh=3
test_data=test_data[np.abs(stats.zscore(test_data["age"]))<thresh]
#vizualizing outliers
sns.set(style="whitegrid")
ax=sns.boxplot(data=test_data["age"])

test_data=test_data[np.abs(stats.zscore(test_data["balance"]))<thresh]
test_data=test_data[np.abs(stats.zscore(test_data["day"]))<thresh]
test_data=test_data[np.abs(stats.zscore(test_data["duration"]))<thresh]
test_data=test_data[np.abs(stats.zscore(test_data["campaign"]))<thresh]
test_data=test_data[np.abs(stats.zscore(test_data["pdays"]))<thresh]

#string conversion , label encoding, one hot encoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#test_data['marital'] = le.fit_transform(test_data['marital'])
test_data['loan'] = le.fit_transform(test_data['loan'])
test_data['class'] = le.fit_transform(test_data['class'])
test_data['housing'] = le.fit_transform(test_data['housing'])
test_data.info()
test_data['loan'].head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers = [('encoder', OneHotEncoder(),[1,2, 7])],remainder='passthrough')
test_data = np.array(ct.fit_transform(test_data))
test_data
#why should we transform it to numpy arrayt? 
#test_data is numpy array now not a dataframe, will it be a problem?
#test_data.info()
test_data=pd.DataFrame(test_data)
test_data.head()
test_data.info()

#correlation check
train_copy = test_data.copy()
corr_matrix = train_copy.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper
to_drop = [var for var in upper.columns if any(upper[var] > .90)]
to_drop

#x y split
x_test= test_data.iloc[:,:-1]
y_test= test_data.iloc[:,-1]
x_test.head()
x_test.info()


#load the model from disk
from sklearn.externals import joblib
model = joblib.load('finalized_model.sav')
result = model.score(x_test, y_test)
