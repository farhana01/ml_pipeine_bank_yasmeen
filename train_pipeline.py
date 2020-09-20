# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:13 2020

@author: Farhana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data= pd.read_csv("bank_train.csv")
print(train_data.head())

print("===============describe================")
print(train_data.describe())

print("================info===================")
print(train_data.info())

# null values
print(train_data.isnull())
#null values per column
print(train_data.isnull().sum())
# total null values
print(train_data.isnull().sum().sum())

#Axis 0 will act on all the ROWS in each COLUMN
#Axis 1 will act on all the COLUMNS in each ROW
#So a mean on axis 0 will be the mean of all the rows in each column, and a mean on axis 1 will be a mean of all the columns in each row.

#drop the missing value
# train_data_copy=train_data.copy()
# print(train_data_copy.head())
# print(train_data_copy.isnull().sum())
# train_data_copy=train_data_copy.dropna()
# print(train_data_copy.isnull().sum().sum())
# print(train_data_copy.shape)

train_data.drop(train_data.columns[0], axis=1, inplace= True)
train_data.info()

#fill the missing value with mean/mode
#train_data.fillna(train_data.mode(),inplace=True)
train_data['balance'].fillna(train_data['balance'].mean(),inplace=True)
print(train_data['balance'].isnull().sum())

# fill column with mode
train_data['education'].fillna(train_data['education'].mode().iloc[0], inplace=True)
print(train_data['education'].isnull().sum())
train_data['month'].fillna(train_data['month'].mode().iloc[0], inplace=True)
print(train_data['month'].isnull().sum())
print(train_data.isnull().sum())

train_data.info()
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



#string conversion , label encoding, one hot encoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#train_data['marital'] = le.fit_transform(train_data['marital'])
train_data['loan'] = le.fit_transform(train_data['loan'])
train_data['class'] = le.fit_transform(train_data['class'])
train_data['housing'] = le.fit_transform(train_data['housing'])
train_data.info()
train_data['loan'].head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers = [('encoder', OneHotEncoder(),[1,2, 7])],remainder='passthrough')
train_data = np.array(ct.fit_transform(train_data))
train_data
#why should we transform it to numpy arrayt? 
#train_data is numpy array now not a dataframe, will it be a problem?
#train_data.info()
train_data=pd.DataFrame(train_data)
train_data.head()
train_data.info()

#correlation check
train_copy = train_data.copy()
corr_matrix = train_copy.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper
to_drop = [var for var in upper.columns if any(upper[var] > .90)]
to_drop

#x y split
x_train= train_data.iloc[:,:-1]
y_train= train_data.iloc[:,-1]
x_train.head()
x_train.info()
#scalling

#model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(x_train, y_train)
#predictions = clf.predict(x_test)

#save the model
#from sklearn.externals import joblib
#filename='finalized_model.sav'
#joblib.dump(model, filename)

#load the model from disk
#model = joblib.load('finalized_model.sav')
