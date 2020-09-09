# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:13 2020

@author: Farhana
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
train_data_copy=train_data.copy()
print(train_data_copy.head())
print(train_data_copy.isnull().sum())
train_data_copy=train_data_copy.dropna()
print(train_data_copy.isnull().sum().sum())
print(train_data_copy.shape)


#fill the missing value with mean/mode


