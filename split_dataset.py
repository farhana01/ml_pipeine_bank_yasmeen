#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset= pd.read_csv("data/bank.csv")
print(dataset)
#print(dataset.info())
#print(dataset.describe())

#import train_test library
from sklearn.model_selection import train_test_split
train, test= train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
# what is random_state ,shuffle 
print("==============train data ===================")
print(train)
print("===============test data =======================")
print(test)
train.to_csv("bank_train.csv")
test.to_csv("bank_test.csv")

