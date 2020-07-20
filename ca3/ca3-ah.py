# -*- coding: utf-8 -*-

__author__ = 'Anders Mølmen Høst'
__email__ = 'anders.molmen.host@nmbu.no'

"""
Compulsory assignment 3
"""

"""
- Read the data
- Split dataset into training and test datasets, of the training data
- Standardize (necessary for all but decision trees/random forest?)
- Train the model, start with Perceptron
- Print out accuracy
- Make changes

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv('CA3-train.csv')

# Search for missing values
# print(df.isnull().sum()). Output = 0

# Assign features to X matrix and corresponding labels to vector y
X, y = df.iloc[:, 1:25].values, df.iloc[:, 25]
# Split the dataset by using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                    random_state=0)

# Standardizing our data to make algorithms behave better
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# PCA

