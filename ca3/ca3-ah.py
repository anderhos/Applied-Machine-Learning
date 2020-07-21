# -*- coding: utf-8 -*-

__author__ = 'Anders Mølmen Høst'
__email__ = 'anders.molmen.host@nmbu.no'

"""
Compulsory assignment 3
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

"""
- Read the data
- Split dataset into training and test datasets, of the training data
- Standardize (necessary for all but decision trees/random forest?)
- Train the model, start with Perceptron
- Print out accuracy
- Make changes

"""


# Read data
df = pd.read_csv('CA3-train.csv')

# Search for missing values
# print(df.isnull().sum()). Output = 0

# Assign features to X matrix and corresponding labels to vector y
X, y = df.iloc[:, 1:25].values, df.iloc[:, 25]
# Split the dataset by using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, stratify=y,
                                                    random_state=1)
# Different test_sizes
test_size = [0.6, 0.3, 0.1, 0.05, 0.01]


# Standardizing our data to make algorithms behave better
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Initialize the PCA transformer
n_components = 10
pca = PCA(n_components=n_components, random_state=1)
# Dimensionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)
#print(pca.explained_variance_ratio_)

# plot cumulative sum of explained variances
def plot_var_exp(n_components):
    pca2 = PCA(n_components=n_components, random_state=1)
    X_train_pca = pca2.fit_transform(X_train_std)
    X_test_pca = pca2.fit_transform(X_test_std)
    var_exp = pca2.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1, n_components + 1), var_exp, alpha=0.5, align='center',
            label='Individual explained variances')
    plt.step(range(1, n_components + 1), cum_var_exp, where='mid',
             label='Cumulative explained variances')
    plt.xlabel('Explained variance ratio')
    plt.ylabel('Principal component index')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_var_exp(n_components=10)

# Fitting the Perceptron on the reduced dataset
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_pca, y_train)
y_pred = ppn.predict(X_test_pca)
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print('Accuracy: {:.3}'.format(ppn.score(X_test_pca, y_test)))

# Fitting the Perceptron on the original dataset
ppn2 = Perceptron(eta0=0.1, random_state=1)
ppn2.fit(X_train_std, y_train)
y_pred = ppn2.predict(X_test_std)
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print('Accuracy: {:.3}'.format(ppn2.score(X_test, y_test)))


# Task. Plot accuracy from different train_test_split. test_size.


# Test with different train_test_splits and different eta
# Different number of PCAs
# Test with feature selection
# Modify parameters see the sklearn website
# make plots