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
import numpy as np
import matplotlib.pyplot as plt

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
# Constructing the covariance matrix
cov_mat = np.cov(X_train_std.T)
# Eigenvalue, eigenvector pairs
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# plot the cumulative sum of explained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,25), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1,25), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Comment: Use two PCs will explain little. 10 is better but hard to visualize

# Sort eigenpairs by decreasing order of eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# collect 10 eigenvectors corresponds to the 10 largest eigenvalues to capture about
# 70 percent of the variance in the dataset
# Projection matrix
w = np.hstack([(eigen_pairs[i][1][:, np.newaxis]) for i in range(10)])
