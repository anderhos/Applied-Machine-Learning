# -*- coding: utf-8 -*-

__author__ = 'Anders Mølmen Høst'
__email__ = 'anders.molmen.host@nmbu.no'

"""
Compulsory assignment 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('CA4-glassTrain.csv')

# Number of features to be plotted

num_vars = 9

# Create frame with features and types
glass_df = df.iloc[:, 1:num_vars+2]
# assign types to variable y
y_train = glass_df.iloc[:, 9]
# assign feature to variable X
X_train = glass_df.iloc[:, 0:num_vars]

# perform a nested cross-validation

# Classify our data using an estimator an varying
# the parameters


def scores(GridSearchCV):
    gs = GridSearchCV(estimator=DecisionTreeClassifier(
                        random_state=0),
                        param_grid=[{'max_depth': [1, 2, 3,
                               4, 5, 6,
                               7, None]}],
                        scoring='accuracy',
                        cv=2)
    return cross_val_score(gs, X_train, y_train,
                           scoring='accuracy', cv=6)


if __name__ == "__main__":
    acc_scores = scores(GridSearchCV)
    print(f'CV accuracy: {np.mean(acc_scores)} +/- {np.std(acc_scores)}')
    print(f'Max accuracy: {max(acc_scores)}')