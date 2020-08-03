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
from matplotlib.colors import ListedColormap

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
# Sum missing values for each column
missing_values = df.isnull().sum()
if missing_values.any():
    print("Dataset has missing values")
else:
    print("No missing values in dataset!")

# Assign features to X matrix and corresponding labels to vector y

# features
c_first = 1
c_last = 25    # not included

X, y = df.iloc[:, c_first:c_last].values, df.iloc[:, 25]
#print(df.iloc[:, 21:24])
print(f"Selected features:", df.iloc[:, c_first:c_last].columns)


# Default parameters
seed = 1
test_size = 0.3

# Splitting data with default parameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                    random_state=seed)

# Standardizing our data to make algorithms behave better
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Function to plot decision regions. Works only when two features are selected
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # Source Python for Machine Learning ch05
    # setup marker generator and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx2.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')
    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')
    plt.xlabel('First feature [standardized]')
    plt.ylabel('Second feature [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def combined(X_train, X_test, y_train, y_test):
    # Function for stacking training and test data
    # To be used in plot_decision_regions
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    return X_combined, y_combined


def fit_test_size(ppn, X, y, test_size_list, seed, feature_extraction=False, n_components=None):
    # Accuracy for different test_train_splits
    for size in test_size_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, stratify=y,
                                                            random_state=seed)
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.fit_transform(X_test)
        if feature_extraction:
            # Dimensionality reduction through PCA
            pca = PCA(n_components=n_components, random_state=1)
            X_train_pca = pca.fit_transform(X_train_std)
            X_test_pca = pca.fit_transform(X_test_std)
            ppn.fit(X_train_pca, y_train)
            y_pred = ppn.predict(X_test_pca)
            print(f'Misclassified examples PCA: {(y_test != y_pred).sum()}')
            print('Accuracy PCA: {:.3}'.format(ppn.score(X_test_pca, y_test)))
            print(f'Test size: {size}')
        else:
            ppn.fit(X_train_std, y_train)
            y_pred = ppn.predict(X_test_std)
            print(f'Misclassified examples: {(y_test != y_pred).sum()}')
            print('Training Accuracy: {:.3}'.format(ppn.score(X_train_std, y_train)))
            print('Accuracy: {:.3}'.format(ppn.score(X_test_std, y_test)))
            print(f'Test size: {size}')
    # Note: After function call test size is the last index of test_size_list

#

# Plot results of classification



# plot cumulative sum of explained variances
def plot_var_exp(n_components):
    pca = PCA(n_components=n_components, random_state=1)
    pca.fit_transform(X_train_std)
    pca.fit_transform(X_test_std)
    var_exp = pca.explained_variance_ratio_
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


# Plot accuracy at different learning rates, eta, for Perceptron
def plot_learning_rate(eta):
    ppn_score = []
    for learning_rate in eta:
        ppn = Perceptron(eta0=learning_rate, random_state=1)
        ppn.fit(X_train_std, y_train)
        ppn_score.append(ppn.score(X_test, y_test))
    plt.plot(range(len(ppn_score)), ppn_score)
    # Note: Consider changing from index to learning rate
    plt.xlabel('Learning rate index')
    plt.ylabel('Accuracy')
    plt.show()





if __name__ == "__main__":
    # plot_var_exp(10)
    # print(f"test size for plot learning rates: {i}")
    # Plot Learning Perceptron
    #learning_rates = [0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]

    # Split the dataset by using train_test_split
    test_size_list = [0.6, 0.3, 0.1, 0.05, 0.01]
    ppn = Perceptron(penalty='l1', alpha=0.001, eta0=0.01, random_state=seed)
    fit_test_size(ppn, X, y, test_size_list, seed)


    #X_combined, y_combined = combined(X_train_pca, X_test_pca, y_train, y_test)

    #plot_decision_regions(X=X_combined, y=y_combined, classifier=ppn)
    #plot_learning_rate(learning_rates)


# Notes: learning rates entirely different at different seeds.
# Might be prone to overfitting? Low learning rate.
# Random_state = 1, good learning rate at around 0.01

#plot_var_exp(n_components=10)


# Test with different train_test_splits and different eta
# Different number of PCAs
# Test with feature selection
# Modify parameters see the sklearn website
# make plots