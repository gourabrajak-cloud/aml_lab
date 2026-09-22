import numpy as np
import pandas as pd

def design_matrix():
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))
    return X

def r_square(y, y_pred, y_mean):
    n = np.sum((y-y_pred)**2)
    d = np.sum((y-y_mean)**2)
    return 1 - (n)/(d)

def fit_multiple_lr(X, y):
    return np.linalg.inv(X.T@X)@X.T@y

def fit_multiple_lr_beta(X, y):
    XT = X.T
    XTX = X.T @ X
    XTXI = np.linalg.inv(XTX)
    XTY = XT @ y
    beta = XTXI @ XTY
    return beta

def predict_multiple_lr(X, beta):
    return X @ beta

## Functions for Ploynomial regression below:

def polynomial_features(X, degree):
    return np.column_stack(
        [X ** power for power in range(degree + 1)]
    )