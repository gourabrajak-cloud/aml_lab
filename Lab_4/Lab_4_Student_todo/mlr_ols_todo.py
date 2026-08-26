import numpy as np
import pandas as pd


# ============================================================
# Lab 4: Multiple Linear Regression
# Student TODO — Implement ONLY the core logic below.
#
# Data loading and validation are provided and must NOT be changed.
# ============================================================

def mae(y, y_pred):
    # TODO:
    # Calculate Mean Absolute Error:
    #
    # Return the MAE.
    pass


def mse(y, y_pred):
    # TODO:
    # Calculate Mean Squared Error:
    #
    # Return the MSE.
    pass


def rmse(y, y_pred):
    # TODO:
    # Calculate Root Mean Squared Error:
    #
    # Return the RMSE.
    pass


def r_square(y, y_pred, y_mean):
    # TODO:
    # Calculate R²:
    #
    # Return R².
    pass


def fit_multiple_lr(X, y):
    # TODO:
    # Implement the compact OLS Normal Equation:
    # beta = (X^T X)^(-1) X^T y
    #
    # Add an intercept column of ones before applying the equation.
    # Return the coefficient vector beta.
    pass


def fit_multiple_lr_beta(X, y):
    # TODO:
    # Implement the broken-down Normal Equation:
    # 1. Add an intercept column of ones.
    # 2. Calculate X.T.
    # 3. Calculate X.T @ X.
    # 4. Calculate the inverse of X.T @ X.
    # 5. Calculate X.T @ y.
    # 6. Calculate beta.
    #
    # Return the coefficient vector beta.
    pass


def predict_multiple_lr(X, beta):
    # TODO:
    # Add an intercept column of ones and calculate predictions:
    # y_pred = X @ beta
    #
    # Return the predicted values.
    pass


# ============================================================
# Provided Data Loading Logic — DO NOT IMPLEMENT
# ============================================================

def load_mlr_dataset(which=True):
    import os
    import gdown
    clean_csv = 'student_clean_dataset.csv'
    raw_csv = 'student_raw_dataset.csv'
    files = [
        {'file_id': '1t5mmVocO1_fGXGqRftpekRom9yxq-ML4', 'file_name': clean_csv},
        {'file_id': '1FDlcHX8C1tFCVr6T7g6Nd3EWKh4-N_Ww', 'file_name': raw_csv}
    ]
    for file in files:
        if not os.path.exists(file['file_name']):
            print(f"Downloading {file['file_name']}...")
            gdown.download(f"https://drive.google.com/uc?id={file['file_id']}", file['file_name'], quiet=True)
        else:
            print(f"{file['file_name']} already exists. Skipping download.")
    df_clean = pd.read_csv(clean_csv)
    df_raw = pd.read_csv(raw_csv)
    return df_clean if which else df_raw


# ============================================================
# Provided Validation Logic — DO NOT IMPLEMENT
# ============================================================

def validate_xy(X, y):
    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('\nX data type:')
    print(type(X), X.dtype)
    print('\ny data type:')
    print(type(y), y.dtype)
    print('\nMissing values:')
    print('X:', np.isnan(X).sum())
    print('y:', np.isnan(y).sum())
    print('\nNumber of observations:')
    print('X:', len(X))
    print('y:', len(y))
    if len(X) != len(y):
        raise ValueError('X and y must have the same number of observations.')
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError('X and y contain missing values.')
    print('\nX and y validation successful.')
