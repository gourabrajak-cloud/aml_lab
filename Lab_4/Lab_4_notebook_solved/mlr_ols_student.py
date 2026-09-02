import numpy as np
import pandas as pd

# Lab 4: Multiple Linear Regression using OLS

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def r_square(y, y_pred, y_mean):
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def fit_multiple_lr(X, y):
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def fit_multiple_lr_beta(X, y):
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))
    XT = X.T
    XTX = XT @ X
    XTX_inverse = np.linalg.inv(XTX)
    XTy = XT @ y
    beta = XTX_inverse @ XTy
    return beta

def predict_multiple_lr(X, beta):
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))
    return X @ beta

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
