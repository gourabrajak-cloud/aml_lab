import numpy as np
import pandas as pd


# ============================================================
# Lab 3: Simple Linear Regression
# Student TODO — Implement ONLY the core logic below.
#
# Data loading and validation are provided and must NOT be changed.
# ============================================================


def find_mean(values):
    # TODO:
    # Calculate and return the arithmetic mean of the values.
    pass


def find_slope_intercept(x, y, x_mean, y_mean):
    # TODO:
    # Implement the OLS formulas:
    #
    # Return: m, b
    pass


def predict_lr(x, m, b):
    # TODO:
    # Generate predictions:
    #
    #
    # Return the predicted values.
    pass


def calculate_residuals(y, y_pred):
    # TODO:
    # Calculate the prediction error for every observation:
    #
    # Return the residuals.
    pass


# ============================================================
# Error / Evaluation Metrics — Student TODO
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
    #
    # Return R².
    pass


# ============================================================
# Provided Data Loading Logic — DO NOT IMPLEMENT
# ============================================================

def load_slr_dataset(which):
    import os
    import gdown

    clean_csv = "student_clean_dataset.csv"
    raw_csv = "student_raw_dataset.csv"

    files = [{"file_id": "1t5mmVocO1_fGXGqRftpekRom9yxq-ML4", "file_name": clean_csv},
        {"file_id": "1FDlcHX8C1tFCVr6T7g6Nd3EWKh4-N_Ww", "file_name": raw_csv}]

    for file in files:
        if not os.path.exists(file["file_name"]):
            print(f"Downloading {file['file_name']}...")

            gdown.download(f"https://drive.google.com/uc?id={file['file_id']}", file["file_name"], quiet=True)
        else:
            print(f"{file['file_name']} already exists. Skipping download.")

    df_clean = pd.read_csv(clean_csv)
    df_raw = pd.read_csv(raw_csv)
    return df_clean if which else df_raw


# ============================================================
# Provided Validation Logic — DO NOT IMPLEMENT
# ============================================================

def validate_xy(X, y):
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nX data type:")
    print(X.dtypes)

    print("\ny data type:")
    print(y.dtypes)

    print("\nMissing values:")
    print("X:", X.isna().sum().sum())
    print("y:", y.isna().sum())

    print("\nNumber of observations:")
    print("X:", len(X))
    print("y:", len(y))

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of observations.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("X and y contain missing values.")

    print("\nX and y validation successful.")
