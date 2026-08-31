import numpy as np


def find_mean(values):
    return np.mean(values)


def find_slope_intercept(x, y, x_mean, y_mean):
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        raise ValueError("Cannot calculate slope: x has zero variance.")

    m = numerator / denominator
    b = y_mean - m * x_mean
    return m, b


def predict_lr(x, m, b):
    return m * x + b


def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def r_square(y, y_pred, y_mean):
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    if ss_total == 0:
        raise ValueError("R^2 is undefined when y has zero variance.")

    return 1 - (ss_residual / ss_total)


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
