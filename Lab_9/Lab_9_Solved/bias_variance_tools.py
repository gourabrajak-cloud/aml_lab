"""
Lab 9 helpers: bias, variance, and model complexity.

This module intentionally contains only the "plumbing" that every part of
the notebook needs but that isn't itself part of the lesson:

- fitting/predicting with a polynomial regression model
- scaling an input variable
- loading and splitting the real dataset

Everything that IS the lesson -- drawing repeated samples, estimating bias
and variance, cross-validation, learning curves -- is written directly in
the notebook, one step at a time, so students can read exactly how each
quantity is computed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def scale_values(x, minimum, maximum):
    """Scale one input variable into the range [-1, 1].

    Polynomial features blow up quickly for large x, so we rescale before
    fitting. minimum/maximum are passed in explicitly (rather than computed
    from x itself) so that a model can be scaled consistently even when it
    is evaluated on different data than it was trained on.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if minimum == maximum:
        return np.zeros_like(x)
    return 2 * (x - minimum) / (maximum - minimum) - 1


def fit_polynomial_model(x, y, degree, x_minimum=None, x_maximum=None):
    """Fit a polynomial regression model of the given degree.

    Returns a small dictionary bundling everything predict_polynomial()
    needs later: the fitted regression, the polynomial feature
    transformer, and the scaling range used during training.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of values.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    if x_minimum is None:
        x_minimum = x.min()
    if x_maximum is None:
        x_maximum = x.max()

    x_scaled = scale_values(x, x_minimum, x_maximum).reshape(-1, 1)
    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    x_polynomial = polynomial.fit_transform(x_scaled)

    regression = LinearRegression()
    regression.fit(x_polynomial, y)

    return {
        "degree": degree,
        "polynomial": polynomial,
        "regression": regression,
        "x_minimum": x_minimum,
        "x_maximum": x_maximum,
    }


def predict_polynomial(model, x):
    """Predict with a model returned by fit_polynomial_model()."""
    x_scaled = scale_values(x, model["x_minimum"], model["x_maximum"]).reshape(-1, 1)
    x_polynomial = model["polynomial"].transform(x_scaled)
    return model["regression"].predict(x_polynomial)


def load_academic_data(file_path):
    """Read the academic dataset and check that the expected columns exist."""
    data = pd.read_csv(file_path)
    needed_columns = ["Study_Hours", "Exam_Score"]

    for column in needed_columns:
        if column not in data.columns:
            raise ValueError("The dataset is missing the column: " + column)

    return data


def split_regression_data(data, seed=42):
    """Make a 60% training, 20% validation, and 20% test split."""
    x = data["Study_Hours"].to_numpy()
    y = data["Exam_Score"].to_numpy()

    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development, y_development, test_size=0.25, random_state=seed
    )

    return {
        "x_train": x_train,
        "x_validation": x_validation,
        "x_test": x_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }
