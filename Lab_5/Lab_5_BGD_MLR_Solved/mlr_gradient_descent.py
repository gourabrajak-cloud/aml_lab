import numpy as np


def predict_mlr(X, weights):
    """Generate MLR predictions: y_hat = H @ weights."""

    # Add a column of ones for the intercept.
    ones = np.ones((X.shape[0], 1))
    H = np.hstack((ones, X))

    return H @ weights


def compute_mlr_residual(y, predictions):
    """Compute prediction error: predicted minus actual."""

    return predictions - y


def compute_mlr_gradient(X, residual):
    """Compute the full batch gradient for all MLR parameters."""

    n = X.shape[0]

    # Add a column of ones for the intercept.
    ones = np.ones((X.shape[0], 1))
    H = np.hstack((ones, X))

    # Gradient = (2 / n) H transpose residual.
    gradient = (2 / n) * (H.T @ residual)

    return gradient


def compute_mlr_loss(residual):
    """Compute Mean Squared Error loss."""

    return np.mean(residual ** 2)


def update_mlr_parameters(weights, gradient, alpha):
    """Update all MLR parameters using the learning rate."""

    weights = weights - alpha * gradient

    return weights


def batch_gradient_descent_mlr(X, y, weights, alpha, epochs):
    """Train MLR parameters using Batch Gradient Descent."""

    loss_history = []

    for i in range(epochs):

        # 1. Make predictions.
        predictions = predict_mlr(X, weights)

        # 2. Calculate residual.
        residual = compute_mlr_residual(y, predictions)

        # 3. Calculate the gradient.
        gradient = compute_mlr_gradient(X, residual)

        # 4. Update parameters.
        weights = update_mlr_parameters(
            weights, gradient, alpha
        )

        # 5. Calculate loss.
        loss = compute_mlr_loss(residual)
        loss_history.append(loss)

        # Print progress.
        if i % 1000 == 0:
            print(f"Iteration {i:5d} | Loss = {loss:.4f}")

    return weights, loss_history
