import numpy as np


def predict_mlr(X, weights):
    """Generate MLR predictions: y_hat = H @ weights."""

    # Add a column of ones for the intercept.
    pass


def compute_mlr_residual(y, predictions):
    """Compute prediction error: predicted minus actual."""

    pass


def compute_mlr_gradient(X, residual):
    """Compute the full batch gradient for all MLR parameters."""

    # return gradient
    pass


def compute_mlr_loss(residual):
    """Compute Mean Squared Error loss."""

    pass


def update_mlr_parameters(weights, gradient, alpha):
    """Update all MLR parameters using the learning rate."""

    # return weights
    pass

def batch_gradient_descent_mlr(X, y, weights, alpha, epochs):
    """Train MLR parameters using Batch Gradient Descent."""

    loss_history = []

    for i in range(epochs):

        # 1. Make predictions.

        # 2. Calculate residual.

        # 3. Calculate the gradient.

        # 4. Update parameters.

        # 5. Calculate loss.
        # loss = 
        loss_history.append(loss)

        # Print progress.
        if i % 1000 == 0:
            print(f"Iteration {i:5d} | Loss = {loss:.4f}")

    return weights, loss_history
