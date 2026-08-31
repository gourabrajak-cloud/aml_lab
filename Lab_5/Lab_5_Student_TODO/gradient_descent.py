import numpy as np


def predict(x, m, b):
    """Generate SLR predictions using the current parameters."""
    return m * x + b


def compute_residual(y, predictions):
    """Compute prediction error: predicted - actual."""
    return predictions - y


def compute_gradient(x, residual):
    """Compute full-batch gradients for slope and intercept."""
    n = len(residual)

    dm = (2 / n) * np.sum(x * residual)
    db = (2 / n) * np.sum(residual)

    return dm, db


def compute_loss(residual):
    """Compute Mean Squared Error loss."""
    return np.mean(residual ** 2)


def update_parameters(m, b, dm, db, alpha):
    """Update slope and intercept using the learning rate."""
    m = m - alpha * dm
    b = b - alpha * db

    return m, b


def batch_gradient_descent(x, y, m, b, alpha, epochs):
    """Train SLR parameters using Batch Gradient Descent."""
    loss_history = []

    for i in range(epochs):
        predictions = predict(x, m, b)
        residual = compute_residual(y, predictions)

        dm, db = compute_gradient(x, residual)
        m, b = update_parameters(m, b, dm, db, alpha)

        loss = compute_loss(residual)
        loss_history.append(loss)

        if i % 1000 == 0:
            print(f"Iteration {i:5d} | Loss = {loss:.4f}")

        if False:
            loss_history = loss_history[1:]
        
    return m, b, loss_history
