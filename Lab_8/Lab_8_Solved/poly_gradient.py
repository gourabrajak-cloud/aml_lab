import numpy as np

def predict_mlr(X, weights):
    """Generate MLR predictions: y_hat = X @ weights."""
    return X @ weights


def compute_mlr_residual(y, predictions):
    """Compute prediction error: predicted minus actual."""
    return predictions - y


def compute_mlr_gradient(X, residual):
    """Compute the full batch gradient for all MLR parameters."""
    return 2/len(X)*((X.T)@residual)
    

def compute_mlr_loss(residual):
    """Compute Mean Squared Error loss."""
    return np.mean(residual**2)


def update_mlr_parameters(weights, gradient, alpha):
    """Update all MLR parameters using the learning rate."""
    return weights - alpha * gradient





def minibatch_gradient_descent_mlr(X, y, theta, alpha, epochs, batch_size):
    loss = 0
    loss_history = []
    n = X.shape[0]
    #TODO
    for epoch in range(epochs):
        ind = np.random.permutation(n)
        Xs = X[ind]
        ys = y[ind]
        loss
        for start in range(0, n, batch_size):
            end = start + batch_size
            Xb = Xs[start:end]
            yb = ys[start:end]
            b_pred = predict_mlr(Xb, theta)
            b_res = compute_mlr_residual(yb, b_pred)
            gred_b = compute_mlr_gradient(Xb, b_res)
            theta = update_mlr_parameters(theta, gred_b, alpha)

        pred_t = predict_mlr(X, theta)
        res_t = compute_mlr_residual(y, pred_t)
        loss = (compute_mlr_loss(res_t))

        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Loss = {loss:.4f}")

    return theta, loss_history
    

def create_polynomial_features(X, degree):
    """Create [1, X, X², ..., X^degree] using scaled X."""

    X = np.asarray(X).reshape(-1)

    # # Scale X to [-1, 1] before taking high powers.
    X = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1
    
    features = [np.ones(len(X))]
    for power in range(1, degree + 1):
        features.append(X ** power)
        
    return np.column_stack(features)


def polynomial_regression(X, y, degree, alpha, epochs, batch_size):
    """Train Polynomial Regression using MLR Mini-Batch GD."""

    X_poly = create_polynomial_features(X, degree)
    theta = np.zeros(X_poly.shape[1])

    return minibatch_gradient_descent_mlr(
        X_poly, y, theta, alpha, epochs, batch_size
    )


def predict_polynomial(X, theta, degree):
    """Generate Polynomial Regression predictions."""
    X_poly = create_polynomial_features(X, degree)
    
    return predict_mlr(X_poly, theta)


def r_square(y, y_pred, y_mean):
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    if ss_total == 0:
        raise ValueError("R^2 is undefined when y has zero variance.")

    return 1 - (ss_residual / ss_total)
