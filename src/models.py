import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    """Squash any number into a probability between 0 and 1."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def init_weights(n_features):
    """One weight per feature, plus one bias at the end. All start at zero."""
    return np.zeros(n_features + 1)


def predict_proba(w, X):
    """Probability that each patient is benign (label 1)."""
    return sigmoid(X @ w[:-1] + w[-1])


def train(w, X, y, epochs=200, lr=0.1):
    """Improve the weights with gradient descent. Returns a new weight array."""
    w = w.copy()
    for _ in range(epochs):
        p = predict_proba(w, X)
        error = p - y
        grad_w = X.T @ error / len(y)
        grad_b = error.mean()
        w[:-1] -= lr * grad_w
        w[-1] -= lr * grad_b
    return w


def evaluate(w, X, y):
    """Return accuracy and AUC on the given data."""
    p = predict_proba(w, X)
    accuracy = ((p >= 0.5) == y).mean()
    auc = roc_auc_score(y, p)
    return accuracy, auc