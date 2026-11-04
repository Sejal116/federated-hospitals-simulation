import numpy as np
from models import sigmoid


def train_private(w, X, y, epochs, lr, clip, noise_multiplier, rng):
    """Gradient descent where no single patient can change the model much.

    For every patient we compute their own gradient, shrink it if its length
    is above `clip`, add up all the gradients, then add random noise.
    """
    w = w.copy()
    n = len(y)
    X_with_bias = np.hstack([X, np.ones((n, 1))])

    for _ in range(epochs):
        p = sigmoid(X_with_bias @ w)
        grads = (p - y)[:, None] * X_with_bias

        lengths = np.linalg.norm(grads, axis=1, keepdims=True)
        grads = grads * np.minimum(1.0, clip / np.maximum(lengths, 1e-12))

        noise = rng.normal(0, noise_multiplier * clip, size=w.shape)
        w -= lr * (grads.sum(axis=0) + noise) / n

    return w


def make_private_trainer(clip, noise_multiplier, seed):
    """Return a trainer with the same inputs as `train`, for federated use."""
    rng = np.random.default_rng(seed)

    def trainer(w, X, y, epochs, lr):
        return train_private(w, X, y, epochs, lr, clip, noise_multiplier, rng)

    return trainer