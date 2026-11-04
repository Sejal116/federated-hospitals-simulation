import numpy as np
from models import init_weights, train


def federated_averaging(hospitals, n_features, rounds=20, local_epochs=10, lr=0.1):
    """Train one shared model without moving any patient data.

    Each round, every hospital improves a copy of the shared model on its
    own patients. The server then averages the copies, giving bigger
    hospitals more say, and sends the result back out.
    """
    global_w = init_weights(n_features)
    sizes = np.array([len(y_h) for _, y_h in hospitals])

    for _ in range(rounds):
        local_weights = []
        for X_h, y_h in hospitals:
            w = train(global_w, X_h, y_h, epochs=local_epochs, lr=lr)
            local_weights.append(w)
        global_w = np.average(local_weights, axis=0, weights=sizes)

    return global_w