import numpy as np
from sklearn.metrics import roc_auc_score
from models import predict_proba


def per_patient_loss(w, X, y):
    """How surprised the model is by each patient's true label."""
    p = np.clip(predict_proba(w, X), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def membership_attack_auc(w, X_members, y_members, X_outsiders, y_outsiders):
    """Guess who was in the training data: lower loss means more likely a member.

    0.5 means the attack is no better than a coin flip. Higher means leakage.
    """
    loss_in = per_patient_loss(w, X_members, y_members)
    loss_out = per_patient_loss(w, X_outsiders, y_outsiders)
    scores = -np.concatenate([loss_in, loss_out])
    truth = np.concatenate([np.ones(len(loss_in)), np.zeros(len(loss_out))])
    return roc_auc_score(truth, scores)