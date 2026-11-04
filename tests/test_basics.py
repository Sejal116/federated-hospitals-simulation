import numpy as np

from split import load_and_split, split_into_hospitals
from models import init_weights, train, evaluate
from federated import federated_averaging


def test_train_test_sizes():
    X_train, X_test, y_train, y_test = load_and_split(42)
    assert X_train.shape == (455, 30)
    assert X_test.shape == (114, 30)


def test_hospitals_hold_every_training_patient_exactly_once():
    X_train, _, y_train, _ = load_and_split(42)
    hospitals = split_into_hospitals(X_train, y_train, seed=42)

    assert sum(len(y_h) for _, y_h in hospitals) == len(y_train)

    in_hospitals = np.sort(np.concatenate([X_h.sum(axis=1) for X_h, _ in hospitals]))
    original = np.sort(X_train.sum(axis=1))
    assert np.allclose(in_hospitals, original)


def test_hospitals_have_different_patient_mixes():
    X_train, _, y_train, _ = load_and_split(42)
    hospitals = split_into_hospitals(X_train, y_train, seed=42)
    shares = [1 - y_h.mean() for _, y_h in hospitals]
    assert max(shares) - min(shares) > 0.1


def test_training_beats_an_untrained_model():
    X_train, X_test, y_train, y_test = load_and_split(42)
    untrained = init_weights(X_train.shape[1])
    trained = train(untrained, X_train, y_train)
    assert evaluate(trained, X_test, y_test)[0] > evaluate(untrained, X_test, y_test)[0]


def test_federated_is_close_to_pooled():
    X_train, X_test, y_train, y_test = load_and_split(42)
    hospitals = split_into_hospitals(X_train, y_train, seed=42)
    n_features = X_train.shape[1]

    pooled = train(init_weights(n_features), X_train, y_train)
    federated = federated_averaging(hospitals, n_features)

    pooled_acc = evaluate(pooled, X_test, y_test)[0]
    federated_acc = evaluate(federated, X_test, y_test)[0]
    assert abs(pooled_acc - federated_acc) < 0.03
    