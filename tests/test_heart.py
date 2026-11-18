import numpy as np
import pandas as pd

from heart_data import (COLUMNS, N_FEATURES, apply_preparation,
                        fit_preparation)
from models import init_weights, predict_proba
from run_heart import train_bias_only


def make_patients(n=30, chol_missing=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(100, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(90, 190, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.random(n) * 3,
    })
    df.loc[:4, "thalach"] = np.nan
    df.loc[:4, "cp"] = np.nan
    if chol_missing:
        df["chol"] = np.nan
    return df


def test_preparation_gives_fixed_size_table_without_gaps():
    train_df, test_df = make_patients(30), make_patients(10)
    stats = fit_preparation(train_df)
    X = apply_preparation(test_df, stats)
    assert X.shape == (10, N_FEATURES)
    assert not np.isnan(X).any()


def test_a_measurement_missing_everywhere_becomes_zeros_not_errors():
    train_df = make_patients(30, chol_missing=True)
    stats = fit_preparation(train_df)
    X = apply_preparation(train_df, stats)
    assert not np.isnan(X).any()
    assert np.allclose(X[:, 2], 0)  # chol is the third numeric column


def test_preparation_learns_only_from_the_given_training_patients():
    train_df = make_patients(30)
    stats = fit_preparation(train_df)
    shifted = train_df.copy()
    shifted["age"] = shifted["age"] + 100
    X = apply_preparation(shifted, stats)
    assert X[:, 0].mean() > 5  # still scaled with the original training mean


def test_bias_only_training_leaves_every_weight_unchanged():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, N_FEATURES))
    y = (rng.random(50) < 0.9).astype(float)
    w = init_weights(N_FEATURES)
    w[:-1] = rng.normal(size=N_FEATURES)
    new_w = train_bias_only(w, X, y)
    assert np.array_equal(new_w[:-1], w[:-1])
    assert new_w[-1] != w[-1]
    assert predict_proba(new_w, X).mean() > predict_proba(w, X).mean()
