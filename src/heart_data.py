"""Load the UCI Heart Disease data as four separate hospitals.

The dataset comes from four hospitals: Cleveland and VA Long Beach (USA),
Hungary, and Switzerland. Each hospital is already a separate file, so this
is a more natural federated setting than slicing one dataset at random.

Each hospital prepares its own data using only its own training patients
(filling gaps and scaling), so nothing is shared between hospitals before
training. This also avoids the scaler problem noted for the breast cancer
experiment.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
DISPLAY_NAMES = {
    "cleveland": "Cleveland",
    "hungarian": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.{}.data")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# slope, ca and thal are left out: they are missing for most patients at
# three of the four hospitals.
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2]}

# In these columns a value of 0 is impossible, so it means "not recorded".
ZERO_MEANS_MISSING = ["trestbps", "chol"]


def load_hospital(name, data_dir="data/heart"):
    """Return one hospital's patients. Downloads the file on first use."""
    path = os.path.join(data_dir, f"processed.{name}.data")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path)
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # Some copies of these files contain a stray text line. Turn every value
    # into a number and drop any row that has no outcome label.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["num"]).reset_index(drop=True)
    for col in ZERO_MEANS_MISSING:
        df.loc[df[col] == 0, col] = np.nan
    # Label 1 = any heart disease (original values 1 to 4), 0 = none.
    df["disease"] = (df["num"] > 0).astype(int)
    return df


def load_all(data_dir="data/heart"):
    return {name: load_hospital(name, data_dir) for name in HOSPITALS}


def missing_table(hospitals):
    """Share of patients with each measurement missing, per hospital."""
    columns = NUMERIC + BINARY + list(CATEGORICAL)
    rows = {DISPLAY_NAMES[n]: df[columns].isna().mean() * 100
            for n, df in hospitals.items()}
    return pd.DataFrame(rows).round(0).astype(int)


def fit_preparation(train_df):
    """Learn how to fill gaps and scale, from one hospital's own patients."""
    stats = {}
    for col in NUMERIC:
        observed = train_df[col].dropna()
        median = observed.median() if len(observed) else 0.0
        filled = train_df[col].fillna(median)
        std = filled.std()
        stats[col] = (median, filled.mean(), std if std and std > 0 else 1.0)
    for col in BINARY:
        observed = train_df[col].dropna()
        stats[col] = observed.mode().iloc[0] if len(observed) else 0.0
    return stats


def apply_preparation(df, stats):
    """Turn a hospital's patients into a number-only table of 17 features."""
    parts = []
    for col in NUMERIC:
        median, mean, std = stats[col]
        parts.append(((df[col].fillna(median) - mean) / std).to_numpy()[:, None])
    for col in BINARY:
        parts.append(df[col].fillna(stats[col]).to_numpy()[:, None])
    for col, levels in CATEGORICAL.items():
        for level in levels:
            # A missing value gives zeros in every column of this group.
            parts.append((df[col] == level).astype(float).to_numpy()[:, None])
    return np.hstack(parts).astype(float)


N_FEATURES = len(NUMERIC) + len(BINARY) + sum(len(v) for v in CATEGORICAL.values())
