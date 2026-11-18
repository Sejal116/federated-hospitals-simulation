"""Does a shared model help or hurt each hospital? UCI Heart Disease, 4 sites.

For every hospital we compare:
  local          trained only on that hospital's own patients
  federated      one shared model (federated averaging) across all four
  fed+finetune   the shared model, then adjusted on the hospital's own patients
  fed+bias       the shared model with only its starting point (bias) re-fitted
                 on the hospital's own patients. Added after the first run
                 showed that Switzerland has a very different disease rate.
  majority       always guess the hospital's most common outcome (a floor)

Every patient is tested using 5-fold cross-validation inside their own
hospital, and the whole thing is repeated with different random folds.
Numbers are errors per 100 patients (lower is better).

Run from the main folder:
    python src/run_heart.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from federated import federated_averaging
from heart_data import (DISPLAY_NAMES, HOSPITALS, N_FEATURES, apply_preparation,
                        fit_preparation, load_all, missing_table)
from models import init_weights, predict_proba, train

LOCAL_EPOCHS = 200
FINETUNE_EPOCHS = 20      # chosen before looking at results
FINETUNE_LR = 0.05
METHODS = ["local", "federated", "fed+finetune", "fed+bias", "majority"]


def train_bias_only(w, X, y, epochs=200, lr=0.1):
    """Keep every weight of the shared model fixed and re-fit only the bias."""
    w = w.copy()
    for _ in range(epochs):
        w[-1] -= lr * (predict_proba(w, X) - y).mean()
    return w


def fit_all_methods(prepared_train):
    """Train every method on one set of training folds. Returns weights."""
    hospitals = [prepared_train[name] for name in HOSPITALS]
    shared = federated_averaging(hospitals, N_FEATURES)
    models = {"federated": {}, "local": {}, "fed+finetune": {}, "fed+bias": {}}
    for name in HOSPITALS:
        X, y = prepared_train[name]
        models["federated"][name] = shared
        models["local"][name] = train(init_weights(N_FEATURES), X, y,
                                      epochs=LOCAL_EPOCHS, lr=0.1)
        models["fed+finetune"][name] = train(shared, X, y,
                                             epochs=FINETUNE_EPOCHS,
                                             lr=FINETUNE_LR)
        models["fed+bias"][name] = train_bias_only(shared, X, y)
    return models


def one_repeat(data, repeat, folds=5):
    """Out-of-fold predictions for every patient, for every method."""
    splitters = {
        name: list(StratifiedKFold(folds, shuffle=True, random_state=repeat)
                   .split(df, df["disease"]))
        for name, df in data.items()
    }
    predictions = {m: {n: np.zeros(len(data[n])) for n in HOSPITALS}
                   for m in METHODS}

    for fold in range(folds):
        prepared_train, prepared_test, test_index = {}, {}, {}
        for name, df in data.items():
            train_idx, test_idx = splitters[name][fold]
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            stats = fit_preparation(train_df)
            prepared_train[name] = (apply_preparation(train_df, stats),
                                    train_df["disease"].to_numpy())
            prepared_test[name] = apply_preparation(test_df, stats)
            test_index[name] = test_idx
        models = fit_all_methods(prepared_train)

        for name in HOSPITALS:
            X_test = prepared_test[name]
            for method in ("local", "federated", "fed+finetune", "fed+bias"):
                predictions[method][name][test_index[name]] = predict_proba(
                    models[method][name], X_test)
            majority = prepared_train[name][1].mean() >= 0.5
            predictions["majority"][name][test_index[name]] = float(majority)

    rows = []
    for name in HOSPITALS:
        y = data[name]["disease"].to_numpy()
        for method in METHODS:
            p = predictions[method][name]
            error = 100 * ((p >= 0.5) != y).mean()
            auc = roc_auc_score(y, p) if method != "majority" else np.nan
            balanced = balanced_accuracy_score(y, p >= 0.5)
            rows.append({"repeat": repeat, "hospital": DISPLAY_NAMES[name],
                         "method": method, "errors_per_100": error,
                         "balanced_acc": balanced, "auc": auc})
    return rows


def summarise(results):
    """Mean and the middle 95% of the repeats, per hospital and method."""
    table = (results.groupby(["hospital", "method"])["errors_per_100"]
             .agg(mean="mean", low=lambda s: s.quantile(0.025),
                  high=lambda s: s.quantile(0.975)).round(1).reset_index())
    for extra in ("balanced_acc", "auc"):
        mean = results.groupby(["hospital", "method"])[extra].mean().round(3)
        table[extra] = [mean[(h, m)] for h, m in zip(table["hospital"], table["method"])]
    return table


def paired_gap(results):
    """Shared minus local errors per 100, repeat by repeat."""
    wide = results.pivot_table(index=["hospital", "repeat"], columns="method",
                               values="errors_per_100").reset_index()
    rows = []
    for hospital, group in wide.groupby("hospital"):
        for other in ("federated", "fed+finetune", "fed+bias"):
            gap = group[other] - group["local"]
            rows.append({"hospital": hospital, "method": other,
                         "gap_mean": round(gap.mean(), 1),
                         "gap_low": round(gap.quantile(0.025), 1),
                         "gap_high": round(gap.quantile(0.975), 1),
                         "worse_than_local_in": f"{(gap > 0).sum()} of {len(gap)}"})
    return pd.DataFrame(rows)


def plot(summary, path):
    hospitals = [DISPLAY_NAMES[n] for n in HOSPITALS]
    width = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, method in enumerate(METHODS):
        sub = summary[summary["method"] == method].set_index("hospital").loc[hospitals]
        err = [sub["mean"] - sub["low"], sub["high"] - sub["mean"]]
        ax.bar(np.arange(4) + (i - 2) * width, sub["mean"], width,
               yerr=err, capsize=2, label=method)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(hospitals)
    ax.set_ylabel("Wrong results per 100 patients (lower is better)")
    ax.set_title("UCI Heart Disease: each hospital, five methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/heart")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    data = load_all(args.data_dir)
    print("Patients and disease share per hospital:")
    for name, df in data.items():
        print(f"  {DISPLAY_NAMES[name]:14s} {len(df):4d} patients, "
              f"{df['disease'].mean():.0%} have heart disease")
    print("\nPercent of patients with each measurement missing:")
    print(missing_table(data).to_string())

    rows = []
    for repeat in range(args.repeats):
        rows += one_repeat(data, repeat)
    results = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/heart_results.csv", index=False)
    summary = summarise(results)
    summary.to_csv("results/heart_summary.csv", index=False)
    gaps = paired_gap(results)
    gaps.to_csv("results/heart_gap.csv", index=False)
    plot(summary, "results/heart.png")

    print(f"\nWrong results per 100 patients, mean and middle 95% of "
          f"{args.repeats} repeats:")
    print(summary.to_string(index=False))
    print("\nShared model minus local model (positive = shared is worse):")
    print(gaps.to_string(index=False))
