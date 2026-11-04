import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from split import load_and_split, split_into_hospitals
from models import evaluate
from federated import federated_averaging
from privacy import make_private_trainer
from run_experiments import hospitals_are_usable, N_SEEDS

CLIP = 1.0
NOISE_LEVELS = [0, 10, 25, 50, 100, 200, 400, 800]


def main():
    rows = []
    for seed in range(N_SEEDS):
        X_train, X_test, y_train, y_test = load_and_split(seed)
        hospitals = split_into_hospitals(X_train, y_train, seed=seed)
        if not hospitals_are_usable(hospitals):
            continue
        n_features = X_train.shape[1]

        w = federated_averaging(hospitals, n_features)
        acc, auc = evaluate(w, X_test, y_test)
        rows.append({"seed": seed, "noise": "none", "acc": acc, "auc": auc})

        for z in NOISE_LEVELS:
            trainer = make_private_trainer(CLIP, z, seed)
            w = federated_averaging(hospitals, n_features, trainer=trainer)
            acc, auc = evaluate(w, X_test, y_test)
            rows.append({"seed": seed, "noise": z, "acc": acc, "auc": auc})

    table = pd.DataFrame(rows)
    table.to_csv("results/privacy_results.csv", index=False)

    summary = table.groupby("noise", sort=False)[["acc", "auc"]].agg(["mean", "std"])
    print(f"Seeds used: {table['seed'].nunique()} of {N_SEEDS}")
    for noise, row in summary.iterrows():
        print(f"noise {str(noise):>5}: accuracy {row[('acc','mean')]:.3f} "
              f"+/- {row[('acc','std')]:.3f}   AUC {row[('auc','mean')]:.3f}")

    private = table[table["noise"] != "none"].copy()
    private["noise"] = private["noise"].astype(float)
    grouped = private.groupby("noise")["acc"].agg(["mean", "std"])
    baseline = table[table["noise"] == "none"]["acc"].mean()

    plt.figure(figsize=(6, 4))
    plt.errorbar(range(len(grouped)), grouped["mean"], yerr=grouped["std"],
                 marker="o", capsize=4, label="Private federated")
    plt.axhline(baseline, linestyle="--", color="gray", label="No privacy")
    plt.xticks(range(len(grouped)), [f"{z:g}" for z in grouped.index])
    plt.xlabel("Noise level (0 = clipping only)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy as privacy noise increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/privacy.png", dpi=200)


if __name__ == "__main__":
    main()