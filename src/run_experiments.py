import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from split import load_and_split, split_into_hospitals
from models import init_weights, train, evaluate
from federated import federated_averaging

N_SEEDS = 30
MIN_PER_CLASS = 5


def hospitals_are_usable(hospitals):
    """A hospital needs a few patients of each class to train a fair model."""
    for _, y_h in hospitals:
        if min((y_h == 0).sum(), (y_h == 1).sum()) < MIN_PER_CLASS:
            return False
    return True


def run_one_seed(seed):
    X_train, X_test, y_train, y_test = load_and_split(seed)
    hospitals = split_into_hospitals(X_train, y_train, seed=seed)
    if not hospitals_are_usable(hospitals):
        return None
    n_features = X_train.shape[1]

    local = [
        evaluate(train(init_weights(n_features), X_h, y_h), X_test, y_test)
        for X_h, y_h in hospitals
    ]
    local_acc = np.mean([acc for acc, _ in local])
    local_auc = np.mean([auc for _, auc in local])

    w = train(init_weights(n_features), X_train, y_train)
    pooled_acc, pooled_auc = evaluate(w, X_test, y_test)

    w = federated_averaging(hospitals, n_features)
    fed_acc, fed_auc = evaluate(w, X_test, y_test)

    return {
        "seed": seed,
        "local_acc": local_acc, "local_auc": local_auc,
        "pooled_acc": pooled_acc, "pooled_auc": pooled_auc,
        "federated_acc": fed_acc, "federated_auc": fed_auc,
    }


def main():
    rows = []
    for seed in range(N_SEEDS):
        row = run_one_seed(seed)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv("results/results.csv", index=False)
    print(f"Seeds used: {len(table)} of {N_SEEDS}")

    methods = [("local", "Hospital alone"), ("pooled", "All data pooled"),
               ("federated", "Federated averaging")]
    for key, name in methods:
        acc, auc = table[f"{key}_acc"], table[f"{key}_auc"]
        print(f"{name:20s} accuracy {acc.mean():.3f} +/- {acc.std():.3f}   "
              f"AUC {auc.mean():.3f} +/- {auc.std():.3f}")

    means = [table[f"{key}_acc"].mean() for key, _ in methods]
    stds = [table[f"{key}_acc"].std() for key, _ in methods]
    plt.figure(figsize=(6, 4))
    plt.bar([name for _, name in methods], means, yerr=stds, capsize=6)
    plt.ylim(0.90, 1.0)
    plt.ylabel("Test accuracy (axis starts at 0.90)")
    plt.title(f"Average over {len(table)} random splits")
    plt.tight_layout()
    plt.savefig("results/accuracy.png", dpi=200)


if __name__ == "__main__":
    main()