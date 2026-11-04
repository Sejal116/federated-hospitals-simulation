import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from split import load_and_split, split_into_hospitals
from models import init_weights, train, evaluate
from federated import federated_averaging
from run_experiments import hospitals_are_usable, N_SEEDS

# Hospital size bands (number of training patients) used to group the results.
BANDS = [(0, 100), (100, 150), (150, 200), (200, 10_000)]


def collect_hospital_results():
    """For every hospital in every usable split, record its size and the
    test accuracy of its own model and of the shared federated model."""
    rows = []
    for seed in range(N_SEEDS):
        X_train, X_test, y_train, y_test = load_and_split(seed)
        hospitals = split_into_hospitals(X_train, y_train, seed=seed)
        if not hospitals_are_usable(hospitals):
            continue
        n_features = X_train.shape[1]

        shared = federated_averaging(hospitals, n_features)
        shared_acc, _ = evaluate(shared, X_test, y_test)

        for X_h, y_h in hospitals:
            own = train(init_weights(n_features), X_h, y_h)
            own_acc, _ = evaluate(own, X_test, y_test)
            rows.append({"seed": seed, "patients": len(y_h),
                         "alone_acc": own_acc, "shared_acc": shared_acc})
    return pd.DataFrame(rows)


def summarise_by_size(table):
    """Average error of a hospital alone vs in the shared model, by size band."""
    out = []
    for low, high in BANDS:
        part = table[(table.patients >= low) & (table.patients < high)]
        if part.empty:
            continue
        out.append({
            "band_low": low,
            "band_high": high,
            "hospitals": len(part),
            "error_alone": 1 - part.alone_acc.mean(),
            "error_shared": 1 - part.shared_acc.mean(),
            "share_better_off": (part.shared_acc > part.alone_acc).mean(),
            "share_worse_off": (part.shared_acc < part.alone_acc).mean(),
        })
    return pd.DataFrame(out)


def main():
    table = collect_hospital_results()
    summary = summarise_by_size(table)
    summary.to_csv("results/size_results.csv", index=False)

    print(f"{len(table)} hospital results from {table.seed.nunique()} splits")
    for _, r in summary.iterrows():
        label = (f"{int(r.band_low)}-{int(r.band_high)}"
                 if r.band_high < 10_000 else f"{int(r.band_low)}+")
        print(f"{label:>8s} patients: wrong {r.error_alone:.1%} alone, "
              f"{r.error_shared:.1%} shared "
              f"({int(r.hospitals)} hospitals)")

    labels = [f"{int(r.band_low)}-{int(r.band_high)}" if r.band_high < 10_000
              else f"{int(r.band_low)}+" for _, r in summary.iterrows()]
    x = np.arange(len(summary))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, summary.error_alone * 100, 0.4, label="Hospital alone")
    plt.bar(x + 0.2, summary.error_shared * 100, 0.4, label="Shared model")
    plt.xticks(x, labels)
    plt.xlabel("Patients at the hospital")
    plt.ylabel("Wrong diagnoses (%)")
    plt.title("Smaller hospitals gain the most from sharing")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/size.png", dpi=200)


if __name__ == "__main__":
    main()
