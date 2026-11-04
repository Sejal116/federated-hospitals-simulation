import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

from split import load_and_split, split_into_hospitals
from models import init_weights, train
from federated import federated_averaging
from privacy import make_private_trainer, train_private
from attack import membership_attack_auc
from run_experiments import hospitals_are_usable, N_SEEDS

CLIP = 1.0
NOISE_LEVELS = [50, 200]


def main():
    scores = {}

    def add(name, value):
        scores.setdefault(name, []).append(value)

    poly = PolynomialFeatures(2, include_bias=False)

    for seed in range(N_SEEDS):
        X_train, X_test, y_train, y_test = load_and_split(seed)
        hospitals = split_into_hospitals(X_train, y_train, seed=seed)
        if not hospitals_are_usable(hospitals):
            continue
        n_features = X_train.shape[1]

        # Experiment A: the simple model we used everywhere else.
        w = federated_averaging(hospitals, n_features)
        add("A federated, no noise",
            membership_attack_auc(w, X_train, y_train, X_test, y_test))
        for z in NOISE_LEVELS:
            trainer = make_private_trainer(CLIP, z, seed)
            w = federated_averaging(hospitals, n_features, trainer=trainer)
            add(f"A federated, noise {z}",
                membership_attack_auc(w, X_train, y_train, X_test, y_test))

        # Experiment B: a model made to overfit on purpose, one hospital alone.
        X_test_poly = poly.fit_transform(X_test)
        for X_h, y_h in hospitals:
            X_poly = poly.transform(X_h)
            n_poly = X_poly.shape[1]

            w = train(init_weights(n_poly), X_poly, y_h, epochs=2000, lr=0.1)
            add("B overfit hospital, no noise",
                membership_attack_auc(w, X_poly, y_h, X_test_poly, y_test))
            for z in NOISE_LEVELS:
                rng = np.random.default_rng(seed)
                w = train_private(init_weights(n_poly), X_poly, y_h,
                                  2000, 0.1, CLIP, z, rng)
                add(f"B overfit hospital, noise {z}",
                    membership_attack_auc(w, X_poly, y_h, X_test_poly, y_test))

    names = list(scores)
    means = [np.mean(scores[n]) for n in names]
    stds = [np.std(scores[n], ddof=1) for n in names]
    pd.DataFrame({"setting": names, "attack_auc_mean": means,
                  "attack_auc_std": stds}).to_csv(
        "results/attack_results.csv", index=False)

    for name, mean, std in zip(names, means, stds):
        print(f"{name:32s} attack AUC {mean:.3f} +/- {std:.3f}")

    plt.figure(figsize=(7, 4.5))
    plt.barh(names[::-1], means[::-1], xerr=stds[::-1], capsize=4)
    plt.axvline(0.5, linestyle="--", color="gray")
    plt.xlim(0.4, 0.7)
    plt.xlabel("Attack AUC (0.5 = coin flip, axis starts at 0.4)")
    plt.title("Can an attacker tell who was in the training data?")
    plt.tight_layout()
    plt.savefig("results/attack.png", dpi=200)


if __name__ == "__main__":
    main()