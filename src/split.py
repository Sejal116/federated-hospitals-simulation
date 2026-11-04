import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_split(seed=42):
    """Load the dataset, split it into train and test, and scale it."""
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def split_into_hospitals(X, y, n_hospitals=3, alpha=1.0, seed=42):
    """Divide training data into hospitals with different patient mixes."""
    rng = np.random.default_rng(seed)
    hospital_indexes = [[] for _ in range(n_hospitals)]

    for label in (0, 1):
        indexes = np.where(y == label)[0]
        rng.shuffle(indexes)

        shares = rng.dirichlet([alpha] * n_hospitals)
        cut_points = (np.cumsum(shares) * len(indexes)).astype(int)[:-1]
        pieces = np.split(indexes, cut_points)

        for hospital, piece in enumerate(pieces):
            hospital_indexes[hospital].extend(piece.tolist())

    hospitals = []
    for idx in hospital_indexes:
        idx = np.array(idx, dtype=int)
        hospitals.append((X[idx], y[idx]))
    return hospitals


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    hospitals = split_into_hospitals(X_train, y_train)

    total = 0
    for number, (X_h, y_h) in enumerate(hospitals, start=1):
        malignant_share = 1 - y_h.mean()
        total += len(y_h)
        print(f"Hospital {number}: {len(y_h)} patients, "
              f"{malignant_share:.0%} malignant")
    print("Total:", total, "(training set size:", len(y_train), ")")