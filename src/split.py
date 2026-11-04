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


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Share of benign in train:", y_train.mean())
    print("Share of benign in test:", y_test.mean())