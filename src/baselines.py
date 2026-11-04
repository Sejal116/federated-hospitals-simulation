from split import load_and_split, split_into_hospitals
from models import init_weights, train, evaluate
from federated import federated_averaging


def run_baselines(seed=42):
    X_train, X_test, y_train, y_test = load_and_split(seed)
    hospitals = split_into_hospitals(X_train, y_train, seed=seed)
    n_features = X_train.shape[1]

    results = {}
    for number, (X_h, y_h) in enumerate(hospitals, start=1):
        w = train(init_weights(n_features), X_h, y_h)
        results[f"Hospital {number} alone"] = evaluate(w, X_test, y_test)

    w = train(init_weights(n_features), X_train, y_train)
    results["All data pooled"] = evaluate(w, X_test, y_test)
    w = federated_averaging(hospitals, n_features)
    results["Federated averaging"] = evaluate(w, X_test, y_test)
    return results


if __name__ == "__main__":
    for name, (accuracy, auc) in run_baselines().items():
        print(f"{name:20s} accuracy {accuracy:.3f}   AUC {auc:.3f}")