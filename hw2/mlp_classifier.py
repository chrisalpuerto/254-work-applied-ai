"""
CPSC 481 Assignment #2 - Part 2: Supervised learning for classification

Classifies the Iris flower dataset with scikit-learn's MLPClassifier:
  2.3  Baseline: one hidden layer with 3 neurons, un-tuned hyperparameters.
  2.4  Tuning: keep the (3,) architecture, search over scaling / activation /
       solver / learning rate / regularization with GridSearchCV.
  2.5  Bigger network: add neurons / layers using the best settings from 2.4.
Because a 3-neuron network is sensitive to its random initialization and the
test set has only 30 samples, every configuration is also re-trained with
5 different seeds and reported as mean +/- std test accuracy.

Also exposes:
  load_iris_data()  - shared Iris loader (used by kmeans_clustering.py)
  train_best_mlp()  - returns the best model for Part 3, without printing
"""
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
SEEDS = [0, 1, 2, 3, 4]  # model seeds for the robustness check (split stays fixed)
LARGER_ARCHITECTURES = [(10,), (16, 8), (32, 16), (64, 32, 16)]


# Data loading
@lru_cache(maxsize=1)
def _fetch_iris():
    # Download Iris from the UCI ML repository once per process
    return fetch_ucirepo(id=53)


def load_iris_data(return_feature_names=False):
    # Load Iris from the UCI repository and return (X, y) as numpy arrays
    """
    X: (150, 4) float features, y: (150,) string class labels.
    If return_feature_names=True, also returns the list of feature names.
    """
    iris = _fetch_iris()
    X = iris.data.features            # DataFrame, 4 numeric columns
    y = iris.data.targets             # DataFrame, one column ("class")
    y = y.iloc[:, 0].to_numpy()       # flatten to a 1-D array of string labels
    if return_feature_names:
        return X.to_numpy(dtype=float), y, list(X.columns)
    return X.to_numpy(dtype=float), y


def split_data(X, y):
    # 80% training / 20% testing, stratified so each class keeps its proportion.
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


# Model helpers

def make_pipeline():
    return Pipeline([
        ("scaler", "passthrough"),
        ("mlp", MLPClassifier(hidden_layer_sizes=(3,), random_state=RANDOM_STATE)),
    ])


# 2.3 Baseline: one hidden layer with 3 neurons, defaults + max_iter=500, no scaling
BASELINE_PARAMS = {
    "scaler": "passthrough",
    "mlp__hidden_layer_sizes": (3,),
    "mlp__max_iter": 500,
}

# 2.4 Search space - the architecture stays fixed at (3,) for every candidate.
# lbfgs ignores learning_rate_init, so it gets its own (smaller) sub-grid.
TUNING_GRID = [
    {   # Adam
        "scaler": ["passthrough", StandardScaler()],
        "mlp__solver": ["adam"],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__learning_rate_init": [0.001, 0.01, 0.1],
        "mlp__alpha": [1e-4, 1e-2],
        "mlp__max_iter": [2000],
    },
    {   # L-BFGS (quasi-Newton; works well on small datasets)
        "scaler": ["passthrough", StandardScaler()],
        "mlp__solver": ["lbfgs"],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": [1e-4, 1e-2, 1.0],
        "mlp__max_iter": [2000],
    },
    {   # SGD with momentum
        "scaler": ["passthrough", StandardScaler()],
        "mlp__solver": ["sgd"],
        "mlp__momentum": [0.9],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__learning_rate_init": [0.01, 0.1],
        "mlp__alpha": [1e-4, 1e-2],
        "mlp__max_iter": [2000],
    },
]


def build_model(params, seed=RANDOM_STATE):
    # Create a fresh, unfitted pipeline with the given parameters and seed
    model = clone(make_pipeline())
    model.set_params(**{k: clone(v) if hasattr(v, "fit") else v for k, v in params.items()})
    model.set_params(mlp__random_state=seed)
    return model


def fit_and_score(model, X_train, y_train, X_test, y_test):
    # Fit the model and return (train_acc %, test_acc %, converged?).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    train_acc = 100 * accuracy_score(y_train, model.predict(X_train))
    test_acc = 100 * accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc, converged


def seed_robustness(params, X_train, y_train, X_test, y_test):
    # Mean and std of test accuracy (%) over SEEDS, with the split held fixed.
    accs = [fit_and_score(build_model(params, seed), X_train, y_train, X_test, y_test)[1]
            for seed in SEEDS]
    return float(np.mean(accs)), float(np.std(accs))


def describe_params(params):
    # Readable one-line summary of a parameter dict.
    parts = []
    for key, val in params.items():
        key = key.replace("mlp__", "")
        if key == "scaler":
            val = "StandardScaler" if val != "passthrough" else "none"
        parts.append(f"{key}={val}")
    return ", ".join(parts)


# Experiments (no printing here, so train_best_mlp() can reuse them quietly)
def run_experiments():
    # Run baseline, tuning and larger-network experiments; return all results.
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    data = (X_train, y_train, X_test, y_test)
    results = {"data": data, "configs": {}}

    def record(name, params, model, scores):
        mean, std = seed_robustness(params, *data)
        results["configs"][name] = {
            "params": params, "model": model,
            "train_acc": scores[0], "test_acc": scores[1], "converged": scores[2],
            "seed_mean": mean, "seed_std": std,
        }

    # 2.3 Baseline
    baseline = build_model(BASELINE_PARAMS)
    record("Baseline (3,)", BASELINE_PARAMS, baseline, fit_and_score(baseline, *data))

    # 2.4 Tuning with 5-fold cross-validation on the TRAINING set only
    search = GridSearchCV(make_pipeline(), TUNING_GRID, cv=5, scoring="accuracy", n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X_train, y_train)
    tuned_params = dict(search.best_params_)
    tuned_params["mlp__hidden_layer_sizes"] = (3,)
    results["cv_best_score"] = 100 * search.best_score_
    results["n_candidates"] = len(search.cv_results_["params"])
    tuned = build_model(tuned_params)
    record("Tuned (3,)", tuned_params, tuned, fit_and_score(tuned, *data))

    # 2.5 Larger architectures, re-using the best non-architecture settings
    tuned_test = results["configs"]["Tuned (3,)"]["test_acc"]
    arch_rows = []
    for arch in LARGER_ARCHITECTURES:
        params = {**tuned_params, "mlp__hidden_layer_sizes": arch}
        model = build_model(params)
        tr, te, conv = fit_and_score(model, *data)
        arch_rows.append({"arch": arch, "params": params, "model": model,
                          "scores": (tr, te, conv)})
    results["arch_rows"] = arch_rows

    # Choose the first architecture that beats the tuned test accuracy;
    # if none does, take the best one found (earliest/simplest wins ties).
    improving = [r for r in arch_rows if r["scores"][1] > tuned_test]
    chosen = improving[0] if improving else max(arch_rows, key=lambda r: r["scores"][1])
    results["larger_improved"] = bool(improving)
    record(f"Larger {chosen['arch']}", chosen["params"], chosen["model"], chosen["scores"])

    # Best overall: highest test accuracy, ties broken by the 5-seed mean
    best_name = max(results["configs"],
                    key=lambda n: (results["configs"][n]["test_acc"],
                                   results["configs"][n]["seed_mean"]))
    results["best_name"] = best_name
    return results


def train_best_mlp():
    # Return (model, X_test, y_test, test_accuracy_percent) for the best MLP found.
    results = run_experiments()
    best = results["configs"][results["best_name"]]
    _, _, X_test, y_test = results["data"]
    return best["model"], X_test, y_test, best["test_acc"]


# Reporting
def improvement_sentence(label, before_name, after_name, configs):
    # Auto-generated sentence: did accuracy improve, and by how much?
    b, a = configs[before_name], configs[after_name]
    diff = a["test_acc"] - b["test_acc"]
    seed_diff = a["seed_mean"] - b["seed_mean"]
    if diff > 0:
        verdict = f"YES - test accuracy improved by {diff:+.2f} points"
    elif diff == 0:
        verdict = "NO CHANGE on the fixed split - test accuracy stayed the same"
    else:
        verdict = f"NO - test accuracy dropped by {diff:+.2f} points"
    return (f"{label}: {verdict} ({b['test_acc']:.2f}% -> {a['test_acc']:.2f}%); "
            f"averaged over 5 seeds the change is {seed_diff:+.2f} points "
            f"({b['seed_mean']:.2f}% -> {a['seed_mean']:.2f}%).")


def main():
    X, y, feature_names = load_iris_data(return_feature_names=True)
    print("=" * 74)
    print("PART 2: MLP classification of the Iris dataset")
    print("=" * 74)
    print(f"Loaded Iris from the UCI ML repository (id=53): X shape {X.shape}, y shape {y.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes:  {sorted(set(y))}")

    results = run_experiments()
    X_train, y_train, X_test, y_test = results["data"]
    configs = results["configs"]
    names = list(configs)
    base_name, tuned_name, large_name = names
    print(f"Split: {len(y_train)} training / {len(y_test)} testing samples (stratified)")

    def show(name):
        c = configs[name]
        print(f"  Settings: {describe_params(c['params'])}")
        print(f"  Train accuracy: {c['train_acc']:.2f}%")
        print(f"  Test  accuracy: {c['test_acc']:.2f}%")
        print(f"  Test accuracy over seeds {SEEDS}: "
              f"{c['seed_mean']:.2f}% +/- {c['seed_std']:.2f}%")
        if not c["converged"]:
            print("  Note: ConvergenceWarning - optimizer hit max_iter before converging.")

    print("\n--- 2.3 Baseline: 1 hidden layer, 3 neurons, un-tuned ---")
    show(base_name)

    print("\n--- 2.4 Hyperparameter tuning (architecture fixed at (3,)) ---")
    print(f"  GridSearchCV: {results['n_candidates']} candidates x 5 folds on the training set")
    print(f"  Best 5-fold CV accuracy: {results['cv_best_score']:.2f}%")
    show(tuned_name)

    print("\n--- 2.5 Larger architectures (using the tuned settings) ---")
    print(f"  {'Architecture':<16}{'Train %':>10}{'Test %':>10}")
    for row in results["arch_rows"]:
        tr, te, _ = row["scores"]
        print(f"  {str(row['arch']):<16}{tr:>10.2f}{te:>10.2f}")
    if results["larger_improved"]:
        print(f"  -> First architecture beating the tuned (3,) test accuracy: {large_name}")
    else:
        print(f"  -> None beat the tuned (3,) test accuracy; best found: {large_name}")
    show(large_name)

    print("\n--- Summary ---")
    header = f"  {'Configuration':<22}{'Train %':>10}{'Test %':>10}{'5-seed test % (mean +/- std)':>32}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in names:
        c = configs[name]
        seed_str = f"{c['seed_mean']:.2f} +/- {c['seed_std']:.2f}"
        print(f"  {name:<22}{c['train_acc']:>10.2f}{c['test_acc']:>10.2f}{seed_str:>32}")
    print()
    print("  " + improvement_sentence("2.4 (tuning vs. baseline)", base_name, tuned_name, configs))
    print("  " + improvement_sentence("2.5 (larger vs. tuned)", tuned_name, large_name, configs))

    best_name = results["best_name"]
    best = configs[best_name]["model"]
    y_pred = best.predict(X_test)
    labels = sorted(set(y))
    print(f"\n--- Final (best) model: {best_name} ---")
    print("Confusion matrix (rows = true class, columns = predicted class):")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels)
    print(cm.to_string())
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, digits=4))


if __name__ == "__main__":
    main()
