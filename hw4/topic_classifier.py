"""
CPSC 254 - Assignment 4, Part 1
Text Data Preprocessing, Vectorization, and Classification on AG News.

Run:
    python topic_classifier.py            # full run (all 4 configurations)
    python topic_classifier.py --quick    # small subset smoke test

Outputs (written to ./results/):
    part1_output.txt          - everything printed to the console
    part1_summary.csv/.md     - results table for the four configurations
    confusion_*.png           - confusion matrix for each configuration
    misclassified_examples.txt- misclassified test examples (Count, no stop words)
"""

import argparse
import html
import json
import os
import re
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
)
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
LABEL_IDS = [1, 2, 3, 4]
LABEL_NAMES = [LABELS[i] for i in LABEL_IDS]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

# MLP settings - identical for every run in Part 1 (2) and (3).
MLP_PARAMS = dict(
    hidden_layer_sizes=(128,),  # one hidden layer, 128 units
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    batch_size=256,
    max_iter=5,                 # 5 epochs (full passes over the 120k training docs)
    alpha=1e-4,                 # L2 penalty (sklearn default, listed for reproducibility)
    shuffle=True,
    early_stopping=False,
    random_state=SEED,
)

# Vectorizer settings - identical for every run except the stop-word flag.
VEC_PARAMS = dict(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")


class Tee:
    """Write to the console and a log file at the same time."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# (1a) Load data
# ---------------------------------------------------------------------------
def clean_text(s):
    """Light cleanup of AG News artifacts (escaped newlines, HTML entities)."""
    s = s.replace("\\", " ")                       # '\' marks escaped newlines
    s = re.sub(r"&?#(\d+);", lambda m: chr(int(m.group(1))), s)  # ' #39;' -> "'"
    s = html.unescape(s.replace("quot;", '"').replace("amp;", "&"))
    s = re.sub(r"<[^>]+>", " ", s)                 # stray HTML tags
    return re.sub(r"\s+", " ", s).strip()


def load_with_torchtext():
    """Official loader. torchtext is deprecated and does not install on
    Python >= 3.13, so this usually fails and we fall back to local files."""
    from torchtext.datasets import AG_NEWS  # noqa: F401

    train = [(int(lbl), txt) for lbl, txt in AG_NEWS(split="train")]
    test = [(int(lbl), txt) for lbl, txt in AG_NEWS(split="test")]
    # torchtext already joins title + description into one string
    to_df = lambda rows: pd.DataFrame(
        {"label": [r[0] for r in rows], "title": "", "description": [r[1] for r in rows]}
    )
    return to_df(train), to_df(test)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def load_data():
    try:
        train_df, test_df = load_with_torchtext()
        source = "torchtext.datasets.AG_NEWS"
    except Exception as e:  # ImportError, missing torchdata, network, ...
        print(f"torchtext AG_NEWS unavailable ({type(e).__name__}: {e}).")
        print(f"Falling back to the official AG News split in {DATA_DIR}/*.jsonl")
        train_df = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
        test_df = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
        source = "local JSONL (official AG News v3 train/test split)"

    # (1b) Combine title + description into a single text field
    for df in (train_df, test_df):
        df["text"] = (df["title"].fillna("") + ". " + df["description"].fillna("")).map(clean_text)
        df["text"] = df["text"].str.lstrip(". ")
        df["category"] = df["label"].map(LABELS)
    return train_df, test_df, source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sparsity_pct(X):
    total = X.shape[0] * X.shape[1]
    return (1 - X.nnz / total) * 100, X.nnz, total


def make_vectorizer(kind, stop_words):
    cls = CountVectorizer if kind == "Count" else TfidfVectorizer
    return cls(stop_words="english" if stop_words else None, **VEC_PARAMS)


def plot_confusion(cm, title, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABEL_NAMES)
    ax.set_yticks(range(4), LABEL_NAMES)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_confusion(cm):
    df = pd.DataFrame(cm, index=[f"true {n}" for n in LABEL_NAMES],
                      columns=[f"pred {n}" for n in LABEL_NAMES])
    print(df.to_string())


def train_and_evaluate(name, X_train, y_train, X_test, y_test):
    clf = MLPClassifier(**MLP_PARAMS)
    t0 = time.time()
    with warnings.catch_warnings():
        # max_iter is a deliberate fixed epoch budget, not a convergence failure
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=LABEL_IDS)

    print(f"\n--- {name} ---")
    print(f"Features: {X_train.shape[1]:,} | epochs run: {clf.n_iter_} | "
          f"final training loss: {clf.loss_:.4f} | train time: {train_time:.1f}s")
    print(f"Test accuracy:          {acc:.4f}")
    print(f"Test macro precision:   {prec:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print_confusion(cm)
    return dict(clf=clf, y_pred=y_pred, acc=acc, prec=prec, cm=cm,
                train_time=train_time, n_features=X_train.shape[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="use a 10k/2k subset for a fast smoke test")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, "part1_output.txt"))
    np.random.seed(SEED)

    # ------------------------------------------------------------------ (1)
    section("PART 1 (1a-b): Load AG News and combine title + description")
    train_df, test_df, source = load_data()
    if args.quick:
        train_df = train_df.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_df = test_df.sample(2_000, random_state=SEED).reset_index(drop=True)
        print("[--quick] using a random subset")
    print(f"Data source: {source}")
    print(f"Training examples: {len(train_df):,}")
    print(f"Test examples:     {len(test_df):,}")

    section("PART 1 (1c): One example article per category")
    for lid in LABEL_IDS:
        row = train_df[train_df["label"] == lid].iloc[0]
        print(f"\n[{lid}: {LABELS[lid]}]\n  {row['text']}")

    section("PART 1 (1c): Number of examples per category")
    counts = pd.DataFrame({
        "Train": train_df["category"].value_counts(),
        "Test": test_df["category"].value_counts(),
    }).reindex(LABEL_NAMES)
    counts.loc["Total"] = counts.sum()
    print(counts.to_string())

    section("PART 1 (1d-e): Document-term matrix (CountVectorizer) and TF-IDF matrix")
    print("Both vectorizers are fit on TRAINING text only, then used to transform test text.")
    print(f"Vectorizer settings: {VEC_PARAMS}, stop_words=None")

    count_vec = make_vectorizer("Count", stop_words=False)
    Xc_train = count_vec.fit_transform(train_df["text"])
    Xc_test = count_vec.transform(test_df["text"])

    tfidf_vec = make_vectorizer("Tfidf", stop_words=False)
    Xt_train = tfidf_vec.fit_transform(train_df["text"])
    Xt_test = tfidf_vec.transform(test_df["text"])

    sp, nnz, total = sparsity_pct(Xc_train)
    print(f"\nTraining DTM shape (documents x features): "
          f"{Xc_train.shape[0]:,} x {Xc_train.shape[1]:,}")
    print(f"Nonzero entries: {nnz:,}")
    print(f"Total entries:   {total:,}")
    print(f"Sparsity: (1 - {nnz:,} / {total:,}) x 100 = {sp:.4f}%")
    print(f"Average distinct terms per training document: {nnz / Xc_train.shape[0]:.1f}")
    print(f"Test DTM shape:  {Xc_test.shape[0]:,} x {Xc_test.shape[1]:,}")
    print(f"TF-IDF train shape: {Xt_train.shape[0]:,} x {Xt_train.shape[1]:,} "
          f"(sparsity {sparsity_pct(Xt_train)[0]:.4f}%)")
    dense_gb = total * 8 / 1e9
    sparse_mb = (Xc_train.data.nbytes + Xc_train.indices.nbytes + Xc_train.indptr.nbytes) / 1e6
    print(f"Memory if stored dense (float64): {dense_gb:.1f} GB vs. CSR sparse: {sparse_mb:.1f} MB")

    # Small illustration of a DTM cell vs a TF-IDF cell for the same document
    doc0 = Xc_train[0]
    vocab = count_vec.get_feature_names_out()
    print("\nDTM vs TF-IDF cells for training document 0:")
    print(f"  text: {train_df['text'].iloc[0]}")
    t_vocab_idx = {w: i for i, w in enumerate(tfidf_vec.get_feature_names_out())}
    for j in doc0.indices[np.argsort(-doc0.data)][:8]:
        w = vocab[j]
        print(f"  term '{w:<12}' count = {doc0[0, j]}   tf-idf = {Xt_train[0, t_vocab_idx[w]]:.4f}")

    # ------------------------------------------------------------------ (2)
    section("PART 1 (2): MLP classifier on CountVectorizer features")
    print("Model: sklearn.neural_network.MLPClassifier")
    print("  Input layer:      one unit per vocabulary term (bag-of-words counts)")
    print("  Hidden layers:    1 hidden layer x 128 units, ReLU activation")
    print("  Output layer:     4 units, softmax (multi-class)")
    print("  Loss:             categorical cross-entropy (log-loss) + L2 penalty alpha=1e-4")
    print("  Optimizer:        Adam, learning rate 0.001 (constant), betas (0.9, 0.999)")
    print("  Batch size:       256")
    print("  Epochs:           5 (max_iter=5, early_stopping=False)")
    print(f"  random_state:     {SEED}, shuffle=True")
    print(f"  Full parameter dict: {MLP_PARAMS}")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    results = {}
    key = ("CountVectorizer", "No")
    results[key] = train_and_evaluate("CountVectorizer, no stop-word removal",
                                      Xc_train, y_train, Xc_test, y_test)
    r = results[key]
    print("\nMacro-averaged precision = unweighted mean of the per-class precisions "
          "(each of the 4 classes counts equally).")
    print("\nPer-class report:")
    print(classification_report(y_test, r["y_pred"], labels=LABEL_IDS,
                                target_names=LABEL_NAMES, digits=4))
    plot_confusion(r["cm"], "MLP - CountVectorizer, no stop-word removal",
                   os.path.join(RESULTS_DIR, "confusion_count_nostop.png"))

    section("PART 1 (2): Misclassified test examples (CountVectorizer, no stop words)")
    wrong = np.where(r["y_pred"] != y_test)[0]
    print(f"Misclassified: {len(wrong):,} of {len(y_test):,} test documents")
    # Show a few per (true -> predicted) error type so the variety is visible
    picked, seen = [], Counter()
    for i in wrong:
        pair = (y_test[i], r["y_pred"][i])
        if seen[pair] < 1:
            picked.append(i)
            seen[pair] += 1
        if len(picked) == 8:
            break
    with open(os.path.join(RESULTS_DIR, "misclassified_examples.txt"), "w", encoding="utf-8") as f:
        for n, i in enumerate(picked, 1):
            block = (f"\n#{n}  test index {i}\n"
                     f"  True: {LABELS[y_test[i]]:<9} Predicted: {LABELS[r['y_pred'][i]]}\n"
                     f"  Text: {test_df['text'].iloc[i]}")
            print(block)
            f.write(block + "\n")

    # ------------------------------------------------------------------ (3)
    section("PART 1 (3): Compare feature representations and stop-word removal")
    print("Same split, same MLP settings, each vectorizer fit on training text only.")
    feats = {("CountVectorizer", "No"): (Xc_train, Xc_test),
             ("TfidfVectorizer", "No"): (Xt_train, Xt_test)}
    for kind, stop in [("Tfidf", False), ("Count", True), ("Tfidf", True)]:
        key = (f"{kind}Vectorizer", "Yes" if stop else "No")
        if key not in feats:
            vec = make_vectorizer(kind, stop)
            feats[key] = (vec.fit_transform(train_df["text"]), vec.transform(test_df["text"]))
        if key not in results:
            X_tr, X_te = feats[key]
            results[key] = train_and_evaluate(
                f"{key[0]}, stop-word removal = {key[1]}", X_tr, y_train, X_te, y_test)
        slug = f"{kind.lower()}_{'stop' if stop else 'nostop'}"
        plot_confusion(results[key]["cm"], f"MLP - {key[0]}, stop words removed: {key[1]}",
                       os.path.join(RESULTS_DIR, f"confusion_{slug}.png"))

    order = [("CountVectorizer", "No"), ("TfidfVectorizer", "No"),
             ("CountVectorizer", "Yes"), ("TfidfVectorizer", "Yes")]
    rows = []
    for key in order:
        r = results[key]
        rows.append({
            "Vectorizer": key[0],
            "Stop-word removal": key[1],
            "# features": r["n_features"],
            "Test accuracy": round(r["acc"], 4),
            "Macro precision": round(r["prec"], 4),
            "Train time (s)": round(r["train_time"], 1),
        })
    summary = pd.DataFrame(rows)
    best_key = order[int(np.argmax([results[k]["acc"] for k in order]))]

    section("PART 1 (3): Summary table")
    print(summary.to_string(index=False))
    print(f"\nBest configuration by test accuracy: {best_key[0]}, stop-word removal = {best_key[1]}")
    print("Confusion matrix of the best configuration (rows = true, cols = predicted):")
    print_confusion(results[best_key]["cm"])

    summary.to_csv(os.path.join(RESULTS_DIR, "part1_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "part1_summary.md"), "w", encoding="utf-8") as f:
        try:
            f.write(summary.to_markdown(index=False))  # needs 'tabulate'
        except ImportError:
            f.write(summary.to_string(index=False))
        f.write(f"\n\nBest: {best_key[0]}, stop-word removal = {best_key[1]}\n")

    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
