"""
CPSC 481 Assignment #2 - Part 3: Unsupervised learning for clustering

3.1  Clusters the Iris dataset with K-means (K=3) WITHOUT using the labels, and
     computes the RMSE of each cluster with respect to its own centroid.
3.2  Cluster quality validation: brings in the true labels (for evaluation only)
     to measure how well the clusters line up with the real species, and
     compares the result with the supervised MLP from mlp_classifier.py.
"""
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from mlp_classifier import load_iris_data, train_best_mlp

FIG_DIR = "figures"
RANDOM_STATE = 42
K = 3


def save_fig(name):
    # Save the current figure into figures/ (always before plt.show()).
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")


def run_kmeans(X):
    # Fit K-means with K=3 on X (no labels involved).
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(X)
    return km, clusters


def cluster_rmse(X, clusters, centroids):
    # RMSE_k = sqrt( mean over points in cluster k of ||x_i - centroid_k||^2 ).
    per_cluster = {}
    for k in range(len(centroids)):
        sq_dist = np.sum((X[clusters == k] - centroids[k]) ** 2, axis=1)
        per_cluster[k] = np.sqrt(np.mean(sq_dist))
    # Overall RMSE over all points, each measured to its own centroid
    overall = np.sqrt(np.mean(np.sum((X - centroids[clusters]) ** 2, axis=1)))
    return per_cluster, overall


def validate_clusters(clusters, y):
    """Compare cluster IDs with the true labels (evaluation only).

    Returns the contingency table, the Hungarian cluster->class mapping,
    mapped accuracy %, purity %, ARI and NMI.
    """
    table = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(y, name="true class"))
    # Hungarian algorithm: one-to-one cluster->class mapping maximizing agreement
    rows, cols = linear_sum_assignment(-table.to_numpy())
    mapping = {table.index[r]: table.columns[c] for r, c in zip(rows, cols)}
    y_mapped = np.array([mapping[c] for c in clusters])
    accuracy = 100 * np.mean(y_mapped == y)
    # Purity: each cluster votes for its majority class
    purity = 100 * table.max(axis=1).sum() / len(y)
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    return table, mapping, y_mapped, accuracy, purity, ari, nmi


def print_clusters(X, clusters, km, feature_names):
    # Print size, centroid and RMSE for every cluster.
    per_cluster, overall = cluster_rmse(X, clusters, km.cluster_centers_)
    for k in range(K):
        centroid = ", ".join(f"{n}={v:.4f}" for n, v in zip(feature_names, km.cluster_centers_[k]))
        print(f"  Cluster {k}: size={np.sum(clusters == k):3d}  RMSE={per_cluster[k]:.4f}")
        print(f"             centroid: {centroid}")
    print(f"  Total points: {len(clusters)}")
    print(f"  Overall RMSE (all points to their own centroid): {overall:.4f}")
    print(f"  K-means inertia (sum of squared distances): {km.inertia_:.4f}")


def main():
    X, y, feature_names = load_iris_data(return_feature_names=True)
    print("=" * 74)
    print("PART 3: K-means clustering of the Iris dataset (K=3)")
    print("=" * 74)
    print(f"X shape: {X.shape}  (labels are NOT used for clustering)")

    # ---------------- 3.1 Clustering on the raw features (primary answer) -------
    print("\n--- 3.1 K-means on the RAW features (primary result) ---")
    km, clusters = run_kmeans(X)
    print_clusters(X, clusters, km, feature_names)

    # ---------------- 3.2 Cluster quality validation ----------------------------
    print("\n--- 3.2 Cluster quality validation (true labels used for evaluation only) ---")
    table, mapping, y_mapped, accuracy, purity, ari, nmi = validate_clusters(clusters, y)
    print("Contingency table (cluster ID vs. true class):")
    print(table.to_string())
    print("\nHungarian cluster -> class mapping:")
    for k, cls in sorted(mapping.items()):
        print(f"  Cluster {k} -> {cls}")
    print(f"\nMapped accuracy (Hungarian): {accuracy:.2f}%")
    print(f"Purity:                      {purity:.2f}%")
    print(f"Adjusted Rand Index (ARI):   {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

    # Per-class view: how many of each true class landed in its mapped cluster
    print("\nPer-class agreement with the mapped cluster:")
    clean, confused = [], []
    for cls in sorted(set(y)):
        mask = y == cls
        correct = np.sum(y_mapped[mask] == cls)
        pct = 100 * correct / mask.sum()
        print(f"  {cls:<16} {correct:2d}/{mask.sum()} correct ({pct:.1f}%)")
        (clean if correct == mask.sum() else confused).append(cls)
    print(f"Separated cleanly: {', '.join(clean) if clean else 'none'}")
    print(f"Confused:          {', '.join(confused) if confused else 'none'}")

    # Optional: the same analysis on standardized features
    print("\n--- (Optional) K-means on STANDARDIZED features, for comparison ---")
    X_std = StandardScaler().fit_transform(X)
    km_std, clusters_std = run_kmeans(X_std)
    per_std, overall_std = cluster_rmse(X_std, clusters_std, km_std.cluster_centers_)
    _, _, _, acc_std, pur_std, ari_std, nmi_std = validate_clusters(clusters_std, y)
    sizes_std = [int(np.sum(clusters_std == k)) for k in range(K)]
    print(f"  Cluster sizes: {sizes_std}")
    print("  Per-cluster RMSE (in standardized units): "
          + ", ".join(f"C{k}={per_std[k]:.4f}" for k in range(K))
          + f"; overall={overall_std:.4f}")
    print(f"  Mapped accuracy={acc_std:.2f}%  Purity={pur_std:.2f}%  "
          f"ARI={ari_std:.4f}  NMI={nmi_std:.4f}")

    # ---------------- Compare with the supervised MLP ---------------------------
    print("\n--- Comparison with the MLP classifier (mlp_classifier.py) ---")
    print("Training the best MLP from Part 2 (this takes a few seconds)...")
    mlp_model, X_test, y_test, mlp_test_acc = train_best_mlp()
    mlp_all_acc = 100 * np.mean(mlp_model.predict(X) == y)
    print(f"  {'Method':<44}{'Accuracy %':>12}")
    print(f"  {'K-means (raw), Hungarian-mapped, all 150 pts':<44}{accuracy:>12.2f}")
    print(f"  {'MLP (supervised), 30-sample test set':<44}{mlp_test_acc:>12.2f}")
    print(f"  {'MLP (supervised), all 150 pts (incl. train)':<44}{mlp_all_acc:>12.2f}")
    comparison = (
        f"The supervised MLP ({mlp_test_acc:.2f}% on unseen test data) outperforms K-means "
        f"({accuracy:.2f}%) because it is trained with the true labels and can learn curved "
        f"decision boundaries exactly where the classes meet, while K-means never sees the "
        f"labels and simply groups points around the nearest centroid. K-means also assumes "
        f"roughly spherical, similar-sized clusters, so it separates "
        f"{', '.join(clean) or 'no class'} cleanly but splits the overlapping "
        f"{' and '.join(confused) or 'remaining'} region along a straight boundary that does "
        f"not match the species boundary."
    )
    print()
    print(textwrap.fill(comparison, width=90, initial_indent="  ", subsequent_indent="  "))

    # ---------------- Plots: petal length vs. petal width -----------------------
    fx, fy = 2, 3  # petal length, petal width - the most discriminative pair
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(7, 5))
    for k in range(K):
        pts = X[clusters == k]
        plt.scatter(pts[:, fx], pts[:, fy], s=25, color=colors[k], alpha=0.75,
                    label=f"Cluster {k} -> {mapping[k]}")
    plt.scatter(km.cluster_centers_[:, fx], km.cluster_centers_[:, fy], marker="X",
                s=250, color="black", edgecolor="white", label="Centroids")
    plt.title("K-means clusters (K=3) on Iris")
    plt.xlabel(feature_names[fx])
    plt.ylabel(feature_names[fy])
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig("kmeans_clusters.png")
    plt.show()

    # Side by side: clusters vs. true classes. Each cluster is drawn in the color
    # of the class it maps to, so mismatched points stand out.
    class_colors = {cls: colors[i] for i, cls in enumerate(sorted(set(y)))}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for k in range(K):
        pts = X[clusters == k]
        axes[0].scatter(pts[:, fx], pts[:, fy], s=25, alpha=0.75,
                        color=class_colors[mapping[k]], label=f"Cluster {k} (~{mapping[k]})")
    axes[0].scatter(km.cluster_centers_[:, fx], km.cluster_centers_[:, fy], marker="X",
                    s=250, color="black", edgecolor="white", label="Centroids")
    axes[0].set_title(f"K-means clusters (mapped accuracy {accuracy:.2f}%)")
    for cls in sorted(set(y)):
        pts = X[y == cls]
        axes[1].scatter(pts[:, fx], pts[:, fy], s=25, alpha=0.75,
                        color=class_colors[cls], label=cls)
    wrong = y_mapped != y
    axes[1].scatter(X[wrong, fx], X[wrong, fy], s=90, facecolors="none",
                    edgecolors="red", linewidths=1.5, label=f"Mis-clustered ({wrong.sum()})")
    axes[1].set_title("True classes")
    for ax in axes:
        ax.set_xlabel(feature_names[fx])
        ax.set_ylabel(feature_names[fy])
        ax.grid(alpha=0.3)
        ax.legend()
    save_fig("kmeans_vs_true.png")
    plt.show()


if __name__ == "__main__":
    main()
