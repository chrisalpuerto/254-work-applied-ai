# Assignment #2 — Written Answers

Only three items in `questions.md` ask for a written explanation: **2(4)**, **2(5)**, and **3(2)**. The
others (1(1)–(6), 2(1)–(3), 3(1), and 4.1(a)–(c)) are programming tasks. For those, the report needs
screenshots of the program output and figures, not written answers.

All numbers below come from running `mlp_classifier.py` and `kmeans_clustering.py`. Setup: Iris dataset,
80/20 stratified split (120 train / 30 test), `random_state=42`.

---

## 2(4) — Tuning hyperparameters without adding layers or neurons

**Yes, performance improved a lot.**

| Model (1 hidden layer, 3 neurons) | Train acc. | Test acc. | Test acc. over 5 seeds |
|---|---|---|---|
| Baseline: relu, adam, lr=0.001, max_iter=500, no scaling | 70.83% | 76.67% | 54.67% ± 17.59% |
| Tuned: **logistic**, adam, lr=0.001, alpha=1e-4, **max_iter=2000** | 98.33% | 100.00% | 92.00% ± 16.00% |

I kept the architecture fixed at one hidden layer with 3 neurons. I ran `GridSearchCV` with 5-fold
cross-validation on the training set only. It tried 78 combinations of feature scaling (none or
StandardScaler), activation function (relu, tanh, logistic), solver (adam, lbfgs, sgd with momentum 0.9),
learning rate, L2 regularization (`alpha`), and `max_iter`. The best combination had a CV accuracy of
98.33%. It used the **logistic (sigmoid) activation** and allowed **2000 iterations**.

On the test set, accuracy rose from **76.67% to 100.00% (+23.33 points)**. Training accuracy rose from
70.83% to 98.33%.

Why the baseline was weak:
- It stopped at 500 iterations before training had converged.
- With only 3 ReLU units, a unit can get stuck outputting 0 for every input (a "dead" unit). That leaves
  the network too little capacity to separate all three species.
- It depended heavily on the random starting weights: averaged over 5 seeds it scored only 54.67% ± 17.59%.

Sigmoid units don't die this way, and the extra iterations let training finish. The 5-seed average rose to
92.00% (+37.33 points). The model is still somewhat sensitive to initialization (± 16.00%), which is
expected for a network with only 3 hidden neurons.

---

## 2(5) — Adding more layers and/or neurons

**On the 30-sample test set, accuracy could not go any higher, but the larger networks were much more reliable.**

All architectures below reuse the tuned settings from 2(4):

| Architecture | Train acc. | Test acc. |
|---|---|---|
| (3,) tuned, from 2(4) | 98.33% | 100.00% |
| (10,) | 97.50% | 100.00% |
| (16, 8) | 97.50% | 100.00% |
| (32, 16) | 98.33% | 100.00% |
| (64, 32, 16) | 98.33% | 100.00% |

The tuned 3-neuron model already classified all 30 test flowers correctly, so no bigger network could show
higher test accuracy on this split. Every larger architecture also reached 100.00% test accuracy, with
97.50–98.33% training accuracy. The final (10,) model has a perfectly diagonal confusion matrix (10/10/10),
and precision, recall, and F1 are 1.00 for every class.

The improvement shows up in **robustness**. I retrained each model with 5 different random seeds:

| Model | Test acc. over 5 seeds |
|---|---|
| (3,) tuned | 92.00% ± 16.00% |
| (10,) | **100.00% ± 0.00%** |

That is **+8.00 points on average, and the variation disappears**. The 3-neuron network only works when it
starts from a good initialization. With 10 neurons there is enough capacity that every seed learns to
separate versicolor from virginica. So the more complex architecture did improve performance: it gets
100% consistently instead of only on lucky initializations.

Adding layers beyond one hidden layer of 10 neurons brought no further gain. Iris has only 4 features and
3 classes, so a small network is enough.

---

## 3(2) — Cluster quality validation: K-means vs. the MLP classifier

K-means (K=3) was run on the 4 raw features **without using the labels**. The true labels were used
afterward only to check how well the clusters line up with the real species.

**Contingency table (cluster vs. true species):**

| Cluster | Iris-setosa | Iris-versicolor | Iris-virginica | Size | RMSE to centroid |
|---|---|---|---|---|---|
| 1 | **50** | 0 | 0 | 50 | 0.5521 |
| 0 | 0 | **48** | 14 | 62 | 0.8014 |
| 2 | 0 | 2 | **36** | 38 | 0.7927 |

**Cluster quality metrics:**

| Metric | Value |
|---|---|
| Accuracy (each cluster matched one-to-one to its best species with the Hungarian algorithm) | **89.33%** (134/150) |
| Purity | 89.33% |
| Adjusted Rand Index (ARI) | 0.7302 |
| Normalized Mutual Information (NMI) | 0.7582 |
| Overall RMSE (all points to their own centroid) | 0.7254 |

For ARI and NMI, 1 means a perfect match with the species and 0 means no better than random.

**How well do the clusters align with the actual classes?**

- **Iris-setosa is recovered perfectly.** Cluster 1 is exactly the 50 setosa flowers and nothing else.
  Setosa's petals are much smaller than the other two species' (centroid petal length 1.46 vs. 4.39 and
  5.74). That makes it a tight, isolated group, which also has the lowest RMSE (0.5521).
- **Versicolor and virginica are partly mixed.**
  - Versicolor: 48/50 (96%) land in their cluster.
  - Virginica: only 36/50 (72%). Fourteen virginica flowers fall into the versicolor cluster.
  - In feature space the two species overlap around petal length ≈ 5 and petal width ≈ 1.5–2.0. K-means
    assigns each point to its nearest centroid, so the boundary it draws between the two clusters is a
    straight line halfway between the centroids. That line doesn't follow the true species boundary.
  - As a result, cluster 0 grows to 62 points and cluster 2 shrinks to 38, and both have higher RMSE
    (≈0.80) than the setosa cluster. All 16 mis-clustered points are in this overlap region (circled in
    `figures/kmeans_vs_true.png`).

**Comparison with `mlp_classifier.py`:**

| Method | Accuracy |
|---|---|
| K-means (unsupervised), all 150 flowers | 89.33% |
| MLP (supervised), 30 unseen test flowers | 100.00% |
| MLP (supervised), all 150 flowers | 98.00% |

The MLP does about 9–11 points better.
- **The MLP is supervised.** It learns from the labels exactly where one species ends and the next
  begins, and its hidden layer can bend the decision boundary to fit the overlap between versicolor and
  virginica.
- **K-means has no labels.** It only finds groups of points that are close together. It also assumes the
  clusters are roughly round and similar in size, which is true for setosa but not for the two
  overlapping species.

Even so, an ARI of 0.73 and 89% agreement show that the natural structure of the data matches the species
fairly well. Without ever seeing a label, K-means finds one species perfectly and mostly separates the
other two.

Standardizing the features first made K-means worse: 83.33% accuracy and ARI 0.6201. Scaling gives the
sepal measurements as much influence as the petal measurements, but sepal measurements separate the species
less well.
