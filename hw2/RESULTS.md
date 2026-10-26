# CPSC 481 Assignment #2 — Results (Parts 1 – 4.1)

All numbers below were copied from actual runs (`MPLBACKEND=Agg python <file>.py`, Python 3.13.5,
scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, `random_state=42`). Re-running gives identical output.

## Figures (`figures/`)

| File | Content |
|---|---|
| `linreg_scatter.png` | Part 1a — scatter plot of the small training set |
| `linreg_fit.png` | Part 1a — linear fit (red) + degree-2 extra fit (green, dashed) |
| `gas_pred_vs_actual.png` | Part 1b — predicted vs. actual `Idx` on the test set (5000 sampled points) |
| `kmeans_clusters.png` | Part 3 — K-means clusters on petal length vs. petal width, centroids as X |
| `kmeans_vs_true.png` | Part 3 — clusters (left) vs. true classes (right), mis-clustered points circled |
| `gridworld_policy_path.png` | Part 4.1 — Viewer after "Run to convergence" + "Show Path" (γ=0.95, noise=0), rendered headlessly |

---

## Part 1 — `linear_reg.py`

### 1a. Small dataset (scikit-learn `LinearRegression`)

- Coefficient vector `[w0, w1] = [11.4362, 0.9936]`
- Equation: **y = 11.4362 + 0.9936x**
- Training RMSE (own numpy formula) = **4.7477** (sklearn cross-check: 4.7477)

**Extra — degree-2 polynomial** (`PolynomialFeatures(2)` + `LinearRegression`):

- `[w0, w1, w2] = [4.7878, 1.2333, 1.691]`
- Equation: **y = 4.7878 + 1.2333x + 1.6910x^2**
- Training RMSE = **0.6348** (vs. 4.7477 for the straight line)

Discussion: the data is U-shaped, so a straight line cannot follow it — the linear model's RMSE is
4.75, about 7.5× larger than the quadratic's 0.63. The quadratic form `y = w0 + w1x + w2x²` is still
"linear regression" (linear in the weights), just with the extra feature x².

### 1b. GasProperties.csv (normal equation)

- Shape: (420000, 5); columns `['T', 'P', 'TC', 'SV', 'Idx']`; no missing values.
- **Target: `Idx`** (last column). It is a continuous value (range 53.03 – 54.44, not monotonic), not a
  row ID, so no column was dropped. Features: `T, P, TC, SV`.
- Split: 336000 train / 84000 test (`test_size=0.2, random_state=42`).
- Solved with `w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train` (matrix was invertible; pinv fallback not needed).

| Weight | Feature | Normal equation | sklearn | abs diff |
|---|---|---|---|---|
| w0 | bias | 67.9449107045 | 67.9449106551 | 4.94e-08 |
| w1 | T | 0.0372071415 | 0.0372071414 | 1.68e-10 |
| w2 | P | -0.0042198922 | -0.0042198921 | 1.08e-11 |
| w3 | TC | -5.9909560204 | -5.9909556956 | 3.25e-07 |
| w4 | SV | -0.0590808856 | -0.0590808854 | 2.11e-10 |

- Equation: **Idx = 67.9449 + 0.0372·T − 0.0042·P − 5.9910·TC − 0.0591·SV**
- **Training RMSE = 0.136523**, **Testing RMSE = 0.135499** (sklearn: identical, 0.136523 / 0.135499)
- Max |w_normal − w_sklearn| = 3.25e-07 → the normal equation matches sklearn to ~1e-6.
- Train and test RMSE are almost equal, so the linear model is not overfitting (with 4 features and
  336k samples it can't); the remaining error is what a linear function can't capture.

---

## Part 2 — `mlp_classifier.py`

Iris loaded via `fetch_ucirepo(id=53)`; 150 × 4, classes Iris-setosa / versicolor / virginica.
Split: 120 train / 30 test, stratified, `random_state=42`. The 5-seed column re-trains the same
configuration with model seeds 0–4 on the same split.

| Configuration | Settings | Train % | Test % | 5-seed test % (mean ± std) |
|---|---|---|---|---|
| Baseline (3,) | defaults (relu, adam, lr=0.001), no scaling, max_iter=500 | 70.83 | 76.67 | 54.67 ± 17.59 |
| Tuned (3,) | logistic, adam, lr=0.001, alpha=1e-4, max_iter=2000, no scaling | 98.33 | 100.00 | 92.00 ± 16.00 |
| Larger (10,) | same as tuned, hidden_layer_sizes=(10,) | 97.50 | 100.00 | 100.00 ± 0.00 |

Baseline and tuned models both raised a `ConvergenceWarning` (hit max_iter before the loss tolerance was met).

**2.4 tuning:** `GridSearchCV` (78 candidates × 5 folds, training set only) over scaler (none / StandardScaler),
activation (relu / tanh / logistic), solver (adam / lbfgs / sgd+momentum 0.9), `learning_rate_init`,
`alpha`, `max_iter`. Best 5-fold CV accuracy: 98.33%.

**2.5 larger architectures** (with the tuned settings):

| Architecture | Train % | Test % |
|---|---|---|
| (10,) | 97.50 | 100.00 |
| (16, 8) | 97.50 | 100.00 |
| (32, 16) | 98.33 | 100.00 |
| (64, 32, 16) | 98.33 | 100.00 |

Final (best) model: Larger (10,). Confusion matrix on the test set is perfectly diagonal (10/10/10);
precision = recall = F1 = 1.0000 for every class.

### Draft answers

**2.4 — Were you able to improve performance without changing the architecture?** Yes. The un-tuned
3-neuron network (relu, 500 iterations, raw features) reached only 70.83% train / 76.67% test accuracy and
was very unstable across random initializations (54.67% ± 17.59% over 5 seeds) — with only 3 ReLU units,
some initializations leave units "dead" and the network cannot separate all three classes, and 500 Adam
iterations were not enough to converge. Switching to the logistic (sigmoid) activation and allowing 2000
iterations (selected by 5-fold cross-validation on the training set) raised accuracy to 98.33% train /
100.00% test, an improvement of +23.33 points on the test set and +37.33 points averaged over 5 seeds
(54.67% → 92.00%). It is still somewhat seed-sensitive (std 16.00%), because 3 neurons is a very small capacity.

**2.5 — Did a bigger network improve performance?** On the fixed 30-sample test split, no further
improvement was possible: the tuned 3-neuron model already scored 100.00%, and every larger architecture
tried — (10,), (16, 8), (32, 16), (64, 32, 16) — also scored 100.00% test (97.50–98.33% train). The benefit
shows up in robustness instead: the (10,) network scored 100.00% ± 0.00% over 5 seeds versus 92.00% ± 16.00%
for the 3-neuron network (+8.00 points on average), because extra neurons make it much less likely that a
bad initialization leaves the network unable to separate versicolor from virginica.

---

## Part 3 — `kmeans_clustering.py`

`KMeans(n_clusters=3, n_init=10, random_state=42)` on the 4 raw features (labels not used).

| Cluster | Size | RMSE | Centroid (sepal length, sepal width, petal length, petal width) |
|---|---|---|---|
| 0 | 62 | 0.8014 | 5.9016, 2.7484, 4.3935, 1.4339 |
| 1 | 50 | 0.5521 | 5.0060, 3.4180, 1.4640, 0.2440 |
| 2 | 38 | 0.7927 | 6.8500, 3.0737, 5.7421, 2.0711 |

Sizes sum to 150. Overall RMSE = 0.7254; inertia = 78.9408.

### 3.2 Cluster quality validation

Contingency table (cluster vs. true class):

| cluster | Iris-setosa | Iris-versicolor | Iris-virginica |
|---|---|---|---|
| 0 | 0 | 48 | 14 |
| 1 | 50 | 0 | 0 |
| 2 | 0 | 2 | 36 |

- Hungarian mapping: cluster 0 → versicolor, 1 → setosa, 2 → virginica
- **Mapped accuracy = 89.33%**, **Purity = 89.33%**, **ARI = 0.7302**, **NMI = 0.7582**
- Per class: setosa 50/50 (100.0%), versicolor 48/50 (96.0%), virginica 36/50 (72.0%)
- Separated cleanly: Iris-setosa. Confused: Iris-versicolor and Iris-virginica (16 points mis-clustered).

Optional — K-means on standardized features: sizes [53, 50, 47], mapped accuracy 83.33%, purity 83.33%,
ARI 0.6201, NMI 0.6595 (worse than raw features, because standardizing inflates the weight of the sepal
measurements, which separate the species less well than the petal measurements).

Comparison with the MLP:

| Method | Accuracy % |
|---|---|
| K-means (raw), Hungarian-mapped, all 150 points | 89.33 |
| MLP (supervised, best = (10,)), 30-sample test set | 100.00 |
| MLP (supervised), all 150 points incl. training | 98.00 |

### Draft answer (3.2)

K-means recovers the species structure fairly well without ever seeing a label: 89.33% of the flowers fall
in the cluster that corresponds to their species (ARI 0.7302, NMI 0.7582). Cluster 1 is exactly the 50
setosa flowers — setosa's petals are much smaller than the other two species, so it forms a compact,
well-separated group (it also has the smallest RMSE, 0.5521). The other two clusters are less pure:
cluster 0 contains 48 versicolor but also 14 virginica, and cluster 2 contains 36 virginica and 2
versicolor. Versicolor and virginica overlap in feature space (especially around petal length ≈ 5,
petal width ≈ 1.5–2.0), and K-means just assigns each point to the nearest centroid, i.e. it draws a
straight boundary between two roughly spherical clusters; that boundary sits too far toward virginica,
so the larger virginica flowers that are near the middle are pulled into the versicolor cluster (cluster 0
ends up with 62 points vs. 38 in cluster 2, and higher RMSE). The supervised MLP does better (100.00% on
the unseen test set, 98.00% on all 150 points) because it is trained with the true labels and can place
its decision boundary exactly where the species meet, instead of where the data happens to split into
compact blobs.

---

## Part 4.1 — `gridworld_policy_iteration.py`

Only the three TODO functions were filled in (`policy_evaluation`, `policy_improvement` with a new optional
`pi_old=None` parameter, `policy_iteration`). Terminals `(0,4)`, `(4,4)`, `(1,3)` and obstacle `(2,2)` are
skipped (value 0, all-zero policy row). Start = `(4, 0)`.

### γ = 0.95, noise = 0.0 (Viewer defaults) — converged in 3 outer iterations

```
V (rounded):
[[10.01 11.59 13.25 15.    0.  ]
 [11.59 13.25 15.    0.   15.  ]
 [10.01 11.59  0.   15.   13.25]
 [ 8.51 10.01 11.59 13.25 11.59]
 [ 7.08  8.51 10.01 11.59  0.  ]]

Greedy policy:
→ → → ↓ T
→ → → T ←
↑ ↑ # ↑ ↑
↑ ↑ → ↑ ↑
↑ ↑ ↑ ↑ T
```

Path: `(4,0) → (3,0) → (2,0) → (1,0) → (1,1) → (1,2) → (1,3)` — 6 steps to the flag-terminal (+15)
rather than 4 steps to `(4,4)` (+10), because 15·0.95⁵ outweighs the two extra −1 step costs.

### γ = 0.95, noise = 0.2 — converged in 3 outer iterations

```
V (rounded):
[[ 8.61 10.38 12.21 14.06  0.  ]
 [ 9.52 11.78 14.32  0.   14.06]
 [ 7.9   9.62  0.   14.32 12.21]
 [ 6.36  7.88  9.65 11.79 10.38]
 [ 5.1   6.6   8.19  9.86  0.  ]]

Greedy policy:
→ → → ↓ T
→ → → T ←
↑ ↑ # ↑ ↑
↑ → → ↑ ↑
→ → → → T
```

Path: `(4,0) → (4,1) → (4,2) → (4,3) → (4,4)` — with slippery moves, the long route to the flag is less
attractive and the agent takes the short route to the bottom-right terminal.

### Other settings (all verified)

| γ | noise | outer iters | path from (4,0) |
|---|---|---|---|
| 0.5 | 0.0 | 2 | (4,0) → (4,1) → (4,2) → (4,3) → (4,4) |
| 0.99 | 0.0 | 3 | (4,0) → (3,0) → (2,0) → (1,0) → (1,1) → (1,2) → (1,3) |
| 0.5 | 0.2 | 3 | (4,0) → (4,1) → (4,2) → (4,3) → (4,4) |
| 0.99 | 0.2 | 5 | (4,0) → (4,1) → (4,2) → (4,3) → (3,3) → (2,3) → (1,3) |

A low γ makes the agent short-sighted (takes the nearest +10); a high γ makes the distant +15 flag worth
the detour.

### Verification checks (headless, all passed)

1. `policy_iteration` finishes for every setting above; `V` is finite and exactly 0 at all terminals and the obstacle; policies are one-hot.
2. `extract_path` from `(4,0)` always ends on a terminal and never enters `(2,2)`.
3. Values next to the terminals/flag are higher than at the start corner in every setting.
4. "Iterate once" path (15× `policy_evaluation` then 4-argument `policy_improvement`) returns `(pi, True)` without error.
5. The Viewer's "Run to convergence" + "Show Path" works (see `figures/gridworld_policy_path.png`).
