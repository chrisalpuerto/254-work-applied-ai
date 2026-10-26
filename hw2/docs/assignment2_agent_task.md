# Agent Task: CPSC 481 Assignment #2 — Machine Learning Exercise (Parts 1 – 4.1)

You are implementing four Python programs for a machine-learning assignment. Work in the project root. Do **not** do Part 4.2 (Snake DQN) — it is out of scope for this task.

## Deliverables

| File | Status | Part |
|---|---|---|
| `linear_reg.py` | create | 1 — Linear regression |
| `mlp_classifier.py` | create | 2 — MLP classification |
| `kmeans_clustering.py` | create | 3 — K-means clustering |
| `gridworld_policy_iteration.py` | **already exists — fill in the TODOs only** | 4.1 — Policy iteration |
| `requirements.txt` | create | all |
| `RESULTS.md` | create after running everything | report material |
| `figures/` | create; save every plot here as PNG | report screenshots |

`GasProperties.csv` should be in the project root. If it is missing, stop and ask the user for it rather than inventing data.

## Global rules

- Python 3.10+. Dependencies: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `ucimlrepo`. Put them in `requirements.txt` and `pip install -r requirements.txt`.
- Use `random_state=42` everywhere randomness is involved (splits, models, KMeans) so results are reproducible.
- Every script must be runnable as `python <file>.py`, wrap top-level code in `main()` with `if __name__ == "__main__":`, and print clearly labeled results to the terminal (the student screenshots terminal output for the report).
- For every plot: `plt.savefig("figures/<name>.png", dpi=150, bbox_inches="tight")` **before** `plt.show()`. When verifying headlessly, run with `MPLBACKEND=Agg` so `plt.show()` doesn't block.
- Keep code readable and commented — this is graded on understanding, not just output.

## Loading the Iris dataset (Parts 2 and 3)

Load Iris **exactly this way** (the student's required method), not from `sklearn.datasets`:

```python
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)
X = iris.data.features            # DataFrame, 4 numeric columns
y = iris.data.targets             # DataFrame, one column ("class")
y = y.iloc[:, 0].to_numpy()       # flatten to a 1-D array of string labels
```

Put this in a shared helper `load_iris_data()` inside `mlp_classifier.py` that returns `(X, y)` as numpy arrays, and import it from `kmeans_clustering.py` so both scripts use identical data. Printing `iris.metadata` / `iris.variables` is optional (one short print is fine).

---

## Part 1 — `linear_reg.py` [20 pts]

### 1a. Small dataset with scikit-learn
1. Hard-code the training data:
   ```python
   X = [-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7]
   Y = [15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4]
   ```
2. Scatter plot of the data with matplotlib (title, axis labels). Save as `figures/linreg_scatter.png`.
3. Fit `sklearn.linear_model.LinearRegression()` on `x` reshaped to `(-1, 1)`.
4. Print the coefficient vector `[w0, w1]` (`w0 = intercept_`, `w1 = coef_[0]`) and print the equation as a string, e.g. `y = 9.83 + 1.23x`.
5. Compute and print training RMSE = `sqrt( sum((y_i - ŷ_i)^2) / N )`. Implement the formula with numpy yourself; you may also print sklearn's value as a cross-check.
6. Overlay the fitted line on the scatter plot and save as `figures/linreg_fit.png`.
7. **Extension (clearly labeled as extra):** the data is visibly U-shaped, so a straight line fits poorly. Also fit a degree-2 polynomial (`PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression`), print its coefficients `[w0, w1, w2]` and equation `y = w0 + w1x + w2x^2`, its RMSE, and overlay that curve on the same plot. This matches the assignment's `y = w0 + w1x1 + … + wnxn` polynomial form and gives the student something meaningful to discuss. Keep the plain linear model as the primary required answer.

### 1b. GasProperties.csv with the normal equation
1. Load `GasProperties.csv` with pandas. Print `df.head()`, `df.shape`, and the column names.
2. **Choose the target:** use the **last column** as `y` and all other numeric columns as features `X`. Print which column was chosen. (If the CSV has an obvious index/ID column that isn't the target, drop it and note this in the output. If it's genuinely ambiguous which column is the target, stop and ask.)
3. Split 80/20 with `train_test_split(..., test_size=0.2, random_state=42)`.
4. Build the design matrix with a bias column of ones prepended: `X_b = np.c_[np.ones(n), X]`.
5. Solve with the normal equation **using explicit numpy matrix ops** as the assignment requires:
   `w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train`
   (If the matrix is singular, fall back to `np.linalg.pinv` and print a note saying so.)
6. Print `w` with each weight labeled by its feature name (`w0` = bias), and print the fitted equation.
7. Compute and print **training RMSE** and **testing RMSE** with your own RMSE function.
8. Cross-check: fit sklearn `LinearRegression` on the same split and print its coefficients/RMSE next to yours to show they match (to ~1e-6).
9. Optional plot: predicted vs. actual for the test set → `figures/gas_pred_vs_actual.png`.

---

## Part 2 — `mlp_classifier.py` [30 pts]

Note: the assignment text says "train the decision tree algorithm … using MLPClassifier" — that's a typo. Use `sklearn.neural_network.MLPClassifier` throughout.

1. Load Iris with `load_iris_data()` (see above).
2. `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
3. **Baseline (2.3):** `MLPClassifier(hidden_layer_sizes=(3,), random_state=42, max_iter=...)` — one hidden layer, 3 neurons. Pick reasonable but un-tuned settings (e.g. defaults with `max_iter=500`). Print **train accuracy %** and **test accuracy %**. Suppress or catch `ConvergenceWarning` but print a one-line note if it fired.
4. **Tuning without changing architecture (2.4):** keep `hidden_layer_sizes=(3,)` fixed. Try changes such as: `StandardScaler` in a `Pipeline`, `activation` (`relu`, `tanh`, `logistic`), `solver` (`adam`, `lbfgs`, `sgd` with `momentum`), `learning_rate_init`, `alpha`, `max_iter`, `batch_size`. Use `GridSearchCV` (cv=5, on the training set only) or a small explicit loop — either is fine. Print the best hyperparameters and the resulting train/test accuracy %.
5. **Bigger network (2.5):** try a few larger architectures, e.g. `(10,)`, `(16, 8)`, `(32, 16)`, `(64, 32, 16)`, using the best non-architecture settings from step 4. Print a small table of architecture → train %, test %. Stop at the first one that improves on step 4's test accuracy, or report the best one found.
6. **Robustness:** a 3-neuron network is sensitive to initialization and Iris's test set is only 30 samples. For each of the three configurations (baseline, tuned, larger), also report mean ± std test accuracy over 5 seeds (`random_state` in 0–4 for the model; keep the split fixed). This makes the "did it improve?" answer honest.
7. Print a **summary table** of all three configurations and a short auto-generated sentence for each of 2.4 and 2.5 stating whether accuracy improved and by how much.
8. Print a confusion matrix and `classification_report` for the final (best) model.
9. Expose a function `train_best_mlp()` that returns `(model, X_test, y_test, test_accuracy)` so Part 3 can import it without re-running the whole script's printing. All printing/plotting stays in `main()`.

---

## Part 3 — `kmeans_clustering.py` [30 pts]

1. Load Iris via `from mlp_classifier import load_iris_data`. **Do not use the labels for clustering.**
2. Fit `KMeans(n_clusters=3, n_init=10, random_state=42)` on the 4 raw features. (Optionally also run on standardized features and report both — label which is which; raw features are the primary answer.)
3. **Per-cluster RMSE:** for each cluster k, `RMSE_k = sqrt( mean over points in k of ||x_i - centroid_k||^2 )` (Euclidean distance to the cluster's own centroid). Print cluster size, centroid coordinates (labeled by feature name), and RMSE for each cluster, plus the overall RMSE.
4. **Cluster quality validation (3.2)** — now bring in the true labels for evaluation only:
   - Contingency table (pandas `crosstab`) of cluster ID vs. true class.
   - Map each cluster to a class with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on the negated contingency matrix) and report the resulting accuracy %.
   - Purity, Adjusted Rand Index (`adjusted_rand_score`), and Normalized Mutual Information (`normalized_mutual_info_score`).
   - Identify which class(es) K-means separates cleanly and which it confuses (on Iris, setosa is typically perfect while versicolor/virginica overlap — but report what *your* run shows).
5. **Compare to the MLP:** call `train_best_mlp()` from `mlp_classifier.py`, and print K-means mapped accuracy (on all 150 points) next to the MLP test accuracy. Print a 2–4 sentence comparison explaining why a supervised classifier outperforms unsupervised clustering here (labels vs. none; K-means assumes spherical, similar-size clusters; overlapping species in feature space).
6. Plots → `figures/`:
   - `kmeans_clusters.png`: 2-D scatter (petal length vs. petal width, or first two PCA components) colored by cluster, centroids marked with an X.
   - `kmeans_vs_true.png`: same axes side-by-side, colored by true class.

---

## Part 4.1 — `gridworld_policy_iteration.py` [20 pts]

**Edit the existing file in place.** Only fill in the three TODO functions (`policy_evaluation`, `policy_improvement`, `policy_iteration`). Do not change the `Gridworld` class, `transition_dist`, `extract_path`, `debug_policy_along_path`, or the `Viewer` class/callbacks, and keep every function signature compatible with how `Viewer` calls it.

### Environment facts you must respect
- `pi` has shape `(H, W, 4)`; `pi[r, c, a]` = probability of action `a` in state `(r, c)`. Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
- `V` has shape `(H, W)`.
- `transition_dist(env, s, a, noise)` returns a list of `(s_next, prob)` pairs. Use it — don't reimplement the dynamics.
- `env.reward(s, a, s_next)` gives the reward for a transition.
- **Terminal states and obstacles must be skipped** (their value stays 0, their policy row stays all-zeros). This matters: terminal `(1, 3)` is also a reward flag, so a terminal self-loop would otherwise collect +5 forever and blow up the values.

### (a) `policy_evaluation(env, V, pi, gamma, noise)` — one synchronous sweep
For each non-terminal, non-obstacle state `s`:

```
V_new[s] = Σ_a  pi(a|s) · Σ_{s'} P(s'|s,a) · [ R(s,a,s') + gamma · V[s'] ]
```

Read from the old `V`, write into `V_new` (synchronous update — `V_new = V.copy()` is already there). Return `V_new`. One call = exactly one sweep; `Viewer.on_iter` calls it 15 times in a row.

### (b) `policy_improvement(env, V, gamma, noise, pi_old=None)`
Add `pi_old=None` as an **optional** trailing parameter (keeps `Viewer.on_iter`'s 4-argument call working).

For each non-terminal, non-obstacle state, compute action values:

```
Q(s,a) = Σ_{s'} P(s'|s,a) · [ R(s,a,s') + gamma · V[s'] ]
```

Make the new policy **deterministic (one-hot)** on the best action. Break ties toward the lowest action index among actions within `1e-9` of the max (`np.isclose`), so the policy is stable and `extract_path`'s `argmax` agrees with it.

Return `(pi_new, stable)`:
- If `pi_old is None`: `stable = True` (preserves current behavior for the Viewer).
- Otherwise `stable = True` iff, for every state, the action `pi_old` picks (its argmax) already has `Q` within `1e-9` of the max `Q`. Comparing Q-values rather than raw arrays prevents endless flip-flopping between equally good actions.

### (c) `policy_iteration(env, gamma, noise, max_eval_iters=100, tol=1e-6)`
```
V = zeros; pi = random_policy(env)
repeat (cap at e.g. 1000 outer iterations as a safety net):
    # Policy evaluation: sweep until converged or max_eval_iters reached
    for i in range(max_eval_iters):
        V_new = policy_evaluation(env, V, pi, gamma, noise)
        delta = max |V_new - V|
        V = V_new
        if delta < tol: break
    # Policy improvement
    pi, stable = policy_improvement(env, V, gamma, noise, pi_old=pi)
    if stable: break
return V, pi
```

The initial `random_policy` is uniform, so the first improvement step will never be judged stable against it unless it happens to agree — that's expected. Print the number of outer iterations taken when the loop ends (a single short line).

### Verification for 4.1 (run headlessly — don't open the GUI)
Write a temporary check script (or a `python -c` snippet) with `MPLBACKEND=Agg` that imports from the module and confirms:
1. `policy_iteration(Gridworld(), 0.95, 0.0)` finishes, returns finite `V`, and `V` is 0 at every terminal and obstacle.
2. `extract_path(env, pi, (4, 0))` ends on a terminal state and never enters the obstacle `(2, 2)`.
3. The same holds with `noise=0.2` and with `gamma=0.5` and `0.99`.
4. Values are larger near the terminals / the flag-terminal `(1, 3)` than far away.
5. The 15-sweep "Iterate once" path also works: run `policy_evaluation` 15× then `policy_improvement(env, V, gamma, noise)` with 4 args, and confirm it returns `(pi, True)` without error.

Print the final V (rounded to 2 decimals), the greedy action arrows as a text grid (↑ → ↓ ←, `T` for terminals, `#` for obstacles), and the path. Delete the temporary check script afterwards (leave the module itself unchanged apart from the TODO implementations).

---

## Final verification (all parts)

1. `pip install -r requirements.txt`
2. Run each script with `MPLBACKEND=Agg` and confirm it exits with code 0 and all expected figures exist in `figures/`.
3. Sanity checks:
   - Part 1: normal-equation weights match sklearn to ~1e-6; test RMSE is printed; degree-2 RMSE is lower than linear RMSE on the small dataset.
   - Part 2: all accuracies are between 0 and 100 %; summary table printed.
   - Part 3: cluster sizes sum to 150; ARI/NMI are between 0 and 1.
   - Part 4.1: checks above pass.
4. Write `RESULTS.md` containing, per part, the key numbers actually produced by your runs (coefficients, equations, RMSEs, accuracies, cluster metrics, gridworld V and path), the list of figures generated, and short draft answers to the written questions (2.4, 2.5, 3.2) based on those real numbers. Do not invent or round away results — copy them from the actual output.
5. Report back: which files were created/modified, any assumptions made (especially the GasProperties target column), and anything that didn't work.
