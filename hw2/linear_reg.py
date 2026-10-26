"""
CPSC 481 Assignment #2 - Part 1: Supervised learning for linear regression
Author: Christopher Alpuerto

1a. Fits a line to a small hard-coded dataset with scikit-learn's LinearRegression,
    prints the coefficients / equation / training RMSE, and plots the result.
    (Extra: also fits a degree-2 polynomial, since the data is U-shaped.)
1b. Loads GasProperties.csv, splits it 80/20, and solves linear regression with
    the normal equation  w = (X^T X)^-1 X^T y  using explicit numpy matrix ops.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

FIG_DIR = "figures"
GAS_CSV = "GasProperties.csv"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    # Root Mean Squared Error: sqrt( sum((y_i - y_hat_i)^2) / N )
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))


def save_fig(name):
    # save the current figure into figures/ (always before plt.show()).
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")


def format_equation(weights, terms):
    # build a readable equation string like 'y = 1.00 + 2.00x - 3.00x^2'
    eq = f"y = {weights[0]:.4f}"
    for w, term in zip(weights[1:], terms):
        sign = "-" if w < 0 else "+"
        eq += f" {sign} {abs(w):.4f}{term}"
    return eq

# Part 1a: small dataset with scikit-learn
def part_1a():
    print("PART 1a: Linear regression on the small dataset (scikit-learn)")
    

    # (1) Load the training dataset
    X = [-3.0, -2.5, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 2.7]
    Y = [15.5, 12.9, 9.5, 6.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 21.4]
    x = np.array(X).reshape(-1, 1)  # sklearn expects a 2-D feature matrix
    y = np.array(Y)
    print(f"N = {len(y)} training examples")
    # (2) Scatter plot of the raw data
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="tab:blue", label="Training data")
    plt.title("Training data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig("linreg_scatter.png")
    plt.show()
    # (3) Train a linear model y = w0 + w1*x
    model = LinearRegression()
    model.fit(x, y)
    # (4) Coefficient vector [w0, w1] and the equation
    w = np.array([model.intercept_, model.coef_[0]])
    print(f"\nCoefficient vector [w0, w1] = {np.round(w, 4).tolist()}")
    print(f"Linear equation: {format_equation(w, ['x'])}")
    # (5) Training RMSE using our own formula (plus sklearn as a cross-check)
    y_hat = model.predict(x)
    train_rmse = rmse(y, y_hat)
    sk_rmse = np.sqrt(mean_squared_error(y, y_hat))
    print(f"Training RMSE (numpy formula)  = {train_rmse:.4f}")
    print(f"Training RMSE (sklearn check)  = {sk_rmse:.4f}")

    # degree-2 polynomial, since the data is clearly U-shaped 
    print("\n[EXTRA] Degree-2 polynomial regression y = w0 + w1x + w2x^2")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly.fit_transform(x)  # columns: [x, x^2]
    poly_model = LinearRegression()
    poly_model.fit(x_poly, y)
    w_poly = np.array([poly_model.intercept_, *poly_model.coef_])
    poly_rmse = rmse(y, poly_model.predict(x_poly))
    print(f"[EXTRA] Coefficient vector [w0, w1, w2] = {np.round(w_poly, 4).tolist()}")
    print(f"[EXTRA] Quadratic equation: {format_equation(w_poly, ['x', 'x^2'])}")
    print(f"[EXTRA] Training RMSE = {poly_rmse:.4f}  (linear RMSE = {train_rmse:.4f})")
    # (6) Overlay both fitted functions on the scatter plot
    x_line = np.linspace(min(X) - 0.2, max(X) + 0.2, 200).reshape(-1, 1)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="tab:blue", label="Training data")
    plt.plot(x_line, model.predict(x_line), color="tab:red",
             label=f"Linear: {format_equation(w, ['x'])}")
    plt.plot(x_line, poly_model.predict(poly.transform(x_line)), color="tab:green",
             linestyle="--", label=f"Quadratic (extra): RMSE={poly_rmse:.3f}")
    plt.title(f"Linear regression fit (linear RMSE = {train_rmse:.3f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig("linreg_fit.png")
    plt.show()

# Part 1b: GasProperties.csv with the normal equation
def part_1b():
    
    print("PART 1b: GasProperties.csv with the normal equation")

    df = pd.read_csv(GAS_CSV)
    print("First rows:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Target = last column, features = every other numeric column.
    # "Idx" is a continuous measured value (not a row ID), so it is the target.
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns[:-1] if pd.api.types.is_numeric_dtype(df[c])]
    print(f"\nTarget column (y): '{target_col}'")
    print(f"Feature columns (X): {feature_cols}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # 80% training / 20% testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")

    # Design matrices: prepend a column of ones so w0 acts as the bias/intercept
    Xb_train = np.c_[np.ones(len(X_train)), X_train]
    Xb_test = np.c_[np.ones(len(X_test)), X_test]

    # Normal equation: w = (X^T X)^-1 X^T y  (transpose, inverse, multiplication)
    XtX = Xb_train.T @ Xb_train
    try:
        w = np.linalg.inv(XtX) @ Xb_train.T @ y_train
    except np.linalg.LinAlgError:
        print("NOTE: X^T X is singular - falling back to the pseudo-inverse (np.linalg.pinv).")
        w = np.linalg.pinv(XtX) @ Xb_train.T @ y_train

    names = ["bias"] + feature_cols
    print("\nWeights from the normal equation:")
    for i, (name, wi) in enumerate(zip(names, w)):
        print(f"  w{i} ({name:>4}) = {wi: .10f}")
    print(f"Equation: {format_equation(w, [f'*{c}' for c in feature_cols])}")

    # RMSE on the training and testing sets
    yhat_train = Xb_train @ w
    yhat_test = Xb_test @ w
    train_rmse = rmse(y_train, yhat_train)
    test_rmse = rmse(y_test, yhat_test)
    print(f"\nTraining RMSE = {train_rmse:.6f}")
    print(f"Testing  RMSE = {test_rmse:.6f}")

    # Cross-check against scikit-learn's LinearRegression on the same split
    sk = LinearRegression().fit(X_train, y_train)
    w_sk = np.r_[sk.intercept_, sk.coef_]
    print("\nCross-check with sklearn LinearRegression:")
    print(f"  {'weight':<10}{'normal eq':>20}{'sklearn':>20}{'abs diff':>14}")
    for i, name in enumerate(names):
        print(f"  w{i:<9}{w[i]:>20.10f}{w_sk[i]:>20.10f}{abs(w[i] - w_sk[i]):>14.2e}")
    print(f"  sklearn train RMSE = {rmse(y_train, sk.predict(X_train)):.6f}, "
          f"test RMSE = {rmse(y_test, sk.predict(X_test)):.6f}")
    print(f"  Max |w_normal - w_sklearn| = {np.max(np.abs(w - w_sk)):.2e}")

    # Predicted vs. actual on the test set (subsample points so the plot stays readable)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_test), size=min(5000, len(y_test)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[idx], yhat_test[idx], s=4, alpha=0.4, label="Test samples (5000 shown)")
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], color="tab:red", label="Perfect prediction")
    plt.title(f"GasProperties: predicted vs actual {target_col} (test RMSE={test_rmse:.4f})")
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig("gas_pred_vs_actual.png")
    plt.show()


def main():
    part_1a()
    part_1b()


if __name__ == "__main__":
    main()
