"""Linear Regression from Scratch
Author: Aditya Srivatsav Lolla
Date: 2023-10-01"""
import numpy as np
import pandas as pd

# =========================
# Imports
# =========================
# Import NumPy for fast vectorized linear algebra (arrays, dot products, norms).
# Import pandas for tabular data handling (DataFrame/Series) and easy column ops.
# Import Matplotlib to visualize training loss or any diagnostic charts.
# =========================

# Utilities
# -------------------------
# train_test_split: deterministic shuffle/split that returns copies of train/test DataFrames/Series.
# - X: features as a pandas DataFrame
# - y: target as a pandas Series
# - test_size: fraction of rows to put into the test split (default 20%)
# - random_state: seed for reproducibility
# Steps:
# 1) Build a RandomState with the seed.
# 2) Create a range of row indices [0..n-1].
# 3) Shuffle indices in-place for randomness.
# 4) Compute the cut index for the train portion based on test_size.
# 5) Slice indices into train/test sets.
# 6) Return .iloc slices of X/y using those index arrays, calling .copy() to avoid SettingWithCopy issues.
# -------------------------

# standardize_fit: compute per-feature mean and std on the TRAIN data only.
# - Returns (mu, sigma) that will be reused to standardize both train and test consistently.
# - Replace any zero std with 1 to prevent divide-by-zero when a column is constant.

# standardize_transform: apply z-score standardization: (X - mu) / sigma
# - Uses the mu/sigma learned from training data to transform any data (train/test/new).

# =========================
# Regression Metrics
# =========================
# MSE: Mean Squared Error — average of squared residuals; penalizes large errors heavily.
# RMSE: Root MSE — in same units as target; easier to interpret.
# MAE: Mean Absolute Error — median-robust; less sensitive to outliers than MSE.
# R2: Coefficient of Determination — 1 - (SSE/SST); fraction of variance explained by the model.
# R2_adjusted: R² adjusted for number of features; discourages overfitting with many predictors.
# explained_variance: 1 - Var(residuals)/Var(y); similar to R² but uses variances.
# Each metric converts inputs to flat NumPy arrays and handles basic numerical stability.
# =========================

# =========================
# Linear Regression Model (Gradient Descent + Normal Equation + optional L2)
# =========================
# __init__:
# - lr: learning rate (step size) for gradient descent updates.
# - n_iters: max number of GD iterations.
# - l2: L2 regularization strength (Ridge). 0.0 means no regularization.
# - tol: early-stopping tolerance on improvement in loss between iterations.
# - verbose: whether to print progress (loss every N steps).
# - Internal attributes:
#   * w: full weight vector including bias as first entry after we add a bias column.
#   * mu/sigma: feature scaling parameters learned from training data.
#   * n_features: number of input features (without bias).

# _add_bias(X_np):
# - Utility to prepend a column of ones to the standardized feature matrix.
# - Converts X of shape (n, d) into Xb of shape (n, d+1) where column 0 is 1 (bias term).

# fit(X, y):
# - End-to-end training via batch gradient descent.
# Steps:
# 1) Fit standardization on X (mu, sigma) to normalize input scale and speed up convergence.
# 2) Transform X using those statistics; convert standardized DataFrame to float NumPy ndarray.
# 3) Reshape y to (n, 1) so matrix math broadcasts cleanly.
# 4) Cache n_features for downstream metrics like adjusted R².
# 5) Add a bias column of ones to X to learn bias inside the weight vector.
# 6) Initialize weights w to zeros with shape (d+1, 1) (the +1 accounts for bias).
# 7) Initialize last_loss to +inf to enable first early-stopping comparison.
# 8) For each iteration:
#    a) Forward pass: yhat = Xb @ w (vectorized predictions for all rows).
#    b) Residuals: resid = yhat - y (prediction error per row).
#    c) Gradient of MSE w.r.t. weights: grad = (Xb.T @ resid) / n (batch gradient).
#    d) If L2 > 0: add L2 gradient to non-bias weights only (bias should not be penalized).
#    e) Weight update: w = w - lr * grad (take a descent step).
#    f) Compute scalar loss = 0.5 * mean(resid^2); add L2 penalty term if enabled.
#    g) If verbose, print loss every few iterations (e.g., every 100 steps).
#    h) Early stopping: if |last_loss - loss| < tol, stop training to avoid overfitting/wasted steps.
#    i) Update last_loss for next iteration comparison.
# 9) Return self to allow fluent chaining.

# fit_normal_equation(X, y):
# - Closed-form solution (Normal Equation) that directly solves for weights in one shot.
# - Also supports L2 (Ridge) by adding λI to XᵀX except for bias row/column.
# Steps:
# 1) Fit and apply standardization (like in fit).
# 2) Add bias column to standardized matrix.
# 3) If l2 == 0:
#    * Use Moore–Penrose pseudo-inverse: w = pinv(Xb) @ y; numerically stable for collinearity.
#    Else (l2 > 0):
#    * Build identity matrix I and set I[0,0]=0 to avoid penalizing bias.
#    * Solve (XbᵀXb + λI) w = Xbᵀ y using a linear solver (more stable than explicit inverse).
# 4) Return self.

# predict(X):
# - Apply the stored mu/sigma to standardize new data the same way as training.
# - Add bias column to match the weight vector shape.
# - Return the vectorized predictions (flattened to 1-D).

# evaluate(X, y):
# - Convenience wrapper: runs predict and returns a dict of common regression metrics:
#   MSE, RMSE, MAE, R2, Adjusted R2 (uses cached n_features), and Explained Variance.

# =========================
# Example Usage (script main)
# =========================
# 1) Create a reproducible RNG and sample size n.
# 2) Build a synthetic housing-like DataFrame with three features:
#    - sqft: normally distributed square footage
#    - bedrooms: random small integer (cast to float for math consistency)
#    - age: exponential to mimic skew in property age
# 3) Define a "true" linear relationship in terms of features (scaled by constants for realism),
#    then add Gaussian noise to simulate measurement error and unobserved factors.
# 4) Split the dataset into train/test with the utility (80/20 by default).
# 5) Train a gradient-descent model:
#    - Initialize with learning rate, iterations, L2=0 (pure OLS), small tolerance, verbose logging.
#    - Fit on training data; evaluate on both train and test to check generalization.
# 6) Train a normal-equation (closed-form) model as a fast baseline and evaluate.
# 7) Train a Ridge (L2>0) model via normal equation to see the effect of regularization.
#    Compare train vs test metrics to observe bias–variance trade-offs.


# ---------- Utilities ----------
def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X.iloc[tr].copy(), X.iloc[te].copy(), y.iloc[tr].copy(), y.iloc[te].copy()

def standardize_fit(X: pd.DataFrame):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0).replace(0, 1.0)  # avoid divide-by-zero
    return mu, sigma

def standardize_transform(X: pd.DataFrame, mu: pd.Series, sigma: pd.Series):
    return (X - mu) / sigma

# ---------- Metrics (NumPy-only) ----------
class RegressionMetrics:
    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        sse = np.sum((y_true - y_pred)**2)
        sst = np.sum((y_true - y_true.mean())**2) + 1e-15
        return 1 - sse/sst

    @staticmethod
    def r2_adjusted(y_true, y_pred, n_features):
        n = np.asarray(y_true).size
        r2 = RegressionMetrics.r2(y_true, y_pred)
        denom = max(n - n_features - 1, 1)
        return 1 - (1 - r2) * (n - 1) / denom

    @staticmethod
    def explained_variance(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        var_res = np.var(y_true - y_pred)
        var_y = np.var(y_true) + 1e-15
        return 1 - var_res/var_y

# ---------- Linear Regression (GD + Normal Eq + optional L2) ----------
class LinearRegressionScratch:
    def __init__(self, lr=0.05, n_iters=2000, l2=0.0, tol=1e-7, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.w_ = None              # includes bias at index 0
        self.mu_ = None             # feature means (for scaling)
        self.sigma_ = None          # feature stds (for scaling)
        self.n_features_ = None

    @staticmethod
    def _add_bias(X_np):
        # X_np: (n, d)
        return np.c_[np.ones((X_np.shape[0], 1)), X_np]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # save scaling params
        self.mu_, self.sigma_ = standardize_fit(X)
        Xs = standardize_transform(X, self.mu_, self.sigma_)
        X_np = Xs.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float).reshape(-1, 1)
        self.n_features_ = X_np.shape[1]

        Xb = self._add_bias(X_np)  # (n, d+1)
        n, d1 = Xb.shape

        # init weights
        self.w_ = np.zeros((d1, 1))
        last_loss = np.inf

        for i in range(self.n_iters):
            yhat = Xb @ self.w_
            resid = yhat - y_np
            # gradient (no L2 on bias term)
            grad = (Xb.T @ resid) / n
            if self.l2 > 0:
                reg = np.r_[np.array([[0.0]]), self.w_[1:]]  # exclude bias
                grad += (self.l2 / n) * reg
            # step
            self.w_ -= self.lr * grad

            # compute loss for early stopping
            loss = 0.5 * np.mean(resid**2)
            if self.l2 > 0:
                loss += 0.5 * self.l2 * np.mean((self.w_[1:])**2)
            if self.verbose and (i % 100 == 0 or i == self.n_iters - 1):
                print(f"iter {i:5d}  loss={loss:.6f}")
            if np.abs(last_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i} (Δloss<{self.tol})")
                break
            last_loss = loss
        return self

    def fit_normal_equation(self, X: pd.DataFrame, y: pd.Series):
        # Closed-form solution using pseudo-inverse; supports L2 (Ridge)
        self.mu_, self.sigma_ = standardize_fit(X)
        Xs = standardize_transform(X, self.mu_, self.sigma_)
        X_np = Xs.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float).reshape(-1, 1)
        self.n_features_ = X_np.shape[1]

        Xb = self._add_bias(X_np)
        n, d1 = Xb.shape

        if self.l2 > 0:
            I = np.eye(d1)
            I[0, 0] = 0.0  # no penalty on bias
            A = Xb.T @ Xb + self.l2 * I
            b = Xb.T @ y_np
            self.w_ = np.linalg.solve(A, b)
        else:
            # Moore-Penrose pinv is stable
            self.w_ = np.linalg.pinv(Xb) @ y_np
        return self

    def predict(self, X: pd.DataFrame):
        Xs = standardize_transform(X, self.mu_, self.sigma_)
        X_np = Xs.to_numpy(dtype=float)
        Xb = self._add_bias(X_np)
        return (Xb @ self.w_).ravel()

    # convenience: evaluate all common metrics
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X)
        return {
            "MSE": RegressionMetrics.mse(y, y_pred),
            "RMSE": RegressionMetrics.rmse(y, y_pred),
            "MAE": RegressionMetrics.mae(y, y_pred),
            "R2": RegressionMetrics.r2(y, y_pred),
            "R2_adjusted": RegressionMetrics.r2_adjusted(y, y_pred, self.n_features_),
            "ExplainedVariance": RegressionMetrics.explained_variance(y, y_pred),
        }

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example data (replace with your DataFrame)
    # Synthetic: y = 3 + 2*x1 - 1*x2 + noise
    rng = np.random.RandomState(0)
    n = 1000
    X_df = pd.DataFrame({
        "sqft": rng.normal(1500, 500, n),
        "bedrooms": rng.randint(1, 5, n).astype(float),
        "age": rng.exponential(30, n)
    })
    true_w = np.array([3.0, 2.0, -1.0])  # for features (after scaling the model still recovers relationship)
    noise = rng.normal(0, 5, n)
    y_ser = pd.Series(
        200_000 + 100 * (true_w[0] * X_df["sqft"]/1000 + true_w[1]*X_df["bedrooms"] + true_w[2]*X_df["age"]/10) + noise,
        name="price"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_ser, test_size=0.2, random_state=42)

    # 1) Gradient Descent
    lr_gd = LinearRegressionScratch(lr=0.05, n_iters=5000, l2=0.0, tol=1e-9, verbose=True)
    lr_gd.fit(X_tr, y_tr)
    print("GD Train:", lr_gd.evaluate(X_tr, y_tr))
    print("GD Test :", lr_gd.evaluate(X_te, y_te))

    # 2) Closed-form (Normal Equation) — fast baseline
    lr_ne = LinearRegressionScratch(l2=0.0)
    lr_ne.fit_normal_equation(X_tr, y_tr)
    print("NE  Train:", lr_ne.evaluate(X_tr, y_tr))
    print("NE  Test :", lr_ne.evaluate(X_te, y_te))

    # 3) Ridge (L2) via Normal Equation (set l2>0)
    lr_ridge = LinearRegressionScratch(l2=1.0)
    lr_ridge.fit_normal_equation(X_tr, y_tr)
    print("Ridge Train:", lr_ridge.evaluate(X_tr, y_tr))
    print("Ridge Test :", lr_ridge.evaluate(X_te, y_te))
