import numpy as np
import pandas as pd

# =====================================================
# Metrics (binary + a bit of multiclass-ready structure)
# =====================================================
class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        # fraction of correct predictions
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return (y_true == y_pred).mean()

    @staticmethod
    def precision_recall_f1_binary(y_true, y_pred, positive_label=1):
        # compute TP/FP/FN for positive class to get precision/recall/F1
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
        fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
        fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
        precision = tp / (tp + fp + 1e-15)
        recall    = tp / (tp + fn + 1e-15)
        f1        = 2 * precision * recall / (precision + recall + 1e-15)
        return precision, recall, f1

    @staticmethod
    def log_loss(y_true, y_prob):
        # cross-entropy loss for binary classification; y_prob = P(y=1)
        y_true = np.asarray(y_true).ravel().astype(float)
        p = np.clip(np.asarray(y_prob).ravel().astype(float), 1e-15, 1-1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    @staticmethod
    def roc_auc(y_true, scores):
        # trapezoidal AUC by sorting predicted scores descending and accumulating TPR/FPR
        y_true = np.asarray(y_true).ravel().astype(int)
        scores = np.asarray(scores).ravel().astype(float)

        # sort by score (desc)
        order = np.argsort(-scores)
        y = y_true[order]

        P = y.sum()
        N = y.size - P
        if P == 0 or N == 0:
            return 0.5  # undefined; fallback to chance

        tps = np.cumsum(y)          # cumulative true positives as threshold moves
        fps = np.cumsum(1 - y)      # cumulative false positives

        tpr = tps / P
        fpr = fps / N

        # add (0,0) start and (1,1) end for completeness
        tpr = np.r_[0.0, tpr, 1.0]
        fpr = np.r_[0.0, fpr, 1.0]

        return np.trapz(tpr, fpr)


# ============================================
# Data helpers: split + standardization (z-score)
# ============================================
def train_test_split(X, y, test_size=0.2, random_state=42):
    # shuffle indices and slice into train/test portions
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X.iloc[tr].copy(), X.iloc[te].copy(), y.iloc[tr].copy(), y.iloc[te].copy()

def standardize_fit(X: pd.DataFrame):
    # compute column-wise mean and std for training features
    mu = X.mean()
    sigma = X.std(axis=0).replace(0, 1.0)  # guard against constant columns
    return mu, sigma

def standardize_transform(X: pd.DataFrame, mu: pd.Series, sigma: pd.Series):
    # apply z-score: (x - mean) / std to any dataset using training stats
    return (X - mu) / sigma


# ======================================
# Logistic Regression (NumPy, from scratch)
# ======================================
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=5000, l2=0.0, tol=1e-7, verbose=False):
        """
        lr      : learning rate for gradient descent
        n_iters : maximum iterations
        l2      : L2 regularization strength (0 = none)
        tol     : early-stop threshold on loss improvement
        verbose : print loss periodically if True
        """
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.w = None              # weights including bias at index 0
        self.mu = None             # feature means for scaling
        self.sigma = None          # feature stds for scaling
        self.n_features = None

    @staticmethod
    def _sigmoid(z):
        # numerically stable sigmoid to avoid overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, X_np):
        # prepend a column of ones for the bias term
        return np.c_[np.ones((X_np.shape[0], 1)), X_np]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train via batch gradient descent on the cross-entropy objective:
        J(w) = -1/n Σ [ y log p + (1-y) log(1-p) ] + (λ/2n)||w_no_bias||^2
        where p = σ(Xw).
        Steps:
          1) standardize features (store mu/sigma)
          2) add bias column (so bias is learned as w[0])
          3) iterate: forward pass -> gradient -> weight update
          4) early stop when loss improvement < tol
        """
        # 1) standardize with training stats
        self.mu, self.sigma = standardize_fit(X)
        Xs = standardize_transform(X, self.mu, self.sigma)
        X_np = Xs.to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float).reshape(-1, 1)
        self.n_features = X_np.shape[1]

        # 2) add bias column
        Xb = self._add_bias(X_np)                  # shape (n, d+1)
        n, d1 = Xb.shape

        # 3) init weights
        self.w = np.zeros((d1, 1), dtype=float)
        last_loss = np.inf

        # 4) gradient descent loop
        for i in range(self.n_iters):
            # forward pass: logits -> probabilities
            logits = Xb @ self.w                   # linear response
            p = self._sigmoid(logits)              # probability of class 1

            # gradient of cross-entropy; add L2 for non-bias terms
            # grad = X^T (p - y) / n  (+ λ/n * w_no_bias)
            error = (p - y_np)
            grad = (Xb.T @ error) / n
            if self.l2 > 0:
                reg = np.r_[np.array([[0.0]]), self.w[1:]]  # do not regularize bias
                grad += (self.l2 / n) * reg

            # weight update
            self.w -= self.lr * grad

            # compute loss for monitoring / early stopping
            # base cross-entropy
            ce = ClassificationMetrics.log_loss(y_np, p)
            # add L2 penalty (no bias)
            if self.l2 > 0:
                ce += (self.l2 / (2 * n)) * float(np.sum(self.w[1:] ** 2))
            loss = ce

            # optional logging
            if self.verbose and (i % 200 == 0 or i == self.n_iters - 1):
                print(f"iter {i:5d}  loss={loss:.6f}")

            # early stopping check
            if np.abs(last_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i} (Δloss<{self.tol})")
                break
            last_loss = loss

        return self

    def predict_proba(self, X: pd.DataFrame):
        """
        Return class probabilities P(y=1|x).
        Steps:
          1) standardize using stored mu/sigma
          2) add bias
          3) apply sigmoid(Xb @ w)
        """
        Xs = standardize_transform(X, self.mu, self.sigma)
        X_np = Xs.to_numpy(dtype=float)
        Xb = self._add_bias(X_np)
        return self._sigmoid(Xb @ self.w).ravel()

    def predict(self, X: pd.DataFrame, threshold=0.5):
        """
        Convert probabilities to class labels using a threshold (default 0.5).
        """
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold=0.5, positive_label=1):
        """
        Convenience method: computes common metrics given features/labels.
        """
        p = self.predict_proba(X)
        y_hat = (p >= threshold).astype(int)
        acc = ClassificationMetrics.accuracy(y, y_hat)
        prec, rec, f1 = ClassificationMetrics.precision_recall_f1_binary(y, y_hat, positive_label)
        logloss = ClassificationMetrics.log_loss(y, p)
        auc = ClassificationMetrics.roc_auc(y, p)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "log_loss": logloss, "roc_auc": auc}


# ======================================
# Example: Real-world style (Titanic-like)
# ======================================
if __name__ == "__main__":
    # --- Load your CSV (ensure it has a binary target column) ---
    # For Titanic Kaggle CSV, typical columns include: Survived (target), Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, etc.
    # Replace the path with your actual dataset file.
    df = pd.read_csv("/Users/adityasrivatsav/Documents/GitHub/ML-DL-CV/Machine_learning/ML_From_Scratch/Dataset/tested.csv")

    # --- Basic cleaning / target selection ---
    # Choose target (binary 0/1). Here we assume 'Survived' is already 0/1.
    target_col = "Survived"
    y = df[target_col].astype(int)

    # --- Feature engineering (example choices) ---
    # Select a few numeric and categorical features commonly useful for Titanic.
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    cat_cols = ["Pclass", "Sex", "Embarked"]

    # Fill simple missing values for numerics (median) and categoricals (mode)
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # One-hot encode categoricals; drop_first=True to avoid dummy trap (not required but cleaner)
    X = pd.get_dummies(df[numeric_cols + cat_cols], drop_first=True)

    # --- Train/test split ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train logistic regression (GD) ---
    # L2 small value helps generalization; tune lr/n_iters as needed.
    clf = LogisticRegressionScratch(lr=0.1, n_iters=8000, l2=0.01, tol=1e-7, verbose=True)
    clf.fit(X_tr, y_tr)

    # --- Evaluate on train and test ---
    print("Train:", clf.evaluate(X_tr, y_tr, threshold=0.5, positive_label=1))
    print("Test :", clf.evaluate(X_te, y_te, threshold=0.5, positive_label=1))

    # --- Threshold tuning note ---
    # You can adjust the decision threshold based on ROC/PR trade-offs:
    #   y_hat_03 = (clf.predict_proba(X_te) >= 0.3).astype(int)
    # and recompute metrics if recall is more important than precision, etc.
