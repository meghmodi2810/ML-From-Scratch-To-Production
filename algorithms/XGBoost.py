import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


class XGBoostNode:
    """Single node in an XGBoost tree structure."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, weight=0.0):
        self.feature = feature          # Split feature index
        self.threshold = threshold      # Split threshold value
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.weight = weight            # Optimal leaf weight w*


class XGBoostTree:
    """Individual regression tree built using second-order Taylor expansion (gradients + Hessians)."""

    def __init__(self, max_depth: int = 3, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def _calc_leaf_weight(self, g: np.ndarray, h: np.ndarray) -> float:
        """Calculates optimal leaf weight w* = - sum(g) / (sum(h) + lambda)."""
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)

    def _calc_structure_score(self, G: float, H: float) -> float:
        """Calculates quality score: 1/2 * G^2 / (H + lambda)."""
        return 0.5 * (G ** 2) / (H + self.reg_lambda)

    def _best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        m, d = X.shape
        G_total, H_total = np.sum(g), np.sum(h)

        best_gain = 0.0
        best_feat, best_thresh = None, None

        for feat in range(d):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                G_L, H_L = np.sum(g[left_mask]), np.sum(h[left_mask])
                G_R, H_R = np.sum(g[right_mask]), np.sum(h[right_mask])

                # Split Gain Formula = 1/2 * [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G_total^2/(H_total + lambda) ] - gamma
                score_L = self._calc_structure_score(G_L, H_L)
                score_R = self._calc_structure_score(G_R, H_R)
                score_parent = self._calc_structure_score(G_total, H_total)

                gain = score_L + score_R - score_parent - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int):
        m = X.shape[0]

        if depth >= self.max_depth or m < 2:
            leaf_w = self._calc_leaf_weight(g, h)
            return XGBoostNode(weight=leaf_w)

        feat, thresh, gain = self._best_split(X, g, h)

        if feat is None or gain <= 0.0:  # Post-pruning via gamma penalty
            leaf_w = self._calc_leaf_weight(g, h)
            return XGBoostNode(weight=leaf_w)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        return XGBoostNode(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        self.root = self._build_tree(X, g, h, depth=0)
        return self

    def _predict_sample(self, node: XGBoostNode, x: np.ndarray) -> float:
        if node.left is None and node.right is None:
            return node.weight
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        return self._predict_sample(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(self.root, x) for x in X])


class XGBoostClassifierScratch:
    """XGBoost Binary Classifier from scratch using second-order Taylor expansion and L2 regularization."""

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, max_depth: int = 3, reg_lambda: float = 1.0, gamma: float = 0.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.base_pred = 0.0
        self.loss_history = []  # Log-loss per iteration

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifierScratch":
        X = np.array(X, dtype=np.float64)
        m = X.shape[0]

        self.classes_ = np.unique(y)
        y_binary = np.where(y == self.classes_[0], 0, 1)

        # Initialize base prediction (log-odds)
        p_avg = np.mean(y_binary)
        self.base_pred = np.log(p_avg / (1.0 - p_avg + 1e-10))
        y_pred = np.full(m, self.base_pred)

        self.trees = []
        self.loss_history = []

        for _ in range(self.n_estimators):
            p = self._sigmoid(y_pred)
            self.loss_history.append(self._log_loss(y_binary, p))

            # First-order gradient: g_i = p_i - y_i
            g = p - y_binary

            # Second-order Hessian: h_i = p_i * (1 - p_i)
            h = p * (1.0 - p)
            h = np.maximum(h, 1e-10)

            # Fit regularized tree on gradients and Hessians
            tree = XGBoostTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X, g, h)

            # Update predictions with learning rate shrinkage
            tree_pred = tree.predict(X)
            y_pred += self.lr * tree_pred

            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        raw_scores = np.full(X.shape[0], self.base_pred)

        for tree in self.trees:
            raw_scores += self.lr * tree.predict(X)

        prob_positive = self._sigmoid(raw_scores)
        return np.column_stack([1.0 - prob_positive, prob_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.5, self.classes_[1], self.classes_[0])


# =====================================================================
# DEMO & VISUAL PLOTS
# =====================================================================

def run_xgboost_scratch_demo():
    np.random.seed(42)

    # 1. High-Dimensional Dataset Benchmark
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_scratch = XGBoostClassifierScratch(n_estimators=30, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
    xgb_scratch.fit(X_train, y_train)

    acc = np.mean(xgb_scratch.predict(X_test) == y_test)

    print("=" * 65)
    print("DAY 11: SCRATCH XGBOOST BENCHMARK")
    print("=" * 65)
    print(f"Custom XGBoost Test Accuracy (30 Trees) : {acc:.4f}")
    print("=" * 65 + "\n")

    # 2. Fit 2D Dataset for Decision Boundary Visualisation
    X_2d, y_2d = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=42)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.2, random_state=42)

    xgb_2d = XGBoostClassifierScratch(n_estimators=40, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
    xgb_2d.fit(X_train_2d, y_train_2d)

    # 3. Create Plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("XGBoost from Scratch: Training Trajectory & Decision Surface", fontsize=14, fontweight="bold")

    # Plot 1: Log-Loss Convergence
    axes[0].plot(range(1, len(xgb_scratch.loss_history) + 1), xgb_scratch.loss_history, marker="o", color="teal", linewidth=2)
    axes[0].set_title("Training Loss Convergence (Log-Loss vs. Boosting Iteration)", fontsize=11)
    axes[0].set_xlabel("Boosting Round (Tree Index)")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: 2D Decision Boundary
    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = xgb_2d.predict(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    axes[1].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, cmap="coolwarm", edgecolors="k", label="Train Points", alpha=0.8)
    axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d, cmap="coolwarm", marker="x", s=60, label="Test Points", alpha=0.9)
    axes[1].set_title("2D XGBoost Decision Boundary Surface", fontsize=11)
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_xgboost_scratch_demo()