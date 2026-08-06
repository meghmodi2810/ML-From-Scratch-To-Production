import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split


class DecisionNode:
    """Helper class representing a single node in a Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, impurity_drop=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity_drop = impurity_drop


class DecisionTreeSubsampled:
    """Decision Tree Classifier supporting Random Forest feature subsampling (sqrt(d))."""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_importances: np.ndarray):
        m, d = X.shape
        if m < self.min_samples_split:
            return None, None, 0.0

        if self.max_features is None:
            feat_indices = np.arange(d)
        elif isinstance(self.max_features, str) and self.max_features == "sqrt":
            n_feats = max(1, int(np.sqrt(d)))
            feat_indices = np.random.choice(d, n_feats, replace=False)
        elif isinstance(self.max_features, int):
            feat_indices = np.random.choice(d, min(d, self.max_features), replace=False)
        else:
            feat_indices = np.arange(d)

        parent_gini = self._gini(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat in feat_indices:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / m
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain > 0 and best_feat is not None:
            feature_importances[best_feat] += best_gain * (m / float(len(y)))

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, feature_importances: np.ndarray):
        m, n_labels = X.shape[0], len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or m < self.min_samples_split:
            leaf_value = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_value)

        feat, thresh, gain = self._best_split(X, y, feature_importances)
        if feat is None:
            leaf_value = np.argmax(np.bincount(y))
            return DecisionNode(value=leaf_value)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_importances)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_importances)

        return DecisionNode(feature=feat, threshold=thresh, left=left_child, right=right_child, impurity_drop=gain)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_importances: np.ndarray):
        self.root = self._build_tree(X, y, depth=0, feature_importances=feature_importances)
        return self

    def _predict_sample(self, node: DecisionNode, x: np.ndarray) -> int:
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        return self._predict_sample(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(self.root, x) for x in X])


class RandomForestClassifierScratch:
    """Random Forest Classifier implemented from scratch using parallel Bootstrap Aggregation (Bagging)."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 10, min_samples_split: int = 2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierScratch":
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        m, d = X.shape
        self.trees = []
        self.feature_importances_ = np.zeros(d)

        oob_predictions = [[] for _ in range(m)]

        for b in range(self.n_estimators):
            boot_indices = np.random.choice(m, m, replace=True)
            oob_indices = np.setdiff1d(np.arange(m), boot_indices)

            X_boot, y_boot = X[boot_indices], y[boot_indices]

            tree = DecisionTreeSubsampled(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot, feature_importances=self.feature_importances_)
            self.trees.append(tree)

            if len(oob_indices) > 0:
                preds_oob = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, preds_oob):
                    oob_predictions[idx].append(pred)

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        oob_correct = 0
        oob_total = 0
        for idx in range(m):
            if len(oob_predictions[idx]) > 0:
                majority_vote = np.bincount(oob_predictions[idx]).argmax()
                if majority_vote == y[idx]:
                    oob_correct += 1
                oob_total += 1

        self.oob_score_ = oob_correct / float(oob_total) if oob_total > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)


# =====================================================================
# LEARN-IN-PUBLIC VISUAL DASHBOARD GENERATOR
# =====================================================================

def run_day10_visual_experiments():
    """Generates a 4-panel visual dashboard comparing Single Trees vs Random Forests."""
    np.random.seed(42)

    # Dataset 1: High Dimensional for Feature Importance & Variance Sweeps
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6, n_redundant=2, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Track Variance Reduction vs Tree Count
    tree_counts = [1, 3, 5, 10, 20, 30, 40, 50]
    scratch_accuracies = []
    
    for n in tree_counts:
        rf = RandomForestClassifierScratch(n_estimators=n, max_depth=8, max_features="sqrt").fit(X_train, y_train)
        acc = np.mean(rf.predict(X_test) == y_test)
        scratch_accuracies.append(acc)

    # Fit Full Model
    rf_scratch = RandomForestClassifierScratch(n_estimators=50, max_depth=10, max_features="sqrt").fit(X_train, y_train)
    rf_sklearn = RandomForestClassifier(n_estimators=50, max_depth=10, max_features="sqrt", oob_score=True, random_state=42).fit(X_train, y_train)

    # Dataset 2: 2D Synthetic Data for Boundary Plots
    X_2d, y_2d = make_blobs(n_samples=250, centers=2, cluster_std=2.2, random_state=42)
    single_tree_2d = DecisionTreeSubsampled(max_depth=10).fit(X_2d, y_2d, np.zeros(2))
    rf_2d = RandomForestClassifierScratch(n_estimators=50, max_depth=10, max_features="sqrt").fit(X_2d, y_2d)

    # -----------------------------------------------------------------
    # RENDER 4-PANEL DASHBOARD
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Day 10: Ensemble Learning & Random Forest Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Variance Reduction Curve (Accuracy vs Number of Trees)
    axes[0, 0].plot(tree_counts, scratch_accuracies, marker="o", linewidth=2.5, color="navy", label="Random Forest Accuracy")
    axes[0, 0].axhline(scratch_accuracies[0], color="crimson", linestyle="--", label=f"Single Tree (N=1): {scratch_accuracies[0]:.2f}")
    axes[0, 0].set_title("1. Variance Reduction: Accuracy vs Number of Trees (B)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Number of Estimators (B)")
    axes[0, 0].set_ylabel("Test Set Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: MDI Feature Importances (Scratch vs Sklearn)
    indices = np.arange(X.shape[1])
    axes[0, 1].bar(indices - 0.2, rf_scratch.feature_importances_, width=0.4, label="Scratch RF", color="navy")
    axes[0, 1].bar(indices + 0.2, rf_sklearn.feature_importances_, width=0.4, label="sklearn RF", color="teal")
    axes[0, 1].set_title("2. Mean Decrease in Impurity (MDI) Feature Importances", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Feature Index")
    axes[0, 1].set_ylabel("Normalized Gini Drop")
    axes[0, 1].set_xticks(indices)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Single Tree 2D Decision Boundary (High Variance)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z1 = single_tree_2d.predict(grid).reshape(xx.shape)
    axes[1, 0].contourf(xx, yy, Z1, alpha=0.3, cmap="coolwarm")
    axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap="coolwarm", edgecolors="k", s=35)
    axes[1, 0].set_title("3. Single Decision Tree Boundary (Overfit & Rigid)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("x₁")
    axes[1, 0].set_ylabel("x₂")

    # Panel 4: Random Forest 2D Decision Boundary (Smoothed & Low Variance)
    Z2 = rf_2d.predict(grid).reshape(xx.shape)
    axes[1, 1].contourf(xx, yy, Z2, alpha=0.3, cmap="coolwarm")
    axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap="coolwarm", edgecolors="k", s=35)
    axes[1, 1].set_title("4. Random Forest Boundary (Smooth & Robust)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("x₁")
    axes[1, 1].set_ylabel("x₂")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day10_visual_experiments()