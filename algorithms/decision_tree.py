import numpy as np
import matplotlib.pyplot as plt


class Node:
    """Represents a single node in the Decision Tree binary tree structure."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of feature to split on (e.g., column 0)
        self.threshold = threshold      # Threshold value for split (e.g., x_0 <= 2.5)
        self.left = left                # Left child Node (samples <= threshold)
        self.right = right              # Right child Node (samples > threshold)
        self.value = value              # Predicted class label (ONLY populated for leaf nodes)

    def is_leaf_node(self) -> bool:
        """Returns True if the node is a terminal leaf node."""
        return self.value is not None


class DecisionTreeScratch:
    """Recursive Decision Tree Classifier built from scratch in pure NumPy.
    
    Supports both Entropy and Gini Impurity split criteria along with pre-pruning options.
    """

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: int = 10,
        min_samples_split: int = 2
    ):
        """Initializes decision tree hyperparameters.

        Args:
            criterion (str): Impurity metric ('entropy' or 'gini'). Defaults to 'entropy'.
            max_depth (int): Maximum depth allowed for the tree (pre-pruning). Defaults to 10.
            min_samples_split (int): Minimum samples required to attempt a split. Defaults to 2.
        """
        assert criterion in ["entropy", "gini"], "Criterion must be 'entropy' or 'gini'."
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculates Node Impurity (Entropy or Gini Impurity)."""
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]  # Filter out zeroes to avoid log2(0) errors

        if self.criterion == "entropy":
            # H(S) = -sum(p_i * log2(p_i))
            return float(-np.sum(ps * np.log2(ps)))
        else:
            # Gini(S) = 1 - sum(p_i^2)
            return float(1.0 - np.sum(ps ** 2))

    def _split(self, X_column: np.ndarray, split_thresh: float):
        """Splits data indices into left (<= thresh) and right (> thresh) branches."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        """Calculates Information Gain from splitting on a specific feature and threshold.
        
        Formula:
            IG(S, A) = H(S) - [(|S_left| / |S|) * H(S_left) + (|S_right| / |S|) * H(S_right)]
        """
        parent_impurity = self._calculate_impurity(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        # Weighted average impurity of child nodes
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        imp_l = self._calculate_impurity(y[left_idxs])
        imp_r = self._calculate_impurity(y[right_idxs])
        child_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r

        # Information Gain = Parent Impurity - Weighted Child Impurity
        return parent_impurity - child_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Iterates over all features and unique candidate thresholds to find maximum Information Gain."""
        best_gain = -1.0
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _most_common_label(self, y: np.ndarray) -> int:
        """Returns the majority class label in y."""
        return int(np.bincount(y).argmax())

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively builds the Decision Tree using CART algorithm."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Base Case: Pre-pruning stopping criteria for leaf creation
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)

        # Find optimal feature and threshold split
        best_feat, best_thresh = self._best_split(X, y)

        # If no split yields positive information gain, make current node a leaf
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        # Recursive step: partition dataset and build left and right subtrees
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeScratch":
        """Fits the Decision Tree classifier."""
        self.root = self._build_tree(X, y)
        return self

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """Recursively traverses tree for a single feature vector x until reaching a leaf."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for a sample matrix X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def print_tree(self, node: Node = None, depth: int = 0):
        """Prints a clean ASCII text representation of the decision tree structure."""
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{'  ' * depth}└─► Predict Class {node.value}")
            return

        print(f"{'  ' * depth}├─► [Feature x_{node.feature} <= {node.threshold:.3f}]")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)


# =====================================================================
# OVERFITTING DEMONSTRATION & VISUALIZATION PIPELINE
# =====================================================================

def run_day06_decision_tree_demo():
    """Builds two decision trees (pruned vs unconstrained) to demonstrate overfitting."""
    np.random.seed(42)
    m = 120

    # Generate non-linear concentric circle dataset
    r1 = np.random.normal(1.0, 0.2, m // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, m // 2)
    c0 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

    r2 = np.random.normal(2.5, 0.3, m // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, m // 2)
    c1 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

    X = np.vstack([c0, c1])
    y = np.hstack([np.zeros(m // 2), np.ones(m // 2)]).astype(int)

    # Model 1: Pruned Tree (Max Depth = 3)
    tree_pruned = DecisionTreeScratch(criterion="entropy", max_depth=3)
    tree_pruned.fit(X, y)

    # Model 2: Overfitted Deep Tree (Max Depth = 15)
    tree_deep = DecisionTreeScratch(criterion="entropy", max_depth=15, min_samples_split=2)
    tree_deep.fit(X, y)

    print("=" * 60)
    print("DAY 6: DECISION TREE STRUCTURE (MAX DEPTH = 3)")
    print("=" * 60)
    tree_pruned.print_tree()
    print("=" * 60 + "\n")

    # Plot Decision Boundaries
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Day 6: Decision Tree Splitting & Overfitting Analysis", fontsize=14, fontweight='bold')

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Subplot 1: Pruned Tree (Max Depth 3)
    Z1 = tree_pruned.predict(grid).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z1, alpha=0.3, cmap="RdBu")
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], color="red", label="Class 0", edgecolors="k")
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Class 1", edgecolors="k")
    axes[0].set_title("Pruned Tree (max_depth=3) — Smooth Generalization")
    axes[0].set_xlabel("Feature x₁")
    axes[0].set_ylabel("Feature x₂")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Overfitted Deep Tree (Max Depth 15)
    Z2 = tree_deep.predict(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z2, alpha=0.3, cmap="RdBu")
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], color="red", label="Class 0", edgecolors="k")
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Class 1", edgecolors="k")
    axes[1].set_title("Deep Unconstrained Tree (max_depth=15) — Overfitting")
    axes[1].set_xlabel("Feature x₁")
    axes[1].set_ylabel("Feature x₂")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day06_decision_tree_demo()