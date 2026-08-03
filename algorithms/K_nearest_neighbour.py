import numpy as np
import matplotlib.pyplot as plt


class KNNScratch:
    """K-Nearest Neighbors Classifier implemented from scratch in pure NumPy.
    
    Supports Euclidean, Manhattan, Minkowski, and Cosine distance metrics.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", p: float = 3.0):
        """Initializes KNN parameters.

        Args:
            k (int): Number of nearest neighbors to consider. Defaults to 5.
            metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine'). Defaults to 'euclidean'.
            p (float): Norm degree for Minkowski distance when metric='minkowski'. Defaults to 3.0.
        """
        assert metric in ["euclidean", "manhattan", "minkowski", "cosine"], f"Unsupported metric '{metric}'."
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        """Lazy learning fit: stores training dataset in memory."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        return self

    def _compute_distances(self, x_query: np.ndarray) -> np.ndarray:
        """Computes distances between a single query sample and all training samples."""
        if self.metric == "euclidean":
            # sqrt( sum((x - z)^2) )
            return np.sqrt(np.sum((self.X_train - x_query) ** 2, axis=1))

        elif self.metric == "manhattan":
            # sum( |x - z| )
            return np.sum(np.abs(self.X_train - x_query), axis=1)

        elif self.metric == "minkowski":
            # ( sum( |x - z|^p ) )^(1/p)
            return np.sum(np.abs(self.X_train - x_query) ** self.p, axis=1) ** (1.0 / self.p)

        elif self.metric == "cosine":
            # 1 - (x . z) / (||x|| * ||z||)
            norm_query = np.linalg.norm(x_query)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            # Avoid division by zero
            denom = norm_query * norm_train
            denom = np.where(denom == 0, 1e-15, denom)
            dot_product = np.dot(self.X_train, x_query)
            return 1.0 - (dot_product / denom)

    def _predict_single(self, x_query: np.ndarray) -> int:
        """Finds k-nearest neighbors and returns majority class vote."""
        distances = self._compute_distances(x_query)

        # Get indices of the k smallest distances
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]

        # Majority vote
        counts = np.bincount(k_nearest_labels)
        return int(np.argmax(counts))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for query matrix X."""
        X_query = np.array(X, dtype=np.float64)
        return np.array([self._predict_single(x) for x in X_query])


# =====================================================================
# DEMO, ACCURACY VS K, & CURSE OF DIMENSIONALITY EXPERIMENTS
# =====================================================================

def run_day07_knn_experiments():
    """Runs KNN benchmarks: Accuracy vs k curve and Curse of Dimensionality degradation test."""
    np.random.seed(42)

    # 1. Dataset Generation: 2D Synthetic Binary Dataset
    m = 200
    c0 = np.random.multivariate_normal([-1, -1], [[1.0, 0.4], [0.4, 1.0]], m // 2)
    c1 = np.random.multivariate_normal([1.5, 1.5], [[1.0, -0.3], [-0.3, 1.0]], m // 2)

    X_base = np.vstack([c0, c1])
    y = np.hstack([np.zeros(m // 2), np.ones(m // 2)]).astype(int)

    # Train/Test Split (80/20)
    indices = np.random.permutation(m)
    train_size = int(m * 0.8)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train_base, y_train = X_base[train_idx], y[train_idx]
    X_test_base, y_test = X_base[test_idx], y[test_idx]

    # --- EXPERIMENT 1: Accuracy vs k Curve ---
    k_values = list(range(1, 31, 2))  # Odd numbers to prevent tie votes
    accuracies = []

    for k in k_values:
        knn = KNNScratch(k=k, metric="euclidean")
        knn.fit(X_train_base, y_train)
        preds = knn.predict(X_test_base)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)

    # --- EXPERIMENT 2: Curse of Dimensionality ---
    # Append uninformative random Gaussian noise dimensions to X_base
    noise_dimensions = [0, 5, 20, 50, 100, 200]
    curse_accuracies = []

    for num_noise in noise_dimensions:
        if num_noise == 0:
            X_noisy = X_base
        else:
            noise = np.random.normal(0, 1, size=(m, num_noise))
            X_noisy = np.hstack([X_base, noise])

        X_tr_noisy, X_te_noisy = X_noisy[train_idx], X_noisy[test_idx]

        knn = KNNScratch(k=5, metric="euclidean")
        knn.fit(X_tr_noisy, y_train)
        preds = knn.predict(X_te_noisy)
        acc = np.mean(preds == y_test)
        curse_accuracies.append(acc)

    print("=" * 60)
    print("DAY 7: K-NEAREST NEIGHBORS (KNN) BENCHMARKS")
    print("=" * 60)
    print(f"Optimal k found         : {k_values[np.argmax(accuracies)]} (Accuracy: {max(accuracies):.4f})")
    print("-" * 60)
    print("Curse of Dimensionality Impact (k=5):")
    for dims, acc in zip(noise_dimensions, curse_accuracies):
        print(f"  Total Features: {dims + 2:3d} (Noise Features: {dims:3d}) -> Test Accuracy: {acc:.4f}")
    print("=" * 60 + "\n")

    # Plot Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Day 7: KNN Hyperparameter Tuning & Curse of Dimensionality", fontsize=14, fontweight='bold')

    # Plot 1: Accuracy vs k Curve
    axes[0].plot(k_values, accuracies, color="#1f77b4", marker="o", linewidth=2.5)
    axes[0].set_title("Test Accuracy vs. Hyperparameter k")
    axes[0].set_xlabel("Number of Neighbors (k)")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Curse of Dimensionality
    axes[1].plot([d + 2 for d in noise_dimensions], curse_accuracies, color="#d62728", marker="s", linewidth=2.5)
    axes[1].set_title("Curse of Dimensionality: Performance Degradation")
    axes[1].set_xlabel("Total Feature Dimensions (d)")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day07_knn_experiments()