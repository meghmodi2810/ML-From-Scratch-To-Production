import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class DecisionStump:
    """Weak Learner: Single-level decision tree (depth=1)."""

    def __init__(self):
        self.polarity = 1          # Threshold direction (+1 or -1)
        self.feature_idx = None    # Index of feature to split on
        self.threshold = None      # Threshold value for split
        self.alpha = None          # Estimator voting weight in final ensemble

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(m)

        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AdaBoostClassifierScratch:
    """Adaptive Boosting (AdaBoost) Classifier built from scratch in pure NumPy.
    
    Sequential sample re-weighting with estimator confidence weighting (alpha).
    """

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifierScratch":
        X = np.array(X, dtype=np.float64)
        m, d = X.shape

        # Map binary labels to {-1, +1}
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "AdaBoost supports binary classification only."
        y_encoded = np.where(y == self.classes_[0], -1, 1)

        # 1. Initialize sample weights uniformly: w_i = 1 / m
        w = np.ones(m, dtype=np.float64) / m
        self.estimators = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")

            # Search across all features and unique thresholds
            for feat_idx in range(d):
                X_column = X[:, feat_idx]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(m)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1

                        # Calculate weighted classification error: epsilon = sum(w_i * I(y_i != h(x_i)))
                        error = np.sum(w[y_encoded != predictions])

                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_idx = feat_idx

            # Safeguard numerical division
            min_error = np.clip(min_error, 1e-10, 1.0 - 1e-10)

            # 2. Calculate voting weight: alpha = 1/2 * ln((1 - epsilon) / epsilon)
            alpha = 0.5 * np.log((1.0 - min_error) / min_error)
            stump.alpha = alpha

            # 3. Update sample weights: w_i = w_i * exp(-alpha * y_i * h(x_i))
            predictions = stump.predict(X)
            w *= np.exp(-alpha * y_encoded * predictions)

            # Normalize sample weights: sum(w_i) = 1
            w /= np.sum(w)

            self.estimators.append(stump)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculates raw confidence scores: f(x) = sum(alpha_t * h_t(x))."""
        X = np.array(X, dtype=np.float64)
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.estimators])
        return np.sum(stump_preds, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary target labels: sign(sum(alpha_t * h_t(x)))."""
        raw_scores = self.decision_function(X)
        return np.where(raw_scores >= 0, self.classes_[1], self.classes_[0])


# =====================================================================
# DEMO & SEQUENTIAL BOUNDARY EVOLUTION PLOTS
# =====================================================================

def run_adaboost_demo():
    np.random.seed(42)

    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    y_binary = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    adaboost_1 = AdaBoostClassifierScratch(n_estimators=1).fit(X_train, y_train)
    adaboost_10 = AdaBoostClassifierScratch(n_estimators=10).fit(X_train, y_train)
    adaboost_50 = AdaBoostClassifierScratch(n_estimators=50).fit(X_train, y_train)

    acc_1 = np.mean(adaboost_1.predict(X_test) == y_test)
    acc_10 = np.mean(adaboost_10.predict(X_test) == y_test)
    acc_50 = np.mean(adaboost_50.predict(X_test) == y_test)

    print("=" * 65)
    print("DAY 11: ADABOOST FROM SCRATCH BENCHMARKS")
    print("=" * 65)
    print(f"AdaBoost (1 Stump) Accuracy    : {acc_1:.4f}")
    print(f"AdaBoost (10 Stumps) Accuracy  : {acc_10:.4f}")
    print(f"AdaBoost (50 Stumps) Accuracy  : {acc_50:.4f}")
    print("=" * 65 + "\n")

    # Generate 2D Grid Mesh for Plotting Decision Boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AdaBoost Decision Boundary Evolution", fontsize=14, fontweight="bold")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    models = [
        ("1 Estimator (Single Stump)", adaboost_1, acc_1),
        ("10 Estimators", adaboost_10, acc_10),
        ("50 Estimators", adaboost_50, acc_50),
    ]

    for idx, (title, model, acc) in enumerate(models):
        # Obtain predictions for the mesh grid
        Z = model.predict(grid).reshape(xx.shape)
        
        # Plot decision regions
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        
        # Scatter plot dataset points
        axes[idx].scatter(
            X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            color="red", label="Class -1 (Train)", edgecolors="k", alpha=0.7
        )
        axes[idx].scatter(
            X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            color="blue", label="Class +1 (Train)", edgecolors="k", alpha=0.7
        )
        axes[idx].scatter(
            X_test[:, 0], X_test[:, 1],
            c=y_test, cmap="coolwarm", marker="x", s=50, label="Test Points", alpha=0.9
        )
        
        axes[idx].set_title(f"{title}\nTest Acc: {acc:.2f}")
        axes[idx].set_xlabel("x₁")
        axes[idx].set_ylabel("x₂")
        axes[idx].legend(loc="upper right")
        axes[idx].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_adaboost_demo()