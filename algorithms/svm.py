import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split


class LinearSVMScratch:
    """Soft-Margin Linear Support Vector Machine implemented from scratch using SGD.
    
    Optimizes the primal objective:
        J(w, b) = 1/2 * ||w||^2 + C * sum( max(0, 1 - y_i * (w^T * x_i + b)) )
    """

    def __init__(self, C: float = 1.0, lr: float = 0.001, epochs: int = 1000):
        """Initializes SVM hyperparameters.

        Args:
            C (float): Regularization parameter balancing margin maximization and hinge loss penalty.
            lr (float): Learning rate for Stochastic Gradient Descent.
            epochs (int): Number of complete passes over the training dataset.
        """
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        """Fits the SVM model weights (w) and bias (b) using Stochastic Gradient Descent."""
        X = np.array(X, dtype=np.float64)
        m, d = X.shape

        # Map labels to {-1, +1}
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "LinearSVMScratch supports binary classification only."
        y_encoded = np.where(y == self.classes_[0], -1, 1)

        # Initialize parameters
        self.w = np.zeros(d)
        self.b = 0.0

        # Optimization loop via SGD
        for epoch in range(self.epochs):
            for i in range(m):
                # Margin condition check: y_i * (w^T * x_i + b) >= 1
                functional_margin = y_encoded[i] * (np.dot(X[i], self.w) + self.b)

                if functional_margin >= 1.0:
                    # Case 1: Point is safely on/outside the margin boundary
                    grad_w = (1.0 / self.C) * self.w
                    grad_b = 0.0
                else:
                    # Case 2: Point violates the margin or is misclassified
                    grad_w = (1.0 / self.C) * self.w - y_encoded[i] * X[i]
                    grad_b = -y_encoded[i]

                # Update weights and bias
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

        # Extract Support Vectors: Points satisfying y_i * f(x_i) <= 1.0 + tolerance
        margins = y_encoded * (np.dot(X, self.w) + self.b)
        self.support_vector_indices_ = np.where(margins <= 1.001)[0]
        self.support_vectors_ = X[self.support_vector_indices_]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculates signed Euclidean distance/confidence score from hyperplane: f(x) = w^T * x + b."""
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary target class labels."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimates class probabilities using Platt-style Sigmoidal transformation of decision scores."""
        scores = self.decision_function(X)
        prob_positive = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - prob_positive, prob_positive])

    @property
    def margin_width(self) -> float:
        """Calculates total geometric margin width: M = 2 / ||w||_2."""
        norm_w = np.linalg.norm(self.w)
        return 2.0 / norm_w if norm_w > 0 else 0.0


# =====================================================================
# DEMO 1: LINEAR SVM & SKLEARN VALIDATION
# =====================================================================

def run_day09_linear_svm_demo():
    """Validates custom Linear SVM against scikit-learn's SVC on synthetic data."""
    np.random.seed(42)

    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=42)
    y_binary = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    svm_scratch = LinearSVMScratch(C=1.0, lr=0.001, epochs=2500)
    svm_scratch.fit(X_train, y_train)

    preds_scratch = svm_scratch.predict(X_test)
    probs_scratch = svm_scratch.predict_proba(X_test)
    acc_scratch = np.mean(preds_scratch == y_test)

    svm_sklearn = SVC(kernel="linear", C=1.0, probability=True)
    svm_sklearn.fit(X_train, y_train)

    preds_sklearn = svm_sklearn.predict(X_test)
    acc_sklearn = np.mean(preds_sklearn == y_test)

    print("=" * 60)
    print("DAY 9: COMPLETE SVM FROM SCRATCH BENCHMARKS")
    print("=" * 60)
    print(f"Scratch Accuracy             : {acc_scratch:.4f}")
    print(f"sklearn SVC Accuracy         : {acc_sklearn:.4f}")
    print(f"Learned Weights (w)          : {svm_scratch.w}")
    print(f"Learned Bias (b)             : {svm_scratch.b:.4f}")
    print(f"Geometric Margin Width (M)   : {svm_scratch.margin_width:.4f} units")
    print(f"Number of Support Vectors    : {len(svm_scratch.support_vectors_)}")
    print(f"Sample Probabilities (First 3):")
    for i in range(3):
        print(f"  Sample {i+1}: Class -1 = {probs_scratch[i, 0]:.3f}, Class +1 = {probs_scratch[i, 1]:.3f}")
    print("=" * 60 + "\n")


# =====================================================================
# DEMO 2: KERNEL TRICK VISUALIZATION (2D NON-LINEAR TO 3D LINEAR HYPERPLANE)
# =====================================================================

def plot_kernel_trick_3d_demo():
    """Demonstrates how a 2D non-linearly separable dataset becomes linearly separable 
    when projected into 3D feature space (z = x1^2 + x2^2).
    """
    np.random.seed(42)

    # 1. Generate Non-Linearly Separable Concentric Circles
    X_2d, y = make_circles(n_samples=300, factor=0.3, noise=0.08, random_state=42)
    y_encoded = np.where(y == 0, -1, 1)

    # 2. Kernel Transformation: Project into 3D Space via phi(x1, x2) = [x1, x2, x1^2 + x2^2]
    z = (X_2d[:, 0] ** 2) + (X_2d[:, 1] ** 2)
    X_3d = np.column_stack([X_2d, z])

    # 3. Fit Linear SVM directly on 3D Transformed Feature Space
    svm_3d = LinearSVMScratch(C=10.0, lr=0.001, epochs=3000)
    svm_3d.fit(X_3d, y_encoded)

    # 4. Plot 2D vs 3D Figures Side-by-Side
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Day 9: Kernel Trick Transformation (2D Non-Linear to 3D Linear Hyperplane)", fontsize=14, fontweight="bold")

    # Plot 1: Original 2D Space (Non-Linearly Separable)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], color="red", label="Class -1 (Outer)", edgecolors="k", alpha=0.8)
    ax1.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], color="green", label="Class +1 (Inner)", edgecolors="k", alpha=0.8)
    ax1.set_title("Original 2D Input Space\n(No straight line can separate these!)")
    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: 3D Feature Space Projection & Separating Hyperplane
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        X_3d[y == 0, 0], X_3d[y == 0, 1], X_3d[y == 0, 2],
        color="red", label="Class -1", edgecolors="k", alpha=0.8, s=30
    )
    ax2.scatter(
        X_3d[y == 1, 0], X_3d[y == 1, 1], X_3d[y == 1, 2],
        color="green", label="Class +1", edgecolors="k", alpha=0.8, s=30
    )

    # Generate 3D Decision Hyperplane Meshgrid: w1*x1 + w2*x2 + w3*z + b = 0 => z = (-w1*x1 - w2*x2 - b) / w3
    x1_grid = np.linspace(-1.2, 1.2, 30)
    x2_grid = np.linspace(-1.2, 1.2, 30)
    XX1, XX2 = np.meshgrid(x1_grid, x2_grid)

    w1, w2, w3 = svm_3d.w
    b = svm_3d.b

    # Hyperplane equation for z
    ZZ = (-w1 * XX1 - w2 * XX2 - b) / (w3 + 1e-9)

    # Surface plot for 3D separating plane
    ax2.plot_surface(XX1, XX2, ZZ, alpha=0.4, color="cyan", edgecolor="navy", linewidth=0.5)

    ax2.set_title("Transformed 3D Feature Space ϕ(x)\nLinear Hyperplane (Cyan) Separates Classes!")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_zlabel("z = x₁² + x₂²")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day09_linear_svm_demo()
    plot_kernel_trick_3d_demo()