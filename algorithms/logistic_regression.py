import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionScratch:
    """Binary Logistic Regression implemented from scratch in pure NumPy.
    
    Supports L2 Regularization (Ridge) and numerical stabilization techniques.
    
    Hypothesis:
        h_theta(x) = sigmoid(X_b @ theta) = 1 / (1 + exp(-X_b @ theta))
        
    Cost Function (Binary Cross-Entropy / Log Loss):
        J(theta) = -(1/m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat)) + L2_penalty
        
    Vectorized Gradient:
        grad = (1/m) * X_b^T @ (y_hat - y) + (alpha_reg / m) * theta_weights
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1000,
        threshold: float = 0.5,
        l2_ratio: float = 0.0,
        alpha_reg: float = 0.0
    ):
        """Initializes Logistic Regression hyperparameters.

        Args:
            learning_rate (float): Step size alpha for gradient descent updates. Defaults to 0.05.
            epochs (int): Total optimization iterations over dataset. Defaults to 1000.
            threshold (float): Decision probability cutoff for class 1 prediction. Defaults to 0.5.
            l2_ratio (float): L2 regularization toggle multiplier (0.0 or 1.0). Defaults to 0.0.
            alpha_reg (float): Regularization strength parameter lambda. Defaults to 0.0.
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.l2_ratio = l2_ratio
        self.alpha_reg = alpha_reg
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the Sigmoid activation function with overflow protection.

        Formula:
            sigma(z) = 1 / (1 + exp(-z))
        """
        # Clip z to range [-500, 500] to prevent math overflow in np.exp()
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Prepends a column of ones for dummy bias feature x_0 = 1.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Augmented matrix X_b of shape (m, d + 1).
        """
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]

    def compute_cost(self, X_b: np.ndarray, y: np.ndarray) -> float:
        """Computes Binary Cross-Entropy (Log Loss) with clipping stabilization.

        Args:
            X_b (np.ndarray): Augmented feature matrix of shape (m, d + 1).
            y (np.ndarray): Target binary vector of shape (m,).

        Returns:
            float: Scalar cost value J(theta).
        """
        m = len(y)
        y_hat = self.sigmoid(X_b @ self.theta)

        # Clip predictions to range [1e-15, 1 - 1e-15] to prevent log(0) resulting in NaN
        eps = 1e-15
        y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)

        # Binary Cross-Entropy Loss
        bce_loss = -(1.0 / m) * np.sum(
            y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped)
        )

        # L2 Regularization penalty (strictly excluding bias parameter theta[0])
        weights = self.theta[1:]
        l2_penalty = self.l2_ratio * (self.alpha_reg / (2.0 * m)) * np.sum(weights ** 2)

        return float(bce_loss + l2_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Fits optimal parameters theta using Gradient Descent on Log Loss.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).
            y (np.ndarray): Binary labels array of shape (m,).

        Returns:
            LogisticRegressionScratch: Fitted model instance.
        """
        m, d = X.shape
        X_b = self._add_bias(X)
        self.theta = np.zeros(d + 1)
        self.cost_history = []

        for _ in range(self.epochs):
            # Forward Pass: Compute predictions y_hat via Sigmoid
            z = X_b @ self.theta
            y_hat = self.sigmoid(z)

            # Residual error (y_hat - y)
            residuals = y_hat - y

            # Base gradient: (1/m) * X_b^T @ (y_hat - y)
            grad = (1.0 / m) * (X_b.T @ residuals)

            # L2 Regularization Gradient (zero out bias penalty)
            weights = self.theta.copy()
            weights[0] = 0.0
            l2_grad = self.l2_ratio * (self.alpha_reg / m) * weights

            # Parameter update step: theta := theta - alpha * total_gradient
            self.theta -= self.lr * (grad + l2_grad)

            # Record cost
            cost = self.compute_cost(X_b, y)
            self.cost_history.append(cost)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability P(y=1|X) in range [0, 1].

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Predicted probabilities of shape (m,).
        """
        assert self.theta is not None, "Model must be fitted before predict_proba()."
        X_b = self._add_bias(X)
        return self.sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts discrete binary labels (0 or 1) using decision threshold.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Binary predictions of shape (m,).
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)


# =====================================================================
# DEMONSTRATION & 2D DECISION BOUNDARY PLOTTING
# =====================================================================

def run_logistic_regression_demo():
    """Generates synthetic 2D data, trains Logistic Regression, and plots results."""
    np.random.seed(42)
    m = 100

    # 1. Generate 2D synthetic dataset with two Gaussian clusters
    cluster_0 = np.random.multivariate_normal(
        mean=[-1.5, -1.5], cov=[[1, 0.4], [0.4, 1]], size=m // 2
    )
    cluster_1 = np.random.multivariate_normal(
        mean=[1.5, 1.5], cov=[[1, -0.2], [-0.2, 1]], size=m // 2
    )

    X = np.vstack([cluster_0, cluster_1])
    y = np.hstack([np.zeros(m // 2), np.ones(m // 2)])

    # 2. Fit model
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
    model.fit(X, y)

    print("=" * 60)
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)
    print(f"Learned Bias  (theta_0): {model.theta[0]:.4f}")
    print(f"Learned Weight 1 (theta_1): {model.theta[1]:.4f}")
    print(f"Learned Weight 2 (theta_2): {model.theta[2]:.4f}")
    print(f"Final Log Loss J(theta)  : {model.cost_history[-1]:.6f}")
    print("=" * 60 + "\n")

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Day 4: Logistic Regression & Decision Boundary Dynamics", fontsize=14, fontweight='bold')

    # Subplot 1: Convergence trajectory (Log Loss vs Epochs)
    axes[0].plot(range(len(model.cost_history)), model.cost_history, color="#d62728", linewidth=2)
    axes[0].set_title("Binary Cross-Entropy Loss (Log Loss) vs Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Log Loss J(θ)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: 2D Dataset and Linear Decision Boundary
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], color="#1f77b4", label="Class 0 (y=0)", edgecolors="k", alpha=0.8)
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], color="#ff7f0e", label="Class 1 (y=1)", edgecolors="k", alpha=0.8)

    # Decision Boundary Line Equation: theta_0 + theta_1*x1 + theta_2*x2 = 0
    # Rearranging for x2: x2 = -(theta_0 + theta_1 * x1) / theta_2
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_vals = np.array([x1_min, x1_max])
    theta_0, theta_1, theta_2 = model.theta
    x2_vals = -(theta_0 + theta_1 * x1_vals) / theta_2

    axes[1].plot(x1_vals, x2_vals, color="black", linestyle="--", linewidth=2.5, 
                 label=f"Decision Boundary\n({theta_0:.2f} + {theta_1:.2f}x₁ + {theta_2:.2f}x₂ = 0)")

    axes[1].set_title("2D Classification Decision Boundary (p = 0.5)")
    axes[1].set_xlabel("Feature x₁")
    axes[1].set_ylabel("Feature x₂")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_logistic_regression_demo()