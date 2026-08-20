import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. PREPROCESSING MODULES (FROM SCRATCH)
# =====================================================================

class StandardScalerScratch:
    """Standardizes features by centering mean to 0 and scaling variance to 1 (Z-score)."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        """Calculates column-wise mean and standard deviation.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            StandardScalerScratch: Fitted scaler instance.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevent division by zero for constant feature columns
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms X using fitted training mean and std.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Standardized feature matrix (m, d).
        """
        assert self.mean is not None, "Scaler must be fitted before calling transform()."
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits scaler parameters and returns transformed matrix."""
        return self.fit(X).transform(X)


class MinMaxScalerScratch:
    """Transforms feature values strictly into the bounded range [0, 1]."""

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: np.ndarray) -> "MinMaxScalerScratch":
        """Calculates column-wise min and max values.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            MinMaxScalerScratch: Fitted scaler instance.
        """
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms X to [0, 1] range using stored min/max bounds."""
        assert self.min is not None, "Scaler must be fitted before calling transform()."
        range_val = self.max - self.min
        range_val[range_val == 0.0] = 1.0  # Avoid zero-division
        return (X - self.min) / range_val

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits scaler parameters and returns transformed matrix."""
        return self.fit(X).transform(X)


class PolynomialFeaturesScratch:
    """Generates polynomial features up to specified degree for 1D/2D arrays."""

    def __init__(self, degree: int = 2):
        """Initializes generator with desired degree."""
        self.degree = degree

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Expands input feature array X into polynomial matrix [X, X^2, ..., X^degree].

        Args:
            X (np.ndarray): Input feature array of shape (m, 1) or (m,).

        Returns:
            np.ndarray: Polynomial feature matrix of shape (m, degree).
        """
        X_flat = X.squeeze()
        poly_cols = [X_flat ** d for d in range(1, self.degree + 1)]
        return np.column_stack(poly_cols)


# =====================================================================
# 2. MULTIVARIATE & REGULARIZED REGRESSION MODEL (FROM SCRATCH)
# =====================================================================

class RegularizedLinearRegressionScratch:
    """Multivariate Linear Regression model supporting L1 (Lasso), L2 (Ridge), 
    and ElasticNet regularization using pure NumPy Gradient Descent.
    
    Hypothesis:
        h_theta(X) = X_b @ theta
        
    Cost Function:
        J(theta) = MSE(theta) + L1_penalty + L2_penalty
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
        alpha_reg: float = 0.1
    ):
        """Initializes model hyperparameters.

        Args:
            learning_rate (float): Step size alpha for gradient descent updates.
            epochs (int): Number of optimization passes through dataset.
            l1_ratio (float): L1 regularization multiplier (1.0 = Pure Lasso).
            l2_ratio (float): L2 regularization multiplier (1.0 = Pure Ridge).
            alpha_reg (float): Regularization strength lambda.
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.alpha_reg = alpha_reg
        self.theta = None
        self.cost_history = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Prepends a column of ones for dummy bias feature x_0 = 1."""
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]

    def compute_cost(self, X_b: np.ndarray, y: np.ndarray) -> float:
        """Computes regularized MSE cost value J(theta)."""
        m = len(y)
        predictions = X_b @ self.theta
        mse_cost = (1.0 / (2.0 * m)) * np.sum((predictions - y) ** 2)

        # Do NOT regularize bias parameter theta[0]
        weights = self.theta[1:]
        l1_penalty = self.l1_ratio * (self.alpha_reg / m) * np.sum(np.abs(weights))
        l2_penalty = self.l2_ratio * (self.alpha_reg / (2.0 * m)) * np.sum(weights ** 2)

        return float(mse_cost + l1_penalty + l2_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedLinearRegressionScratch":
        """Fits parameters theta using regularized batch gradient descent."""
        m, d = X.shape
        X_b = self._add_bias(X)
        self.theta = np.zeros(d + 1)
        self.cost_history = []

        for _ in range(self.epochs):
            predictions = X_b @ self.theta
            residuals = predictions - y

            # 1. Base MSE Gradient: (1/m) * X_b^T * (y_hat - y)
            grad = (1.0 / m) * (X_b.T @ residuals)

            # 2. Compute Regularization Gradients (zero out bias penalty)
            weights = self.theta.copy()
            weights[0] = 0.0

            l1_grad = self.l1_ratio * (self.alpha_reg / m) * np.sign(weights)
            l2_grad = self.l2_ratio * (self.alpha_reg / m) * weights

            # 3. Update Parameters: theta := theta - alpha * total_gradient
            total_grad = grad + l1_grad + l2_grad
            self.theta -= self.lr * total_grad

            cost = self.compute_cost(X_b, y)
            self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes predictions for given feature matrix X."""
        assert self.theta is not None, "Model must be fitted before predict()."
        X_b = self._add_bias(X)
        return X_b @ self.theta


# =====================================================================
# 3. EXPERIMENTATION & VISUALIZATION PIPELINE
# =====================================================================

def run_day03_experiments():
    """Executes experiments for scaling, polynomial fitting, and regularization."""
    np.random.seed(42)
    m = 35

    # Generate synthetic non-linear data: y = 0.5*x^2 - x + 2 + Gaussian noise
    X_raw = np.sort(np.random.uniform(-3, 3, m)).reshape(-1, 1)
    true_y = 0.5 * (X_raw.squeeze() ** 2) - X_raw.squeeze() + 2.0
    y_noisy = true_y + np.random.normal(0, 1.0, size=m)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Day 3: Multivariate, Polynomials, & Regularization Dynamics", fontsize=14, fontweight='bold')

    # -------------------------------------------------------------
    # Plot 1: Unregularized vs Regularized High-Degree Polynomials
    # -------------------------------------------------------------
    X_plot = np.linspace(-3.2, 3.2, 200).reshape(-1, 1)

    # Transform data to Degree-8 Polynomial
    poly_deg8 = PolynomialFeaturesScratch(degree=8)
    X_p8 = poly_deg8.fit_transform(X_raw)
    X_plot_p8 = poly_deg8.fit_transform(X_plot)

    # Scale polynomial features
    scaler = StandardScalerScratch()
    X_p8_scaled = scaler.fit_transform(X_p8)
    X_plot_scaled = scaler.transform(X_plot_p8)

    # Fit Unregularized Model (High Variance / Overfitting)
    unreg_model = RegularizedLinearRegressionScratch(learning_rate=0.01, epochs=3000, alpha_reg=0.0)
    unreg_model.fit(X_p8_scaled, y_noisy)

    # Fit Ridge Model (L2 Regularization)
    ridge_model = RegularizedLinearRegressionScratch(learning_rate=0.01, epochs=3000, l2_ratio=1.0, alpha_reg=5.0)
    ridge_model.fit(X_p8_scaled, y_noisy)

    # Fit Lasso Model (L1 Regularization)
    lasso_model = RegularizedLinearRegressionScratch(learning_rate=0.01, epochs=3000, l1_ratio=1.0, alpha_reg=2.0)
    lasso_model.fit(X_p8_scaled, y_noisy)

    # Plot predictions
    axes[0].scatter(X_raw, y_noisy, color="black", alpha=0.7, label="Noisy Training Data")
    axes[0].plot(X_plot, unreg_model.predict(X_plot_scaled), "r--", linewidth=2, label="Degree 8 Unregularized (Overfit)")
    axes[0].plot(X_plot, ridge_model.predict(X_plot_scaled), "g-", linewidth=2.5, label="Degree 8 + Ridge (L2)")
    axes[0].plot(X_plot, lasso_model.predict(X_plot_scaled), "b-.", linewidth=2.5, label="Degree 8 + Lasso (L1)")

    axes[0].set_title("Overfitting vs Regularized Curve Fitting")
    axes[0].set_xlabel("Feature X")
    axes[0].set_ylabel("Target y")
    axes[0].set_ylim(-2, 10)
    axes[0].legend(loc="upper center", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Coefficient Shrinkage Curves (Ridge vs Lasso)
    # -------------------------------------------------------------
    alphas = np.logspace(-3, 2, 40)
    ridge_coeffs = []
    lasso_coeffs = []

    for a in alphas:
        r = RegularizedLinearRegressionScratch(learning_rate=0.01, epochs=2000, l2_ratio=1.0, alpha_reg=a)
        r.fit(X_p8_scaled, y_noisy)
        ridge_coeffs.append(r.theta[1:].copy())

        l = RegularizedLinearRegressionScratch(learning_rate=0.01, epochs=2000, l1_ratio=1.0, alpha_reg=a)
        l.fit(X_p8_scaled, y_noisy)
        lasso_coeffs.append(l.theta[1:].copy())

    ridge_coeffs = np.array(ridge_coeffs)
    lasso_coeffs = np.array(lasso_coeffs)

    # Plot Ridge weight decay
    for feat_idx in range(ridge_coeffs.shape[1]):
        axes[1].plot(alphas, ridge_coeffs[:, feat_idx], label=f"θ_{feat_idx+1}" if feat_idx < 3 else "")

    axes[1].set_xscale("log")
    axes[1].set_title("Ridge (L2) Weight Shrinkage vs Regularization Lambda (α)")
    axes[1].set_xlabel("Lambda α (Log Scale)")
    axes[1].set_ylabel("Coefficient Values (θ_j)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("DAY 3 COMPLETE: MULTIVARIATE, SCALING, POLYS, & REGULARIZATION")
    print("=" * 60)

    # Run quick scaling test
    X_test = np.array([[1000.0, 1.0], [3000.0, 4.0], [5000.0, 2.0]])
    scaler_std = StandardScalerScratch()
    print("\nStandardized Sample Matrix:\n", scaler_std.fit_transform(X_test))

    # Run full experiment pipeline
    run_day03_experiments()
