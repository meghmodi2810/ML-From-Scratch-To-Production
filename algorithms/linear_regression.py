import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """Linear Regression implemented from first principles using pure NumPy.
    
    Supports Batch Gradient Descent, Stochastic Gradient Descent (SGD),
    Mini-Batch Gradient Descent, and the analytical Normal Equation.
    
    Hypothesis: 
        h_theta(x) = theta^T * x = w^T * x + b
        
    Mean Squared Error Cost Function:
        J(theta) = (1 / 2m) * sum_{i=1}^m (h_theta(x^(i)) - y^(i))^2
        
    Gradient Formula:
        dJ / d_theta_j = (1 / m) * sum_{i=1}^m (h_theta(x^(i)) - y^(i)) * x_j^(i)
        
    Parameter Update Rule:
        theta := theta - alpha * grad(J(theta))
        
    Normal Equation (Analytical Closed-Form Solution):
        theta = (X_b^T * X_b)^(-1) * X_b^T * y
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, method: str = "batch", batch_size: int = 32):
        """Initializes the Linear Regression model parameters.

        Args:
            learning_rate (float): Learning rate alpha for gradient descent updates. Defaults to 0.01.
            epochs (int): Number of optimization iterations. Defaults to 1000.
            method (str): Optimization method. Choices: 'batch', 'sgd', 'mini_batch', 'normal_eq'. Defaults to 'batch'.
            batch_size (int): Size of mini-batches when method='mini_batch'. Defaults to 32.
        """
        self.alpha = learning_rate
        self.epochs = epochs
        self.method = method.lower()
        self.batch_size = batch_size
        self.theta = None  # Parameter vector containing [bias, weight_1, weight_2, ..., weight_d]
        self.cost_history = []

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Prepends a column of ones to feature matrix X to represent the bias term (x_0 = 1).

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Augmented matrix X_b of shape (m, d + 1).
        """
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]

    def compute_cost(self, X_b: np.ndarray, y: np.ndarray) -> float:
        """Computes the Mean Squared Error (MSE) Cost J(theta).

        Formula:
            J(theta) = (1 / 2m) * ||X_b * theta - y||^2

        Args:
            X_b (np.ndarray): Augmented feature matrix of shape (m, d + 1).
            y (np.ndarray): Target vector of shape (m,).

        Returns:
            float: Evaluated scalar cost value J(theta).
        """
        m = len(y)
        predictions = X_b @ self.theta
        residuals = predictions - y
        cost = (1.0 / (2.0 * m)) * np.sum(residuals ** 2)
        return float(cost)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """Fits the linear model to training data using the specified optimization strategy.

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).
            y (np.ndarray): Target array of shape (m,).

        Returns:
            LinearRegressionScratch: Fitted model instance.
        """
        m, d = X.shape
        X_b = self._add_bias_term(X)
        self.cost_history = []

        if self.method == "normal_eq":
            # -------------------------------------------------------------
            # NORMAL EQUATION: theta = (X_b^T * X_b)^(-1) * X_b^T * y
            # Closed-form analytical solution due to convexity of MSE Loss
            # -------------------------------------------------------------
            self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            cost = self.compute_cost(X_b, y)
            self.cost_history.append(cost)
            print(f"[Normal Equation] Analytical solution computed | Cost J(theta): {cost:.6f}")
            return self

        # Initialize parameter vector theta to zeros [b, w_1, ..., w_d]
        self.theta = np.zeros(d + 1)

        # -------------------------------------------------------------
        # BATCH GRADIENT DESCENT (Uses full dataset per update)
        # -------------------------------------------------------------
        if self.method == "batch":
            for epoch in range(self.epochs):
                predictions = X_b @ self.theta
                residuals = predictions - y
                grad = (1.0 / m) * (X_b.T @ residuals)
                
                # Update rule: theta := theta - alpha * grad
                self.theta -= self.alpha * grad
                
                cost = self.compute_cost(X_b, y)
                self.cost_history.append(cost)

        # -------------------------------------------------------------
        # STOCHASTIC GRADIENT DESCENT (SGD - 1 random sample per update)
        # -------------------------------------------------------------
        elif self.method == "sgd":
            for epoch in range(self.epochs):
                # Shuffle data at each epoch
                indices = np.random.permutation(m)
                X_b_shuffled = X_b[indices]
                y_shuffled = y[indices]

                for i in range(m):
                    xi = X_b_shuffled[i:i+1]  # Shape (1, d+1)
                    yi = y_shuffled[i:i+1]    # Shape (1,)

                    prediction = xi @ self.theta
                    residual = prediction - yi
                    grad = xi.T @ residual  # Gradient for 1 sample (m=1)
                    
                    self.theta -= self.alpha * grad.squeeze()

                cost = self.compute_cost(X_b, y)
                self.cost_history.append(cost)

        # -------------------------------------------------------------
        # MINI-BATCH GRADIENT DESCENT (Batch size B per update)
        # -------------------------------------------------------------
        elif self.method == "mini_batch":
            for epoch in range(self.epochs):
                indices = np.random.permutation(m)
                X_b_shuffled = X_b[indices]
                y_shuffled = y[indices]

                for i in range(0, m, self.batch_size):
                    X_mini = X_b_shuffled[i:i + self.batch_size]
                    y_mini = y_shuffled[i:i + self.batch_size]
                    b_size = len(y_mini)

                    predictions = X_mini @ self.theta
                    residuals = predictions - y_mini
                    grad = (1.0 / b_size) * (X_mini.T @ residuals)

                    self.theta -= self.alpha * grad

                cost = self.compute_cost(X_b, y)
                self.cost_history.append(cost)

        else:
            raise ValueError(f"Unknown optimization method '{self.method}'. "
                             f"Choose from: 'batch', 'sgd', 'mini_batch', 'normal_eq'.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target outputs for given input features X using hypothesis h_theta(x).

        Args:
            X (np.ndarray): Feature matrix of shape (m, d).

        Returns:
            np.ndarray: Predicted values array of shape (m,).
        """
        assert self.theta is not None, "Model must be fitted before calling predict()."
        X_b = self._add_bias_term(X)
        return X_b @ self.theta


def plot_linear_regression_comparison():
    """Generates synthetic dataset, fits all 4 optimization variants, and plots results."""
    # 1. Generate Synthetic Data
    np.random.seed(42)
    m = 100
    true_w = 3.5
    true_b = 2.0

    X = np.linspace(0, 10, m).reshape(-1, 1)
    noise = np.random.normal(0, 1.5, size=(m, 1))
    y = (true_w * X + true_b + noise).squeeze()

    # 2. Train All Variants
    models = {
        "Batch GD": LinearRegressionScratch(learning_rate=0.01, epochs=300, method="batch"),
        "SGD": LinearRegressionScratch(learning_rate=0.005, epochs=300, method="sgd"),
        "Mini-Batch GD": LinearRegressionScratch(learning_rate=0.01, epochs=300, method="mini_batch", batch_size=16),
        "Normal Equation": LinearRegressionScratch(method="normal_eq")
    }

    print("=" * 60)
    print("TRAINING LINEAR REGRESSION OPTIMIZATION VARIANTS")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X, y)
        b_learned = model.theta[0]
        w_learned = model.theta[1]
        print(f"[{name:15s}] Learned theta: [bias={b_learned:.4f}, weight={w_learned:.4f}]")

    print("=" * 60 + "\n")

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Linear Regression Optimization & Convex Cost Analysis", fontsize=14, fontweight='bold')

    # Subplot 1: Convergence Trajectories J(theta) vs Epochs
    colors = {"Batch GD": "#1f77b4", "SGD": "#ff7f0e", "Mini-Batch GD": "#2ca02c"}
    for name, model in models.items():
        if name != "Normal Equation":
            axes[0].plot(range(len(model.cost_history)), model.cost_history, 
                         color=colors[name], label=name, linewidth=2, alpha=0.85)

    # Add Normal Equation benchmark horizontal line
    normal_cost = models["Normal Equation"].cost_history[0]
    axes[0].axhline(y=normal_cost, color='black', linestyle='--', linewidth=1.5, 
                    label=f'Normal Eq Benchmark (J={normal_cost:.4f})')

    axes[0].set_title("Cost Function Convergence J(θ) vs Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Cost J(θ)")
    axes[0].set_ylim(0, 15)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Fitted Lines vs True Data Distribution
    axes[1].scatter(X, y, color="#7f7f7f", alpha=0.6, edgecolors="k", label="Target Data (y)")
    
    line_styles = {"Batch GD": "-", "SGD": "--", "Mini-Batch GD": "-.", "Normal Equation": ":"}
    line_colors = {"Batch GD": "#1f77b4", "SGD": "#ff7f0e", "Mini-Batch GD": "#2ca02c", "Normal Equation": "#d62728"}

    for name, model in models.items():
        y_pred = model.predict(X)
        axes[1].plot(X, y_pred, label=f"{name} (b={model.theta[0]:.2f}, w={model.theta[1]:.2f})", 
                     linestyle=line_styles[name], color=line_colors[name], linewidth=2)

    axes[1].plot(X, true_w * X.squeeze() + true_b, color="black", linestyle="-", 
                 linewidth=1, label=f"True Generator (b={true_b}, w={true_w})")

    axes[1].set_title("Fitted Regression Lines Comparison")
    axes[1].set_xlabel("Feature X")
    axes[1].set_ylabel("Target y")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_linear_regression_comparison()