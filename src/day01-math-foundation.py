from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def loss_function(w: np.ndarray) -> float:
    """Computes a 2D scalar loss function for housing/model error.

    Args:
        w (np.ndarray): A 1D array of shape (2,) containing weights [w1, w2].

    Returns:
        float: The calculated scalar loss value L(w1, w2) = 2*w1^2 + 3*w2^2 - 4*w1.
    """
    return float(2 * (w[0] ** 2) + 3 * (w[1] ** 2) - 4 * w[0])


def compute_gradient(w: np.ndarray) -> np.ndarray:
    """Computes the analytical gradient vector of the loss function.

    Args:
        w (np.ndarray): A 1D array of shape (2,) representing current weights [w1, w2].

    Returns:
        np.ndarray: A 1D array of shape (2,) containing partial derivatives [dL/dw1, dL/dw2].
    """
    grad_w1 = 4 * w[0] - 4
    grad_w2 = 6 * w[1]
    return np.array([grad_w1, grad_w2])


def plot_math_foundations(current_w: np.ndarray, updated_w: np.ndarray, X: np.ndarray, y_hat: np.ndarray, weights_vec: np.ndarray, mu: float, sigma: float):
    """Generates visual plots for all 5 mathematical concepts taught on Day 1.

    Args:
        current_w (np.ndarray): Initial weights before optimization step.
        updated_w (np.ndarray): Weights after applying a single gradient descent step.
        X (np.ndarray): Feature matrix of shape (2, 2) containing distance and passengers.
        y_hat (np.ndarray): Predicted taxi fares for each sample.
        weights_vec (np.ndarray): Weight vector used for L1 and L2 norm calculations.
        mu (float): Mean of the Gaussian probability density function.
        sigma (float): Standard deviation of the Gaussian probability density function.

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Day 1: Mathematical Foundations Visualized", fontsize=14, fontweight='bold')

    # -------------------------------------------------------------
    # Plot 1: Loss Surface & Gradient Step
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    w1_vals = np.linspace(-1, 3, 100)
    w2_vals = np.linspace(-1, 2, 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = 2 * (W1**2) + 3 * (W2**2) - 4 * W1

    cp = ax1.contour(W1, W2, Z, levels=15, cmap='Blues')
    ax1.clabel(cp, inline=True, fontsize=8)
    ax1.plot(current_w[0], current_w[1], 'ro', label='Initial w (Old Loss: 3.0)')
    ax1.plot(updated_w[0], updated_w[1], 'go', label='Updated w (New Loss: 0.88)')
    ax1.annotate('', xy=updated_w, xytext=current_w,
                 arrowprops=dict(facecolor='red', edgecolor='red', shrink=0, width=1.5, headwidth=8))
    ax1.set_title("1 & 2. Gradient Descent Step")
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Matrix Forward Pass (Taxi Fare Predictions)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    distances = X[:, 0]
    ax2.bar(["Ride 1 (3 mi)", "Ride 2 (10 mi)"], y_hat, color=['#1f77b4', '#ff7f0e'], edgecolor='k')
    for i, val in enumerate(y_hat):
        ax2.text(i, val + 0.8, f"${val:.2f}", ha='center', fontweight='bold')
    ax2.set_ylim(0, 35)
    ax2.set_title(r"3. Forward Pass ($\hat{y} = Xw + b$)")
    ax2.set_ylabel("Predicted Fare ($)")
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Plot 3: Regularization Norms (L1 vs L2)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    l1_val = np.sum(np.abs(weights_vec))
    l2_val = np.linalg.norm(weights_vec)
    ax3.bar(["L1 Norm (Lasso)", "L2 Norm (Ridge)"], [l1_val, l2_val], color=['#2ca02c', '#d62728'], edgecolor='k')
    ax3.text(0, l1_val + 0.2, f"{l1_val:.2f}", ha='center', fontweight='bold')
    ax3.text(1, l2_val + 0.2, f"{l2_val:.2f}", ha='center', fontweight='bold')
    ax3.set_ylim(0, 9)
    ax3.set_title("4. Regularization Norms for w=[3, -4, 0]")
    ax3.set_ylabel("Penalty Value")
    ax3.grid(axis='y', linestyle=':', alpha=0.6)

    # -------------------------------------------------------------
    # Plot 4: Gaussian Probability Density Function
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    x_axis = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf_axis = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (((x_axis - mu) / sigma) ** 2))
    peak_pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma))

    ax4.plot(x_axis, pdf_axis, color='purple', linewidth=2, label=r'$\mathcal{N}(\mu=30, \sigma=5)$')
    ax4.scatter([mu], [peak_pdf], color='red', zorder=5, label=f'Peak at x=30 ({peak_pdf:.4f})')
    ax4.set_title("5. Gaussian Probability Density")
    ax4.set_xlabel("Time (mins)")
    ax4.set_ylabel("Probability Density")
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, linestyle=':', alpha=0.6)

    # Clean up empty subplots
    fig.delaxes(axes[1, 1])
    fig.delaxes(axes[1, 2])

    plt.tight_layout()

    output_dir = Path(__file__).resolve().parents[1] / "src" / "servings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "day01_math_foundations.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    plt.show()


def run_math_examples():
    """Runs all 5 numerical math foundation examples and visualizes the results."""
    print("=" * 60)
    print("DAY 1: MATHEMATICAL FOUNDATIONS FOR MACHINE LEARNING")
    print("=" * 60 + "\n")

    # 1. VECTOR CALCULUS & GRADIENT VECTOR
    print("--- 1. VECTOR CALCULUS: GRADIENT VECTOR ---")
    current_weights = np.array([2.0, 1.0])
    grad = compute_gradient(current_weights)
    print(f"Current Weights (w1, w2): {current_weights}")
    print(f"Computed Gradient Vector:  {grad}\n")

    # 2. GRADIENT DESCENT UPDATE STEP
    print("--- 2. GRADIENT DESCENT UPDATE STEP ---")
    learning_rate = 0.1
    new_weights = current_weights - (learning_rate * grad)
    print(f"Learning Rate (eta): {learning_rate}")
    print(f"Updated Weights:    {new_weights}")
    print(f"Old Loss Value:     {loss_function(current_weights):.4f}")
    print(f"New Loss Value:     {loss_function(new_weights):.4f} (Decreased!)\n")

    # 3. MATRIX FORWARD PASS
    print("--- 3. MATRIX FORWARD PASS (TAXI FARE PREDICTION) ---")
    X = np.array([
        [3.0, 1.0],   # Sample 1
        [10.0, 2.0]   # Sample 2
    ])
    w = np.array([2.5, 0.5])
    b = 2.0
    y_hat = (X @ w) + b
    print(f"Feature Matrix X:\n{X}")
    print(f"Weights w: {w}, Bias b: {b}")
    print(f"Predicted Taxi Fares y_hat: ${y_hat}\n")

    # 4. REGULARIZATION NORMS
    print("--- 4. REGULARIZATION NORMS ---")
    weights_vec = np.array([3.0, -4.0, 0.0])
    l1_norm = np.sum(np.abs(weights_vec))
    l2_norm = np.linalg.norm(weights_vec)
    print(f"Weight Vector: {weights_vec}")
    print(f"L1 Norm (Lasso Penalty): {l1_norm:.2f}")
    print(f"L2 Norm (Ridge Penalty): {l2_norm:.2f}\n")

    # 5. GAUSSIAN PDF
    print("--- 5. GAUSSIAN PROBABILITY DENSITY FUNCTION ---")
    x = 30.0
    mu = 30.0
    sigma = 5.0
    gaussian_pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    print(f"Target x: {x}, Mean mu: {mu}, Std Dev sigma: {sigma}")
    print(f"Probability Density at x={x}: {gaussian_pdf:.4f}")
    print("=" * 60)

    # Trigger Matplotlib Visualization
    plot_math_foundations(current_weights, new_weights, X, y_hat, weights_vec, mu, sigma)


if __name__ == "__main__":
    run_math_examples()
