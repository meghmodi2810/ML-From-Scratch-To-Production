# Day 9: Support Vector Machines (SVM) & Kernel Transformations
# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. HAND-CALCULATED TOY DATASET VERIFICATION
# =====================================================================
X_toy = np.array([[1, 1], [2, 0], [2, 3], [3, 2]], dtype=np.float64)
y_toy = np.array([-1, -1, 1, 1])

# Fit Hard Margin Linear SVM (C very large)
svm_toy = SVC(kernel="linear", C=1e5)
svm_toy.fit(X_toy, y_toy)

print("=" * 60)
print("HAND-CALCULATION VERIFICATION (4-POINT DATASET)")
print("=" * 60)
print(f"Learned Weights (w) : {svm_toy.coef_[0]}")
print(f"Learned Bias (b)    : {svm_toy.intercept_[0]:.4f}")
print(f"Support Vectors     :\n{svm_toy.support_vectors_}")

norm_w = np.linalg.norm(svm_toy.coef_[0])
margin_width = 2.0 / norm_w
print(f"Geometric Margin (M): {margin_width:.4f} units")
print("=" * 60 + "\n")

# =====================================================================
# 2. DECISION BOUNDARY PLOTTING FUNCTION
# =====================================================================
def plot_svm_boundary(clf, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#ffaaaa', '#aaffaa'], alpha=0.3)
    ax.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors=['k', 'k', 'k'], 
               linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
    
    # Plot samples
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color="red", label="Class -1", edgecolors="k", alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="green", label="Class +1", edgecolors="k", alpha=0.7)
    
    # Highlight Support Vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none", edgecolors="black", linewidths=2.0, label="Support Vectors")
    
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

# =====================================================================
# 3. EXPERIMENT: HARD VS. SOFT MARGIN SWEEP (C HYPERPARAMETER)
# =====================================================================
X_blobs, y_blobs = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
y_blobs = np.where(y_blobs == 0, -1, 1)

c_values = [0.01, 1.0, 100.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, C in enumerate(c_values):
    clf = SVC(kernel="linear", C=C).fit(X_blobs, y_blobs)
    plot_svm_boundary(clf, X_blobs, y_blobs, axes[idx], f"Linear Kernel (C = {C})")

plt.tight_layout()
plt.show()

# =====================================================================
# 4. EXPERIMENT: NON-LINEAR RBF KERNEL SWEEP (GAMMA & C)
# =====================================================================
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)
y_moons = np.where(y_moons == 0, -1, 1)

gamma_values = [0.1, 1.0, 10.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, gamma in enumerate(gamma_values):
    clf = SVC(kernel="rbf", C=1.0, gamma=gamma).fit(X_moons, y_moons)
    plot_svm_boundary(clf, X_moons, y_moons, axes[idx], f"RBF Kernel (C=1.0, γ={gamma})")

plt.tight_layout()
plt.show()