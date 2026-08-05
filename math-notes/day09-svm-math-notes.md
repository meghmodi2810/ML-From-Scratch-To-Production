# Day 9: Support Vector Machines & Max-Margin Geometry

## 1. Hand-Calculations: 4-Point Toy Dataset

Consider a 2D binary classification dataset with four points:
* $x_1 = (1, 1)^T$, $y_1 = -1$
* $x_2 = (2, 0)^T$, $y_2 = -1$
* $x_3 = (2, 3)^T$, $y_3 = +1$
* $x_4 = (3, 2)^T$, $y_4 = +1$

### A. Finding the Optimal Hyperplane
The optimal decision line separating Class -1 and Class +1 passes midway between the convex hulls. 

Let the decision boundary be:
$$w_1 x_1 + w_2 x_2 + b = 0$$

For a Hard Margin SVM, the support vectors satisfy $y_i(w^T x_i + b) = 1$:
* Point $x_1 = (1, 1)$: $-(w_1 + w_2 + b) = 1 \implies w_1 + w_2 + b = -1$
* Point $x_4 = (3, 2)$: $+(3w_1 + 2w_2 + b) = 1 \implies 3w_1 + 2w_2 + b = 1$

Subtracting the two equations:
$$(3w_1 + 2w_2 + b) - (w_1 + w_2 + b) = 1 - (-1)$$
$$2w_1 + w_2 = 2 \implies w_2 = 2 - 2w_1$$

By geometric symmetry across the midpoint between $(1,1)$ and $(3,2)$, the normal vector $w$ has equal components ($w_1 = w_2$):
$$2w_1 + w_1 = 2 \implies w_1 = \frac{2}{3}, \quad w_2 = \frac{2}{3}$$

Substituting back to find $b$:
$$\frac{2}{3} + \frac{2}{3} + b = -1 \implies b = -\frac{7}{3}$$

### B. Weight Vector & Bias
$$w = \begin{bmatrix} 2/3 \\ 2/3 \end{bmatrix}, \quad b = -\frac{7}{3}$$

### C. Calculating the Geometric Margin ($M$)
The $L_2$ norm of $w$:
$$\|w\|_2 = \sqrt{\left(\frac{2}{3}\right)^2 + \left(\frac{2}{3}\right)^2} = \sqrt{\frac{8}{9}} = \frac{2\sqrt{2}}{3}$$

The maximum margin width $M$:
$$M = \frac{2}{\|w\|_2} = \frac{2}{\frac{2\sqrt{2}}{3}} = \frac{3}{\sqrt{2}} \approx 2.1213 \text{ units}$$

---

## 2. Key Mathematical Terms

* **Primal Objective:** 
  $$\min_{w, b, \xi} \left( \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^m \xi_i \right) \quad \text{s.t. } y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* **Hinge Loss:** $L(y, f(x)) = \max(0, 1 - y \cdot f(x))$
* **C Hyperparameter:** Tradeoff between margin width ($\frac{1}{2}\|w\|^2$) and slack violations ($\sum \xi_i$).
  * Large $C$: Hard margin (narrow margin, zero error tolerance $\to$ Overfitting).
  * Small $C$: Soft margin (wider margin, error tolerant $\to$ Underfitting).
* **RBF Kernel (Gaussian):** $K(x, z) = \exp(-\gamma \|x - z\|^2)$.
  * High $\gamma$: Short reach, decision boundary tightly wraps individual points ($\to$ Overfitting).
  * Low $\gamma$: Long reach, smooth boundary ($\to$ Underfitting).