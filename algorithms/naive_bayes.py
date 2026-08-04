import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class GaussianNaiveBayesScratch:
    """Gaussian Naive Bayes Classifier implemented from scratch in pure NumPy.
    
    Designed for continuous numerical features.
    Feature likelihoods are modeled using Gaussian probability density functions (PDFs).
    """

    def __init__(self, eps: float = 1e-9):
        """Initializes model.

        Args:
            eps (float): Small variance smoothing term added to variances to prevent zero-division.
        """
        self.eps = eps
        self.classes = None
        self.priors = None      # Shape: (num_classes,)
        self.means = None       # Shape: (num_classes, num_features)
        self.variances = None   # Shape: (num_classes, num_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        """Computes class priors, feature means (mu), and feature variances (sigma^2) per class."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        m, d = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.priors = np.zeros(num_classes)
        self.means = np.zeros((num_classes, d))
        self.variances = np.zeros((num_classes, d))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Prior P(y) = count(y) / m
            self.priors[idx] = X_c.shape[0] / float(m)
            # Feature means mu_{y, j}
            self.means[idx, :] = np.mean(X_c, axis=0)
            # Feature variances sigma^2_{y, j} + eps for stability
            self.variances[idx, :] = np.var(X_c, axis=0) + self.eps

        return self

    def _gaussian_log_pdf(self, class_idx: int, X: np.ndarray) -> np.ndarray:
        """Computes log Gaussian likelihoods for all query samples under a given class.
        
        Formula:
            log P(x | y) = -0.5 * log(2 * pi * sigma^2) - ((x - mu)^2 / (2 * sigma^2))
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]

        # Log PDF evaluation
        log_pdf = -0.5 * np.log(2.0 * np.pi * var) - ((X - mean) ** 2) / (2.0 * var)
        # Sum log probabilities across feature dimensions (Naive independence assumption)
        return np.sum(log_pdf, axis=1)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates unnormalized log joint probability [log P(y) + sum log P(x_j | y)]."""
        X = np.array(X, dtype=np.float64)
        m = X.shape[0]
        num_classes = len(self.classes)

        log_posteriors = np.zeros((m, num_classes))

        for idx in range(num_classes):
            log_prior = np.log(self.priors[idx])
            log_likelihood = self._gaussian_log_pdf(idx, X)
            log_posteriors[:, idx] = log_prior + log_likelihood

        return log_posteriors

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates normalized class probabilities using log-sum-exp trick for numerical stability."""
        log_posteriors = self.predict_log_proba(X)
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        return exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels with maximum posterior log-probability."""
        log_posteriors = self.predict_log_proba(X)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes[class_indices]


class MultinomialNaiveBayesScratch:
    """Multinomial Naive Bayes Classifier with Laplace Smoothing implemented from scratch.
    
    Designed for discrete feature count matrices (e.g., Bag-of-Words in NLP).
    """

    def __init__(self, alpha: float = 1.0):
        """Initializes model.

        Args:
            alpha (float): Laplace smoothing parameter (alpha = 1.0 for standard Laplace smoothing).
        """
        self.alpha = alpha
        self.classes = None
        self.priors = None                # Shape: (num_classes,)
        self.feature_log_prob = None     # Shape: (num_classes, num_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayesScratch":
        """Computes class priors and Laplace-smoothed word likelihoods."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        m, d = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.priors = np.zeros(num_classes)
        self.feature_log_prob = np.zeros((num_classes, d))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # Prior P(y) = count(y) / m
            self.priors[idx] = X_c.shape[0] / float(m)

            # Laplace Smoothed Likelihood Formula:
            # P(w_j | c) = (N_{c, j} + alpha) / (N_c + alpha * |V|)
            word_counts_c = np.sum(X_c, axis=0)          # N_{c, j} for each feature
            total_words_c = np.sum(word_counts_c)         # N_c total words in class c

            smoothed_likelihoods = (word_counts_c + self.alpha) / (total_words_c + self.alpha * d)
            self.feature_log_prob[idx, :] = np.log(smoothed_likelihoods)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates unnormalized log joint probability [log P(y) + X @ log P(w | y)^T]."""
        X = np.array(X, dtype=np.float64)
        m = X.shape[0]
        num_classes = len(self.classes)

        log_posteriors = np.zeros((m, num_classes))

        for idx in range(num_classes):
            log_prior = np.log(self.priors[idx])
            # Matrix multiplication sums word log likelihoods weighted by word counts in X
            log_likelihood = X @ self.feature_log_prob[idx, :]
            log_posteriors[:, idx] = log_prior + log_likelihood

        return log_posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels with maximum posterior log-probability."""
        log_posteriors = self.predict_log_proba(X)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes[class_indices]


# =====================================================================
# DEMO & VALIDATION AGAINST SKLEARN
# =====================================================================

def run_day08_naive_bayes_experiments():
    """Runs tests for both Gaussian and Multinomial Naive Bayes models against scikit-learn."""
    np.random.seed(42)

    # -----------------------------------------------------------------
    # TEST 1: Gaussian Naive Bayes on Continuous Features
    # -----------------------------------------------------------------
    X_cont, y_cont = make_classification(
        n_samples=500, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=42
    )
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cont, y_cont, test_size=0.2, random_state=42)

    gnb_scratch = GaussianNaiveBayesScratch().fit(X_tr_c, y_tr_c)
    gnb_sklearn = GaussianNB().fit(X_tr_c, y_tr_c)

    preds_gnb_scratch = gnb_scratch.predict(X_te_c)
    preds_gnb_sklearn = gnb_sklearn.predict(X_te_c)

    # -----------------------------------------------------------------
    # TEST 2: Multinomial Naive Bayes on Discrete Word Counts
    # -----------------------------------------------------------------
    # Simulate a discrete Bag-of-Words matrix (500 documents, 20 vocabulary words)
    X_disc = np.random.poisson(lam=2.0, size=(500, 20))
    y_disc = np.random.choice([0, 1], size=500)
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_disc, y_disc, test_size=0.2, random_state=42)

    mnb_scratch = MultinomialNaiveBayesScratch(alpha=1.0).fit(X_tr_d, y_tr_d)
    mnb_sklearn = MultinomialNB(alpha=1.0).fit(X_tr_d, y_tr_d)

    preds_mnb_scratch = mnb_scratch.predict(X_te_d)
    preds_mnb_sklearn = mnb_sklearn.predict(X_te_d)

    print("=" * 65)
    print("DAY 8: NAIVE BAYES BENCHMARKS & SKLEARN VALIDATION")
    print("=" * 65)
    print("Gaussian Naive Bayes (Continuous Features):")
    print(f"  Scratch Accuracy           : {np.mean(preds_gnb_scratch == y_te_c):.4f}")
    print(f"  sklearn Accuracy           : {np.mean(preds_gnb_sklearn == y_te_c):.4f}")
    print(f"  Exact Match vs. Scikit    : {np.all(preds_gnb_scratch == preds_gnb_sklearn)}")
    print("-" * 65)
    print("Multinomial Naive Bayes (Discrete Bag-of-Words + Laplace):")
    print(f"  Scratch Accuracy           : {np.mean(preds_mnb_scratch == y_te_d):.4f}")
    print(f"  sklearn Accuracy           : {np.mean(preds_mnb_sklearn == y_te_d):.4f}")
    print(f"  Exact Match vs. Scikit    : {np.all(preds_mnb_scratch == preds_mnb_sklearn)}")
    print("=" * 65 + "\n")

    # Plot Decision Regions for Gaussian Naive Bayes
    X_2d = X_cont[:, :2]
    X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = train_test_split(X_2d, y_cont, test_size=0.2, random_state=42)

    viz_model = GaussianNaiveBayesScratch().fit(X_tr_2d, y_tr_2d)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = viz_model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set1")
    ax.scatter(X_te_2d[y_te_2d == 0, 0], X_te_2d[y_te_2d == 0, 1], color="red", label="Class 0", edgecolors="k")
    ax.scatter(X_te_2d[y_te_2d == 1, 0], X_te_2d[y_te_2d == 1, 1], color="blue", label="Class 1", edgecolors="k")
    ax.scatter(X_te_2d[y_te_2d == 2, 0], X_te_2d[y_te_2d == 2, 1], color="green", label="Class 2", edgecolors="k")

    ax.set_title("Day 8: Gaussian Naive Bayes Decision Regions")
    ax.set_xlabel("Feature x₁")
    ax.set_ylabel("Feature x₂")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day08_naive_bayes_experiments()