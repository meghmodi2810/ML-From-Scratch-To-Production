from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.logistic_regression import LogisticRegressionScratch


class ClassificationMetricsScratch:
    """Comprehensive Classification Evaluation Metrics & Curve Generator implemented from scratch in pure NumPy.
    """

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[tn, fp], [fn, tp]])

    @classmethod
    def accuracy(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy = (TP + TN) / Total"""
        return float(np.mean(y_true == y_pred))

    @classmethod
    def precision(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision = TP / (TP + FP)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    @classmethod
    def recall(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall (Sensitivity / TPR) = TP / (TP + FN)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    @classmethod
    def specificity(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Specificity (True Negative Rate) = TN / (TN + FP)"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    @classmethod
    def f1_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        prec = cls.precision(y_true, y_pred)
        rec = cls.recall(y_true, y_pred)
        return float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

    @classmethod
    def roc_curve(cls, y_true: np.ndarray, y_probs: np.ndarray, num_thresholds: int = 100):
        """Generates (FPR, TPR) coordinates across probability decision thresholds from 1.0 down to 0.0."""
        thresholds = np.linspace(1.0, 0.0, num_thresholds)
        fprs, tprs = [], []

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            tpr = cls.recall(y_true, y_pred)
            fpr = 1.0 - cls.specificity(y_true, y_pred)
            tprs.append(tpr)
            fprs.append(fpr)

        return np.array(fprs), np.array(tprs), thresholds

    @classmethod
    def precision_recall_curve(cls, y_true: np.ndarray, y_probs: np.ndarray, num_thresholds: int = 100):
        """Generates (Recall, Precision) coordinates across probability decision thresholds."""
        thresholds = np.linspace(1.0, 0.0, num_thresholds)
        precisions, recalls = [], []

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            precisions.append(cls.precision(y_true, y_pred))
            recalls.append(cls.recall(y_true, y_pred))

        return np.array(recalls), np.array(precisions), thresholds

    @classmethod
    def auc_score(cls, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Computes Area Under Curve (AUC) using trapezoidal numerical integration."""
        return float(np.trapz(y_coords, x_coords))

    @classmethod
    def tune_threshold_by_f1(cls, y_true: np.ndarray, y_probs: np.ndarray, num_thresholds: int = 200):
        """Finds optimal decision threshold t that maximizes F1-score on validation data."""
        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        best_threshold = 0.5
        best_f1 = 0.0

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            f1 = cls.f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        return best_threshold, best_f1


# =====================================================================
# DEMO & VISUALIZATION PIPELINE
# =====================================================================

def run_day05_comprehensive_demo():
    """Generates synthetic dataset, fits Logistic Regression, and plots ROC & PR curves."""
    np.random.seed(42)
    m = 200

    # Imbalanced Synthetic Dataset: 80% Class 0 (Negative), 20% Class 1 (Positive)
    m_pos = int(m * 0.2)
    m_neg = m - m_pos

    cluster_neg = np.random.multivariate_normal([-0.5, -0.5], [[1.5, 0.3], [0.3, 1.5]], m_neg)
    cluster_pos = np.random.multivariate_normal([1.5, 1.5], [[0.8, -0.1], [-0.1, 0.8]], m_pos)

    X = np.vstack([cluster_neg, cluster_pos])
    y = np.hstack([np.zeros(m_neg), np.ones(m_pos)])

    # Fit Logistic Regression Model
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=1500)
    model.fit(X, y)
    y_probs = model.predict_proba(X)

    # 1. Metrics at default threshold 0.5
    y_pred_default = model.predict(X)
    cm = ClassificationMetricsScratch.confusion_matrix(y, y_pred_default)
    acc = ClassificationMetricsScratch.accuracy(y, y_pred_default)
    prec = ClassificationMetricsScratch.precision(y, y_pred_default)
    rec = ClassificationMetricsScratch.recall(y, y_pred_default)
    spec = ClassificationMetricsScratch.specificity(y, y_pred_default)
    f1 = ClassificationMetricsScratch.f1_score(y, y_pred_default)

    # 2. Threshold Tuning
    opt_t, opt_f1 = ClassificationMetricsScratch.tune_threshold_by_f1(y, y_probs)

    # 3. Compute Curves
    fprs, tprs, _ = ClassificationMetricsScratch.roc_curve(y, y_probs)
    roc_auc = ClassificationMetricsScratch.auc_score(fprs, tprs)

    recalls, precisions, _ = ClassificationMetricsScratch.precision_recall_curve(y, y_probs)
    pr_auc = ClassificationMetricsScratch.auc_score(recalls, precisions)

    # 4. Confusion Matrix for default threshold
    cm_labels = np.array([["TN", "FP"], ["FN", "TP"]])

    # Print Full Report
    print("=" * 60)
    print("DAY 5: CLASSIFICATION EVALUATION METRICS REPORT")
    print("=" * 60)
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")
    print(f"Accuracy                 : {acc:.4f}")
    print(f"Precision                : {prec:.4f}")
    print(f"Recall (Sensitivity/TPR) : {rec:.4f}")
    print(f"Specificity (TNR)        : {spec:.4f}")
    print(f"F1-Score (Threshold 0.5) : {f1:.4f}")
    print("-" * 60)
    print(f"Optimal Threshold (F1)   : {opt_t:.4f}")
    print(f"Maximized F1-Score       : {opt_f1:.4f}")
    print(f"ROC-AUC Score            : {roc_auc:.4f}")
    print(f"PR-AUC Score             : {pr_auc:.4f}")
    print("=" * 60 + "\n")

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Day 5: ROC Curve vs. Precision-Recall Curve Analysis", fontsize=14, fontweight='bold')

    # Subplot 1: ROC Curve
    axes[0].plot(fprs, tprs, color="#1f77b4", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Baseline (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Precision-Recall Curve
    pos_ratio = m_pos / m
    axes[1].plot(recalls, precisions, color="#ff7f0e", linewidth=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[1].axhline(y=pos_ratio, color="grey", linestyle="--", label=f"Random Baseline (Ratio = {pos_ratio:.2f})")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Confusion Matrix Heatmap
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Pred 0", "Pred 1"])
    axes[2].set_yticklabels(["True 0", "True 1"])
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(
                j,
                i,
                f"{cm_labels[i, j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_day05_comprehensive_demo()