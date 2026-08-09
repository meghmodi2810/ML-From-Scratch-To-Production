import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Generator


class StratifiedKFoldScratch:
    """Stratified k-Fold Cross-Validator implemented from scratch in pure NumPy.
    
    Ensures exact class distribution proportions are preserved across every fold.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates train/val index splits preserving class distributions."""
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)

        if self.shuffle:
            np.random.seed(self.random_state)

        # Group indices by target class
        classes, y_indices = np.unique(y, return_inverse=True)
        class_indices = [np.where(y_indices == c)[0] for c in range(len(classes))]

        fold_indices = [[] for _ in range(self.n_splits)]

        # Distribute class indices evenly across folds
        for cls_idx in class_indices:
            if self.shuffle:
                np.random.shuffle(cls_idx)

            # Split class indices into n_splits roughly equal chunks
            splits = np.array_split(cls_idx, self.n_splits)
            for fold_idx, chunk in enumerate(splits):
                fold_indices[fold_idx].extend(chunk)

        # Yield train and validation indices for each fold
        all_indices = np.arange(n_samples)
        for fold_idx in range(self.n_splits):
            val_idx = np.array(fold_indices[fold_idx], dtype=np.int64)
            train_idx = np.setdiff1d(all_indices, val_idx)
            yield train_idx, val_idx


def plot_learning_curves(model, X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> None:
    """Plots diagnostic Learning Curves (Training Size vs Accuracy)."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores_mean, val_scores_mean = [], []

    cv = StratifiedKFoldScratch(n_splits=cv_splits, shuffle=True, random_state=42)

    for train_frac in train_sizes:
        sub_train_scores, sub_val_scores = [], []
        
        for train_idx, val_idx in cv.split(X, y):
            # Subset the training split according to train_frac
            n_sub = max(5, int(len(train_idx) * train_frac))
            sub_train_idx = train_idx[:n_sub]

            X_tr, y_tr = X[sub_train_idx], y[sub_train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            
            tr_acc = np.mean(model.predict(X_tr) == y_tr)
            val_acc = np.mean(model.predict(X_val) == y_val)

            sub_train_scores.append(tr_acc)
            sub_val_scores.append(val_acc)

        train_scores_mean.append(np.mean(sub_train_scores))
        val_scores_mean.append(np.mean(sub_val_scores))

    # Plotting Learning Curves
    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes * 100, train_scores_mean, "o-", color="crimson", linewidth=2, label="Training Score")
    plt.plot(train_sizes * 100, val_scores_mean, "s-", color="navy", linewidth=2, label="Cross-Validation Score")
    
    plt.title("Diagnostic Learning Curves (Bias vs. Variance Diagnosis)", fontsize=12, fontweight="bold")
    plt.xlabel("Percentage of Training Data Used (%)")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


# =====================================================================
# DEMO VALIDATION SCRIPT
# =====================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    # Generate Synthetic Imbalanced Dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, weights=[0.8, 0.2], random_state=42)
    skf = StratifiedKFoldScratch(n_splits=5)

    print("=" * 60)
    print("STRATIFIED K-FOLD SCRATCH VALIDATION")
    print("=" * 60)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        tr_dist = np.bincount(y[tr_idx]) / len(tr_idx)
        val_dist = np.bincount(y[val_idx]) / len(val_idx)
        print(f"Fold {fold+1}: Train Class Ratio = {tr_dist.round(3)}, Val Class Ratio = {val_dist.round(3)}")
    print("=" * 60)

    # Plot Diagnostic Learning Curve
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    plot_learning_curves(tree, X, y, cv_splits=5)