from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class RBPNN(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        # Pre-organize data by class for faster access
        self.X_by_class_ = {c: X[y == c] for c in self.classes_}
        return self

    def predict_proba(self, X):
        # RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)
        # We compute sum of kernels for each class
        probs = []
        for x in X:
            class_scores = []
            for c in self.classes_:
                X_c = self.X_by_class_[c]
                # Compute squared euclidean distances
                # Using broadcasting: (N_c, D) - (D,) -> (N_c, D)
                dists_sq = np.sum((X_c - x)**2, axis=1)
                # Kernel values
                kernels = np.exp(-self.gamma * dists_sq)
                # Sum of kernels (density estimation)
                score = np.sum(kernels)
                class_scores.append(score)
            probs.append(class_scores)
        
        probs = np.array(probs)
        # Normalize to get probabilities
        row_sums = probs.sum(axis=1, keepdims=True)
        # Handle zero sum (numerical underflow)
        row_sums[row_sums == 0] = 1e-10
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
