from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class RBPNN(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, X):
        distances = euclidean_distances(X, self.centroids_)
        return self.classes_[np.argmin(distances, axis=1)]
