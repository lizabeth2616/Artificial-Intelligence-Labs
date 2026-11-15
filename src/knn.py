import numpy as np
from collections import Counter


class KNNClassifier:
    """Собственная реализация KNN для классификации."""

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == "euclidean":
            diff = X[:, None, :] - self.X_train[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
        if self.metric == "manhattan":
            return np.abs(X[:, None, :] - self.X_train[None, :, :]).sum(axis=2)
        raise ValueError("Unsupported metric")

    def predict(self, X):
        distances = self._distances(X)
        k = min(self.k, distances.shape[1])
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        predictions = []
        for row in neighbors:
            labels = self.y_train[row]
            predictions.append(Counter(labels).most_common(1)[0][0])
        return np.array(predictions)


class KNNRegressor:
    """Собственная реализация KNN для регрессии."""

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == "euclidean":
            diff = X[:, None, :] - self.X_train[None, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
        if self.metric == "manhattan":
            return np.abs(X[:, None, :] - self.X_train[None, :, :]).sum(axis=2)
        raise ValueError("Unsupported metric")

    def predict(self, X):
        distances = self._distances(X)
        k = min(self.k, distances.shape[1])
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        return self.y_train[neighbors].mean(axis=1)
