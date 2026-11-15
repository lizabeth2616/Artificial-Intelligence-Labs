import numpy as np


class LinearRegression:
    """Собственная реализация линейной регрессии (градиентный спуск)."""

    def __init__(self, learning_rate=0.05, n_iterations=1500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            pred = X @ self.weights + self.bias
            error = pred - y
            self.weights -= self.learning_rate * (X.T @ error) / n_samples
            self.bias -= self.learning_rate * np.sum(error) / n_samples
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


class LogisticRegression:
    """Собственная реализация логистической регрессии, one-vs-rest."""

    def __init__(self, learning_rate=0.1, n_iterations=1500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.classes_ = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.weights = {}
        self.bias = {}

        for label in self.classes_:
            target = (y == label).astype(float)
            weights = np.zeros(n_features)
            bias = 0.0
            for _ in range(self.n_iterations):
                pred = self._sigmoid(X @ weights + bias)
                error = pred - target
                weights -= self.learning_rate * (X.T @ error) / n_samples
                bias -= self.learning_rate * np.sum(error) / n_samples
            self.weights[label] = weights
            self.bias[label] = bias
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [self._sigmoid(X @ self.weights[label] + self.bias[label]) for label in self.classes_]
        )
        return self.classes_[np.argmax(scores, axis=1)]
