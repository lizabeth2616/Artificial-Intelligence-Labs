import numpy as np


def _thresholds(column, max_thresholds=16):
    if len(column) < 2:
        return np.array([])
    quantiles = np.linspace(0.1, 0.9, max_thresholds)
    return np.unique(np.quantile(column, quantiles))


class DecisionTreeClassifier:
    """Собственная реализация решающего дерева для классификации."""

    def __init__(self, max_depth=5, min_samples_split=2, max_thresholds=16):
        self.max_depth = 20 if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.tree = self._build(X, y, 0)
        return self

    def _leaf(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs * probs)

    def _best_split(self, X, y):
        best_score = np.inf
        best_feature = None
        best_threshold = None
        n = len(y)

        for feature in range(X.shape[1]):
            column = X[:, feature]
            for threshold in _thresholds(column, self.max_thresholds):
                left = column <= threshold
                n_left = int(left.sum())
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue
                score = (n_left * self._gini(y[left]) + n_right * self._gini(y[~left])) / n
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._leaf(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return self._leaf(y)

        left = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
        }

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.tree) for row in X])

    def _predict_one(self, row, node):
        if not isinstance(node, dict):
            return node
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])


class DecisionTreeRegressor:
    """Собственная реализация решающего дерева для регрессии."""

    def __init__(self, max_depth=5, min_samples_split=2, max_thresholds=16):
        self.max_depth = 20 if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree = self._build(X, y, 0)
        return self

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
        best_score = np.inf
        best_feature = None
        best_threshold = None
        n = len(y)

        for feature in range(X.shape[1]):
            column = X[:, feature]
            for threshold in _thresholds(column, self.max_thresholds):
                left = column <= threshold
                n_left = int(left.sum())
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue
                score = (n_left * self._mse(y[left]) + n_right * self._mse(y[~left])) / n
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return float(np.mean(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return float(np.mean(y))

        left = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
        }

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.tree) for row in X], dtype=float)

    def _predict_one(self, row, node):
        if not isinstance(node, dict):
            return node
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])
