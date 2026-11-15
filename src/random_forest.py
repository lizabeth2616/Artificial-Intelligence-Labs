import numpy as np
from collections import Counter

try:
    from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
except ImportError:
    from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """Собственная реализация случайного леса для классификации."""

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.array([tree.predict(X) for tree in self.trees])
        result = []
        for i in range(X.shape[0]):
            result.append(Counter(votes[:, i]).most_common(1)[0][0])
        return np.array(result)


class RandomForestRegressor:
    """Собственная реализация случайного леса для регрессии."""

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = np.array([tree.predict(X) for tree in self.trees], dtype=float)
        return predictions.mean(axis=0)
