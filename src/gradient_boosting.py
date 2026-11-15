import numpy as np

try:
    from .decision_tree import DecisionTreeRegressor
except ImportError:
    from decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Собственная реализация градиентного бустинга для регрессии."""

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.initial_pred = float(np.mean(y))
        residuals = y - self.initial_pred

        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=2)
            tree.fit(X, residuals)
            self.trees.append(tree)
            residuals = residuals - self.learning_rate * tree.predict(X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.initial_pred, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


class GradientBoostingClassifier:
    """Собственная реализация градиентного бустинга для классификации (one-vs-rest)."""

    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models = {}
        for label in self.classes_:
            target = (y == label).astype(float)
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
            )
            model.fit(X, target)
            self.models[label] = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([self.models[label].predict(X) for label in self.classes_])
        return self.classes_[np.argmax(scores, axis=1)]
