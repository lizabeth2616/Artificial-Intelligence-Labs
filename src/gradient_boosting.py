import numpy as np
from .decision_tree import DecisionTreeRegressor, DecisionTreeClassifier


class GradientBoostingRegressor:
    """Простая собственная реализация градиентного бустинга для регрессии"""
    
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.initial_pred = np.mean(y)
        residuals = y - self.initial_pred
        
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


class GradientBoostingClassifier:
    """Упрощенная собственная реализация градиентного бустинга для классификации (one-vs-rest)"""
    
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = {}
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        
        for c in self.classes_:
            y_binary = (y == c).astype(float)
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth
            )
            model.fit(X, y_binary)
            self.models[c] = model
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            scores[:, i] = self.models[c].predict(X)
        
        return self.classes_[np.argmax(scores, axis=1)]
