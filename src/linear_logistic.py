import numpy as np


class LinearRegression:
    """Собственная реализация линейной регрессии (градиентный спуск)"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class LogisticRegression:
    """Собственная реализация логистической регрессии (градиентный спуск)"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Для многоклассовой задачи используем one-vs-rest
        self.classes_ = np.unique(y)
        self.weights = {}
        self.bias = {}
        
        for c in self.classes_:
            y_binary = (y == c).astype(int)
            w = np.zeros(n_features)
            b = 0
            
            for _ in range(self.n_iterations):
                linear = np.dot(X, w) + b
                y_pred = self._sigmoid(linear)
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_binary))
                db = (1 / n_samples) * np.sum(y_pred - y_binary)
                
                w -= self.learning_rate * dw
                b -= self.learning_rate * db
            
            self.weights[c] = w
            self.bias[c] = b
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            scores = {}
            for c in self.classes_:
                linear = np.dot(x, self.weights[c]) + self.bias[c]
                scores[c] = self._sigmoid(linear)
            predictions.append(max(scores, key=scores.get))
        
        return np.array(predictions)
