import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights + self.bias)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            dw = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db