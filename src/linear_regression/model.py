import numpy as np
class LinearRegression:
    def __init__(self, learning_rate, n_iterations):
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
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            dw = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model not trained yet. Call fit() first.")
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self):
        if self.weights is None:
            raise Exception("Model not trained yet. Call fit() first.")
        return {
            "weights": self.weights,
            "bias": self.bias
    }

if __name__ == "__main__":
    # Dummy data
    X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
    y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

    # Train
    model = LinearRegression(learning_rate=0.2, n_iterations=100000)
    print("Iterations:", model.n_iterations)
    print("Learning rate:", model.learning_rate)
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)
    print("Predictions:", predictions)

    # Params
    params = model.get_params()
    print("Weights:", params["weights"])
    print("Bias:", params["bias"])

    # Loss
    print("Initial loss:", model.loss_history[0])
    print("Final loss:", model.loss_history[-1])