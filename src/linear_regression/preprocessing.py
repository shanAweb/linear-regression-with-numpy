import numpy as np

def train_test_split(X, y, test_size=0.2, seed=None):
    np.random.seed(seed)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def normalize(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    X_scaled = (X - min_val) / (max_val - min_val)
    return X_scaled, min_val, max_val