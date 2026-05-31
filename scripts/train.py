import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linear_regression import LinearRegression, mse, rmse, mae, r2_score, train_test_split, standardize

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'Housing.csv'))

# ── 2. Encode categorical columns ─────────────────────────────────────────────
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# ── 3. Select features and target ─────────────────────────────────────────────
features = ['area', 'bathrooms', 'airconditioning', 'stories', 'parking', 'bedrooms']
target = 'price'

X = df[features].values.astype(np.float64)
y = df[target].values.astype(np.float64)

# ── 4. Standardize features ───────────────────────────────────────────────────
X, mean, std = standardize(X)

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# ── 5. Split into train and test ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")

# ── 6. Train the model ────────────────────────────────────────────────────────
model = LinearRegression(learning_rate=0.01, n_iterations=5000)
model.fit(X_train, y_train)

# ── 7. Predict ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# scale back to real prices
y_pred = y_pred * y_std + y_mean
y_test = y_test * y_std + y_mean

# ── 8. Metrics ────────────────────────────────────────────────────────────────
print("\n── Model Performance ──────────────────────")
print(f"MSE:  {mse(y_test, y_pred):,.2f}")
print(f"RMSE: {rmse(y_test, y_pred):,.2f}")
print(f"MAE:  {mae(y_test, y_pred):,.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"\nWeights: {model.get_params()['weights']}")
print(f"Bias:    {model.get_params()['bias']:,.2f}")

# ── 9. Plot loss curve ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(model.loss_history)
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('MSE Loss')

# ── 10. Plot actual vs predicted ──────────────────────────────────────────────
axes[1].scatter(y_test, y_pred, alpha=0.6, color='steelblue')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title('Actual vs Predicted')
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')

plt.tight_layout()
plt.show()