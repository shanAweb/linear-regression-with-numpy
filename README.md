# Linear Regression from Scratch

A linear regression model built from scratch using only NumPy, trained on real housing data.

## Overview

This project implements linear regression with gradient descent — no scikit-learn, no autograd. Just NumPy for the math, pandas for loading data, and matplotlib for plots. It's a learning project that shows how the algorithm works end to end: preprocessing, training, and evaluation.

## Results

On the held-out test set, the model explains **~54.5%** of house price variance:

| Metric | Value |
| :----- | ----: |
| RMSE   | 1,263,292 |
| MAE    | 956,775 |
| R²     | 0.5456 |

## Project Structure

```text
linear-regression/
├── data/
│   └── Housing.csv                  # Kaggle housing dataset
├── notebooks/
│   └── exploration.ipynb            # Data exploration
├── scripts/
│   └── train.py                     # End-to-end training pipeline
├── src/linear_regression/
│   ├── model.py                     # Gradient-descent linear regression
│   ├── preprocessing.py             # Train/test split, scaling
│   └── metrics.py                   # MSE, RMSE, MAE, R²
├── tests/
│   └── test_model.py
├── pyproject.toml
└── README.md
```

| Module                | Responsibility |
| :-------------------- | :------------- |
| `model.py`            | Linear regression via gradient descent (NumPy only) |
| `preprocessing.py`    | Train/test split, standardization, normalization |
| `metrics.py`          | MSE, RMSE, MAE, R² score |
| `scripts/train.py`    | Full training pipeline on the housing data |

## Dataset

Kaggle Housing Dataset — 545 samples, 13 features.

- **Target:** `price`
- **Features used:** `area`, `bathrooms`, `airconditioning`, `stories`, `parking`, `bedrooms`

## Setup

Requires **Python 3.12+** and **Poetry**.

```bash
# Install dependencies
poetry install

# Run the training pipeline
poetry run python scripts/train.py
```

## How It Works

The model minimizes mean squared error (MSE) loss with batch gradient descent:

```text
y_pred  = X · weights + bias
loss    = mean((y_pred - y)²)

dw      = (1/n) · Xᵀ · (y_pred - y)
db      = mean(y_pred - y)

weights = weights - lr · dw
bias    = bias    - lr · db
```

### Hyperparameters

| Parameter     | Value |
| :------------ | ----: |
| Learning rate | 0.01  |
| Iterations    | 5000  |
| Test size     | 20%   |
| Random seed   | 42    |

## Tests

```bash
poetry run pytest
```

## Explore the Data

```bash
poetry run jupyter notebook notebooks/exploration.ipynb
```

## Dependencies

| Package      | Purpose |
| :----------- | :------ |
| `numpy`      | Model math and arrays |
| `pandas`     | Loading and encoding the dataset |
| `matplotlib` | Loss curve and prediction plots |
