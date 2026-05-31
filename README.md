```markdown
# Linear Regression from Scratch

A linear regression model built from scratch using only NumPy, trained on real housing data.

## Project Structure

```
linear-regression/
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   └── Housing.csv
├── src/
│   └── linear_regression/
│       ├── __init__.py
│       ├── model.py
│       ├── metrics.py
│       └── preprocessing.py
├── scripts/
│   └── train.py
├── notebooks/
│   └── exploration.ipynb
└── tests/
    └── test_model.py
```

## What's Built

- **`model.py`** — Linear regression using gradient descent, numpy only
- **`preprocessing.py`** — Train/test split, standardization, normalization
- **`metrics.py`** — MSE, RMSE, MAE, R² score
- **`scripts/train.py`** — Full training pipeline on real housing data

## Dataset

Kaggle Housing Dataset — 545 samples, 13 features.

Target variable: `price`

Features used: `area`, `bathrooms`, `airconditioning`, `stories`, `parking`, `bedrooms`

## Model Results

| Metric | Value |
|---|---|
| RMSE | 1,263,292 |
| MAE | 956,775 |
| R² | 0.5456 |

Model explains **54.5%** of house price variance using pure numpy gradient descent.

## Setup

Make sure you have Python 3.12 and Poetry installed.

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry env activate

# Run training
poetry run python scripts/train.py
```

## How It Works

### Gradient Descent
The model learns by minimizing MSE loss:

```
y_pred = X · weights + bias
loss   = mean((y_pred - y)²)
dw     = (1/n) · Xᵀ · (y_pred - y)
db     = mean(y_pred - y)

weights = weights - lr · dw
bias    = bias    - lr · db
```

### Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate | 0.01 |
| Iterations | 5000 |
| Test size | 20% |
| Random seed | 42 |

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Model, math, arrays |
| `pandas` | Loading and encoding dataset |
| `matplotlib` | Loss curve and prediction plots |

## Explore the Data

```bash
poetry run jupyter notebook notebooks/exploration.ipynb
```
```