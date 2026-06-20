"""
Dataset loaders for the Phase-05 distributional parity harness.

All loaders return fixed-seed train/test splits with identical hyperparameters
exposed for both Sylva and sklearn.  The fairness protocol requires:
  - Identical hyperparameters + fixed seeds across both implementations.
  - Like-for-like comparisons only (ET-vs-ET, RF-vs-RF — never crossed).
  - No dataset randomness leaking between runs.

Covertype: fetched and cached once by sklearn.datasets.fetch_covtype.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.datasets import fetch_covtype, make_classification, make_regression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Fixed global constants
# ---------------------------------------------------------------------------

# Primary dataset seed — all datasets use this for reproducibility.
DATASET_SEED: int = 2025_06_20

# Primary dataset dimensions.
N_SAMPLES_SYNTHETIC: int = 20_000
N_FEATURES_SYNTHETIC: int = 50

# Covertype subset size (real-data anchor).
COVERTYPE_SUBSET: int = 10_000

# Train/test split ratio.
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Shared hyperparameter sets (identical for both Sylva and sklearn)
# The fairness protocol requires these to be fixed and symmetric.
# ---------------------------------------------------------------------------

# Classification hyperparameters (ExtraTrees and RandomForest)
CLF_HYPERPARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": False,  # ET default; overridden for RF
    "criterion": "gini",
}

# Regression hyperparameters
REG_HYPERPARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": False,  # ET default; overridden for RF
    "criterion": "mse",
}

# RF bootstrap=True (correct sklearn RF default)
RF_CLF_HYPERPARAMS: dict = {**CLF_HYPERPARAMS, "bootstrap": True}
RF_REG_HYPERPARAMS: dict = {**REG_HYPERPARAMS, "bootstrap": True}


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


class Dataset(NamedTuple):
    """A fixed-seed train/test split with metadata."""

    name: str
    X_train: np.ndarray  # shape (n_train, n_features), dtype float32
    X_test: np.ndarray   # shape (n_test, n_features), dtype float32
    y_train: np.ndarray  # shape (n_train,), dtype float32
    y_test: np.ndarray   # shape (n_test,), dtype float32
    task: str            # "classification" or "regression"
    n_classes: int       # 1 for regression


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_make_classification() -> Dataset:
    """
    Primary synthetic classification dataset: make_classification 20k×50.

    Parameters mirror Phase-05 plan spec:
    - n_samples=20_000, n_features=50, n_informative=20, n_redundant=10,
      n_repeated=5, n_classes=2, random_state=DATASET_SEED.
    - 80/20 train/test split with the same seed.
    - Cast to float32 (Sylva is f32 end-to-end per D-05).
    """
    X, y = make_classification(
        n_samples=N_SAMPLES_SYNTHETIC,
        n_features=N_FEATURES_SYNTHETIC,
        n_informative=20,
        n_redundant=10,
        n_repeated=5,
        n_classes=2,
        random_state=DATASET_SEED,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=DATASET_SEED
    )
    return Dataset(
        name="make_classification_20k_50",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="classification",
        n_classes=2,
    )


def load_covertype_subset() -> Dataset:
    """
    Real-data anchor: Covertype subset (first COVERTYPE_SUBSET rows).

    Fetched and cached once by sklearn.  7-class classification with 54
    features.  Used as a real-world complement to the synthetic dataset.
    Cast to float32 (Sylva is f32 end-to-end per D-05).
    """
    data = fetch_covtype()
    X_all = data.data.astype(np.float32)
    y_all = (data.target - 1).astype(np.float32)  # sklearn returns 1-indexed

    # Reproducible subset selection (fixed index, no random state needed).
    rng = np.random.RandomState(DATASET_SEED)
    idx = rng.choice(len(X_all), size=min(COVERTYPE_SUBSET, len(X_all)), replace=False)
    X = X_all[idx]
    y = y_all[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=DATASET_SEED
    )
    return Dataset(
        name="covertype_subset",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="classification",
        n_classes=7,
    )


def load_make_regression() -> Dataset:
    """
    Primary synthetic regression dataset: make_regression 20k×50.

    Parameters:
    - n_samples=20_000, n_features=50, n_informative=20, noise=0.1,
      random_state=DATASET_SEED.
    - 80/20 train/test split with the same seed.
    - Cast to float32.
    """
    X, y = make_regression(
        n_samples=N_SAMPLES_SYNTHETIC,
        n_features=N_FEATURES_SYNTHETIC,
        n_informative=20,
        noise=0.1,
        random_state=DATASET_SEED,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=DATASET_SEED
    )
    return Dataset(
        name="make_regression_20k_50",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task="regression",
        n_classes=1,
    )
