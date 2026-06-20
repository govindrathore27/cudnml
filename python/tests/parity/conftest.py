"""
pytest fixtures for the Phase-05 distributional parity harness.

Provides:
  - Fixed-seed datasets (make_classification 20k×50, Covertype subset, make_regression)
  - Pinned hyperparameter sets (identical for Sylva and sklearn)
  - Version manifest (fairness protocol: versions pinned and recorded)

Fairness protocol (binding, per STATE.md / PITFALLS 1,2,13):
  - ET compared only to ET, RF only to RF (never crossed).
  - Identical hyperparameters + fixed seeds across both implementations.
  - Parity (accuracy/KS) is the GATE; training time is REPORTED, not gated.
  - sklearn/scipy/numpy/Python/Sylva-commit versions pinned + recorded here.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import scipy
import sklearn

from .datasets import (
    CLF_HYPERPARAMS,
    DATASET_SEED,
    RF_CLF_HYPERPARAMS,
    RF_REG_HYPERPARAMS,
    REG_HYPERPARAMS,
    Dataset,
    load_covertype_subset,
    load_make_classification,
    load_make_regression,
)

# ---------------------------------------------------------------------------
# Version manifest — fairness protocol: pinned versions recorded
# ---------------------------------------------------------------------------

def _sylva_commit() -> str:
    """Return the current Sylva git commit hash (first 12 chars)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


VERSION_MANIFEST: dict = {
    "python": sys.version,
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "sylva_commit": _sylva_commit(),
    "dataset_seed": DATASET_SEED,
}


def print_version_manifest() -> None:
    """Print the version manifest for the study report (fairness protocol)."""
    print("\n=== Version Manifest (Fairness Protocol) ===")
    for k, v in VERSION_MANIFEST.items():
        print(f"  {k}: {v}")
    print("=" * 45)


# ---------------------------------------------------------------------------
# Dataset fixtures (session-scoped — expensive to generate/download)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def clf_dataset() -> Dataset:
    """Primary classification dataset: make_classification 20k×50, fixed seed."""
    return load_make_classification()


@pytest.fixture(scope="session")
def covertype_dataset() -> Dataset:
    """Real-data anchor: Covertype 10k subset, 7 classes, fixed seed."""
    return load_covertype_subset()


@pytest.fixture(scope="session")
def reg_dataset() -> Dataset:
    """Primary regression dataset: make_regression 20k×50, fixed seed."""
    return load_make_regression()


# ---------------------------------------------------------------------------
# Hyperparameter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def et_clf_params() -> dict:
    """
    ExtraTrees classifier hyperparameters (identical for Sylva and sklearn).
    bootstrap=False, criterion=gini, n_estimators=200, max_depth=12.
    """
    return dict(CLF_HYPERPARAMS)


@pytest.fixture(scope="session")
def rf_clf_params() -> dict:
    """
    RandomForest classifier hyperparameters (identical for Sylva and sklearn).
    bootstrap=True, criterion=gini, n_estimators=200, max_depth=12.
    """
    return dict(RF_CLF_HYPERPARAMS)


@pytest.fixture(scope="session")
def et_reg_params() -> dict:
    """
    ExtraTrees regressor hyperparameters (identical for Sylva and sklearn).
    bootstrap=False, criterion=mse, n_estimators=200, max_depth=12.
    """
    return dict(REG_HYPERPARAMS)


@pytest.fixture(scope="session")
def rf_reg_params() -> dict:
    """
    RandomForest regressor hyperparameters (identical for Sylva and sklearn).
    bootstrap=True, criterion=mse, n_estimators=200, max_depth=12.
    """
    return dict(RF_REG_HYPERPARAMS)


# ---------------------------------------------------------------------------
# Autouse: print version manifest at session start
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def print_manifest() -> None:
    """Print the version manifest once at session start (fairness protocol)."""
    print_version_manifest()
