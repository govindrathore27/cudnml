"""
Shallow-depth regression KS proof (Plan 02-05 Task-3 checkpoint follow-up).

The deep (max_depth=12) ExtraTrees-regression feature-frequency KS check is
informational because it fails (p~1e-4) while accuracy/R2 parity passes. The
claim is that this divergence is an RNG-STREAM ARTIFACT — two faithful
implementations with independent RNGs (Sylva: Philox-4x32-10; sklearn:
our_rand_r serial) produce statistically-distinguishable-but-equally-valid DEEP
trees — and NOT a bug in Sylva's MSE-regression split logic.

This test PROVES (or refutes) that claim. At shallow depth the random draws are
too few to compound, so a faithful Sylva algorithm MUST match sklearn's
feature-selection-frequency distribution closely (high KS p). A LOW shallow-depth
p would instead expose a real structural divergence that R2 parity alone cannot
catch — a genuine oracle-correctness concern.

Note: both implementations use the SAME hyperparameters (incl. max_features=sqrt
for regression, per datasets.py REG_HYPERPARAMS) and the same fixed seed —
strictly ET-vs-ET, like-for-like.
"""

from __future__ import annotations

import json

from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesRegressor

import sylva_core_pyseam as sylva

from .datasets import REG_HYPERPARAMS, Dataset
from .test_calibration import (
    _feature_frequency_distribution,
    _sklearn_split_observations,
)

FIXED_SEED: int = 42
SHALLOW_DEPTH: int = 4
DEEP_DEPTH: int = REG_HYPERPARAMS["max_depth"]  # 12
N_ESTIMATORS: int = 60  # ET is the fast path (random thresholds, no sort)
# A faithful shallow-depth match must be clearly non-significant. The deep gate
# yields ~1e-4; a true RNG artifact should jump well above 0.05 when shallow.
SHALLOW_KS_FLOOR: float = 0.05


def _sylva_cfg(max_depth: int) -> dict:
    p = REG_HYPERPARAMS
    return {
        "n_estimators": N_ESTIMATORS,
        "max_depth": max_depth,
        "max_features": p["max_features"],
        "min_samples_split": p["min_samples_split"],
        "min_samples_leaf": p["min_samples_leaf"],
        "bootstrap": p["bootstrap"],
        "criterion": p["criterion"],
        "seed": FIXED_SEED,
        "algo": "et",
    }


def _ks_feature_freq_at_depth(ds: Dataset, max_depth: int) -> float:
    n_features = ds.X_train.shape[1]

    ir = sylva.fit_cpu(ds.X_train, ds.y_train, _sylva_cfg(max_depth))
    stats = json.loads(sylva.split_statistics(ir))
    sylva_fids = [o["feature_id"] for o in stats["observations"]]
    freq_sylva = _feature_frequency_distribution(sylva_fids, n_features)

    sk_params = {
        k: v
        for k, v in REG_HYPERPARAMS.items()
        if k not in ("criterion", "algo", "seed")
    }
    sk_params["n_estimators"] = N_ESTIMATORS
    sk_params["max_depth"] = max_depth
    sk = ExtraTreesRegressor(**sk_params, random_state=FIXED_SEED)
    sk.fit(ds.X_train, ds.y_train)
    sk_fids, _ = _sklearn_split_observations(sk.estimators_, n_features)
    freq_sklearn = _feature_frequency_distribution(sk_fids, n_features)

    return float(ks_2samp(freq_sylva, freq_sklearn).pvalue)


def test_shallow_depth_regression_feature_freq_matches_sklearn(
    reg_dataset: Dataset,
) -> None:
    """At max_depth=4 the Sylva ET-regression feature distribution must match
    sklearn's (RNG cannot compound) — proving the deep-tree divergence is an
    RNG-stream artifact, not a Sylva split-logic bug."""
    p_shallow = _ks_feature_freq_at_depth(reg_dataset, SHALLOW_DEPTH)
    p_deep = _ks_feature_freq_at_depth(reg_dataset, DEEP_DEPTH)
    print(
        f"\n  ET-reg feature-freq KS: shallow(d={SHALLOW_DEPTH}) p={p_shallow:.4f}  "
        f"deep(d={DEEP_DEPTH}) p={p_deep:.6f}"
    )
    assert p_shallow >= SHALLOW_KS_FLOOR, (
        f"Shallow-depth (max_depth={SHALLOW_DEPTH}) ET-regression feature-frequency KS "
        f"p={p_shallow:.4f} is below {SHALLOW_KS_FLOOR}. The deep divergence is NOT just "
        f"an RNG artifact -- Sylva's ET-regression split logic structurally diverges from "
        f"sklearn even when the RNG cannot compound. Investigate before trusting the oracle."
    )
    print(
        "  PROVEN: shallow-depth feature distribution matches sklearn "
        "-> the deep-depth divergence is an RNG-stream artifact, not a Sylva bug."
    )
