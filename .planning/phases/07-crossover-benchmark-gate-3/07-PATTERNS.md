# Phase 7: Crossover Benchmark (Gate 3) - Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 9 new files + 1 pre-registration planning artifact
**Analogs found:** 8 / 9 (1 has no code analog — the pre-registration doc)

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `.planning/phases/07-crossover-benchmark-gate-3/07-PRE-REGISTRATION.md` | planning artifact | N/A (doc) | none | no analog — structure from RESEARCH.md |
| `python/benchmarks/__init__.py` | config | N/A (package marker) | `python/tests/parity/__init__.py` | exact |
| `python/benchmarks/grid_spec.py` | utility | transform | `python/tests/parity/datasets.py` | role-match |
| `python/benchmarks/crossover_manifest.py` | config | request-response | `python/tests/parity/conftest.py` (VERSION_MANIFEST) | exact |
| `python/benchmarks/crossover_study.py` | service | batch | `python/benchmarks/comparative_study.py` (Phase-5 design, not yet on disk) | role-match (designed precursor) |
| `python/benchmarks/visualize_surface.py` | utility | transform | none in codebase | no analog — use RESEARCH.md patterns |
| `python/tests/test_crossover_accuracy_parity.py` | test | CRUD | `python/tests/parity/test_distributional_parity.py` | exact |
| `python/tests/test_crossover_manifest.py` | test | CRUD | `python/tests/parity/test_distributional_parity.py` (manifest print pattern) | role-match |
| `python/tests/test_crossover_fairness_rules.py` | test | CRUD | `python/tests/parity/test_distributional_parity.py` (fairness prohibitions) | role-match |
| `results/` directory + JSON/CSV/PNG/MD artifacts | config | batch | none | no analog — harness output |

---

## Pattern Assignments

### `python/benchmarks/grid_spec.py` (utility, transform)

**Analog:** `python/tests/parity/datasets.py`

**Imports pattern** (lines 1–19):
```python
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.datasets import fetch_covtype, make_classification, make_regression
from sklearn.model_selection import train_test_split
```

**Constants pattern** (lines 25–67 of datasets.py):
```python
# Primary dataset seed — all datasets use this for reproducibility.
DATASET_SEED: int = 2025_06_20

# Hyperparameters — IDENTICAL across Sylva and sklearn (fairness protocol).
# n_jobs=-1 REQUIRED for all sklearn baselines (PITFALL 2, PITFALL 13).
CLF_HYPERPARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 12,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": False,  # ET
    "criterion": "gini",
}
RF_CLF_HYPERPARAMS: dict = {**CLF_HYPERPARAMS, "bootstrap": True}
```

**Grid spec pattern** (new — mirrors datasets.py structure, extends to grid):
```python
# (n×d) pre-registered grid — MUST match 07-PRE-REGISTRATION.md exactly.
# Do NOT add or remove cells after the pre-registration is committed.
N_VALUES = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
D_VALUES = [20, 50, 100, 200]

# Full Cartesian product — no cherry-picking.
SYNTHETIC_GRID: list[tuple[int, int]] = [
    (n, d) for n in N_VALUES for d in D_VALUES
]

# Real-data anchors — fixed shapes from sklearn/UCI.
REAL_ANCHORS = [
    {"name": "covertype_full", "n": 581_012, "d": 54},
    {"name": "higgs_1m",       "n": 1_000_000, "d": 28},
]
```

**Dataset loader pattern** (lines 92–124 of datasets.py — adapt for large n):
```python
class Dataset(NamedTuple):
    name: str
    X_train: np.ndarray   # float64 — coercion to float32 is INSIDE timed region
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    task: str
    n_classes: int

def make_grid_dataset(n: int, d: int, seed: int = DATASET_SEED) -> Dataset:
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=max(2, d // 3),
        n_redundant=max(1, d // 10), random_state=seed
    )
    # NOTE: X stays float64 here — dtype coercion to float32 happens INSIDE
    # the timed region in timed_fit(), not here (PITFALL 1).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return Dataset(name=f"synthetic_{n}x{d}", X_train=X_train, X_test=X_test,
                   y_train=y_train, y_test=y_test, task="classification", n_classes=2)
```

---

### `python/benchmarks/crossover_manifest.py` (config, request-response)

**Analog:** `python/tests/parity/conftest.py` lines 42–71

**Imports + VERSION_MANIFEST pattern** (conftest.py lines 17–63):
```python
from __future__ import annotations

import subprocess
import sys

import numpy as np
import sklearn

def _sylva_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

VERSION_MANIFEST: dict = {
    "python": sys.version,
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "sylva_commit": _sylva_commit(),
    "dataset_seed": DATASET_SEED,
}
```

**Extension pattern for crossover_manifest.py** (adds grid + baselines + hardware):
```python
# Extend the conftest VERSION_MANIFEST shape with Phase-7 specifics.
# Every baseline that is "not available on measurement host" is recorded as
# that string, never silently omitted (PITFALL 13 / honesty protocol).

MANIFEST: dict = {
    # --- Core versions (mirror conftest.VERSION_MANIFEST shape) ---
    "python": sys.version,
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "sylva_commit": _sylva_commit(),
    "dataset_seed": DATASET_SEED,

    # --- Baseline versions (resolved by human-verify checkpoint) ---
    "sklearnex_version": "not yet resolved — see checkpoint:human-verify",
    "cuml_version":      "not yet resolved — see checkpoint:human-verify",

    # --- Hardware / driver / CUDA pins ---
    "gpu_model":      "RTX 4060 Ti",
    "cuda_version":   "12.8",
    "driver_version": "595.79",
    "compute_cap":    "sm_89",

    # --- Pre-registered grid spec ---
    "grid_n_values":  [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    "grid_d_values":  [20, 50, 100, 200],
    "n_repeats_warm": 5,
    "n_repeats_cold": 1,
    "win_threshold":  0.05,        # 5% — from 07-PRE-REGISTRATION.md
    "accuracy_threshold": 0.01,    # 1% accuracy delta
}
```

---

### `python/benchmarks/crossover_study.py` (service, batch)

**Analog:** Phase-5 `comparative_study.py` (designed precursor; not yet on disk). Pattern sourced from 05-06-PLAN.md task 1 spec + RESEARCH.md Patterns 1–3.

**Imports pattern** (mirrors test_distributional_parity.py lines 29–54):
```python
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

import sylva_core_pyseam as sylva

from .crossover_manifest import MANIFEST
from .grid_spec import SYNTHETIC_GRID, REAL_ANCHORS, DATASET_SEED, make_grid_dataset
```

**BenchmarkImpl pattern** (RESEARCH.md Pattern 2):
```python
@dataclass
class BenchmarkImpl:
    name: str            # e.g. "sklearn_et", "sylva_gpu_et", "cuml_rf_DIFFERENT_ALGO"
    algorithm: str       # "et" | "rf"  — surface groups by algorithm; NEVER cross
    backend: str         # "cpu" | "gpu"
    is_reference: bool   # True for sklearn (accuracy parity anchor)
    factory: object      # callable() -> unfitted estimator

    def __post_init__(self) -> None:
        # Structural guard — enforce algorithm tag is present.
        assert self.algorithm in ("et", "rf"), f"Unknown algorithm tag: {self.algorithm}"
```

**Core timing pattern** (RESEARCH.md Pattern 1 — the canonical contract):
```python
def timed_fit(model, X_f64: np.ndarray, y: np.ndarray,
              n_cold: int = 1, n_warm: int = 5) -> dict:
    """
    Time model.fit() end-to-end from host float64 numpy array.
    Dtype coercion (float64 -> float32) is INSIDE the timed region — always.
    Anti-pattern: starting the timer after X.astype(np.float32). That omits
    real user cost and constitutes benchmark dishonesty (PITFALL 1).
    """
    cold_times: list[float] = []
    for _ in range(n_cold):
        t0 = time.perf_counter()
        X32 = X_f64.astype(np.float32)   # coercion INSIDE timed region
        model.fit(X32, y)
        cold_times.append(time.perf_counter() - t0)

    warm_times: list[float] = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        X32 = X_f64.astype(np.float32)   # coercion INSIDE every warm call too
        model.fit(X32, y)
        warm_times.append(time.perf_counter() - t0)

    sorted_warm = sorted(warm_times)
    q1 = sorted_warm[len(sorted_warm) // 4]
    q3 = sorted_warm[3 * len(sorted_warm) // 4]
    return {
        "cold_times": cold_times,
        "warm_times": warm_times,
        "median_cold_s": statistics.median(cold_times),
        "median_warm_s": statistics.median(warm_times),
        "iqr_warm_s": (q3 - q1) if len(warm_times) > 2 else 0.0,
    }
```

**OOM-safe wrapper pattern** (RESEARCH.md Pattern 3):
```python
def safe_timed_fit(impl: BenchmarkImpl, X_f64: np.ndarray, y: np.ndarray,
                   **kwargs: Any) -> dict:
    """Wrap timed_fit; catch CUDA OOM and host OOM; continue grid loop."""
    try:
        model = impl.factory()
        times = timed_fit(model, X_f64, y, **kwargs)
        X32 = X_f64.astype(np.float32)
        acc = accuracy_score(y_test, model.predict(X32_test))
        return {"oom": False, "error": None, "accuracy": acc, **times}
    except (MemoryError, Exception) as exc:
        err = str(exc)
        is_oom = "out of memory" in err.lower() or isinstance(exc, MemoryError)
        return {"oom": is_oom, "error": err,
                "accuracy": None, "median_cold_s": None, "median_warm_s": None}
```

**Grid loop pattern** (RESEARCH.md architecture diagram):
```python
def run_crossover_study(output_dir: Path = Path("results")) -> None:
    output_dir.mkdir(exist_ok=True)
    results: list[dict] = []

    for n, d in SYNTHETIC_GRID:
        ds = make_grid_dataset(n, d)
        for impl in ET_IMPLS + RF_IMPLS:
            row = safe_timed_fit(impl, ds.X_train, ds.y_train)
            row.update({"n": n, "d": d, "impl": impl.name,
                        "algorithm": impl.algorithm,
                        "dataset": ds.name})
            # Runtime fairness assertion — no ET-vs-RF in any cell.
            assert impl.algorithm in ("et", "rf")
            results.append(row)

    # Write canonical + tabular outputs.
    (output_dir / "crossover_results.json").write_text(
        json.dumps(results, indent=2))
    # ... CSV + markdown table generation ...
```

**sklearnex patching pattern** (RESEARCH.md Pattern 5):
```python
# Call BEFORE importing sklearn estimators for the session.
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX_AVAILABLE = True
except ImportError:
    SKLEARNEX_AVAILABLE = False
    # Record in manifest: sklearnex_version: "not available on measurement host"
```

---

### `python/tests/test_crossover_accuracy_parity.py` (test, CRUD)

**Analog:** `python/tests/parity/test_distributional_parity.py`

**Imports + header pattern** (test_distributional_parity.py lines 1–54):
```python
"""
Phase-7 crossover accuracy-parity GATE (BENCH-02).

Asserts Sylva ET/RF test-set accuracy is within tolerance of sklearn ET/RF
like-for-like (ET-vs-ET, RF-vs-RF — NEVER crossed).  Runs with device='cpu'
so it is CI-portable (no GPU required).  Speed is NOT asserted here; speed
is reported by crossover_study.py.

Fairness protocol tags (per PITFALLS 1, 2, 13 and binding ROADMAP note):
  [FP-1]  ET-vs-ET, RF-vs-RF only.
  [FP-2]  Identical hyperparameters + fixed seeds.
  [FP-3]  Accuracy is the GATE; speed is reported, not gated.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

import sylva_core_pyseam as sylva

from .parity.conftest import VERSION_MANIFEST
from .parity.datasets import CLF_HYPERPARAMS, RF_CLF_HYPERPARAMS, DATASET_SEED
```

**Tolerance constant pattern** (test_distributional_parity.py — calibrated, not guessed):
```python
# Accuracy parity tolerance — 1% (from 07-PRE-REGISTRATION.md kill criterion).
# Justification: identical hyperparameters + fixed seed; any gap > 1% signals
# an algorithmic divergence bug, not measurement noise.
ACCURACY_TOLERANCE: float = 0.01

FIXED_SEED: int = 42  # deterministic Sylva-vs-sklearn comparison seed
```

**Test body pattern** (mirrors test_distributional_parity.py structure):
```python
def test_et_clf_accuracy_parity(clf_dataset):
    """
    Sylva ET clf test-set accuracy is within ACCURACY_TOLERANCE of sklearn ET clf.
    Like-for-like: identical hyperparams, bootstrap=False, device='cpu'.
    """
    params = {**CLF_HYPERPARAMS, "random_state": FIXED_SEED}
    sk_clf = ExtraTreesClassifier(**params, n_jobs=-1)
    sylva_clf = sylva.ExtraTreesClassifier(**params, device="cpu")

    sk_clf.fit(clf_dataset.X_train, clf_dataset.y_train)
    sylva_clf.fit(clf_dataset.X_train, clf_dataset.y_train)

    sk_acc = accuracy_score(clf_dataset.y_test, sk_clf.predict(clf_dataset.X_test))
    sylva_acc = accuracy_score(clf_dataset.y_test, sylva_clf.predict(clf_dataset.X_test))

    delta = abs(sk_acc - sylva_acc)
    assert delta <= ACCURACY_TOLERANCE, (
        f"ET clf accuracy parity FAILED: |{sylva_acc:.4f} - {sk_acc:.4f}| = "
        f"{delta:.4f} > {ACCURACY_TOLERANCE} (BENCH-02 gate)"
    )
```

---

### `python/tests/test_crossover_manifest.py` (test, CRUD)

**Analog:** `python/tests/parity/conftest.py` (VERSION_MANIFEST shape)

**Pattern** (extends conftest.py print_version_manifest + adds grid completeness check):
```python
from benchmarks.crossover_manifest import MANIFEST

def test_manifest_has_required_keys() -> None:
    required = {
        "python", "sklearn", "numpy", "sylva_commit", "dataset_seed",
        "sklearnex_version", "cuml_version",
        "gpu_model", "cuda_version", "driver_version",
        "grid_n_values", "grid_d_values",
        "n_repeats_warm", "win_threshold", "accuracy_threshold",
    }
    missing = required - set(MANIFEST.keys())
    assert not missing, f"MANIFEST missing required keys: {missing}"

def test_manifest_grid_matches_preregistration() -> None:
    """Grid in manifest must match 07-PRE-REGISTRATION.md exactly."""
    assert MANIFEST["grid_n_values"] == [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    assert MANIFEST["grid_d_values"] == [20, 50, 100, 200]
```

---

### `python/tests/test_crossover_fairness_rules.py` (test, CRUD)

**Analog:** `python/tests/parity/test_distributional_parity.py` (FP-1/FP-3 protocol comments)

**Pattern** (encodes prohibitions from 05-06-PLAN.md prohibitions block as static assertions):
```python
from benchmarks.grid_spec import SYNTHETIC_GRID, ET_IMPLS, RF_IMPLS

def test_no_et_vs_rf_cross() -> None:
    """ET impls must never appear in RF group and vice versa (PITFALL 13)."""
    et_names = {impl.name for impl in ET_IMPLS}
    rf_names = {impl.name for impl in RF_IMPLS}
    assert et_names.isdisjoint(rf_names), "ET and RF impl sets must be disjoint"

def test_sklearn_baselines_have_n_jobs_minus_1() -> None:
    """All sklearn baseline factories must use n_jobs=-1 (PITFALL 2)."""
    for impl in ET_IMPLS + RF_IMPLS:
        if "sklearn" in impl.name:
            m = impl.factory()
            assert getattr(m, "n_jobs", None) == -1, (
                f"{impl.name} must use n_jobs=-1 (weakest-baseline trap, PITFALL 2)"
            )

def test_cuml_labeled_different_algorithm() -> None:
    """cuML RF must not appear in the ET group (PITFALL 4 / PITFALL 7)."""
    et_names = {impl.name for impl in ET_IMPLS}
    assert not any("cuml" in n for n in et_names), (
        "cuML must NOT appear in the ET comparison group — it is RF, a different algorithm"
    )
```

---

### `python/benchmarks/visualize_surface.py` (utility, transform)

**Analog:** none in codebase — use RESEARCH.md patterns + standard matplotlib/seaborn idiom.

**Pattern** (RESEARCH.md architecture diagram output section):
```python
"""
Generate crossover surface visualization from results/crossover_results.json.
Produces:
  - results/crossover_surface.png  (heatmap: win / lose / tie / OOM per cell)
  - results/crossover_table.md     (canonical markdown table — the primary artifact)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/crossover_results.json")

def build_surface_table(results: list[dict]) -> str:
    """Emit markdown table. This is the canonical artifact; the heatmap is supplemental."""
    ...

def build_heatmap(results: list[dict], out_path: Path) -> None:
    """Heatmap: green = GPU ET warm wins, red = CPU wins, grey = OOM."""
    ...
```

---

## Shared Patterns

### Version Manifest Shape
**Source:** `python/tests/parity/conftest.py` lines 42–71
**Apply to:** `crossover_manifest.py`
```python
VERSION_MANIFEST: dict = {
    "python": sys.version,
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "sylva_commit": _sylva_commit(),
    "dataset_seed": DATASET_SEED,
}
```
Extend this dict shape; do not replace it.

### Dataset Container (NamedTuple)
**Source:** `python/tests/parity/datasets.py` lines 75–84
**Apply to:** `grid_spec.py`
```python
class Dataset(NamedTuple):
    name: str
    X_train: np.ndarray  # float64 in grid_spec — coercion is INSIDE timed region
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    task: str
    n_classes: int
```
Note: Phase-5 datasets cast to float32 at load time. Phase-7 grid_spec MUST keep float64 at load time; coercion to float32 happens only inside `timed_fit()` to satisfy PITFALL 1.

### Hyperparameter Constants
**Source:** `python/tests/parity/datasets.py` lines 44–67
**Apply to:** `grid_spec.py` (import and extend, do not duplicate)
```python
from python.tests.parity.datasets import CLF_HYPERPARAMS, RF_CLF_HYPERPARAMS, DATASET_SEED
# Add n_jobs=-1 for all sklearn baseline instantiations — NEVER omit this.
```

### Fairness-Rule Header Comment Block
**Source:** `python/tests/parity/test_distributional_parity.py` lines 9–25
**Apply to:** ALL Phase-7 Python files (crossover_study.py, test_*.py)
Mirror the `[FP-1]` through `[FP-6]` tag pattern at the top of every file that touches comparisons.

### Like-For-Like Fixture Pattern
**Source:** `python/tests/parity/conftest.py` lines 79–135 (session-scoped fixtures)
**Apply to:** `test_crossover_accuracy_parity.py`
Use `scope="session"` for heavy dataset fixtures; `scope="function"` for estimator instances.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `.planning/phases/07-crossover-benchmark-gate-3/07-PRE-REGISTRATION.md` | planning artifact | N/A | Markdown protocol document; no code analog. Structure is fully specified in RESEARCH.md §Pre-Registration Document Structure (10 required sections). |
| `python/benchmarks/visualize_surface.py` | utility | transform | No visualization code exists in the codebase. Use `matplotlib.pyplot.imshow` / `seaborn.heatmap` per RESEARCH.md §Don't Hand-Roll. |
| `results/` directory artifacts | output | batch | No results directory or artifact pattern exists. The canonical artifact is `results/crossover_results.json`; the markdown table is the published surface. |

---

## Metadata

**Analog search scope:** `python/tests/parity/`, `python/benchmarks/` (empty), `.planning/phases/05-full-forest-randomforest-sklearn-estimators/05-06-PLAN.md`
**Files scanned:** 5 (conftest.py, datasets.py, test_distributional_parity.py, __init__.py ×2)
**Pattern extraction date:** 2026-06-27
