# Phase 9: Treelite Export & Packaging — Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 7 new/modified files
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `python/sylva/export.py` | service | transform (ForestIR → treelite.Model) | `crates/sylva-core/src/pyseam.rs` (IR JSON handle → Python output) | role-match |
| `python/tests/test_export.py` | test | request-response (predict round-trip) | `python/tests/parity/test_distributional_parity.py` | exact |
| `python/tests/conftest.py` (update) | config/fixture | — | `python/tests/parity/conftest.py` | exact |
| `scripts/import_smoke_09.py` | utility | request-response | `scripts/import_smoke.py` (Phase 1 pattern; referenced in 01-03-PLAN.md) | role-match |
| `pyproject.toml` (update) | config | — | existing `pyproject.toml` (Phase 1 proven config) | exact |
| `crates/sylva-core/src/pyseam.rs` (update) | middleware/seam | request-response | existing `pyseam.rs` | exact |
| `docs/INSTALL.md` | documentation | — | `VERSIONS.md` (runtime prerequisite doc pattern) | partial |

---

## Pattern Assignments

### `python/sylva/export.py` (service, transform)

**Analog:** `crates/sylva-core/src/pyseam.rs` — the established pattern for reading the ForestIR JSON handle from the FFI boundary and producing Python output.

**Imports pattern** (`pyseam.rs` lines 31–40):
```python
# Python-side mirror of the pyseam FFI contract:
# - consume `ir_json: str` (the opaque handle from fit_cpu / _ir_json attr)
# - call json.loads() to hydrate ForestIR fields (consistent with seam decision OQ-5)
# - no new PyO3 binding needed — pure Python transform

from __future__ import annotations
import json
import numpy as np
import treelite
from treelite.model_builder import (
    Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation,
)
```

**Input validation pattern** (`pyseam.rs` lines 306–316):
```python
# Mirror the seam's validation-at-boundary rule: parse + validate before any
# ModelBuilder call; raise ValueError (not RuntimeError) for caller errors.
def export_to_treelite(ir_json: str) -> treelite.Model:
    try:
        ir = json.loads(ir_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"export_to_treelite: invalid ir_json: {e}") from e
    _validate_ir_dict(ir)   # check required keys and array lengths
    ...
```

**Core transform pattern** (RESEARCH.md Pattern 1, lines 183–263 — the definitive ModelBuilder walk):
```python
# Global-to-tree-local reindex: ForestIR uses global node IDs; ModelBuilder
# uses tree-local IDs. Source: ir.rs tree_offsets[t]..tree_offsets[t+1].
for t in range(ir["n_trees"]):
    start = ir["tree_offsets"][t]
    end   = ir["tree_offsets"][t + 1]
    builder.start_tree()
    for global_id in range(start, end):
        local_id = global_id - start          # ← mandatory reindex
        builder.start_node(local_id)
        if ir["is_leaf"][global_id]:
            ...  # leaf() call with leaf_proba or leaf_value slice
        else:
            lc_local = ir["left_child"][global_id]  - start
            rc_local = ir["right_child"][global_id] - start
            default_left = (
                ir["default_child"][global_id] == ir["left_child"][global_id]
            )
            builder.numerical_test(
                feature_id=int(ir["feature_id"][global_id]),
                threshold=float(ir["threshold"][global_id]),
                default_left=default_left,
                opname="<",                   # always "<" for ET/RF
                left_child_key=lc_local,
                right_child_key=rc_local,
            )
        builder.end_node()
    builder.end_tree()
return builder.commit()
```

**ForestIR leaf_proba slice pattern** (`ir.rs` lines 48–53):
```python
# From ir.rs: leaf_proba[o*n_classes .. (o+1)*n_classes] where o = leaf_offset[node]
# Copy this exact indexing in Python:
o  = ir["leaf_offset"][global_id]          # index into payload block
nc = n_classes
proba = ir["leaf_proba"][o * nc : (o + 1) * nc]
builder.leaf(list(proba))                  # multi-class vector leaf
```

**Error handling pattern** (`pyseam.rs` lines 46–53):
```python
# Mirror sylva_error_to_pyerr: ValueError for caller errors, RuntimeError for
# internal failures (ModelBuilder state errors).
except (KeyError, IndexError) as e:
    raise ValueError(f"export_to_treelite: malformed ForestIR: {e}") from e
except Exception as e:
    raise RuntimeError(f"export_to_treelite: ModelBuilder error: {e}") from e
```

---

### `python/tests/test_export.py` (test, request-response)

**Analog:** `python/tests/parity/test_distributional_parity.py` — the established parity-test idiom: import the seam, call fit + predict, compare to reference with `np.testing.assert_allclose`.

**Imports pattern** (`test_distributional_parity.py` lines 29–50):
```python
from __future__ import annotations

import json
import numpy as np
import pytest
import treelite
import tl2cgen

import sylva_core_pyseam as sylva  # or the Phase-5 estimator import

from .conftest import ...           # shared fixtures (trained IR handles)
```

**Fixture pattern** (`conftest.py` lines 79–95 — session-scoped trained IR):
```python
# Use session scope: training is expensive; share the trained IR across tests.
@pytest.fixture(scope="session")
def trained_clf_ir(clf_dataset, et_clf_params) -> str:
    """Train a small ET classifier and return the opaque IR JSON handle."""
    X, y = clf_dataset.X_train, clf_dataset.y_train
    return sylva.fit_cpu(X, y, {**et_clf_params, "n_estimators": 50, "max_depth": 6})

@pytest.fixture(scope="session")
def trained_reg_ir(reg_dataset, et_reg_params) -> str:
    """Train a small ET regressor and return the opaque IR JSON handle."""
    X, y = reg_dataset.X_train, reg_dataset.y_train
    return sylva.fit_cpu(X, y, {**et_reg_params, "n_estimators": 50, "max_depth": 6})
```

**Core round-trip parity test pattern** (`test_distributional_parity.py` structure; RESEARCH.md Pattern 2):
```python
def test_gtil_round_trip_clf(trained_clf_ir, clf_dataset):
    X = clf_dataset.X_test.astype(np.float32)
    sylva_proba = sylva.predict_cpu(trained_clf_ir, X)       # (n, n_classes)

    from sylva.export import export_to_treelite
    tl_model = export_to_treelite(trained_clf_ir)
    tl_proba = treelite.gtil.predict(tl_model, X)           # pure Python, Windows-portable

    np.testing.assert_allclose(
        sylva_proba, tl_proba, atol=1e-5,
        err_msg="gtil round-trip failed: check postprocessor (identity vs softmax)",
    )
```

**TL2cgen compiled-path test** (`RESEARCH.md` Pattern 2 secondary path):
```python
@pytest.mark.slow
def test_tl2cgen_round_trip(trained_clf_ir, clf_dataset, tmp_path):
    """TL2cgen compiled-CPU round-trip (Windows MSVC; MSVC cl.exe must be on PATH)."""
    import tl2cgen
    X = clf_dataset.X_test.astype(np.float32)
    tl_model = export_to_treelite(trained_clf_ir)
    tl2cgen.generate_c_code(tl_model, str(tmp_path), params={})
    libpath = tl2cgen.create_shared("msvc", str(tmp_path))  # .dll on Windows
    pred = tl2cgen.Predictor(libpath).predict(tl2cgen.DMatrix(X))
    np.testing.assert_allclose(
        sylva.predict_cpu(trained_clf_ir, X), pred, atol=1e-5
    )
```

**FIL gate pattern** (no existing analog — new pattern, document as manual):
```python
@pytest.mark.linux   # skip on Windows CI; checkpoint:human-verify
def test_fil_round_trip(trained_clf_ir, clf_dataset):
    """GPU inference via nvForest — Linux/WSL2 only. Gate: checkpoint:human-verify."""
    import nvforest
    ...
```

---

### `python/tests/conftest.py` (update — add export fixtures)

**Analog:** `python/tests/parity/conftest.py` (lines 1–147) — the full session-scoped fixture pattern is the direct template.

**Session-scoped dataset reuse pattern** (`conftest.py` lines 79–95):
```python
# Reuse the existing clf_dataset / reg_dataset session fixtures.
# Add export-specific fixtures in the top-level conftest.py so test_export.py
# can import them without reaching into the parity subdirectory.
@pytest.fixture(scope="session")
def trained_clf_ir_small(clf_dataset) -> str:
    """50-tree, depth-6 clf IR — small enough for TL2cgen CI timeout."""
    ...
```

---

### `scripts/import_smoke_09.py` (utility, smoke test)

**Analog:** Phase 1 `scripts/import_smoke.py` (referenced in `01-03-PLAN.md` lines 39–55 as `provides: "Clean-venv import + entrypoint-call smoke test"`).

**Core smoke pattern** (01-03-PLAN.md artifact spec):
```python
# Follows the Phase 1 import_smoke.py pattern exactly.
# Run inside a clean venv (no dev dependencies) after: pip install <wheel>
import sylva
print(f"sylva version: {sylva.__version__}")

e = sylva.ExtraTreesClassifier()
print(f"ExtraTreesClassifier(): {e}")
print("OK — sylva imported and estimator instantiated in clean venv")
```

**Dynamic-loading verification** (`01-03-PLAN.md` must_haves line 29):
```python
# The wheel uses cudarc dynamic-loading: CUDA resolves at first GPU use,
# not at import time. Smoke must succeed even without a GPU present.
# Do NOT call any GPU method in the smoke script.
```

---

### `pyproject.toml` (update)

**Analog:** existing `pyproject.toml` (lines 1–22) — the proven Phase 1 maturin abi3 + dynamic-loading config. Phase 9 generalizes the package name and manifest-path.

**Proven config to copy** (`pyproject.toml` lines 1–22):
```toml
[build-system]
requires = ["maturin>=1.14,<2.0"]
build-backend = "maturin"

[project]
name = "sylva"                               # ← rename from "sylva-cuda"
description = "GPU-native forest ensembles — Extra Trees and Random Forest."
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dynamic = ["version"]
dependencies = [
    "numpy>=1.25",
]

[tool.maturin]
manifest-path = "crates/sylva-core/Cargo.toml"   # ← update to full package crate
module-name = "sylva"
# abi3: one cp310-abi3-win_amd64 wheel works on any CPython >= 3.10
# dynamic-loading: CUDA resolves at runtime (D-02 shipping config proven Phase 1)
```

**Key constraint:** `features = ["pyseam"]` MUST NOT be set in the wheel build — pyseam is test-only. The shipping build uses the default feature set plus `cudarc dynamic-loading`.

---

### `crates/sylva-core/src/pyseam.rs` (update — add `export_ir_json` fn if needed)

**Analog:** existing `pyseam.rs` (lines 200–270, `py_fit_cpu`) — the established pattern for exposing an IR handle via `#[pyfunction]`.

**IR JSON passthrough pattern** (`pyseam.rs` lines 262–269):
```rust
// If the Phase-5 estimator stores the IR internally, expose a getter so
// export.py can retrieve it. Mirror py_fit_cpu's return-as-JSON pattern:
#[pyfunction]
#[pyo3(name = "get_ir_json")]
fn py_get_ir_json(estimator: &SylvaEstimator) -> PyResult<String> {
    serde_json::to_string(&estimator.ir)
        .map_err(|e| PyRuntimeError::new_err(format!("get_ir_json: {e}")))
}
```

**Error mapping pattern** (`pyseam.rs` lines 46–53) — copy verbatim for any new pyseam fn:
```rust
fn sylva_error_to_pyerr(err: SylvaError) -> PyErr {
    match err {
        SylvaError::InvalidInput(_) | SylvaError::InvalidConfig(_) => {
            PyValueError::new_err(err.to_string())
        }
        SylvaError::InvalidIr(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
```

**ForestIR serde pattern** (`ir.rs` lines 22 and 181–186):
```rust
// ForestIR already derives Serialize + Deserialize (ir.rs line 22).
// Serialize to JSON for the FFI handle (proven pattern from py_fit_cpu):
serde_json::to_string(&ir)
    .map_err(|e| SylvaError::InvalidIr(format!("ForestIR serialization failed: {e}")))
// Deserialize in Python-side: json.loads(ir_json) — no new PyO3 binding needed.
```

---

### `docs/INSTALL.md` (new, documentation)

**Analog:** `VERSIONS.md` (runtime prerequisite documentation pattern from Phase 1). No close code analog — this is prose documentation.

**Required sections** (from RESEARCH.md Pattern 3 and §CUDA as runtime prerequisite):
- Prerequisites: NVIDIA GPU + driver ≥ CUDA 12.x, CUDA Toolkit 12.x installed, CUDA DLLs on PATH
- Install: `pip install sylva` (wheel), `pip install treelite==4.7.0 tl2cgen==1.0.0` (optional serving)
- Platform notes: FIL/nvForest Linux/WSL2 only; TL2cgen Windows MSVC supported
- Validation: `python -c "import sylva; print(sylva.__version__)"`

---

## Shared Patterns

### IR JSON Handle (FFI Boundary)
**Source:** `crates/sylva-core/src/pyseam.rs` lines 262–269  
**Apply to:** `export.py`, `test_export.py`  
The established contract: ForestIR crosses the PyO3 boundary as a JSON string (`str`). Python side calls `json.loads()`. No pointer or Rust struct crosses. `export_to_treelite(ir_json: str)` follows this contract.

### Validation at Boundary
**Source:** `pyseam.rs` lines 240–257 (`py_fit_cpu` input validation block)  
**Apply to:** `export.py`  
Validate the ForestIR JSON (check required keys, array lengths, `tree_offsets` invariants) before the first `ModelBuilder` call. Raise `ValueError` for caller errors. Mirror `ir.rs` `validate_structure()` logic in Python.

### Session-Scoped Expensive Fixtures
**Source:** `python/tests/parity/conftest.py` lines 79–95  
**Apply to:** `python/tests/conftest.py` update, `test_export.py`  
Training a forest is expensive. Use `scope="session"` fixtures for trained IR handles. Use small forests (50 trees, `max_depth=6`) for CI round-trip tests to avoid TL2cgen MSVC compilation timeout (RESEARCH.md Pitfall 4).

### `np.testing.assert_allclose` Parity Assert
**Source:** `test_distributional_parity.py` (structure)  
**Apply to:** `test_export.py`  
Use `atol=1e-5` as the parity tolerance for gtil and TL2cgen round-trips. Include `err_msg=` with a diagnostic hint (which postprocessor may be wrong).

### abi3 + dynamic-loading Wheel Config
**Source:** `pyproject.toml` lines 15–22 (Phase 1 proven config)  
**Apply to:** `pyproject.toml` update  
Do not change the maturin feature flags from the Phase 1 proof. Only update `name`, `manifest-path`, `module-name`, and add `dependencies = ["numpy>=1.25"]`. The `dynamic-loading` cudarc feature is the D-02 shipping config — do not regress to static linking.

### serde_json Round-Trip
**Source:** `ir.rs` lines 181–186 (`serde_round_trip` test)  
**Apply to:** `test_export.py` (verifying IR JSON hydration)  
The `ForestIR` `serde_json` round-trip is proven in `ir.rs` tests. The Python `json.loads()` counterpart should be validated in `test_export.py` with a structural metadata check (num_feature, n_trees, task) before the prediction parity check.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `docs/INSTALL.md` | documentation | — | No install doc exists yet; closest is `VERSIONS.md` prose but it is a different artifact type |
| FIL/nvForest test path in `test_export.py` | test | event-driven (GPU) | No GPU inference tests exist in the codebase; gate behind `checkpoint:human-verify` and `@pytest.mark.linux` |

---

## Key Anti-Pattern Reminders (from RESEARCH.md)

1. **Global vs tree-local node IDs:** Always subtract `tree_offsets[t]` before passing `left_child_key`/`right_child_key` to `ModelBuilder`. Source: `ir.rs` line 57 (`tree_offsets` field) + RESEARCH.md Pitfall 1.
2. **Postprocessor for probability leaves:** Use `postprocessor="identity"` — ET/RF `leaf_proba` already sums to 1 (proven by `ir.rs` test `classifier_leaf_proba_sums_to_one`, lines 208–218). Using `"softmax"` re-normalizes and breaks parity.
3. **No `import_from_json()`:** Treelite 4.x removed this. Use `ModelBuilder` only.
4. **No FIL in Windows CI:** Gate behind `checkpoint:human-verify`.

---

## Metadata

**Analog search scope:** `crates/sylva-core/src/`, `python/tests/parity/`, `pyproject.toml`, `.planning/phases/01-*/01-03-PLAN.md`
**Files scanned:** 10
**Pattern extraction date:** 2026-06-27
