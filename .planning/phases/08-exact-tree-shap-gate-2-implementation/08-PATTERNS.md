# Phase 8: Exact Tree SHAP (Gate 2 + Implementation) — Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 9
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/sylva-shap/Cargo.toml` | config | — | `crates/sylva-cuda/Cargo.toml` | exact |
| `crates/sylva-shap/src/lib.rs` | config/re-export | — | `crates/sylva-cuda/src/lib.rs` | exact |
| `crates/sylva-shap/src/error.rs` | utility | — | `crates/sylva-core/src/error.rs` (implied by `SylvaError` pattern) | role-match |
| `crates/sylva-shap/src/path.rs` | utility | transform | `crates/sylva-core/src/cpu/predict.rs` `traverse_tree` fn | role-match |
| `crates/sylva-shap/src/base_value.rs` | utility | transform | `crates/sylva-core/src/cpu/predict.rs` tree loop + `leaf_offset` access | role-match |
| `crates/sylva-shap/src/cpu.rs` | service | request-response (batch) | `crates/sylva-core/src/cpu/predict.rs` `predict_forest` | exact |
| `crates/sylva-shap/src/output.rs` | model | — | `crates/sylva-core/src/backend.rs` `Predictions` enum (implied) | role-match |
| `crates/sylva-core/src/pyseam.rs` (add `shap_values` fn) | middleware | request-response | `crates/sylva-core/src/pyseam.rs` `py_predict_cpu` | exact |
| `python/tests/shap/test_shap_parity.py` | test | request-response | `python/tests/parity/test_distributional_parity.py` | exact |
| Gate-2 spike checkpoint (committed decision record) | config | — | n/a — human-verify artifact | no analog |

---

## Pattern Assignments

### `crates/sylva-shap/Cargo.toml` (workspace member config)

**Analog:** `crates/sylva-cuda/Cargo.toml`

**Workspace member header pattern** (lines 1-16):
```toml
[package]
name = "sylva-shap"
version = "0.1.0"
edition.workspace = true
license.workspace = true
rust-version.workspace = true
authors.workspace = true
repository.workspace = true
description = "Exact TreeSHAP attributions from ForestIR (CPU-first; GPU Wave 3)."

[lib]
crate-type = ["rlib"]   # rlib only — no PyO3 extension; exposed via sylva-core pyseam
```

**Dependency pattern** — mirror sylva-cuda's workspace-inherit style:
```toml
[dependencies]
sylva-core = { path = "../sylva-core" }
ndarray = { workspace = true }
rayon = { workspace = true }
thiserror = { workspace = true }

[dev-dependencies]
approx = { workspace = true }
```

**Workspace root addition** (`Cargo.toml` line 8 — only change):
```toml
members = ["crates/sylva-cuda", "crates/sylva-core", "crates/sylva-shap"]
```

---

### `crates/sylva-shap/src/lib.rs` (crate root, re-exports)

**Analog:** `crates/sylva-cuda/src/lib.rs` (lines 1-27 — module declaration + pub use pattern)

**Module declaration + pub-use pattern:**
```rust
//! sylva-shap — exact per-feature TreeSHAP attributions from ForestIR.
//!
//! CPU-first (Wave 1). GPU path is feature-gated (Wave 3).
//! Consume ForestIR READ-ONLY via `&ForestIR`. No IR mutations.

pub mod base_value;
pub mod cpu;
pub mod error;
pub mod output;
pub mod path;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use cpu::compute_shap_cpu;
pub use error::ShapError;
pub use output::ShapOutput;
```

---

### `crates/sylva-shap/src/error.rs` (ShapError enum)

**Analog:** Error enum pattern from `crates/sylva-core/src/pyseam.rs` lines 46-53 (`sylva_error_to_pyerr` reveals the three SylvaError variants)

**thiserror enum pattern — mirror SylvaError's structure:**
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapError {
    #[error("invalid ForestIR: {0}")]
    InvalidIr(String),
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
}
```

**PyO3 error mapping pattern** (copy from `pyseam.rs` lines 46-53):
```rust
// In pyseam.rs — add alongside sylva_error_to_pyerr:
fn shap_error_to_pyerr(err: ShapError) -> PyErr {
    match err {
        ShapError::InvalidIr(_) | ShapError::DimensionMismatch(_) => {
            PyValueError::new_err(err.to_string())
        }
        ShapError::CudaError(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
```

---

### `crates/sylva-shap/src/path.rs` (PathEntry + extend/unwind DP)

**Analog:** `crates/sylva-core/src/cpu/predict.rs` `traverse_tree` fn (lines 84-106) — same IR field access idiom; SHAP path.rs uses the same `ir.is_leaf[node]`, `ir.feature_id[node]`, `ir.threshold[node]`, `ir.default_child[node]`, `ir.left_child[node]`, `ir.right_child[node]` pattern.

**NaN routing pattern to mirror exactly** (lines 93-104 of predict.rs):
```rust
// COPY this NaN-first routing order into tree_recurse's hot/cold assignment:
node = if v.is_nan() {
    ir.default_child[node] as usize   // D-01: NaN goes to default_child
} else if v <= ir.threshold[node] {
    ir.left_child[node] as usize
} else {
    ir.right_child[node] as usize
};
```

**PathEntry struct** (from RESEARCH.md algorithm sketch, lines 187-192):
```rust
/// One entry on the active-feature path for the EXTEND/UNWIND DP.
#[derive(Clone, Debug)]
pub struct PathEntry {
    pub feature_idx: i32,    // -1 for the root/bias slot
    pub zero_fraction: f32,  // node_weighted_count[child] / node_weighted_count[parent]
    pub one_fraction: f32,   // 1.0 when feature IS in the active set
    pub pweight: f32,        // accumulated path weight (EXTEND bookkeeping)
}
```

**Cover fraction access pattern** — derived from `ir.node_weighted_count` (ir.rs lines 42-43):
```rust
let w = ir.node_weighted_count[node];
let hot_zero  = ir.node_weighted_count[hot]  / w;
let cold_zero = ir.node_weighted_count[cold] / w;
// Guard: w > 0.0 is guaranteed by validate_structure() contract
```

---

### `crates/sylva-shap/src/base_value.rs` (E[f(X)] precomputation)

**Analog:** `crates/sylva-core/src/cpu/predict.rs` lines 43-58 (leaf payload access via `leaf_offset`)

**Leaf value access pattern** (predict.rs lines 47-49 for classification; 66-67 for regression):
```rust
// Classification: leaf_proba block
let lo = ir.leaf_offset[leaf_node] as usize;
let src = &ir.leaf_proba[lo * n_classes..(lo + 1) * n_classes];

// Regression: leaf_value scalar
let lo = ir.leaf_offset[leaf_node] as usize;
ir.leaf_value[lo]
```

**Per-tree base value loop** (mirrors predict_forest tree loop, lines 42-58):
```rust
pub fn expected_value_per_tree(ir: &ForestIR) -> Vec<f64> {
    (0..ir.n_trees)
        .map(|t| {
            let root = ir.tree_root[t] as usize;
            let root_count = ir.node_weighted_count[root] as f64;
            // DFS: for each leaf, accumulate leaf_value * (leaf_count / root_count)
            dfs_expected_value(ir, root, root_count)
        })
        .collect()
}
```

---

### `crates/sylva-shap/src/cpu.rs` (row-parallel CPU TreeSHAP)

**Analog:** `crates/sylva-core/src/cpu/predict.rs` `predict_forest` fn (lines 24-75) — the structural template is identical: boundary validation, task dispatch, tree loop, accumulate in mutable output, divide by n_trees at the end.

**Boundary validation pattern** (predict.rs lines 28-37):
```rust
// Mirror exactly — same guard order:
if n_features != ir.n_features {
    return Err(ShapError::DimensionMismatch(format!(
        "compute_shap_cpu: X has {} features but IR was trained on {}",
        n_features, ir.n_features
    )));
}
if ir.n_trees == 0 {
    return Err(ShapError::InvalidIr("ForestIR has 0 trees".into()));
}
// Also call ir.validate_structure() at entry (security control):
ir.validate_structure().map_err(|e| ShapError::InvalidIr(e.to_string()))?;
```

**Task dispatch pattern** (predict.rs lines 39-74 — mirror the `match ir.task` structure):
```rust
match ir.task {
    Task::Classification { n_classes } => { /* ShapOutput with per-class dim */ }
    Task::Regression => { /* ShapOutput with single output dim */ }
}
```

**f64 accumulation pattern** (RESEARCH.md requirement — NOT from predict.rs which uses f32; this is the key difference):
```rust
// phi accumulates in f64 to avoid precision loss at depth > 15
let mut phi = vec![0.0f64; n_features];
let mut phi_bias = 0.0f64;
tree_shap(ir, t, x_row, &mut phi, &mut phi_bias);
// Downcast to f32 only at output stage (after summing all trees):
phi_row[j] += (phi[j] / ir.n_trees as f64) as f32;
```

**rayon row-parallel outer loop** (RESEARCH.md Pattern 1 — same pattern as Phase 5 predict, but with rayon):
```rust
use rayon::prelude::*;
// axis_iter_mut(Axis(0)).into_par_iter() for row parallelism
// (predict.rs uses a sequential row loop — shap.rs upgrades to rayon par_iter)
```

---

### `crates/sylva-shap/src/output.rs` (ShapOutput struct)

**Analog:** `Predictions` enum in `crates/sylva-core/src/backend.rs` (not read but pattern inferred from pyseam.rs lines 327-339 which unpacks it)

**Output struct pattern:**
```rust
use ndarray::{Array2, Array3};

/// SHAP attribution output for a batch of rows.
#[derive(Debug)]
pub struct ShapOutput {
    /// Attribution values.
    /// Regression: shape [n_rows, n_features] (Array2).
    /// Classification: shape [n_rows, n_features, n_classes] (Array3).
    pub values: ShapValues,
    /// Per-row base value E[f(X)].
    /// Regression: shape [n_rows]; Classification: shape [n_rows, n_classes].
    pub base_values: BaseValues,
}

pub enum ShapValues {
    Regression(Array2<f32>),
    Classification(Array3<f32>),
}
```

---

### `crates/sylva-core/src/pyseam.rs` — add `shap_values` PyO3 fn

**Analog:** `py_predict_cpu` in `crates/sylva-core/src/pyseam.rs` (lines 298-340) — this is the exact template to copy for `.shap_values()`.

**GIL-release pattern** (pyseam.rs lines 319-324):
```rust
// pyo3 0.29 idiom — copy verbatim, swap predict for shap:
let result = py.detach(|| -> Result<ShapOutput, ShapError> {
    sylva_shap::compute_shap_cpu(&ir, x_view)
});
let result = result.map_err(shap_error_to_pyerr)?;
```

**IR deserialization pattern** (pyseam.rs lines 306-308):
```rust
let ir: ForestIR = serde_json::from_str(ir_handle)
    .map_err(|e| PyValueError::new_err(format!("shap_values: invalid ir_handle: {e}")))?;
```

**Array validation pattern** (pyseam.rs lines 310-315):
```rust
if x_view.ncols() != ir.n_features {
    return Err(PyValueError::new_err(format!(
        "shap_values: X has {} features but model has {}",
        x_view.ncols(), ir.n_features
    )));
}
```

**numpy return pattern** (pyseam.rs lines 327-339):
```rust
// Classification — list of per-class 2D arrays (sklearn convention):
// shap_values returns Vec<Array2<f32>> (one per class); caller wraps in Python list.
// Regression — single Array2<f32> shape (n_rows, n_features).
Ok(arr.into_pyarray(py))
```

**Module registration pattern** (pyseam.rs lines 391-396):
```rust
// Add to the existing sylva_core_pyseam module fn:
module.add_function(wrap_pyfunction!(py_shap_values, module)?)?;
```

---

### `python/tests/shap/test_shap_parity.py` (SHAP-03 validation harness)

**Analog:** `python/tests/parity/test_distributional_parity.py` (lines 1-60) — same import structure, same `np.allclose` tolerance pattern, same `sylva_core_pyseam` import idiom.

**Import pattern** (mirror test_distributional_parity.py lines 29-50):
```python
from __future__ import annotations

import numpy as np
import pytest
import shap

import sylva_core_pyseam as sylva  # existing seam; shap_values added in Phase 8
from .conftest import VERSION_MANIFEST
```

**Fixture pattern** (mirror test_distributional_parity.py conftest/dataset style):
```python
@pytest.fixture
def shallow_clf_ir(small_clf_dataset):
    """ForestIR JSON for a shallow (max_depth=5) classifier — fast, precise."""
    X, y = small_clf_dataset
    return sylva.fit_cpu(X, y, {"n_estimators": 10, "max_depth": 5, "seed": 42})

@pytest.fixture
def deep_clf_ir(covertype_subset):
    """ForestIR JSON for a deep (max_depth=12) classifier — exercises float64 accum."""
    X, y = covertype_subset
    return sylva.fit_cpu(X, y, {"n_estimators": 100, "max_depth": 12, "seed": 42})
```

**shap custom-dict injection pattern** (RESEARCH.md Pattern 2 + Don't-Hand-Roll section):
```python
# Build shap reference via custom-dict injection (no sklearn estimator needed):
tree_dicts = sylva.ir_to_shap_dict(ir_handle)  # Rust helper via PyO3
model = {"trees": tree_dicts}
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_ref = explainer.shap_values(X)
```

**np.allclose tolerance pattern** (RESEARCH.md Validation Architecture):
```python
# Shallow trees (max_depth <= 8): atol=1e-4 (shap's own test suite tolerance)
assert np.allclose(sylva_vals, shap_ref, atol=1e-4), (
    f"SHAP parity failed: max_abs={np.abs(sylva_vals - shap_ref).max():.2e}"
)

# Deep trees (max_depth > 8): atol=5e-4 (float32 IR precision limit)
assert np.allclose(sylva_vals, shap_ref, atol=5e-4), (
    f"Deep-tree SHAP parity failed: max_abs={np.abs(sylva_vals - shap_ref).max():.2e}"
)
```

**Additivity check pattern** (RESEARCH.md Don't-Hand-Roll — use shap's built-in):
```python
# shap.TreeExplainer check_additivity=True validates sum(phi) + E[f] == f(x)
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
_ = explainer.shap_values(X, check_additivity=True)  # raises if additivity fails
```

---

## Shared Patterns

### IR Consumption (read-only borrow)
**Source:** `crates/sylva-core/src/cpu/predict.rs` line 24, `crates/sylva-core/src/ir.rs` lines 88-146
**Apply to:** `cpu.rs`, `base_value.rs`, `path.rs`
```rust
// All sylva-shap public fns accept &ForestIR — never &mut ForestIR
pub fn compute_shap_cpu(ir: &ForestIR, x: ArrayView2<f32>) -> Result<ShapOutput, ShapError>
// Call validate_structure() at every public entry point:
ir.validate_structure().map_err(|e| ShapError::InvalidIr(e.to_string()))?;
```

### NaN Routing (copy exactly, do not diverge)
**Source:** `crates/sylva-core/src/cpu/predict.rs` lines 93-104
**Apply to:** `path.rs` `tree_recurse` hot/cold assignment
The `v.is_nan()` check MUST come before `v <= ir.threshold[node]`. Any deviation breaks attribution correctness for NaN rows (RESEARCH.md Pitfall 3).

### leaf_offset Access
**Source:** `crates/sylva-core/src/cpu/predict.rs` lines 47-49, 66-67; `crates/sylva-core/src/ir.rs` lines 51-53
**Apply to:** `path.rs` (leaf value at UNWIND), `base_value.rs`
```rust
// Classifier:
let lo = ir.leaf_offset[node] as usize;
let proba = &ir.leaf_proba[lo * n_classes..(lo + 1) * n_classes];
// Regressor:
let lo = ir.leaf_offset[node] as usize;
let v = ir.leaf_value[lo];
```

### GIL Release (PyO3 0.29)
**Source:** `crates/sylva-core/src/pyseam.rs` lines 262-269 and 319-324
**Apply to:** `py_shap_values` in `pyseam.rs`
```rust
// pyo3 0.29: py.allow_threads() was renamed to py.detach()
let result = py.detach(|| -> Result<ShapOutput, ShapError> { ... });
```

### thiserror Error Enum
**Source:** `crates/sylva-core/src/pyseam.rs` lines 46-53 (error mapping idiom)
**Apply to:** `error.rs`, `pyseam.rs` `shap_error_to_pyerr`
`InvalidInput`/`InvalidIr` → `PyValueError`; internal failures → `PyRuntimeError`. No `.unwrap()` across FFI.

### f64 Accumulation
**Source:** RESEARCH.md Pitfall 1 + Anti-Patterns section; NOT present in `predict.rs` (which uses f32)
**Apply to:** `cpu.rs` `tree_shap`, `path.rs` `phi` vector
All intermediate SHAP sums use `f64`. Downcast to `f32` only in `ShapOutput` construction after dividing by `n_trees`. This is the single most important divergence from the predict.rs analog.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Gate-2 spike checkpoint (committed decision record) | checkpoint | — | A human-verify artifact, not a code file; no existing analog in the codebase |
| `crates/sylva-shap/src/gpu.rs` (Wave 3) | service | batch/CUDA | No GPU-SHAP kernel exists; GPU path is reimplemented in CUDA C from the GPUTreeSHAP Apache-2.0 algorithm — use the NVRTC launch pattern from `crates/sylva-cuda/src/nvrtc_launch.rs` when Wave 3 begins |
| `python/tests/shap/__init__.py` | config | — | Trivial empty file; no analog needed |

---

## Metadata

**Analog search scope:** `crates/sylva-cuda/`, `crates/sylva-core/src/`, `python/tests/parity/`
**Files read:** 7 source files (ir.rs, cpu/predict.rs, pyseam.rs, sylva-cuda/Cargo.toml, sylva-cuda/src/lib.rs, python/tests/parity/test_calibration.py, test_distributional_parity.py) + root Cargo.toml
**Pattern extraction date:** 2026-06-27
