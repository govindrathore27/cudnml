# Phase 5: Full Forest, RandomForest & sklearn Estimators — Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 21 new/modified files
**Analogs found:** 21 / 21 (all have direct analogs; greenfield Python package noted explicitly)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `crates/sylva-cuda/src/cuda_backend/mod.rs` | service (impl Backend, dispatch ET/RF) | request-response | `crates/sylva-core/src/cpu/fit.rs` `fit_forest` | role-match (extend) |
| `crates/sylva-cuda/src/cuda_backend/forest.rs` | orchestrator (tree loop, per-tree schedule) | event-driven (tree loop) | `crates/sylva-core/src/cpu/fit.rs` lines 88–116 (par_iter + assemble) | role-match |
| `crates/sylva-cuda/src/cuda_backend/scheduler.rs` | orchestrator (breadth-first frontier) | event-driven (frontier waves) | `crates/sylva-core/src/cpu/fit.rs` `build_tree`/`build_node` | role-match |
| `crates/sylva-cuda/src/cuda_backend/histogram.rs` | kernel launch wrapper (privatized hist) | CUDA transform | `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` | exact |
| `crates/sylva-cuda/src/cuda_backend/rf_split.rs` | kernel launch wrapper (scan + argmax) | CUDA transform | `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` + `crates/sylva-core/src/cpu/split_rf.rs` | role-match |
| `crates/sylva-cuda/src/cuda_backend/sample_weight.rs` | kernel launch wrapper (weighted hist) | CUDA transform | `crates/sylva-cuda/src/cuda_backend/histogram.rs` (histogram shape) | role-match |
| `crates/sylva-cuda/src/cuda_backend/arena.rs` | resource manager (fit-scoped slab) | CRUD (alloc/reuse) | `crates/sylva-cuda/src/cuda_backend/device_buffers.rs` (Phase-4 plan) | role-match |
| `crates/sylva-cuda/src/cuda_backend/cpu_cutover.rs` | utility (small/deep-node CPU finish) | request-response dispatch | `crates/sylva-core/src/cpu/fit.rs` `build_node` | role-match |
| `crates/sylva-cuda/src/cuda_backend/device_buffers.rs` | resource manager (arena-backed alloc) | CRUD | `crates/sylva-cuda/src/nvrtc_launch.rs` H2D/alloc patterns | role-match |
| `crates/sylva-cuda/src/cuda_backend/assemble.rs` | assembler (D2H → ForestIR) | transform | `crates/sylva-core/src/cpu/fit.rs` `assemble_forest` / `TreeFragment` | exact |
| `crates/sylva-cuda/src/kernels.rs` | kernel source strings | CUDA-C via NVRTC | `crates/sylva-cuda/src/kernels.rs` (existing Phase 1) | EXTEND (same file) |
| `crates/sylva-core/src/importance.rs` | utility (MDI from IR) | transform | `crates/sylva-core/src/ir.rs` `tree_node_range` + `crates/sylva-core/src/cpu/predict.rs` | role-match |
| `crates/sylva-core/src/config.rs` | config model | CRUD | `crates/sylva-core/src/config.rs` (existing) | EXTEND (same file) |
| `crates/sylva-core/src/pyseam.rs` | FFI boundary / estimator seam | request-response | `crates/sylva-core/src/pyseam.rs` (existing Phase 2–4) | EXTEND (promote test seam) |
| `python/sylva/__init__.py` | package root | — | `python/tests/parity/conftest.py` (only Python package scaffolding) | partial |
| `python/sylva/_base.py` | base estimator class | request-response | `crates/sylva-core/src/pyseam.rs` `parse_config` (config contract) | role-match |
| `python/sylva/ensemble.py` | estimator classes (4 classes) | request-response | RESEARCH.md Pattern 4 (sklearn BaseEstimator contract) | greenfield (see no-analog table) |
| `python/sylva/_dispatch.py` | utility (device dispatch) | request-response | `crates/sylva-core/src/pyseam.rs` `sylva_error_to_pyerr` error mapping | role-match |
| `python/tests/test_check_estimator.py` | Python test (CI gate) | request-response | `crates/sylva-core/tests/determinism.rs` (parametrized gate pattern) | role-match |
| `python/tests/test_estimator_api.py` | Python test (API invariants) | request-response | `crates/sylva-core/tests/invariants.rs` | role-match |
| `python/tests/test_feature_importances.py` | Python test (MDI invariant) | transform | `crates/sylva-core/tests/determinism.rs` (data fixture + byte-compare pattern) | role-match |
| `python/tests/gpu_forest/test_rf_cpu_gpu_bitexact.py` | Python test (bit-exact GPU gate) | request-response | `crates/sylva-core/tests/determinism.rs` + Phase-4 `gpu_parity/` idiom | exact |
| `python/tests/gpu_forest/test_sample_weight.py` | Python test (weighted hist parity) | transform | Phase-4 `gpu_parity/` harness shape | role-match |
| `python/benchmarks/comparative_study.py` | benchmark harness | batch | `python/tests/parity/conftest.py` VERSION_MANIFEST + datasets fixture | role-match |

---

## Pattern Assignments

### `crates/sylva-cuda/src/cuda_backend/forest.rs` (orchestrator, tree loop)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` lines 88–116

**Tree-loop pattern** (`crates/sylva-core/src/cpu/fit.rs`, lines 88–116):
```rust
// The GPU forest.rs mirrors this structure — replace par_iter with a serial
// or device-parallel loop; replace build_tree with GPU scheduler::fit_tree.
let trees: Vec<TreeFragment> = (0..n_trees)
    .into_par_iter()
    .map(|tree_id| {
        // RF: bootstrap row set keyed by (seed, tree) — order-independent.
        // ET: all rows (no bootstrap).
        let tree_rows: Vec<usize> = if use_bootstrap {
            bootstrap_indices(n_rows, seed, tree_id as u32)   // Philox sentinel keyed
        } else {
            all_rows.clone()
        };
        build_tree(x, &y_vec, &tree_rows, tree_id as u32, cfg, task, criterion, resolved_max_features)
    })
    .collect();

assemble_forest(trees, n_features, task, criterion, cfg.seed)
```

**Bootstrap schedule reuse rule:** GPU `forest.rs` MUST call `bootstrap_indices(n_rows, seed, tree_id as u32)` from `crates/sylva-core/src/cpu/bootstrap.rs` for the host-side row mask, or inline the EXACT same Philox call with `node=BOOTSTRAP_NODE_SENTINEL` (`u32::MAX`) on the device. Never use a different RNG for bootstrap.

---

### `crates/sylva-cuda/src/cuda_backend/scheduler.rs` (orchestrator, breadth-first frontier)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` `build_node` (lines 210+)

The GPU scheduler replaces the CPU recursive call with a breadth-first frontier queue. The **split decision logic** (`criterion`, `min_samples_leaf`, `min_samples_split`, `max_depth` check, leaf-or-split) is a direct port of `build_node`'s guard conditions. Copy these guards verbatim:

```rust
// From crates/sylva-core/src/cpu/fit.rs — build_node guards (port to GPU scheduler)
let n = rows.len() as u64;
// Leaf conditions: depth limit, too few rows for a split, or no valid split found.
let make_leaf = cfg.max_depth.map_or(false, |d| depth >= d)
    || n < cfg.min_samples_split as u64
    || n < 2 * cfg.min_samples_leaf as u64;
```

**Sibling-subtraction bookkeeping** (new in Phase 5, no CPU analog — use RESEARCH.md Pattern 2):
```rust
// After building the smaller child's histogram directly:
let (small_range, large_range, large_is_left) =
    if n_left <= n_right { (left, right, false) } else { (right, left, true) };
launch_build_histogram(small_range, &mut d_small_hist)?;           // direct
launch_sibling_subtract(&d_parent_hist, &d_small_hist, &mut d_large_hist)?; // parent − small
// Integer subtraction only. Never subtract float sums.
```

---

### `crates/sylva-cuda/src/cuda_backend/histogram.rs` (kernel wrapper, privatized hist)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` (lines 124–168 per Phase-4 PATTERNS.md)

**Launch function pattern** — copy the exact template from Phase-4 PATTERNS.md (analog `run_histogram`):

```rust
// V5 validation before any device call:
if n_classes == 0 {
    return Err(CudaError::InvalidInput("n_classes must be > 0".into()));
}
// H2D via clone_htod (not deprecated memcpy_stod):
let d_x    = stream.clone_htod(x_col_major)?;
// shared_mem_bytes = privatized shared array for the hist:
let cfg = LaunchConfig {
    grid_dim:        (n_blocks, 1, 1),
    block_dim:       (BLOCK_SIZE, 1, 1),
    shared_mem_bytes: (n_bins * n_classes * std::mem::size_of::<u32>()) as u32,
};
// SAFETY: args match kernel signature; sh zero-inited before read;
// __syncthreads() before flush; integer atomicAdd only (no float atomics).
unsafe { builder.launch(cfg)?; }
```

**Compile flags** — carry Phase-4 locked decision:
```rust
// NVRTC compile options: -fmad=false is MANDATORY. Never --use_fast_math.
options: vec!["-lineinfo".to_string(), "-fmad=false".to_string()],
```

---

### `crates/sylva-cuda/src/kernels.rs` (EXTEND — add RF/forest kernel source strings)

**Analog:** `crates/sylva-cuda/src/kernels.rs` (existing, lines 1–69)

**New kernel string shape** — mirror `HISTOGRAM_PRIVATIZED_SRC` pattern (lines 47–64):

```rust
// RF scan+argmax kernel — one new pub const per kernel, r#"..."# raw string:
pub const RF_BINNED_HIST_SRC: &str = r#"
// per-(feature,bin) histogram variant; shares privatized sh pattern with ET:
// sh[bin * n_classes + cls] = 0; __syncthreads(); integer atomicAdd; flush.
// -fmad=false (compile flag); integer atomics only (DET-01).
extern "C" __global__ void rf_binned_hist(...) { ... }
"#;

pub const RF_SCAN_ARGMAX_SRC: &str = r#"
// Inclusive prefix-scan over bins → cumulative left counts; argmax with
// (feature, threshold_bits) tie-break (same op order as criterion.rs).
// Single-block scan, Hillis-Steele or Blelloch with FIXED order (associative
// integer counts — deterministic).
extern "C" __global__ void rf_scan_argmax(...) { ... }
"#;

pub const SIBLING_SUBTRACT_SRC: &str = r#"
// child_large[i] = parent[i] - child_small[i]; integer only. Exact.
extern "C" __global__ void sibling_subtract(
    const unsigned int* parent, const unsigned int* child_small,
    unsigned int* child_large, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) child_large[i] = parent[i] - child_small[i];
}
"#;

pub const WEIGHTED_HIST_SRC: &str = r#"
// Weighted integer histogram: atomicAdd scaled u64 fixed-point weights
// into shared (NOT float atomicAdd — non-deterministic + non-subtractable).
extern "C" __global__ void weighted_hist(...) { ... }
"#;

// Named constant additions (no magic numbers at launch sites):
pub const RF_BLOCK_SIZE:    u32   = 256;
pub const MAX_BINS:         usize = 256; // BIN_COUNT already exists; keep consistent
```

---

### `crates/sylva-cuda/src/cuda_backend/rf_split.rs` (kernel wrapper, RF scan+argmax)

**Analog:** `crates/sylva-core/src/cpu/split_rf.rs` `best_split` function (lines 94–213) — the host-side scoring fallback MUST use the exact same criterion op order.

**Host-side scoring fallback pattern** (copy criterion op order from `split_rf.rs` lines 179–186):
```rust
// Return per-feature cumulative counts to host, score with cpu::criterion,
// argmax host-side (zero device-float-parity risk — Phase-4 locked decision).
// Only move on-device if host round-trips dominate profile (Phase-7 concern).
let improvement = proxy_improvement(parent_impurity, left_imp, right_imp, n_left, n_right);
// Tie-break: (feature_id, threshold_bits) total order — from split_rf.rs lines 188–195:
let is_better = match &best {
    None => true,
    Some(b) => improvement > b.improvement
        || (improvement == b.improvement
            && (feat, thr_bits) < (b.feature_id, b.threshold.to_bits()))
};
```

**Binned-canonical contract (OQ2 locked decision):** the binned RF split is canonical. The CPU oracle in `split_rf.rs` gains a `best_split_binned` path in Phase 5 that evaluates `BinEdges` boundaries rather than raw midpoints. The GPU `rf_split.rs` calls the same `BinnedMatrix`/`BinEdges` layout. Both must agree bit-exactly.

---

### `crates/sylva-cuda/src/cuda_backend/arena.rs` (resource manager, fit-scoped slab)

**Analog:** `crates/sylva-cuda/src/cuda_backend/device_buffers.rs` H2D alloc pattern (Phase-4 plan); safe `alloc_zeros` / `clone_htod` on `CudaStream`.

**Locked decision (OQ1 resolved — safe pre-sized slab):** Use the cudarc safe API (`stream.alloc_zeros::<T>(n)?`), not raw `driver::sys cuMemAllocAsync`. Pre-size the slab once at fit start; reuse buffers across waves. Zero `unsafe` in `arena.rs` except where a `// SAFETY:` comment is strictly necessary.

```rust
// Arena shape — pre-size at fit start, reuse across waves:
pub struct FitArena {
    // Per-wave histogram buffer, sized for (max_nodes_per_wave * n_bins * n_classes)
    d_histogram: CudaSlice<u32>,
    // Per-wave row-index buffer, sized for n_rows
    d_row_indices: CudaSlice<u32>,
    // Parent histogram retained for sibling subtraction
    d_parent_hist: CudaSlice<u32>,
}

impl FitArena {
    pub fn new(stream: &CudaStream, max_nodes: usize, n_bins: usize, n_classes: usize, n_rows: usize)
        -> Result<Self, CudaError>
    {
        Ok(Self {
            d_histogram:   stream.alloc_zeros::<u32>(max_nodes * n_bins * n_classes)?,
            d_row_indices: stream.alloc_zeros::<u32>(n_rows)?,
            d_parent_hist: stream.alloc_zeros::<u32>(n_bins * n_classes)?,
        })
    }
    // Reuse across waves: zero-fill with a memset kernel, not re-alloc.
}
```

---

### `crates/sylva-cuda/src/cuda_backend/assemble.rs` (assembler, D2H → ForestIR)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` `assemble_forest` + `TreeFragment` (lines 122–161)

**Global-offset assembly contract** (from `fit.rs` TreeFragment shape):
```rust
// TreeFragment fields the GPU must produce and D2H before assembly:
struct TreeFragment {
    feature_id:          Vec<i32>,   // LEAF_FEATURE = -1
    threshold:           Vec<f32>,
    left_child:          Vec<i32>,   // local id; assemble adds tree_offset
    right_child:         Vec<i32>,
    default_child:       Vec<i32>,
    is_leaf:             Vec<bool>,
    node_sample_count:   Vec<u64>,
    node_weighted_count: Vec<f32>,
    impurity:            Vec<f32>,
    leaf_value:          Vec<f32>,
    leaf_proba:          Vec<f32>,
    leaf_offset:         Vec<i32>,   // local leaf slot; assemble adds leaf_offset_base
    n_leaf_slots:        usize,
}
// assemble_forest adjusts child ids by node_offset and leaf_offsets by leaf_offset_base.
// The GPU assemble.rs must do the SAME global-offset math, producing an identical
// ForestIR layout so the byte-exact serde gate passes.
```

**ForestIR fields required** (`crates/sylva-core/src/ir.rs` lines 23–65): all nine per-node arrays + `tree_offsets`, `tree_root`, `n_trees`, `n_features`, `task`, `criterion`, `seed`. Do not add or rename fields — the MDI code in `importance.rs` reads `impurity`, `node_weighted_count`, `feature_id`, `left_child`, `right_child`, `is_leaf`, `leaf_offset` by their current names.

---

### `crates/sylva-core/src/importance.rs` (utility, MDI from IR)

**Analog:** `crates/sylva-core/src/ir.rs` `tree_node_range` (line 82) + `crates/sylva-core/src/cpu/predict.rs` forest-traversal loop (lines 42–57)

**MDI pattern** (from RESEARCH.md Code Examples — implement verbatim, f64 internally):
```rust
// crates/sylva-core/src/importance.rs — device-neutral; reads only ForestIR fields.
pub fn feature_importances(ir: &ForestIR) -> Vec<f32> {
    let mut imp = vec![0f64; ir.n_features];
    for t in 0..ir.n_trees {
        let mut tree_imp = vec![0f64; ir.n_features];
        for n in ir.tree_node_range(t) {      // ir.rs line 82: tree_offsets[t]..tree_offsets[t+1]
            if ir.is_leaf[n] { continue; }
            let (l, r) = (ir.left_child[n] as usize, ir.right_child[n] as usize);
            let (wn, wl, wr) = (ir.node_weighted_count[n] as f64,
                                ir.node_weighted_count[l] as f64,
                                ir.node_weighted_count[r] as f64);
            let decrease = wn*ir.impurity[n] as f64
                         - wl*ir.impurity[l] as f64
                         - wr*ir.impurity[r] as f64;
            tree_imp[ir.feature_id[n] as usize] += decrease;
        }
        let s: f64 = tree_imp.iter().sum();
        if s > 0.0 { for (a, v) in imp.iter_mut().zip(&tree_imp) { *a += v / s; } }
    }
    let n = ir.n_trees as f64;
    imp.iter().map(|v| (v / n) as f32).collect()
}
// Invariant: result.iter().sum::<f32>() ≈ 1.0 (±float tolerance across trees).
// f64 accumulation matches sklearn's double-precision MDI recipe.
```

---

### `crates/sylva-core/src/config.rs` (EXTEND — add `max_samples`, `class_weight`)

**Analog:** `crates/sylva-core/src/config.rs` (existing, lines 63–110)

**Extend `TrainConfig` struct** — add two fields after `bootstrap` (line 72), following the existing `Option<usize>` / `Option<_>` pattern:
```rust
// Add to TrainConfig (crates/sylva-core/src/config.rs, after line 72):
pub struct TrainConfig {
    // ... existing fields unchanged ...
    pub bootstrap:         bool,
    /// None = n_rows (sklearn default); Some(f32 in (0,1]) = fraction;
    /// Some(usize) encoded as MaxSamples enum. Only meaningful when bootstrap=true.
    pub max_samples:       Option<MaxSamples>,
    /// None = uniform weights; "balanced" folds into per-row sample_weight at fit.
    pub class_weight:      Option<ClassWeightSpec>,
    // ... rest unchanged ...
}

// Add new enums (follow MaxFeatures pattern at lines 34–60):
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaxSamples {
    Fraction(f32),   // in (0, 1]
    Count(usize),    // absolute row count
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassWeightSpec {
    Balanced,
    Map(Vec<(usize, f32)>),  // class_id → weight
}
```

**Validation extension** (follow `validate()` pattern at lines 79–109):
```rust
// Add to validate() — validate max_samples requires bootstrap=true:
if let Some(MaxSamples::Fraction(f)) = self.max_samples {
    if !(f > 0.0 && f <= 1.0) {
        return Err(SylvaError::InvalidConfig("max_samples fraction must be in (0, 1]".into()));
    }
    if !self.bootstrap {
        return Err(SylvaError::InvalidConfig("max_samples requires bootstrap=true".into()));
    }
}
```

---

### `crates/sylva-core/src/pyseam.rs` (EXTEND — promote to full estimator seam)

**Analog:** `crates/sylva-core/src/pyseam.rs` (existing, lines 1–397 — read in full above)

**Error mapping pattern** (lines 46–53) — copy verbatim for all new boundary functions:
```rust
fn sylva_error_to_pyerr(err: SylvaError) -> PyErr {
    match err {
        SylvaError::InvalidInput(_) | SylvaError::InvalidConfig(_) =>
            PyValueError::new_err(err.to_string()),
        SylvaError::InvalidIr(_) =>
            PyRuntimeError::new_err(err.to_string()),
    }
}
```

**Config parsing pattern** (`parse_config`, lines 71–195) — EXTEND to parse new `max_samples` and `class_weight` fields using the same `get_*` closure pattern.

**Function signature pattern** (lines 228–269 `py_fit_cpu`) — new promoted functions `py_fit`, `py_predict_proba`, `py_get_fitted_attrs` follow the same shape:
```rust
#[pyfunction]
#[pyo3(name = "fit")]
fn py_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    sample_weight: Option<PyReadonlyArray1<f32>>,
    cfg_dict: Bound<'_, PyDict>,
) -> PyResult<String> {                   // returns JSON IR handle
    // 1. parse_config(cfg_dict)?;
    // 2. validate shapes → PyValueError (mirrors existing shape checks lines 240–255)
    // 3. py.detach(|| backend.fit(...))?  — GIL released (line 262 pattern)
    // 4. serde_json::to_string(&ir) → handle string
}
```

**GIL release pattern** (line 262):
```rust
let result = py.detach(|| -> Result<..., SylvaError> {
    // all training work here — no Python objects
});
result.map_err(sylva_error_to_pyerr)
```

**Module registration pattern** (lines 390–396) — add new functions with `wrap_pyfunction!`:
```rust
#[pymodule]
pub fn sylva_pyseam(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_fit, module)?)?;
    module.add_function(wrap_pyfunction!(py_predict_proba, module)?)?;
    module.add_function(wrap_pyfunction!(py_get_fitted_attrs, module)?)?;
    // ... keep existing fit_cpu, predict_cpu, split_statistics for parity harness
    Ok(())
}
```

---

### `python/sylva/_base.py` + `python/sylva/ensemble.py` (estimator classes)

**Analog:** RESEARCH.md Pattern 4 (sklearn BaseEstimator contract) + `crates/sylva-core/src/pyseam.rs` `parse_config` defaults (lines 68–70, 113, 162)

**Base class pattern** (from RESEARCH.md Pattern 4 — no codebase analog; copy this exactly):
```python
# python/sylva/_base.py
from sklearn.base import BaseEstimator

class _SylvaForest(BaseEstimator):
    """Base: NO logic in __init__ (check_estimator hard rule)."""
    def __init__(self, n_estimators=100, *, max_depth=None, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 max_samples=None, criterion="gini", random_state=None,
                 n_jobs=None, class_weight=None, device="auto", fallback="error"):
        # ONE assignment per param, same name — BaseEstimator.get_params() reads these.
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.max_features   = max_features
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.bootstrap      = bootstrap
        self.max_samples    = max_samples
        self.criterion      = criterion
        self.random_state   = random_state
        self.n_jobs         = n_jobs
        self.class_weight   = class_weight
        self.device         = device
        self.fallback       = fallback
        # NO other lines in __init__. Zero validation. Zero derived state.

    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y)   # sets n_features_in_, feature_names_in_
        # build cfg_dict from self.* params
        # dispatch via _dispatch.py → Rust seam
        # store: self._ir_handle_, self.feature_importances_
        return self

    def predict_proba(self, X):
        # check_is_fitted(self, '_ir_handle_')
        X = self._validate_data(X, reset=False)
        # call sylva_seam.predict_proba(self._ir_handle_, X)
        ...
```

**Fitted attribute naming** (sklearn trailing-underscore convention):
- `self.classes_` (clf only)
- `self.n_classes_` (clf only)
- `self.n_features_in_` (set by `_validate_data`)
- `self.feature_names_in_` (set by `_validate_data` when DataFrame input)
- `self.estimators_` (list of per-tree handles or a sentinel list of length n_estimators)
- `self.feature_importances_` (np.ndarray from `importance.rs`)

**Four concrete classes** (in `ensemble.py`):
```python
from sklearn.base import ClassifierMixin, RegressorMixin
from ._base import _SylvaForest

class ExtraTreesClassifier(_SylvaForest, ClassifierMixin):
    def __init__(self, n_estimators=100, *, criterion="gini", bootstrap=False, ...):
        super().__init__(n_estimators=n_estimators, criterion=criterion,
                         bootstrap=bootstrap, ...)

class ExtraTreesRegressor(_SylvaForest, RegressorMixin):
    def __init__(self, n_estimators=100, *, criterion="squared_error", max_features=1.0,
                 bootstrap=False, ...):
        super().__init__(...)

class RandomForestClassifier(_SylvaForest, ClassifierMixin):
    def __init__(self, n_estimators=100, *, criterion="gini", bootstrap=True, ...):
        super().__init__(...)

class RandomForestRegressor(_SylvaForest, RegressorMixin):
    def __init__(self, n_estimators=100, *, criterion="squared_error", max_features=1.0,
                 bootstrap=True, ...):
        super().__init__(...)
```

---

### `python/sylva/_dispatch.py` (device dispatch, no silent fallback)

**Analog:** `crates/sylva-core/src/pyseam.rs` `sylva_error_to_pyerr` (lines 46–53) + the RESEARCH.md anti-pattern "no silent fallback"

**Dispatch pattern:**
```python
# python/sylva/_dispatch.py
import importlib

def get_backend(device: str, fallback: str):
    """Returns the Rust seam module for the requested device.
    Raises RuntimeError if device='cuda' and CUDA unavailable (no silent fallback).
    """
    if device == "cpu":
        return importlib.import_module("sylva._sylva_core")   # PyO3 module
    if device == "cuda":
        try:
            import sylva._sylva_cuda as m
            if not m.cuda_available():
                if fallback == "error":
                    raise RuntimeError(
                        "device='cuda' requested but CUDA is not available. "
                        "Set device='cpu' or fallback='warn' to allow CPU fallback."
                    )
        except ImportError:
            raise RuntimeError("sylva CUDA extension not installed.")
        return m
    if device == "auto":
        # Try CUDA, silently fall back to CPU (auto is the ONLY silent mode).
        try:
            import sylva._sylva_cuda as m
            if m.cuda_available():
                return m
        except ImportError:
            pass
        return importlib.import_module("sylva._sylva_core")
    raise ValueError(f"Unknown device={device!r}. Choose 'cpu', 'cuda', or 'auto'.")
```

---

### `python/tests/test_check_estimator.py` (CI gate)

**Analog:** RESEARCH.md Code Examples (check_estimator gate) + `crates/sylva-core/tests/determinism.rs` parametrized structure

**Pattern** (from RESEARCH.md — implement verbatim):
```python
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from sylva.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                            RandomForestClassifier, RandomForestRegressor)

# Document intentional exceptions with a reason (sklearn 1.6+).
# Empty dict = full parity target. Add entries only as needed with reasons.
EXPECTED_FAILED = {
    # "check_sample_weight_equivalence": "GPU fixed-point weights differ at 1 ULP"
}

@parametrize_with_checks(
    [ExtraTreesClassifier(device="cpu"),
     ExtraTreesRegressor(device="cpu"),
     RandomForestClassifier(device="cpu"),
     RandomForestRegressor(device="cpu")],
    expected_failed_checks=lambda est: EXPECTED_FAILED,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)  # device="cpu" so GPU-less CI runs the API gate
```

---

### `python/tests/gpu_forest/test_rf_cpu_gpu_bitexact.py` (GPU bit-exact gate)

**Analog:** `crates/sylva-core/tests/determinism.rs` (full file — byte-compare pattern)

**Data fixture pattern** (from `determinism.rs` lines 32–52):
```python
# Mirror the fixed-seed clf_data / reg_data fixtures:
import numpy as np

def clf_data(n=40, n_features=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y
```

**Byte-exact assertion pattern** (from `determinism.rs` lines 65–80):
```python
# GPU RF must equal CPU oracle bit-for-bit (serde_json serialization):
def test_gpu_rf_forest_matches_cpu_oracle_bit_exact():
    X, y = clf_data()
    cpu_ir = sylva_seam.fit_cpu(X, y, {"algo": "rf", "n_estimators": 8, "seed": 42})
    gpu_ir = sylva_seam.fit_gpu(X, y, {"algo": "rf", "n_estimators": 8, "seed": 42})
    assert cpu_ir == gpu_ir, "GPU RF must equal CPU oracle byte-for-byte"
    # Gated on OQ2 resolution: binned-canonical RF split must be canonical in BOTH paths.
```

---

## Shared Patterns

### Error Mapping (V5 boundary — apply to ALL new Rust FFI boundary functions)

**Source:** `crates/sylva-core/src/pyseam.rs` lines 46–53
```rust
fn sylva_error_to_pyerr(err: SylvaError) -> PyErr {
    match err {
        SylvaError::InvalidInput(_) | SylvaError::InvalidConfig(_) =>
            PyValueError::new_err(err.to_string()),
        SylvaError::InvalidIr(_) =>
            PyRuntimeError::new_err(err.to_string()),
    }
}
```
**Apply to:** `pyseam.rs` extensions, any new `#[pyfunction]` in the estimator seam.

### Input Validation Before Device Work (V5 — apply to all kernel wrappers)

**Source:** `crates/sylva-core/src/pyseam.rs` lines 240–255 + `crates/sylva-core/src/cpu/fit.rs` lines 53–70
```rust
// Shape validation before any H2D or device call:
if x_shape[0] != y_shape[0] {
    return Err(PyValueError::new_err(format!(
        "X has {} rows but y has {} elements", x_shape[0], y_shape[0]
    )));
}
if x_shape[0] == 0 { return Err(PyValueError::new_err("X must have at least 1 row")); }
```
**Apply to:** `pyseam.rs` promoted functions; `sample_weight` length check before weighted histogram launch.

### NVRTC Compile Flags (CUDA kernels — carry Phase-4 lock)

**Source:** Phase-4 PATTERNS.md `nvrtc_launch.rs` pattern
```rust
// MANDATORY: -fmad=false. NEVER --use_fast_math.
options: vec!["-lineinfo".to_string(), "-fmad=false".to_string()],
```
**Apply to:** all new kernels in `kernels.rs` (`RF_BINNED_HIST_SRC`, `RF_SCAN_ARGMAX_SRC`, `SIBLING_SUBTRACT_SRC`, `WEIGHTED_HIST_SRC`).

### Integer-Only Atomics (determinism — apply to all histogram/weighted kernels)

**Source:** `crates/sylva-cuda/src/kernels.rs` lines 39–42
```cuda
// INTEGER atomicAdd only (counts are associative/deterministic).
// Float atomics are BANNED per PITFALLS Pitfall 5 — non-deterministic + non-subtractable.
atomicAdd(&sh[bins[i]], 1u);   // unsigned int, not float
```
**Apply to:** `RF_BINNED_HIST_SRC`, `WEIGHTED_HIST_SRC`, sibling subtraction (integer arrays).

### GIL Release (PyO3 0.29 — apply to all fit/predict functions)

**Source:** `crates/sylva-core/src/pyseam.rs` line 262
```rust
let result = py.detach(|| -> Result<T, SylvaError> { /* work here */ });
result.map_err(sylva_error_to_pyerr)
```
**Apply to:** all promoted seam functions in `pyseam.rs` that call into Rust training or prediction.

### Bootstrap Philox Contract (determinism — apply to GPU bootstrap)

**Source:** `crates/sylva-core/src/cpu/bootstrap.rs` lines 59–79
```rust
// GPU MUST inline the EXACT same Philox call with node=BOOTSTRAP_NODE_SENTINEL (u32::MAX):
let key = [seed as u32, (seed >> 32) as u32];
let ctr = [tree, BOOTSTRAP_NODE_SENTINEL, 0u32, i as u32];
let raw = philox4x32_10(ctr, key)[0];
let u   = u32_to_unit_f32(raw);
let idx = (u * n_f32) as usize;
idx.min(n - 1)   // T-02-09 clamp
```
**Apply to:** `forest.rs` (host-side bootstrap call), GPU kernel bootstrap path.

### ForestIR SoA Field Names (assembly + MDI — apply to assemble.rs + importance.rs)

**Source:** `crates/sylva-core/src/ir.rs` lines 23–65
All nine per-node arrays: `feature_id`, `threshold`, `left_child`, `right_child`, `default_child`, `is_leaf`, `node_sample_count`, `node_weighted_count`, `impurity`, plus `leaf_value`/`leaf_proba`/`leaf_offset`, `tree_offsets`, `tree_root`.
**Apply to:** `assemble.rs` (produce these exact fields), `importance.rs` (read `impurity`, `node_weighted_count`, `feature_id`, `left_child`, `right_child`, `is_leaf`, `leaf_offset`).

### Byte-Compare Determinism Gate (test pattern — apply to all GPU parity tests)

**Source:** `crates/sylva-core/tests/determinism.rs` lines 1–18
```rust
// Gold standard: serde_json::to_string(&cpu_ir) == serde_json::to_string(&gpu_ir)
// NOT approx/allclose — byte-exact string equality on JSON-serialized ForestIR.
```
**Apply to:** `parity_rf_cpu_gpu.rs`, `forest_cpu_gpu.rs`, `test_rf_cpu_gpu_bitexact.py`.

---

## No Analog Found

Files with no close match in the codebase (use RESEARCH.md patterns directly):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `python/sylva/ensemble.py` (four concrete classes) | estimator | request-response | Python package is greenfield (A4: `find python` shows only `tests/`). Pattern from sklearn developer guide + RESEARCH.md Pattern 4. |
| `python/benchmarks/comparative_study.py` | benchmark harness | batch | No prior benchmark harness. Extends `python/tests/parity/conftest.py` VERSION_MANIFEST and dataset fixture shape; fairness protocol from RESEARCH.md Pitfall 7. |
| `python/sylva/__init__.py` | package root | — | Greenfield; standard `from .ensemble import ...` re-export only. |

---

## Metadata

**Analog search scope:** `crates/sylva-core/src/`, `crates/sylva-cuda/src/`, `crates/sylva-core/tests/`, `.planning/phases/04-*/04-PATTERNS.md`
**Files scanned:** 18 Rust source files + 1 prior PATTERNS.md
**Codebase state note:** `crates/sylva-cuda/src/cuda_backend/` does NOT exist yet (Phase 4 was planned, not built). All `cuda_backend/*.rs` files are new in Phase 5; their analogs are in `sylva-core/src/cpu/` and `sylva-cuda/src/nvrtc_launch.rs`.
**Pattern extraction date:** 2026-06-27
