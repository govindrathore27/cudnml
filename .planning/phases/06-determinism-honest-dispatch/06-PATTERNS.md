# Phase 6: Determinism & Honest Dispatch - Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 12 new/modified files
**Analogs found:** 10 / 12

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/sylva-core/src/report.rs` | model | transform | `crates/sylva-core/src/quantize/report.rs` | exact |
| `crates/sylva-core/src/dispatch.rs` | service | request-response | `crates/sylva-core/src/error.rs` + `backend.rs` | role-match |
| `crates/sylva-core/src/config.rs` (extend) | config | — | `crates/sylva-core/src/config.rs` (self) | exact |
| `crates/sylva-core/src/error.rs` (extend) | model | — | `crates/sylva-core/src/error.rs` (self) | exact |
| `crates/sylva-cuda/src/cuda_backend/availability.rs` | utility | request-response | `crates/sylva-cuda/src/nvrtc_launch.rs` (CudaContext::new) | role-match |
| `crates/sylva-cuda/src/cuda_backend/report_build.rs` | utility | transform | `crates/sylva-core/src/quantize/report.rs` | role-match |
| `crates/sylva-cuda/src/cuda_backend/scheduler.rs` (extend) | service | event-driven | `crates/sylva-cuda/src/nvrtc_launch.rs` | partial |
| `crates/sylva-cuda/src/cuda_backend/arena.rs` (extend) | utility | batch | `crates/sylva-cuda/src/nvrtc_launch.rs` | partial |
| `python/sylva/_dispatch.py` | middleware | request-response | `crates/sylva-core/src/pyseam.rs` (error mapping) | role-match |
| `python/tests/test_dispatch.py` | test | request-response | `crates/sylva-core/tests/determinism.rs` | role-match |
| `python/tests/test_execution_report.py` | test | transform | `crates/sylva-core/tests/determinism.rs` | role-match |
| `crates/sylva-cuda/tests/deterministic_cpu_gpu.rs` | test | batch | `crates/sylva-core/tests/determinism.rs` | exact |

---

## Pattern Assignments

### `crates/sylva-core/src/report.rs` (model, transform)

**Analog:** `crates/sylva-core/src/quantize/report.rs` (lines 1–102)

**Imports pattern** (lines 1–2 of analog):
```rust
use serde::{Deserialize, Serialize};
```

**Core struct pattern** (lines 22–38 of analog — mirror derive set, field naming, doc style):
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizeReport {
    pub input_dtype: String,
    pub input_c_contiguous: bool,
    pub binned_bytes: usize,
    pub edges_bytes: usize,
    pub h2d_executed: bool,
    pub h2d_note: String,
}
```
Copy the exact `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` tag. `ExecutionReport` nests `QuantizeReport` as `pub quantize: QuantizeReport` (superset, not replacement). Add: `selected_backend: SelectedBackend`, `selection_reason: String`, `requested_device: String`, `fallback_policy: String`, `fallback_status: String`, `deterministic: bool`, `conversions: Vec<InputConversion>`, `bytes_h2d: usize`, `bytes_d2h: usize`.

**Serde round-trip test pattern** (lines 45–66 of analog):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> QuantizeReport { /* ... */ }

    #[test]
    fn serde_round_trip() {
        let r = sample_report();
        let json = serde_json::to_string(&r).expect("serialize QuantizeReport");
        let back: QuantizeReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(r, back, "serde round-trip must be identity");
    }
}
```
Mirror identically for `ExecutionReport` — one `serde_round_trip` test, one field-assertion test per critical field.

---

### `crates/sylva-core/src/dispatch.rs` (service, request-response)

**Analog:** `crates/sylva-core/src/backend.rs` (trait + enum pattern, lines 1–97) and `crates/sylva-core/src/error.rs` (thiserror enum pattern, lines 1–21)

**Enum pattern** (from `backend.rs` lines 23–29 — use same derive set):
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Predictions { /* ... */ }
```
Apply to: `RequestedDevice { Auto, Cuda, Cpu }`, `FallbackPolicy { Error }`, `SelectedBackend { Cpu, Cuda }` — all with `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]` (copy the `config.rs` enum derive set, lines 11–24, since these cross the serde boundary).

**Pure decision function pattern:** no analog exists in codebase — use the code example from RESEARCH.md Pattern 2 verbatim. The function is pure (no I/O), takes `(requested: RequestedDevice, fallback: FallbackPolicy, cuda_available: bool) -> Result<(SelectedBackend, String), SylvaError>`. Every unmet-cuda branch returns `Err(SylvaError::DeviceUnavailable(...))`, never falls through to CPU.

**Error variant pattern** (from `error.rs` lines 9–21 — copy thiserror style):
```rust
#[derive(Debug, Error)]
pub enum SylvaError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("invalid ForestIR: {0}")]
    InvalidIr(String),
}
```
Add two new variants in the same file:
```rust
    #[error("device unavailable: {0}")]
    DeviceUnavailable(String),

    #[error("unsupported config: {0}")]
    UnsupportedConfig(String),
```

---

### `crates/sylva-core/src/config.rs` — extend `TrainConfig` (config)

**Analog:** `crates/sylva-core/src/config.rs` itself (lines 63–110)

**Field extension pattern** (lines 63–75 of file):
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub n_estimators: usize,
    // ...existing fields...
    pub seed: u64,
    pub algo: Algo,
}
```
Add `pub deterministic: bool` as the final field. Mirror the existing default-in-`validate` style: add a `validate()` arm that rejects `fallback != "error"` when the enum is extended later (Phase 6 MVP: only `FallbackPolicy::Error` is valid — validation is a no-op for a single-variant enum but documents the intent).

---

### `crates/sylva-cuda/src/cuda_backend/availability.rs` (utility, request-response)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 83 and 137 — the `CudaContext::new(0)?` pattern used in every launch function.

**Probe pattern** (from `nvrtc_launch.rs` line 83):
```rust
let ctx = CudaContext::new(0)?;
```
This already returns `Result<Arc<CudaContext>, DriverError>`. The availability probe wraps this without `?` to convert the outcome to `bool`:
```rust
// In availability.rs:
use cudarc::driver::CudaContext;

pub fn cuda_available() -> bool {
    CudaContext::new(0).is_ok()   // no panic; no shell-out; no nvidia-smi
}
```
No new import paths needed — `cudarc::driver::CudaContext` is already the import in `nvrtc_launch.rs` line 17.

---

### `crates/sylva-cuda/src/cuda_backend/report_build.rs` (utility, transform)

**Analog:** `crates/sylva-core/src/quantize/report.rs` (construction pattern, lines 40–43 — the `impl QuantizeReport` block with a const).

**Builder helper pattern** (from `quantize/report.rs` lines 40–43):
```rust
impl QuantizeReport {
    pub const H2D_NOTE: &'static str = "N/A — no device path until Phase 4";
}
```
Mirror: implement `ExecutionReport` builder helpers as free functions (not a formal builder struct — the report is built *incrementally* during fit). Pattern: start with `ExecutionReport::new_cuda(requested_device, deterministic) -> ExecutionReport` that sets defaults, then mutating helper calls (`push_conversion`, `add_h2d_bytes`) that return `&mut Self` or take `&mut self`. Keep it simple — the shipped `QuantizeReport` is a plain struct with no builder; follow that simplicity.

---

### `python/sylva/_dispatch.py` (middleware, request-response)

**Analog:** `crates/sylva-core/src/pyseam.rs` (FFI error mapping pattern, lines 46–53 and lines 235–270)

**Error mapping pattern** (from `pyseam.rs` lines 46–53):
```python
# Mirror the Rust `sylva_error_to_pyerr` match in Python:
# InvalidInput / InvalidConfig → ValueError
# Internal failures → RuntimeError
# New: DeviceUnavailable → RuntimeError (not ValueError — the device is a system concern)
```

**Param-parsing pattern** (from `pyseam.rs` lines 71–195 — get_bool, get_int, get_str, early-validate):
```python
# In _dispatch.py, resolve_backend():
#   1. Call cuda_available() (imports the Rust probe via PyO3 or wraps CudaContext probe)
#   2. Match requested / fallback → (selected_backend, reason) or raise RuntimeError
#   3. Never fall through from cuda→cpu on error — RAISE with message matching
#      "no usable CUDA device" (the monkeypatch test target)
```

**GIL / boundary pattern** (from `pyseam.rs` line 262):
```python
# py.detach() → in Python: the dispatch decision is cheap (just the probe + match),
# so no threading concern; GPU fit is called after dispatch and releases the GIL internally.
```

**check_estimator compliance** (from `pyseam.rs` lines 181–194 — store verbatim, validate in fit):
```python
# In _base.py / ensemble.py __init__:
self.device = device          # store verbatim — no validation here
self.fallback = fallback      # check_estimator: check_no_attributes_set_in_init
self.deterministic = deterministic
# validate/resolve in fit() only
```

---

### `crates/sylva-cuda/tests/deterministic_cpu_gpu.rs` (test, batch)

**Analog:** `crates/sylva-core/tests/determinism.rs` (complete file, lines 1–288) — exact idiom to mirror.

**Fixture pattern** (lines 32–53):
```rust
fn clf_data() -> (Array2<f32>, Array1<f32>) {
    let n = 40usize;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j { /* ... */ });
    let y = Array1::from_iter((0..n).map(|i| if i < 20 { 0.0 } else { 1.0 }));
    (x, y)
}
```
Copy identically — same dataset sizes, same shapes.

**Core byte-compare gate** (lines 101–117):
```rust
fn assert_same_seed_byte_identical(
    x: &Array2<f32>,
    y: &Array1<f32>,
    cfg: &TrainConfig,
    label: &str,
) {
    let backend = CpuBackend;
    let ir1 = backend.fit(x.view(), y.view(), cfg).expect("fit 1");
    let ir2 = backend.fit(x.view(), y.view(), cfg).expect("fit 2");
    let s1 = serde_json::to_string(&ir1).expect("ser 1");
    let s2 = serde_json::to_string(&ir2).expect("ser 2");
    assert_eq!(
        s1, s2,
        "{label}: same-seed fits must produce byte-identical serialized ForestIR \
         (exact string equality — NOT allclose)"
    );
}
```
Replace `CpuBackend` with `CudaBackend::new().expect("cuda")`. Replace `backend.fit(...)` with `backend.fit_with_report(...)` (returns `(ForestIR, ExecutionReport)`); compare only the `ForestIR` JSON. The assert message must include "NOT allclose" — matches the CPU gate wording.

**Multi-seed sweep pattern** (lines 241–253):
```rust
let seeds: &[u64] = &[0, 1, 7, 42, 100, 999, u64::MAX / 2];
for &seed in seeds {
    let cfg = et_clf_cfg(seed);
    // ... fit twice, compare strings
    assert_eq!(s1, s2, "seed={seed}: same-seed must be byte-identical");
}
```
Mirror exactly — same seed values, same loop structure.

**Different-seeds guard** (lines 188–207):
```rust
assert_ne!(sa, sb, "different seeds must produce different serialized ForestIRs");
```
Include this as the "trivial equality" regression guard in the GPU test too.

---

### `python/tests/test_dispatch.py` (test, request-response)

**Analog:** `crates/sylva-core/tests/determinism.rs` (fixture + assert pattern); `crates/sylva-core/src/pyseam.rs` (monkeypatching target is `_dispatch.cuda_available`)

**Monkeypatch pattern** (from RESEARCH.md DET-03 code example — no codebase analog exists yet):
```python
def test_cuda_requested_but_unavailable_raises(monkeypatch):
    monkeypatch.setattr(_dispatch, "cuda_available", lambda: False)
    est = RandomForestClassifier(device="cuda", fallback="error")
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        est.fit(X, y)
```
The `match=` string must match the `SylvaError::DeviceUnavailable` message from the Rust dispatch (which propagates via `sylva_error_to_pyerr` → `PyRuntimeError`). Confirm the exact message in the Rust error variant definition.

---

### `python/tests/test_execution_report.py` (test, transform)

**Analog:** `crates/sylva-core/src/quantize/report.rs` (lines 60–101 — field-assertion test structure)

**Field assertion pattern** (from `quantize/report.rs` lines 68–84):
```rust
#[test]
fn h2d_executed_is_false() {
    let r = sample_report();
    assert!(!r.h2d_executed, "h2d_executed must be false in Phase 3 (D-11)");
}
#[test]
fn h2d_note_is_n_a() {
    let r = sample_report();
    assert_eq!(r.h2d_note, "N/A — no device path until Phase 4", "...");
}
```
Mirror in Python: one assertion per critical field (`selected_backend`, `fallback_status`, `deterministic`, `conversions` kind-set, `bytes_copied > 0`). Feed float64 F-order input to trigger dtype+layout conversion assertions.

---

## Shared Patterns

### No-silent-fallback typed error mapping (all FFI boundary files)
**Source:** `crates/sylva-cuda/src/lib.rs` lines 32–37 (`cuda_error_to_pyerr`); `crates/sylva-core/src/pyseam.rs` lines 46–53 (`sylva_error_to_pyerr`)
**Apply to:** `_dispatch.py`, `python/sylva/_base.py`, any new PyO3 function in `sylva-cuda`
```rust
// Pattern: match on error variant → PyValueError for caller errors, PyRuntimeError for device/internal
fn cuda_error_to_pyerr(err: CudaError) -> PyErr {
    match err {
        CudaError::InvalidInput(_) => PyValueError::new_err(err.to_string()),
        CudaError::Compile(_) | CudaError::Driver(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
// New: SylvaError::DeviceUnavailable → PyRuntimeError (not ValueError)
```

### serde + byte-identical gate (all report and determinism test files)
**Source:** `crates/sylva-core/tests/determinism.rs` lines 110–116; `crates/sylva-core/src/quantize/report.rs` lines 61–65
**Apply to:** `report.rs` (serde round-trip test), `deterministic_cpu_gpu.rs` (byte-compare gate)
```rust
let json = serde_json::to_string(&value).expect("serialize");
let back = serde_json::from_str(&json).expect("deserialize");
assert_eq!(value, back);   // round-trip test

// For byte-compare gate — string equality, NOT float comparison:
assert_eq!(s1, s2, "exact string equality — NOT allclose");
```

### TrainConfig field extension (config.rs)
**Source:** `crates/sylva-core/src/config.rs` lines 63–110 (struct layout + `validate()` method)
**Apply to:** `config.rs` itself (extend `TrainConfig`)
— Add `pub deterministic: bool` using the same plain public field style.
— Add a `validate()` arm for `FallbackPolicy` when that type is introduced.

### CudaContext::new(0) probe pattern (availability.rs)
**Source:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 83 and 137
**Apply to:** `availability.rs`
```rust
use cudarc::driver::CudaContext;
// ... inside every launch fn:
let ctx = CudaContext::new(0)?;   // returns Result<Arc<CudaContext>, DriverError>
// For probe: .is_ok() converts to bool without propagating the error
```

### thiserror enum extension (error.rs)
**Source:** `crates/sylva-core/src/error.rs` lines 5–21
**Apply to:** `error.rs` (add `DeviceUnavailable` and `UnsupportedConfig` variants)
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SylvaError {
    #[error("variant: {0}")]
    VariantName(String),
}
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/sylva-cuda/src/cuda_backend/scheduler.rs` | service | event-driven | No multi-stream scheduler exists yet; Phase 4/5 will create it — Phase 6 extends it. Use RESEARCH.md Pattern 4 (serialize streams under `deterministic=true`). |
| `python/benchmarks/determinism_overhead.py` | bench | batch | No Python benchmark harness exists yet; use RESEARCH.md DET-02 code example as the template. |

---

## Metadata

**Analog search scope:** `crates/sylva-core/src/`, `crates/sylva-core/tests/`, `crates/sylva-cuda/src/`
**Files scanned:** 29
**Pattern extraction date:** 2026-06-27
