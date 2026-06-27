# Phase 8: Exact Tree SHAP (Gate 2 + Implementation) — Research

**Researched:** 2026-06-27
**Domain:** Exact TreeSHAP attributions, GPUTreeSHAP (Apache-2.0), ForestIR consumption, Python shap validation
**Confidence:** MEDIUM

---

## Summary

Phase 8 is a two-stage phase: a feasibility spike (Gate 2) that locks the scope and license path, followed by implementation of exact per-feature SHAP attributions in a new `sylva-shap` crate that consumes the ForestIR read-only.

**Gate 2 feasibility findings (summary):**
WoodelfHD is MIT licensed (not GPL/closed), and SHAP interactions are already deferred to v2 (SHAP-V2-01). The kill criterion in ROADMAP.md ("if WoodelfHD is GPL/closed AND interactions were the target → descope") does NOT apply in either prong: interactions are not the current target (SHAP-02 specifies attributions only), and WoodelfHD is MIT open-source regardless. The kill criterion is therefore moot for the current requirements and the scope is confirmed: exact per-feature SHAP **attributions**, CPU-first then GPU, validated against `shap.TreeExplainer`. [ASSUMED — WoodelfHD license assessed from GitHub via websearch; confirm by reading the LICENSE file directly on github.com/ron-wettenstein/woodelf before the Gate-2 checkpoint]

**Algorithm path:** Reimplement the Lundberg et al. polynomial-time TreeSHAP algorithm (Nature MI 2020) in Rust, consuming the SoA `ForestIR` read-only. The `ForestIR` already carries every field the algorithm needs — no upstream IR changes are required for attributions. For the GPU path, the approach is to port the path-extraction preprocessing step into Rust, then pass `PathElement` vectors to GPUTreeSHAP's CUDA kernels (Apache-2.0 header-only C++) via cudarc NVRTC or direct CUDA C, or alternatively reimplement GPUTreeSHAP's parallel path-enumeration approach in hand-written CUDA C following the same Apache-2.0 algorithm (since it is algorithm reimplementation from an Apache-2.0-described method, not source copy). [ASSUMED — integration path specifics need spike to confirm]

**License discipline:** GPUTreeSHAP (rapidsai/gputreeshap) is Apache-2.0. WoodelfHD / woodelf_explainer is MIT. The `shap` Python reference package is MIT. No GPL-licensed code touches the hot path. Reimplement the algorithm from the paper description and the Apache-2.0 reference — never copy GPL source. [ASSUMED for `shap` license — verify on PyPI]

**Primary recommendation:** Implement TreeSHAP attributions in pure Rust as a new `sylva-shap` crate within the workspace, consuming `ForestIR` via `&ForestIR` reference. CPU-first. GPU second (the GPU path re-uses cudarc already in `sylva-cuda`). Validate against `shap.TreeExplainer` via the Python layer (same pattern as the sklearn parity gate in Phase 2). Skip SHAP interactions entirely — those are v2 (SHAP-V2-01, deferred).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SHAP attribution computation | Rust core (`sylva-shap`) | — | Algorithm is pure numerics over ForestIR; no Python business logic |
| ForestIR consumption | Rust core (`sylva-shap`) | — | ForestIR is read-only; sylva-shap borrows it, never modifies |
| GPU path-extraction + kernel dispatch | Rust core (`sylva-cuda` or `sylva-shap` CUDA sub-module) | — | GPU dispatch lives in the Rust kernel layer per project architecture |
| `.shap_values()` Python API | Python estimator layer (`sylva` package) | — | Delegates to the Rust PyO3 seam, same pattern as `predict` |
| Validation against `shap.TreeExplainer` | Python test layer | — | Comparison harness in Python; imports both sylva and shap |
| Gate-2 feasibility spike | Planning / Wave 0 checkpoint | — | A coding task that produces a human-verified go/no-go decision |

---

## Phase Requirements

<phase_requirements>
| ID | Description | Research Support |
|----|-------------|------------------|
| SHAP-01 | Gate 2 feasibility spike: confirm scope = exact attributions; verify GPUTreeSHAP licensing/integration path before implementation | Gate-2 kill criterion assessed — WoodelfHD is MIT, interactions are deferred; kill criterion does not apply; scope = attributions. GPUTreeSHAP = Apache-2.0 header-only C++. Spike is a Wave-0 human-verify checkpoint. |
| SHAP-02 | `sylva-shap` computes exact per-feature SHAP attributions from `ForestIR` (GPUTreeSHAP approach, Apache-2.0), CPU-first then GPU | Algorithm described in full below; ForestIR carries all required fields; no IR gaps; GPU path uses path-extraction + CUDA parallel algorithm |
| SHAP-03 | `.shap_values()` results validate against `shap.TreeExplainer` within float tolerance | shap 0.52.0 on PyPI; custom-dict injection allows validation without a full sklearn bridge; atol ≈ 1e-4 per SHAP's own test suite |
</phase_requirements>

---

## Gate-2 Feasibility Spike — Decision Framework

### The Kill Criterion (Verbatim from ROADMAP.md)

> "KILL CRITERION: If WoodelfHD is GPL/closed AND interactions were the target — descope to GPUTreeSHAP attributions (Apache-2.0); do not let exact-SHAP block or balloon the MVP"

### Prong-by-Prong Assessment

| Prong | Finding | Confidence | Action |
|-------|---------|------------|--------|
| WoodelfHD is GPL/closed? | WoodelfHD (github.com/ron-wettenstein/woodelf) is **MIT** licensed. | LOW [ASSUMED — websearch; confirm by reading LICENSE file] | No kill on this prong |
| Interactions are the target? | NO. SHAP-02 says "exact per-feature SHAP attributions" (not interactions). Interactions are SHAP-V2-01, explicitly deferred. | HIGH [from REQUIREMENTS.md] | No kill on this prong |
| Kill criterion fires? | **NO** — neither prong is true. | HIGH | Proceed with attributions |

### What Gate-2 Spike Must Produce (before any implementation)

A committed decision record covering:
1. Confirmed WoodelfHD license (read the actual LICENSE file on GitHub)
2. Confirmed GPUTreeSHAP Apache-2.0 (read LICENSE on rapidsai/gputreeshap)
3. Confirmed scope = exact attributions only (log the decision that SHAP-V2-01 interactions are v2)
4. Chosen GPU integration path: (a) call GPUTreeSHAP C++ header via cudarc FFI, OR (b) reimplement GPU path in hand-written CUDA C following the Apache-2.0 algorithm, OR (c) CPU-only for MVP with GPU deferred
5. A human-verify checkpoint — planner must gate implementation Wave 1 on this decision being committed

### Integration Path Options for GPU

| Option | Approach | Pros | Cons | Verdict |
|--------|----------|------|------|---------|
| **A: Port GPUTreeSHAP algorithm to CUDA C** | Reimplement path-extraction + parallel SHAP kernel in hand-written CUDA C (following the Apache-2.0 paper/algorithm, not source-copy) | Matches Sylva's existing NVRTC pattern; no C++ FFI; Apache-2.0-clean via reimplementation | More code to write | **Recommended** — consistent with Sylva's kernel authoring pattern |
| B: Bind to GPUTreeSHAP header-only C++ via cudarc | Include `gpu_treeshap.h`, compile via NVRTC or nvcc | Exact algorithm, minimal code | Requires C++ FFI from Rust; NVRTC compiles C not C++; nvcc AOT is broken on Windows MSVC | **Complex** — NVRTC handles C only; C++ headers would need nvcc or clang, which conflicts with the Windows MSVC constraint |
| C: CPU-only for Phase 8 | Implement pure-Rust CPU TreeSHAP only; GPU deferred to v2 | Simple, guaranteed correct | No GPU speedup for Phase 8 comparative study | **Acceptable fallback** if Option A spike is blocked |

**Recommendation:** Spike Option A (implement GPU path in CUDA C following the GPUTreeSHAP algorithm description) during Gate-2. If blocked within the spike timebox, fall back to Option C and defer GPU to v2. The SHAP-02 requirement says "CPU-first then GPU" — CPU is non-optional; GPU is additive.

---

## Standard Stack

### Core (Phase 8)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `sylva-core` (workspace) | current | ForestIR, SylvaError, rayon, ndarray | ForestIR is the only data source for sylva-shap |
| `rayon` | 1.x (workspace) | CPU parallel path over trees and rows | Already in workspace; data-parallel row/tree loops |
| `ndarray` | 0.16.x (workspace) | SHAP output array [n_rows, n_features] | Consistent with predict output format |
| `shap` (Python) | 0.52.0 [VERIFIED: PyPI] | Validation oracle — shap.TreeExplainer | Reference explainer for SHAP-03 correctness gate |
| `numpy` (Python) | existing in env | Array comparison in validation harness | np.allclose for tolerance comparison |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `approx` | 0.5.x (workspace dev-dep) | Float comparison in Rust unit tests | Leaf value reconstruction, per-tree sanity checks |
| `cudarc` | 0.19.8 (GPU path only) | CUDA driver API + NVRTC for GPU SHAP kernel | Only when GPU path is implemented (Wave 3+) |
| `thiserror` | 1.x (workspace) | `ShapError` enum | Consistent with SylvaError pattern |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure Rust TreeSHAP | Bind to GPUTreeSHAP C++ header | C++ FFI from Rust is feasible but requires NVRTC C mode or nvcc; nvcc is broken on Windows MSVC — Rust reimplementation avoids the FFI entirely |
| CPU-first GPU-second | GPU-only | CPU path is the correctness oracle and the Windows test path; GPU-only would prevent CI without GPU |
| `sylva-shap` as separate crate | Embed in `sylva-core` | Separation keeps `sylva-core` pyo3-free and the SHAP concern isolated; same pattern as `sylva-cuda` |

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| shap | PyPI | ~9 yrs | Very high (widespread ML use) | github.com/shap/shap | [WARNING: seam flagged as SUS due to registry signal mismatch — the real shap package is the established Lundberg SHAP library with millions of users; the PyPI signal anomaly is a registry metadata issue, not a legitimacy issue] | Approved with human-verify note |
| woodelf_explainer | PyPI | New | Low | github.com/ron-wettenstein/woodelf | [SUS — new package; only used as background reading for Gate-2 spike, not installed as a dependency] | NOT a dependency of sylva-shap — reference only |

**Note on `shap` PyPI signal:** The `shap` package (pypi.org/project/shap) is the canonical SHAP library by Lundberg et al., widely used across the ML ecosystem. The seam's `SUS` flag is a registry metadata anomaly (last-published date / download count lookup issue). The planner must add a `checkpoint:human-verify` for the `shap` install step per protocol, but this is the correct package. [ASSUMED — seam verdict is an artifact; confirm by checking pypi.org/project/shap directly]

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** `shap` (flag is almost certainly a seam artifact, not a real concern) — planner inserts checkpoint:human-verify before installing

---

## Algorithm: Exact TreeSHAP Attributions

### What TreeSHAP Computes

For a forest of T trees with M features, exact TreeSHAP computes a per-feature attribution vector φ ∈ R^M for each input row x, such that:

```
Σ_j φ_j(x) = f(x) - E[f(X)]
```

where `f(x)` is the model output and `E[f(X)]` is the expected output over training data (the base value). Each φ_j is the Shapley value for feature j — the fair marginal contribution of feature j to the prediction.

**Attributions only** — φ is a vector of M scalars per row. This is NOT the interaction matrix (which is M×M). Interactions are v2.

### Per-Node Fields Required by TreeSHAP

The algorithm {v, a, b, t, r, d} maps directly to ForestIR:

| Algorithm field | Meaning | ForestIR field | IR type | Gap? |
|----------------|---------|----------------|---------|------|
| `d` | Feature index at node | `feature_id` | `Vec<i32>` (-1 for leaf) | None |
| `t` | Split threshold | `threshold` | `Vec<f32>` | None |
| `a` | Left child index | `left_child` | `Vec<i32>` | None |
| `b` | Right child index | `right_child` | `Vec<i32>` | None |
| `r` | Node cover / weighted sample count | `node_weighted_count` | `Vec<f32>` | None — already present! |
| `v` | Leaf value | `leaf_value` (regression) / `leaf_proba` (classification) | `Vec<f32>` | None |
| NaN routing | Default child direction | `default_child` | `Vec<i32>` | None |

**IR gap assessment: NONE.** The `ForestIR` designed in Phase 2 explicitly carried `node_sample_count` (u64, unweighted) and `node_weighted_count` (f32, weighted) for this exact reason (the comment in ir.rs says "tree-SHAP cover (Treelite data_count)"). No upstream changes needed.

### Computing zero_fraction and one_fraction

These are the path fractions used in the EXTEND/UNWIND dynamic programming:

```
zero_fraction(node → left)  = node_weighted_count[left]  / node_weighted_count[node]
zero_fraction(node → right) = node_weighted_count[right] / node_weighted_count[node]
one_fraction = 1.0  (when the feature IS in the active set, this branch is taken with probability 1)
```

Both values come directly from `node_weighted_count` in the IR. No additional IR fields needed.

### Algorithm Complexity

| Quantity | Complexity |
|----------|-----------|
| Attributions (φ) — this phase | O(T × L × D²) |
| Interactions (φ_ij) — v2, NOT this phase | O(T × L × D² × M) |

For typical parameters (T=100, D=12, L~4096): attribution computation is O(100 × 4096 × 144) ≈ 59M ops per row — fast on CPU with rayon, very fast on GPU. Interactions would add a factor of M (~50–500) — that is why they are v2.

### Algorithm Sketch (Rust CPU Path)

The core is a recursive tree traversal with a "path" of active features maintained as a stack:

```rust
// Source: Lundberg et al. 2020 Nature MI, Algorithm 2 (reimplemented from paper description)
// NOT copied from any GPL source.

struct PathEntry {
    feature_idx: i32,     // -1 for root/bias
    zero_fraction: f32,   // node_weighted_count[child] / node_weighted_count[parent]
    one_fraction: f32,    // 1.0 when feature IS present
    pweight: f32,         // path weight accumulated by EXTEND
}

fn tree_shap(
    ir: &ForestIR,
    tree_idx: usize,
    x: &[f32],
    phi: &mut [f64],   // accumulate into f64, output as f32 — avoids precision loss
    phi_bias: &mut f64,
) {
    let root = ir.tree_root[tree_idx] as usize;
    let mut path: Vec<PathEntry> = Vec::with_capacity(ir.max_depth() + 1);
    tree_recurse(ir, root, 1.0, 1.0, -1, x, phi, phi_bias, &mut path);
}

fn tree_recurse(
    ir: &ForestIR,
    node: usize,
    zero_fraction: f32,
    one_fraction: f32,
    feature_idx: i32,
    x: &[f32],
    phi: &mut [f64],
    phi_bias: &mut f64,
    path: &mut Vec<PathEntry>,
) {
    extend_path(path, zero_fraction, one_fraction, feature_idx);

    if ir.is_leaf[node] {
        // Unwind each feature from the path and accumulate SHAP values
        for i in 1..path.len() {
            let w = sum_path_unwind_weights(path, i);
            let fi = path[i].feature_idx as usize;
            phi[fi] += w * (path[i].one_fraction - path[i].zero_fraction) * leaf_value(ir, node);
        }
        *phi_bias += leaf_value(ir, node) * path[0].pweight;
    } else {
        // Route x to hot and cold child
        let feat = ir.feature_id[node] as usize;
        let v = x[feat];
        let (hot, cold) = if v.is_nan() {
            let dc = ir.default_child[node] as usize;
            if dc == ir.left_child[node] as usize {
                (ir.left_child[node] as usize, ir.right_child[node] as usize)
            } else {
                (ir.right_child[node] as usize, ir.left_child[node] as usize)
            }
        } else if v <= ir.threshold[node] {
            (ir.left_child[node] as usize, ir.right_child[node] as usize)
        } else {
            (ir.right_child[node] as usize, ir.left_child[node] as usize)
        };

        let w = ir.node_weighted_count[node];
        let hot_zero = ir.node_weighted_count[hot] / w;
        let cold_zero = ir.node_weighted_count[cold] / w;

        tree_recurse(ir, hot,  1.0,       1.0,       feat as i32, x, phi, phi_bias, path);
        tree_recurse(ir, cold, cold_zero, 0.0,       feat as i32, x, phi, phi_bias, path);
    }
    unwind_path(path, feature_idx);
}
```

[ASSUMED — sketch based on training knowledge of the Lundberg algorithm description; the actual EXTEND/UNWIND bookkeeping must be implemented from the paper, not from GPL source]

### Expected Value (Base Value)

The base value E[f(X)] is computed as the prediction for the all-features-missing case. For tree ensembles it equals the weighted average leaf value over training data, which can be computed in O(n_nodes) from `node_weighted_count` and leaf values — precomputed once per tree during `sylva-shap` initialization.

---

## Architecture Patterns

### Recommended Crate Layout

```
crates/
├── sylva-core/          # existing — ForestIR, CpuBackend, traits
├── sylva-cuda/          # existing — CudaBackend, NVRTC kernels
└── sylva-shap/          # NEW Phase 8
    ├── Cargo.toml       # depends on sylva-core; optionally sylva-cuda (feature = "cuda")
    └── src/
        ├── lib.rs           # pub re-exports
        ├── cpu.rs           # CPU exact TreeSHAP (pure Rust)
        ├── gpu.rs           # GPU path (feature-gated, Phase 8 Wave 3+)
        ├── path.rs          # PathEntry + extend_path + unwind_path + sum_weights
        ├── base_value.rs    # compute E[f(X)] from IR
        ├── output.rs        # ShapOutput struct: values [n_rows, n_features], base_values [n_rows]
        └── error.rs         # ShapError (thiserror)
```

Expose to Python via `sylva-cuda`'s PyO3 seam (Phase 5 pattern): a `#[pyfunction]` `shap_values(ir_json: &str, x: PyReadonlyArray2<f32>) -> PyResult<...>` that deserializes ForestIR from JSON (the existing FFI handle), calls `sylva_shap::cpu::compute_shap`, and returns a numpy array.

### System Architecture Diagram

```
Python user
    │ .shap_values(X)
    ▼
sylva Python estimator (sylva/__init__.py)
    │ calls PyO3 fn shap_values(ir_json, X_numpy)
    ▼
sylva-cuda (PyO3 seam)  ──deserialize──▶  ForestIR (sylva-core)
    │                                              │ borrow &ForestIR
    ▼                                              ▼
device dispatch ──── CPU ──▶  sylva-shap::cpu::compute_shap(&ForestIR, X)
                 └── GPU ──▶  sylva-shap::gpu::compute_shap(&ForestIR, X, cudarc)
                                              │
                              [PathElement extraction → GPU kernel]
    │
    ▼
ShapOutput { values: Array2<f32>, base_values: Array1<f32> }
    │ → numpy
    ▼
Python user
```

### Project Structure

```
sylva-shap/src/
├── lib.rs              # ShapConfig, pub use
├── cpu.rs              # compute_shap_cpu — row-parallel rayon outer loop
├── gpu.rs              # compute_shap_gpu (cuda feature gate)
├── path.rs             # PathEntry, extend_path, unwind_path
├── base_value.rs       # expected_value_per_tree, aggregate base value
├── output.rs           # ShapOutput { values, base_values }
└── error.rs            # ShapError { InvalidIr, DimensionMismatch, CudaError }
```

### Pattern 1: Row-parallel CPU SHAP

```rust
// Source: [ASSUMED] — rayon data-parallel row loop, same pattern as Phase 5 predict
use rayon::prelude::*;

pub fn compute_shap_cpu(ir: &ForestIR, x: ArrayView2<f32>) -> Result<ShapOutput, ShapError> {
    let (n_rows, n_features) = (x.nrows(), x.ncols());
    let n_classes = ir.n_classes();

    // Allocate output: [n_rows, n_features] for regression / per-class for classification
    let mut values = Array3::<f32>::zeros((n_rows, n_features, n_classes));
    let mut base_vals = Array2::<f32>::zeros((n_rows, n_classes));

    // Precompute per-tree base values (cheap, done once)
    let tree_base = precompute_base_values(ir);

    // Outer parallelism: one task per row
    values
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(base_vals.axis_iter_mut(Axis(0)).into_par_iter())
        .zip(x.rows().into_iter())
        .for_each(|((mut phi_row, mut bias_row), x_row)| {
            for t in 0..ir.n_trees {
                let mut phi = vec![0.0f64; n_features];
                let mut phi_bias = 0.0f64;
                tree_shap(ir, t, x_row.as_slice().unwrap(), &mut phi, &mut phi_bias);
                for j in 0..n_features {
                    // For regression: single output; for classification: per class
                    phi_row[[j, 0]] += (phi[j] / ir.n_trees as f64) as f32;
                }
                bias_row[0] += (phi_bias / ir.n_trees as f64) as f32;
            }
        });

    Ok(ShapOutput { values, base_values: base_vals })
}
```

### Pattern 2: Custom tree dict injection for shap.TreeExplainer validation (Python)

```python
# Source: shap docs — "Example of loading a custom tree model into SHAP"
# Used in SHAP-03 validation harness

import shap
import numpy as np
from sylva_core import _ir_to_dict  # helper that converts ForestIR JSON to shap dict

def validate_shap_values(sylva_ir_json: str, X: np.ndarray, rtol=1e-4):
    """
    Validate sylva-shap output against shap.TreeExplainer.
    Uses the shap custom-dict injection interface.
    """
    # Build shap custom tree dict from ForestIR
    tree_dicts = _ir_to_dict(sylva_ir_json)  # Rust helper via PyO3
    model = {"trees": tree_dicts}
    explainer = shap.TreeExplainer(model)
    shap_ref = explainer.shap_values(X)  # reference values

    # Compute sylva-shap values
    from sylva import ExtraTreesClassifier  # or whichever estimator
    sylva_shap = compute_sylva_shap(sylva_ir_json, X)

    # Compare per-feature attributions: atol=1e-4 matches shap's own test suite tolerance
    assert np.allclose(sylva_shap, shap_ref, atol=1e-4), \
        f"SHAP values differ: max_abs={np.abs(sylva_shap - shap_ref).max():.2e}"
```

### Anti-Patterns to Avoid

- **Using f32 accumulation throughout the SHAP sum:** The recursive summation in tree_shap accumulates many small floating-point additions. Accumulate in f64 internally, downcast to f32 at output. Otherwise deep forests lose precision.
- **Copying GPL source from shap/shap or FastTreeSHAP (MIT but beware attribution):** Reimplement from the Lundberg 2020 paper algorithm description and from GPUTreeSHAP's Apache-2.0 algorithm. The key algorithmic insight (EXTEND/UNWIND DP) is described in the paper; the paper is not GPL.
- **Normalizing by n_trees inside tree_shap:** Do it at the output stage, not per-tree — reduces floating-point error.
- **Single-threaded CPU for large forests:** The outer row loop is embarrassingly parallel; use rayon.
- **Trying to call GPUTreeSHAP's C++ header from NVRTC:** NVRTC compiles CUDA C, not C++. GPUTreeSHAP is a C++ template library. Either write the GPU kernel in CUDA C following the algorithm, or call a C-wrapped binding — do not attempt to compile C++ headers via NVRTC.
- **Implementing interactions:** SHAP-V2-01 is deferred. The interaction complexity is O(TLD²M), which is M times more expensive. Do not implement it in Phase 8.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Custom tree dict injection for shap.TreeExplainer | A "fake sklearn estimator" wrapper | The `{"trees": [...]}` custom-dict interface in `shap.TreeExplainer` | shap already accepts arbitrary tree dicts with the standard node fields; no sklearn wrapper needed |
| SHAP additivity check | Custom sum check | `shap.TreeExplainer(..., check_additivity=True)` already does this | Validates sum(φ) + E[f] == f(x); catches bugs for free |
| Float tolerance comparison in Python | Custom allclose | `np.allclose(a, b, atol=1e-4)` | Standard scientific Python; SHAP's own tests use exactly this |
| GPU path-extraction | Custom path enumeration data structure from scratch | Follow GPUTreeSHAP's PathElement concept (Apache-2.0 algorithm) | GPUTreeSHAP's preprocessing approach is well-described in the paper (PMC9044362) |

**Key insight:** The shap custom-dict interface is the bridge between ForestIR and shap.TreeExplainer. Building a Rust helper that converts ForestIR fields to the dict format (`children_left`, `children_right`, `children_default`, `features`, `thresholds`, `values`, `node_sample_weight`) enables SHAP-03 validation without sklearn dependency.

---

## Common Pitfalls

### Pitfall 1: Floating-Point Accumulation Loss in Deep Trees

**What goes wrong:** At depth 20+, the EXTEND/UNWIND path weight product underflows or loses precision when accumulated in f32.
**Why it happens:** Path weights are products of `node_weighted_count[child] / node_weighted_count[parent]` at each level; 20 multiplications in f32 can lose 6–8 bits.
**How to avoid:** Accumulate `phi` in `f64`, downcast to `f32` at output. The shap reference uses f64 internally.
**Warning signs:** Additivity check (sum of phi ≠ f(x) - E[f]) fails for deep trees but passes for shallow ones.

### Pitfall 2: Classifier Output Shape Mismatch

**What goes wrong:** sklearn SHAP output for binary classifiers is `(n_rows, n_features, 2)` (one array per class). For regressors it is `(n_rows, n_features)`.
**Why it happens:** shap.TreeExplainer uses `predict_proba` output for classifiers.
**How to avoid:** Check `ir.task` and emit the correct shape. For binary classifiers, emit both classes even though φ_class0 = -φ_class1 (they are redundant but expected).
**Warning signs:** np.allclose fails with shape mismatch before even checking values.

### Pitfall 3: NaN Routing in the Hot/Cold Split

**What goes wrong:** If `x[feat]` is NaN, hot/cold assignment must use `default_child` (same policy as predict). Failing to mirror the NaN routing policy from `cpu/predict.rs` gives wrong attributions for rows with missing values.
**Why it happens:** TreeSHAP splits each node into "hot" (the branch x actually takes) and "cold" (the branch x does not take). The NaN case must be handled identically to prediction.
**How to avoid:** Copy the NaN routing logic from `cpu/predict.rs` exactly — `v.is_nan()` check first, then route to `default_child`.
**Warning signs:** Attributions fail the additivity check on NaN-containing rows.

### Pitfall 4: base_value (E[f(X)]) Not Equal to sum-of-root-predictions

**What goes wrong:** The base value is NOT simply the root node value. It is the weighted average leaf value over training, computed from `node_weighted_count` × leaf values traversed from root.
**Why it happens:** The forest is additive over trees, so `base_value = (1/T) × Σ_t tree_expected_value(t)`, where each tree's expected value is computed by a single DFS that weights each leaf by `node_weighted_count[leaf] / node_weighted_count[root]`.
**How to avoid:** Precompute per-tree expected values in `base_value.rs` at initialization time.
**Warning signs:** `sum(phi) + base_value ≠ f(x)` — the additivity property fails.

### Pitfall 5: GPUTreeSHAP's C++ Incompatibility with NVRTC

**What goes wrong:** Attempting to call `gpu_treeshap.h` (C++ template) via NVRTC fails because NVRTC compiles CUDA C, not C++.
**Why it happens:** NVRTC only supports a CUDA C dialect; C++ templates, STL, and CUDA C++ abstractions are not available.
**How to avoid:** Write the GPU SHAP kernel in CUDA C from scratch, following the GPUTreeSHAP algorithm description (path enumeration, parallel Shapley summation). Do not attempt to include the C++ header.
**Warning signs:** NVRTC compile error mentioning templates or namespace std.

### Pitfall 6: License Contamination

**What goes wrong:** Copying SHAP algorithm code from shap/shap (MIT but still), FastTreeSHAP, or any GPL source.
**Why it happens:** The `shap` package's C++ extension `_cext.cpp` implements TreeSHAP; it's MIT but copying it creates a provenance issue.
**How to avoid:** Implement from the Lundberg 2020 paper description (Nature MI) and GPUTreeSHAP (Apache-2.0) algorithm description only. The EXTEND/UNWIND recursive structure is described fully in the supplementary materials of the paper.
**Warning signs:** Code comments that reference "adapted from shap/_cext.cpp" or similar.

---

## IR Field Completeness — No Gaps for Attributions

This is a key finding: the Phase 2 `ForestIR` was deliberately designed for TreeSHAP consumption. [VERIFIED by reading ir.rs directly]

| Required field | ForestIR field | Present? |
|----------------|----------------|---------|
| feature index d | `feature_id: Vec<i32>` | YES — LEAF_FEATURE = -1 |
| split threshold t | `threshold: Vec<f32>` | YES |
| left child a | `left_child: Vec<i32>` | YES |
| right child b | `right_child: Vec<i32>` | YES |
| node cover r (weighted) | `node_weighted_count: Vec<f32>` | YES |
| node cover r (unweighted) | `node_sample_count: Vec<u64>` | YES (bonus) |
| leaf value v (regression) | `leaf_value: Vec<f32>` + `leaf_offset: Vec<i32>` | YES |
| leaf value v (classification) | `leaf_proba: Vec<f32>` + `leaf_offset: Vec<i32>` | YES |
| NaN routing | `default_child: Vec<i32>` | YES |
| leaf flag | `is_leaf: Vec<bool>` | YES |
| tree boundaries | `tree_offsets: Vec<usize>`, `tree_root: Vec<i32>` | YES |

**Conclusion:** zero IR gaps. Phase 8 can consume ForestIR as-is. [VERIFIED: direct reading of crates/sylva-core/src/ir.rs]

---

## Comparative Baseline Study (Phase 8)

Per ROADMAP.md, Phase 8's study is attribution correctness PARITY (primary) + SHAP-compute speedup (reported).

### Correctness Gate

- **Baseline:** `shap.TreeExplainer` with `feature_perturbation="tree_path_dependent"` (the exact TreeSHAP, not the interventional approximation)
- **Dataset:** A forest trained on a medium dense dataset at high `max_depth` (e.g., Covertype subset, max_depth=12 or 15) — high depth makes the explanation cost meaningful
- **Metric:** `max(abs(sylva_shap - shap_ref))` per feature across all rows; must be < 1e-4 for the gate to pass
- **Gate:** Per ROADMAP.md success criterion 5 — attribution agreement within float tolerance is the gate

### Speed Study (reported, not gated)

| Comparison | Metric |
|------------|--------|
| sylva-shap GPU vs `shap.TreeExplainer` CPU | Wall-clock time, row/s |
| sylva-shap GPU vs `rapidsai/GPUTreeSHAP` | Wall-clock time, row/s |
| sylva-shap CPU vs `shap.TreeExplainer` CPU | Wall-clock time, row/s (informational) |

Speed is reported with agreement alongside. A faster-but-wrong result fails the gate. Same fairness rules: pinned hardware/driver/CUDA/package versions, cold + warm, repeated runs, accuracy (agreement) reported alongside speed.

---

## Validation Architecture

Nyquist validation enabled (`workflow.nyquist_validation: true`).

### Test Framework

| Property | Value |
|----------|-------|
| Rust framework | `cargo test` + `cargo nextest` (workspace pattern) |
| Python framework | `pytest` (existing in python/tests/) |
| Quick run | `cargo test -p sylva-shap` |
| Full suite | `cargo test --workspace && pytest python/tests/shap/` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHAP-01 (Gate 2) | Kill criterion evaluated, scope locked, license verified | checkpoint:human-verify (Wave 0) | Manual — committed decision record | — Wave 0 |
| SHAP-01 (GPUTreeSHAP license) | Assert GPUTreeSHAP is Apache-2.0 | manual verify during spike | Read LICENSE at rapidsai/gputreeshap | — Wave 0 |
| SHAP-02 (CPU additive) | sum(phi) + base_value == f(x) for each row | unit | `cargo test -p sylva-shap additivity` | ❌ Wave 1 |
| SHAP-02 (trivial tree) | Single-split tree SHAP values match paper example | unit | `cargo test -p sylva-shap trivial` | ❌ Wave 1 |
| SHAP-02 (NaN routing) | NaN in feature follows default_child same as predict | unit | `cargo test -p sylva-shap nan_routing` | ❌ Wave 1 |
| SHAP-02 (regressor output) | shape [n_rows, n_features], regression task | unit | `cargo test -p sylva-shap regressor_shape` | ❌ Wave 1 |
| SHAP-02 (classifier output) | shape [n_rows, n_features, n_classes] | unit | `cargo test -p sylva-shap clf_shape` | ❌ Wave 1 |
| SHAP-03 (agreement gate) | sylva-shap vs shap.TreeExplainer atol ≤ 1e-4 | integration | `pytest python/tests/shap/test_shap_parity.py` | ❌ Wave 2 |
| SHAP-03 (.shap_values() API) | `.shap_values(X)` exists and returns correct shape | smoke | `pytest python/tests/shap/test_api.py` | ❌ Wave 2 |
| Comparative study (correctness gate) | max_abs_error < 1e-4 on deep-tree Covertype | integration | `pytest python/tests/shap/test_shap_parity.py --deep` | ❌ Wave 3 |
| Comparative study (speedup, reported) | Wall-clock time GPU vs CPU reference | perf | `python scripts/benchmarks/shap_speedup.py` | ❌ Wave 3 |

### Sampling Rate

- **Per task commit:** `cargo test -p sylva-shap`
- **Per wave merge:** `cargo test --workspace && pytest python/tests/shap/`
- **Phase gate:** Full suite green, SHAP-03 parity test passes (atol ≤ 1e-4), additivity check passes before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `crates/sylva-shap/` — new crate scaffolding
- [ ] `python/tests/shap/__init__.py` and `test_shap_parity.py`
- [ ] Gate-2 feasibility spike task (before any other work)
- [ ] `cargo test -p sylva-shap` — framework confirm after Cargo.toml created

*(Existing: `cargo test --workspace` already covers other crates. shap 0.52.0 must be installed in the test venv.)*

---

## Security Domain

Security enforcement enabled (`security_enforcement: true`).

### Applicable ASVS Categories (ASVS Level 1)

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A — no user auth in computation library |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A — library function, no multi-user context |
| V5 Input Validation | Yes | Validate `x.ncols() == ir.n_features` before SHAP computation (same guard as `predict_forest`); validate IR with `ir.validate_structure()` |
| V6 Cryptography | No | N/A |

### Known Threat Patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|-----------|
| Malformed ForestIR (negative node counts, out-of-range children) | Tampering | `ir.validate_structure()` before any SHAP computation |
| Integer overflow in node index arithmetic | Tampering | Use checked casts (`as usize` with bounds check or `try_into()`) for child indices |
| Divide-by-zero in zero_fraction | Tampering | Guard `node_weighted_count[node] > 0.0` before division; root always has count > 0 by IR contract |
| Infinite recursion in malformed trees (cycle in children) | DoS | `validate_structure()` checks acyclicity via index bounds; max recursion depth bounded by IR `max_depth` |

**Key security note for SHAP:** The SHAP computation recurses to tree depth D. With a well-formed IR the recursion depth is bounded and safe. The `validate_structure()` call in ir.rs already catches the main structural violations. `sylva-shap` must call `ir.validate_structure()` at the top of its public API.

---

## State of the Art (SHAP Landscape, 2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| KernelSHAP O(2^M) sampling | Exact TreeSHAP O(TLD²) | 2018 (NeurIPS 2017 SHAP, 2020 Nature MI TreeSHAP) | Attributions are now exact and fast for tree ensembles |
| CPU-only SHAP | GPUTreeSHAP (rapidsai, Apache-2.0) | 2020–2022 | GPU parallelism for large datasets and deep forests |
| Per-row sequential | Row-parallel with rayon / GPU blocks | TreeSHAP 2020+ | Linear scaling in n_rows |
| Float32 throughout | Float64 accumulation → float32 output | Library practice | Required for additivity at depth > 15 |
| TreeSHAP attributions (v1) | SHAP interaction values (v2, WOODELF-HD, shap-IQ) | 2022+ | Interactions are M× more expensive; deferred to v2 in Sylva |

**Deprecated/outdated:**
- **GPUTreeSHAP as a pip-installable Python package:** The standalone rapidsai/gputreeshap repository has not released since v22.02 (Feb 2022). Its algorithm is available and Apache-2.0; the source is usable as an algorithm reference. Do NOT depend on it as an installed Python package.
- **`shap.TreeExplainer(model, feature_perturbation="interventional")`:** The interventional variant uses a different probability model (conditional vs. marginal); use `tree_path_dependent` for validation against Sylva's exact path-dependent implementation.

---

## Open Questions (RESOLVED)

> Resolved by the orchestrator before planning; locked in the plans:
> - **Q1 RESOLVED:** GPU integration path = reimplement the GPUTreeSHAP algorithm in CUDA C via NVRTC (no C++ FFI); expressibility confirmed in the Gate-2 spike (2h timebox), CPU-only fallback if blocked (Plan 08-01 / 08-04 Option A vs C).
> - **Q2 RESOLVED:** WoodelfHD license read from the actual LICENSE file as the first Gate-2 spike action (websearch says MIT — verified there); kill criterion expected NOT to fire (Plan 08-01).
> - **Q3 RESOLVED:** agreement tolerance atol=1e-4 (shallow) / 5e-4 (deep > max_depth 15), pinned by a shallow-then-deep validation run (Plan 08-03).
> - **Q4 RESOLVED:** classifier SHAP output = sklearn-style list of per-class arrays (Plan 08-03).
> - **Q5 RESOLVED:** base_value computed by leaf_offset-weighted DFS over the ForestIR (Plan 08-02).

1. **GPU integration path confirmation**
   - What we know: GPUTreeSHAP is Apache-2.0 header-only C++; NVRTC cannot compile C++; GPU path options are (A) CUDA C reimplementation or (C) CPU-only for Phase 8.
   - What's unclear: Is the GPU path blocked in Phase 8 or achievable within the phase timebox? How much of GPUTreeSHAP's algorithm is expressible in CUDA C without C++ templates?
   - Recommendation: Make this a Gate-2 spike decision. Timebox 2 hours to attempt a CUDA C path-extraction sketch. If blocked, fall back to CPU-only Phase 8 + GPU deferred. **Do not let GPU SHAP block CPU SHAP.**

2. **WoodelfHD exact license (confirm the MIT claim)**
   - What we know: Web search indicates MIT license (github.com/ron-wettenstein/woodelf).
   - What's unclear: The LICENSE file was not directly read in this research session.
   - Recommendation: Read github.com/ron-wettenstein/woodelf/LICENSE as the first task of the Gate-2 spike. This closes the kill criterion check definitively.

3. **shap.TreeExplainer numerical precision for float32 IR**
   - What we know: shap.TreeExplainer uses float64 internally; SHAP's own tests use atol=1e-4 for additivity.
   - What's unclear: When ForestIR data is float32, will the reference and Sylva values agree within 1e-4 or will float32→float64 conversion gaps widen the delta?
   - Recommendation: Run the validation harness first on shallow trees (max_depth=5) where precision is not a concern, then deep (max_depth=20). Adjust atol to 5e-4 if 1e-4 is not achievable for max_depth>15 — document the chosen tolerance in the phase gate.

4. **Classifier SHAP output shape convention**
   - What we know: shap.TreeExplainer returns `(n_rows, n_features, 2)` for sklearn binary classifiers, `(n_rows, n_features)` for raw-output classifiers.
   - What's unclear: What convention should Sylva adopt for `.shap_values()`? Return a list of arrays (sklearn convention) or a 3D array?
   - Recommendation: Match sklearn convention (list of per-class arrays for classifiers, single array for regressors) since Sylva targets sklearn drop-in compatibility. This is a design decision for the planner to lock.

5. **base_value computation for ForestIR**
   - What we know: base_value = (1/T) × Σ_t Σ_leaves (leaf_value × node_weighted_count[leaf] / node_weighted_count[root_t]).
   - What's unclear: Whether `node_weighted_count[root]` = total training samples, or whether it can differ per tree (bootstrap).
   - Recommendation: Bootstrap (RandomForest) means each tree sees ~63.2% of training rows, so node_weighted_count[root_t] < n_training_samples. The computation is still correct — just divide by the tree root's weighted count, not by total n. Verify with a unit test: base_value should equal predict on an all-NaN row (all features missing → all nodes route to default child to a single leaf weighted by cover).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | sylva-shap crate build | ✓ | 1.83+ (workspace) | — |
| cargo nextest | Test runner | check before use | — | `cargo test` |
| Python 3.10+ | Validation harness | ✓ (existing) | — | — |
| shap (Python) | SHAP-03 validation | ✓ | 0.52.0 [VERIFIED: PyPI] | — |
| numpy | Validation comparisons | ✓ | existing in env | — |
| CUDA Toolkit 12.x | GPU path (Wave 3+) | ✓ (from Phase 1) | 12.x | CPU-only path is the fallback |
| cudarc 0.19.8 | GPU kernel dispatch | ✓ (from Phase 1) | 0.19.8 | N/A |

**Missing dependencies with no fallback:** None — CPU path requires only Rust stable.
**Missing dependencies with fallback:** GPU path (Wave 3+) requires CUDA; falls back to CPU-only if Wave 3 is skipped.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | WoodelfHD (github.com/ron-wettenstein/woodelf) is MIT licensed | Gate-2 Feasibility, Kill Criterion | If GPL, kill criterion still doesn't fire (interactions are deferred); but the license check must be explicit in the Gate-2 record |
| A2 | `shap` Python PyPI package is the legitimate Lundberg SHAP library | Package Audit | Seam flagged it SUS — almost certainly a metadata artifact; but human-verify before pip install |
| A3 | TreeSHAP EXTEND/UNWIND algorithm is fully described in Lundberg 2020 (Nature MI) supplementary | Algorithm section | If the paper description is insufficient, may need to reference GPUTreeSHAP's Apache-2.0 paper (PMC9044362) for implementation guidance |
| A4 | GPUTreeSHAP's GPU algorithm (path-extraction + parallel Shapley) is reimplementable in CUDA C without C++ templates | Open Questions (GPU path) | If CUDA C cannot express the algorithm without C++ generics, Option A is blocked and Option C (CPU-only) is the fallback |
| A5 | `node_weighted_count` in ForestIR is sufficient for TreeSHAP's `r` cover field | IR Completeness | If bootstrap causes node_weighted_count to be wrong (e.g., counts are sample-unweighted), SHAP values will be numerically wrong — verify with additivity test |
| A6 | atol=1e-4 is achievable for all depths when ForestIR uses float32 | Validation Architecture | At max_depth > 20, float32 precision loss may push error above 1e-4; may need to document a depth-dependent tolerance |

---

## Sources

### Primary (HIGH confidence)
- `crates/sylva-core/src/ir.rs` — ForestIR field verification (VERIFIED: direct read)
- `crates/sylva-core/src/cpu/predict.rs` — NaN routing policy, traversal pattern (VERIFIED: direct read)
- `.planning/REQUIREMENTS.md` — SHAP-01, SHAP-02, SHAP-03 exact text; SHAP-V2-01 deferred (VERIFIED: direct read)
- `.planning/ROADMAP.md` — Phase 8 success criteria, kill criterion verbatim (VERIFIED: direct read)

### Secondary (MEDIUM confidence)
- `github.com/rapidsai/gputreeshap` README + `gpu_treeshap.h` — Apache-2.0 license confirmed; PathElement structure; C++ template interface
- `shap.readthedocs.io` — TreeExplainer API, custom tree dict format, test tolerances (atol 1e-4 to 1e-5)
- `arxiv/PMC9044362` (GPUTreeSHAP paper) — PathElement fields, tree node representation {v, a, b, t, r, d}, zero_fraction computation

### Tertiary (LOW confidence — websearch only)
- WoodelfHD GitHub (ron-wettenstein/woodelf) — MIT license, pip install woodelf_explainer
- shap PyPI — version 0.52.0 confirmed via `pip index versions shap`
- Lundberg 2020 Nature MI — algorithm complexity O(TLD²), EXTEND/UNWIND DP structure

---

## Metadata

**Confidence breakdown:**
- ForestIR fields / IR gaps: HIGH — direct code read
- Algorithm description: MEDIUM — paper-based, not yet implemented
- GPUTreeSHAP license / integration: MEDIUM — GitHub confirmed Apache-2.0
- WoodelfHD license: LOW [ASSUMED] — websearch result; must be verified in Gate-2 spike
- GPU integration path: LOW [ASSUMED] — option analysis based on known NVRTC/C++ constraints
- Float tolerance: LOW [ASSUMED] — based on shap test file inspection, depth-dependent precision not empirically verified

**Research date:** 2026-06-27
**Valid until:** 2026-09-27 (90 days — stable algorithm domain; shap API changes slowly)
