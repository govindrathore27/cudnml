# Phase 5: Full Forest, RandomForest & sklearn Estimators - Research

**Researched:** 2026-06-27
**Domain:** Scaling a single GPU ExtraTree (Phase 4) into a full GPU forest + GPU RandomForest (prefix-sum/argmax best-split over bins, sibling-histogram subtraction, bootstrap, sample_weight), a stream-ordered fit-scoped arena, a small/deep-node CPU leaf-finishing cutover, and the four sklearn-parity estimator classes (PyO3 0.29 + maturin abi3) — plus the project's FIRST real end-to-end speed comparison (Comparative Baseline Study).
**Confidence:** HIGH for the estimator API contract, the RF best-split-over-bins algorithm, sibling subtraction, MDI feature importances, and the study fairness protocol (all grounded in shipped repo code + sklearn 1.9 docs). MEDIUM for the GPU arena mechanism (cudarc 0.19.8's *safe* API is synchronous; stream-ordered `cudaMallocAsync` requires the raw `driver::sys` FFI — flagged Open Question 1). MEDIUM for the RF binned-vs-raw threshold contract (the bit-exact reconciliation fork carried forward from Phase 4 — Open Question 2).

## Summary

Phase 5 is two distinct bodies of work stitched by one IR. **(A) The GPU forest + RF kernels** scale Phase 4's single-tree `CudaBackend` along three axes: many trees (per-tree Philox schedule already proven in `bootstrap.rs` + `fit.rs`), the RandomForest *best*-split (an inclusive prefix-scan over per-(feature,bin) histograms + argmax, replacing ET's single random threshold), and a performance layer (sibling-histogram subtraction, a fit-scoped arena, and a CPU cutover for small/deep nodes). **(B) The Python estimator layer** wraps the Rust backends in four `BaseEstimator`-conformant classes that pass `sklearn.utils.estimator_checks.check_estimator`. Critically, the CPU oracle already trains all four estimators correctly (`fit.rs` dispatches ET/RF × clf/reg, `split_rf.rs` is the exact best-split, `bootstrap.rs` is the Philox bootstrap), so the *algorithms* are de-risked — Phase 5 is GPU-port + API-surface + the honest speed study, not algorithm invention.

The single highest-leverage architectural insight is that **the GPU RF best-split is where Phase 3's `BinnedMatrix` finally becomes load-bearing.** ET draws one random real threshold per feature (raw-range, Phase-4 Strategy A); RF must evaluate *every candidate threshold*, which on GPU means: build a per-(feature, bin) class-count histogram (the privatized shared-mem integer kernel, already proven), inclusive-scan the bins to get cumulative left/right counts at each bin boundary, score each boundary with the exact `criterion.rs` op order, and argmax with the exact `(feature, threshold_bits)` tie-break. **Sibling subtraction** is the key forest perf win: after splitting a parent, compute only the *smaller* child's histogram directly and derive the larger child by `parent_hist − sibling_hist` — but only the integer count histograms are exactly subtractable; this is why the whole stack is integer-accumulation (DET-01 foundation) and why subtraction must never touch float sums.

**Primary recommendation:** Implement Phase 5 as four plan-sized slices in two waves. **Wave 1 (GPU engine):** (1) the breadth-first forest scheduler extended to many trees with the per-tree Philox + bootstrap schedule, reusing the Phase-4 ET kernels and adding sibling subtraction; (2) the RF binned best-split kernel set (per-(feature,bin) histogram → inclusive scan → argmax) + `sample_weight` weighted-histogram, sharing the privatized histogram engine. **Wave 2 (memory + API + study):** (3) the stream-ordered fit-scoped arena (via `cudarc::driver::sys` `cuMemAllocAsync` pool — see OQ1) + the small/deep-node CPU leaf-finishing cutover; (4) the four Python estimator classes (PyO3 0.29 `#[pyclass]` wrappers over `CpuBackend`/`CudaBackend` with honest dispatch, the full fitted-attribute set incl. real MDI `feature_importances_`, and the `check_estimator` CI gate with documented `expected_failed_checks`), then the Comparative Baseline Study (end-to-end-from-numpy ET/RF vs sklearn `n_jobs=-1` / cuML RF / XGBoost rf-mode on Covertype/Higgs, accuracy beside speed, cold/warm separated, OOM reported). The RF GPU-vs-CPU-oracle bit-exact parity test is the kernel gate; `check_estimator` green is the API gate; accuracy-parity-with-sklearn is the study gate (speed is reported, not gated — Phase 7 is authoritative).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Forest scheduler (many trees, frontier waves) | L3 host orchestration (Rust, `sylva-cuda`) | drives L1 kernel launches | Extends the Phase-4 breadth-first scheduler; per-tree Philox keying already order-independent (`fit.rs`/`bootstrap.rs`). |
| RF best-split over bins (scan + argmax) | L1 CUDA-C via NVRTC (`sylva-cuda`) | host scoring fallback (`cpu::criterion`) | The new GPU-04 kernel; bandwidth/scan-bound; shares the privatized histogram engine with ET. |
| Sibling-histogram subtraction | L3 host (decides which child to compute) + L1 (subtract kernel) | — | `parent − smaller_sibling = larger_sibling`; integer counts only (exactly subtractable). |
| Per-tree RNG schedule (Philox + bootstrap) | L1 CUDA-C inlined + L3 reference (`sylva-core::rng`/`bootstrap`) | — | Counter `(seed,tree,node,feature,draw)` already defined; bootstrap sentinel `node=u32::MAX` already in `bootstrap.rs`. |
| sample_weight → weighted histogram | L1 CUDA-C (weighted atomicAdd into shared) + L3 (validation) | — | Weighted counts are fixed-point integers for determinism (DET-01); see OQ3. |
| Stream-ordered fit-scoped arena | L3 host (`sylva-cuda`, raw `driver::sys` pool) | — | GPU-05; cudarc *safe* alloc is synchronous — pool needs raw FFI (OQ1). |
| Small/deep-node CPU leaf-finishing cutover | L3 host dispatch (`sylva-cuda` → `sylva-core::cpu`) | reuses `cpu::fit` subtree builder | GPU-06; below a row/level threshold, finish on CPU (Pitfall 3 mitigation). |
| Four estimator classes (sklearn parity) | L4 Python package (`sylva/`, PyO3-backed) | wraps L3 backends | EST-01..06; `BaseEstimator` semantics, fitted attrs, `check_estimator`. |
| feature_importances_ (real MDI) | L3 host compute from `ForestIR` | exposed at L4 | EST-04 cuML gap; MDI = normalized impurity-decrease from IR arrays (already stored). |
| Honest device dispatch (`device=`, no silent fallback) | L4 Python + L3 typed errors | full `execution_report_` is Phase 6 | EST contract needs a *minimal* dispatch now; the full report is DET-04 (Phase 6). |
| Comparative Baseline Study | Test/bench tier (Python harness) | reads L4 estimators | First real speed claim; fairness protocol binding (PITFALLS 1,2,13). |

## Standard Stack

> Phase 5 introduces **no new Rust dependencies**. The Rust stack is pinned and proven in Phases 1–4. Phase 5 adds *Python-side* test/benchmark dependencies (cuML, XGBoost, sklearn already present) used only by the Comparative Baseline Study harness — these are external baselines, not Sylva runtime deps.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cudarc | 0.19.8 | CUDA driver API + NVRTC + (for the arena) raw `driver::sys` FFI to `cuMemAllocAsync`/`cuMemPoolCreate` | `[VERIFIED: crates/sylva-cuda/Cargo.toml]` Proven Phases 1–4. Safe alloc (`alloc_zeros`/`clone_htod` on `CudaStream`) is **synchronous**; the stream-ordered pool needs `driver::sys`. `[CITED: docs.rs/cudarc/0.19.8 driver index + driver::sys]` |
| PyO3 | 0.29.0 | The four estimator classes' Rust↔Python FFI (`#[pyclass]`, `#[pymethods]`) | `[VERIFIED: Cargo.toml pyo3 0.29 abi3-py310]` Already used by the `pyseam` seam; Phase 5 promotes that seam into the full estimator API. |
| maturin | ≥1.14,<2.0 | Build the `sylva` abi3 wheel | `[VERIFIED: pyproject.toml build-system]` Canonical PyO3 backend; abi3-py310 one-wheel. |
| rust-numpy (`numpy` crate) | 0.29 | Zero-copy numpy→ndarray at the estimator boundary | `[VERIFIED: sylva-core Cargo.toml numpy 0.29; STATE.md links=python lock]` Must track PyO3 0.29 exactly. |
| sylva-core | (workspace) | `Backend`/`HistogramBackend`, `ForestIR`, Philox, `bootstrap`, `split_rf`, `criterion`, `BinnedMatrix`/`BinEdges`, `TrainConfig` | `[VERIFIED: crates/sylva-core/src]` The device-neutral contract + CPU oracle the GPU matches and the Python layer wraps. |
| scikit-learn | 1.9.0 | `check_estimator` / `parametrize_with_checks` (CI gate) + distributional & speed baseline | `[VERIFIED: .venv-parity scikit_learn-1.9.0.dist-info]` `expected_failed_checks` (1.6+) is the sanctioned way to document intentional exceptions. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cuML (RAPIDS) | pin in study manifest | RandomForest GPU baseline (LABELED a different algorithm — RF, no ET) | Comparative Baseline Study only. Not a Sylva dependency. `[ASSUMED]` install path on Windows may require WSL/conda — verify in study setup (OQ7). |
| XGBoost | pin in study manifest | rf-mode (`num_parallel_tree` + `num_boost_round=1`) random-forest baseline | Study only; the RF-equivalent like-for-like external. |
| pytest | (in `.venv-parity`) | `check_estimator` parametrized run + the study harness | The Python CI/test runner; mirrors `python/tests/parity/`. |
| rayon | 1.x | CPU oracle tree-parallelism + the CPU leaf-finishing cutover path | The GPU-06 cutover reuses the rayon CPU subtree builder. |
| thiserror | 1.x | `CudaError`/`SylvaError` → typed PyErr (no silent fallback) | The estimator dispatch boundary maps device errors to Python exceptions. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cudarc::driver::sys` raw FFI pool (`cuMemAllocAsync`) | A single fit-scoped pre-sized slab from safe `alloc_zeros`, reused across waves (no per-wave alloc) | The slab approach satisfies "no per-node allocation" (GPU-05's real intent) with the *safe* API and zero `unsafe` FFI; it is NOT literally `cudaMallocAsync` stream-ordered. Recommended fallback if the raw pool proves fragile — see OQ1. |
| RF best-split over BINS (scan+argmax) | RF best-split over RAW sorted values (the CPU oracle's exact algorithm) | Bins give O(n_bins) candidates vs O(n_rows) sorted midpoints → far cheaper on GPU, but produces a *different* (binned) tree than the raw-value CPU oracle unless the oracle is also binned. This is the bit-exact fork — OQ2. |
| PyO3 `#[pyclass]` estimator in Rust | Pure-Python estimator wrapping the `pyseam` functions | A thin Python class over the JSON-handle seam is faster to write and easier to make `check_estimator`-clean (Python introspection is native), but adds a serialize/deserialize per call. Recommended: **Python estimator class** wrapping a minimal Rust fit/predict seam (see Pattern 5 / OQ4). |
| Hand-rolled MDI | sklearn's `feature_importances_` | Can't reuse sklearn (different trees); MDI is ~15 lines over IR arrays we already store. Hand-roll from the IR. |

**Installation:** No new Rust packages. Study-only Python baselines (pin exact versions in the study manifest, mirroring `conftest.py::VERSION_MANIFEST`):
```bash
# Study harness only — external baselines, NOT sylva runtime deps:
pip install cuml-cu12  # or via RAPIDS conda; verify Windows availability (OQ7)
pip install xgboost
# sklearn 1.9.0 already pinned in .venv-parity
```

**Version verification:** All Sylva Rust pins are already in `Cargo.lock` + `VERSIONS.md` (cudarc 0.19.8, PyO3 0.29.0, numpy 0.29, CUDA 12.8, sm_89, driver 595.79). sklearn 1.9.0 is confirmed present in `.venv-parity`. cuML/XGBoost versions are study-time pins to be recorded in the study manifest — they are not committed dependencies.

## Package Legitimacy Audit

> Phase 5 adds **no new Rust dependencies**. The only new packages are *Python study baselines* (cuML, XGBoost) that are external comparison targets, not Sylva runtime or build dependencies. Sylva's own stack (cudarc, pyo3, numpy, ndarray, rayon, serde, thiserror) was vetted and pinned in Phases 1–2 and is in the committed `Cargo.lock`.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| cudarc | crates.io | active (0.19.8) | 300k+/version | github.com/coreylowman/cudarc | OK (pinned Phase 1) | Approved (in use) |
| pyo3 / numpy / ndarray / rayon / serde / thiserror | crates.io | mature | very high | (canonical repos) | OK (pinned Phase 1–2) | Approved (in use) |
| scikit-learn | PyPI | mature (1.9.0) | very high | github.com/scikit-learn/scikit-learn | OK | Approved (baseline, present) |
| xgboost | PyPI | mature | very high | github.com/dmlc/xgboost | OK | Approved (study baseline) |
| cuml-cu12 | PyPI/RAPIDS | mature | high | github.com/rapidsai/cuml | OK | Approved (study baseline; Windows install needs verification — OQ7) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*The exact `cuml-cu12` / `xgboost` package names and Windows install paths are `[ASSUMED]` from ecosystem knowledge; the planner should gate the study-harness install behind a `checkpoint:human-verify` task since cuML on native Windows is historically problematic (RAPIDS is Linux-first — OQ7).*

## Architecture Patterns

### System Architecture Diagram

```
  PYTHON USER LAYER (sylva/  — new abi3 wheel, EST-01..06)
  ─────────────────────────────────────────────────────────
  ExtraTreesClassifier / ExtraTreesRegressor /
  RandomForestClassifier / RandomForestRegressor   (BaseEstimator, no logic in __init__)
        │  fit(X, y, sample_weight=None)
        │  predict / predict_proba / predict_log_proba / score
        │  get_params / set_params  (clone-able)
        ▼
  device dispatch  device="auto"|"cuda"|"cpu", fallback="error"
        │   (no silent fallback — typed PyErr on unmet "cuda" request)
        ├──────────────────────────────┐
        ▼                              ▼
  RUST CudaBackend (sylva-cuda)   RUST CpuBackend (sylva-core, shipped)
        │                              │  fit_forest (rayon over trees, exact)
        ▼                              ▼
  ┌─ GPU FOREST FIT (breadth-first, many trees) ─────────────────────────┐
  │  for tree t in 0..n_estimators:                                       │
  │    row set = bootstrap_indices(n,seed,t)  (RF)  | all rows (ET)       │
  │    fit-scoped ARENA (GPU-05): one pool, buffers reused across waves   │
  │    while frontier not empty AND depth < max_depth:                    │
  │      ┌─ per frontier node ────────────────────────────────────────┐  │
  │      │ build privatized __shared__ INTEGER histogram over bins      │  │
  │      │   (or counts per side for ET) — weighted if sample_weight    │  │
  │      │ SIBLING SUBTRACTION: compute smaller child only;             │  │
  │      │   larger = parent_hist − smaller_hist  (integer-exact)       │  │
  │      │ RF: inclusive prefix-SCAN over bins → cumulative L/R counts   │  │
  │      │     argmax over (feature,bin) with criterion.rs op order +   │  │
  │      │     (feature, threshold_bits) tie-break                       │  │
  │      │ ET: single raw-range Philox threshold (Phase-4 Strategy A)    │  │
  │      │ scatter-partition row indices into [left|right]              │  │
  │      │ GPU-06: if node rows < CUTOFF or depth deep → CPU finish      │  │
  │      └─────────────────────────────────────────────────────────────┘  │
  │    append nodes; build next frontier                                  │
  └──────────────────────────────────────────────────────────────────────┘
        │  D2H node arrays per tree
        ▼
  assemble ForestIR  (SAME SoA struct CPU writes — ENG-02; assemble_forest global offsets)
        │
        ├──► predict / predict_proba  (reuse cpu::predict_forest or GPU traversal)
        ├──► feature_importances_  = normalized MDI from IR (impurity, node_sample_count, feature_id)
        └──► Comparative Baseline Study: end-to-end fit(X,y) from numpy vs sklearn/cuML/XGBoost
```

### Recommended Project Structure
```
crates/sylva-cuda/src/cuda_backend/
├── mod.rs              # EXTEND: CudaBackend::fit dispatches ET vs RF, clf vs reg, n_estimators>1
├── scheduler.rs        # EXTEND: many-tree loop; per-tree bootstrap row set; sibling-subtraction bookkeeping
├── forest.rs           # NEW: forest-level orchestration (tree loop, per-tree Philox/bootstrap schedule)
├── histogram.rs        # EXTEND: per-(feature,bin) histogram variant + sibling-subtract kernel wrapper
├── rf_split.rs         # NEW: RF best-split-over-bins (inclusive scan + argmax) launch + host-score path
├── sample_weight.rs    # NEW: weighted-histogram (fixed-point) launch wrapper + validation
├── arena.rs            # NEW: fit-scoped device pool (driver::sys cuMemAllocAsync OR pre-sized slab) — GPU-05
├── cpu_cutover.rs      # NEW: small/deep-node CPU leaf-finishing dispatch (reuses sylva-core::cpu) — GPU-06
├── device_buffers.rs   # EXTEND: arena-backed allocation; per-wave reuse, no per-node cudaMalloc
└── assemble.rs         # EXTEND: multi-tree assembly (mirror cpu/fit.rs assemble_forest global offsets)

crates/sylva-cuda/src/kernels.rs   # EXTEND: RF_BINNED_HIST_SRC, RF_SCAN_ARGMAX_SRC, SIBLING_SUBTRACT_SRC, WEIGHTED_HIST_SRC

crates/sylva-core/src/
├── importance.rs       # NEW: feature_importances_ MDI from ForestIR (device-neutral; EST-04)
├── pyseam.rs           # EXTEND or supersede: promote to full estimator seam (fit/predict/predict_proba/params/attrs)
└── (cpu/, ir.rs, etc. UNCHANGED unless OQ2 picks binned-oracle)

python/sylva/           # NEW: the user-facing package (the abi3 wheel's Python surface)
├── __init__.py
├── _base.py            # SylvaForestBase(BaseEstimator) — shared fit/predict/params/attrs, no logic in __init__
├── ensemble.py         # ExtraTreesClassifier/Regressor, RandomForestClassifier/Regressor
└── _dispatch.py        # device="auto|cuda|cpu", fallback="error" (minimal; full report = Phase 6)

python/tests/
├── test_check_estimator.py   # the EST-06 CI gate (parametrize_with_checks, expected_failed_checks)
├── test_estimator_api.py     # fitted attrs, clone, get/set_params, predict_proba/log_proba/score
├── test_feature_importances.py  # MDI vs sklearn distributional + sums-to-1 invariant
└── gpu_forest/               # RF/forest GPU-vs-CPU-oracle bit-exact (mirror Phase-4 gpu_parity/)
    ├── test_rf_cpu_gpu_bitexact.py
    └── test_sample_weight.py

python/benchmarks/
└── comparative_study.py      # end-to-end study: Sylva ET/RF vs sklearn/cuML/XGBoost (fairness protocol)
```

### Pattern 1: RF best-split over bins — privatized histogram → inclusive scan → argmax
**What:** Replace ET's "one random threshold per feature" with RF's "best of all bin-boundary thresholds." Per candidate feature: build a per-bin class-count histogram (privatized integer, the proven kernel), inclusive-scan the bins to get cumulative left counts (right = total − left), score each boundary with `criterion.rs` op order, argmax with the exact tie-break.
**When to use:** `cfg.algo == RandomForest`. ET stays on the Phase-4 raw-range path.
**Example (CUDA-C sketch):**
```cuda
// Source: original, reimplemented from the histogram-tree best-split algorithm
// (sklearn BestSplitter semantics over bins; Apache-2.0, NOT copied). Shares the
// privatized integer-histogram engine with ET. NVRTC -fmad=false, NO --use_fast_math.
// hist[bin*n_classes + cls] = integer class counts for this (node, feature).
extern "C" __global__ void rf_scan_argmax(
    const unsigned int* __restrict__ hist,  // [n_bins * n_classes], from privatized kernel
    const float* __restrict__ bin_edges,    // real thresholds per bin boundary (BinEdges.flat)
    int n_bins, int n_classes, int feat,
    /* outputs */ float* best_improvement, int* best_bin) {
    // Inclusive prefix-scan over bins of the per-class counts → cumulative left counts.
    // (Single-block scan for n_bins<=256; Hillis-Steele or Blelloch with FIXED order
    //  so the integer cumulants are deterministic — counts are associative.)
    // For each boundary b in 1..n_bins:
    //   left_counts[c]  = prefix[b][c]; right_counts[c] = total[c] - left_counts[c]
    //   improvement = proxy_improvement(parent_imp, gini(left), gini(right), nL, nR)
    //                 computed in the EXACT criterion.rs f32 op order
    //   argmax with tie-break: keep if imp>best || (imp==best && (feat,thr_bits)<(best_feat,best_thr_bits))
}
// Lowest-risk variant: return per-feature cumulative counts to host and score with
// sylva_core::cpu::criterion (zero device-float-parity risk — the Phase-4 locked
// scoring decision), then argmax host-side. Move on-device only if host round-trips
// dominate (Phase-7 perf concern, not a Phase-5 correctness need).
```

### Pattern 2: Sibling-histogram subtraction (the forest perf win)
**What:** After a parent node splits, you need both children's histograms. Compute the histogram only for the child with FEWER rows (cheaper scatter+accumulate), then derive the other by element-wise integer subtraction from the parent's histogram. Halves histogram-build work down the tree.
**When to use:** Every internal split where the parent histogram is retained. **Integer counts only** — never subtract float sums (non-associative; would break determinism/exactness).
**Example (host bookkeeping + subtract kernel):**
```rust
// Source: original (standard hist-tree optimization; reimplemented clean).
// Decide the smaller child by row count; compute it; subtract for the larger.
let (small_range, large_range, large_is_left) =
    if n_left <= n_right { (left, right, false) } else { (right, left, true) };
launch_build_histogram(small_range, &mut d_small_hist)?;        // direct build
launch_sibling_subtract(&d_parent_hist, &d_small_hist, &mut d_large_hist)?; // parent - small
// d_large_hist now equals what a direct build would produce, exactly (integer).
```
```cuda
extern "C" __global__ void sibling_subtract(
    const unsigned int* __restrict__ parent, const unsigned int* __restrict__ child_small,
    unsigned int* __restrict__ child_large, int len) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) child_large[i] = parent[i] - child_small[i]; // exact integer
}
```
**Anti-pattern:** Subtracting weighted FLOAT histograms directly. For `sample_weight`, accumulate weights as **fixed-point integers** (scaled u64) so subtraction stays exact (OQ3); convert to f32 once at scoring.

### Pattern 3: Per-tree RNG + bootstrap schedule (already defined — reuse, don't reinvent)
**What:** The forest extends the single-tree Philox counter `(seed, tree, node, feature, draw)` across `n_estimators` trees. RF bootstrap row sets come from `bootstrap_indices(n, seed, tree)` keyed by the `node=u32::MAX` sentinel (`bootstrap.rs`). Because every draw is a pure function of its coordinate, GPU tree-parallelism is order-independent — identical to the CPU oracle's rayon build.
**When to use:** The forest loop. The GPU must reproduce `bootstrap_indices` exactly (inline the same Philox call with `node = BOOTSTRAP_NODE_SENTINEL`) for RF bit-exactness.
**Example:** see `crates/sylva-core/src/cpu/bootstrap.rs` (the authoritative reference — already documents the Phase-4/5 GPU-repro contract).

### Pattern 4: sklearn-parity estimator (BaseEstimator, no logic in `__init__`)
**What:** Each estimator stores constructor params verbatim in `__init__` (no validation, no derived state — a `check_estimator` hard rule), does all work in `fit`, and exposes fitted attrs with the trailing-underscore convention. `get_params`/`set_params` come free from `BaseEstimator` if every `__init__` arg is a same-named attribute.
**When to use:** All four classes. Share via `SylvaForestBase(BaseEstimator)`.
**Example (Python):**
```python
# Source: sklearn developer guide (BaseEstimator contract), sklearn 1.9.
# https://scikit-learn.org/stable/developers/develop.html
from sklearn.base import BaseEstimator, ClassifierMixin
class _SylvaForest(BaseEstimator):
    def __init__(self, n_estimators=100, *, max_depth=None, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 max_samples=None, criterion="gini", random_state=None,
                 n_jobs=None, class_weight=None, device="auto", fallback="error"):
        # NO logic here — just store. (check_estimator enforces this.)
        self.n_estimators = n_estimators; self.max_depth = max_depth
        self.max_features = max_features; ...  # one assignment per param, same name
    def fit(self, X, y, sample_weight=None):
        X, y = self._validate_data(X, y)         # sets n_features_in_, feature_names_in_
        # build TrainConfig, dispatch to Rust backend, store fitted IR handle, set:
        # self.classes_, self.n_classes_, self.estimators_, self.feature_importances_
        return self
```

### Pattern 5: feature_importances_ = real MDI from the IR (the cuML gap)
**What:** Mean Decrease in Impurity. For each internal node: `importance[feature] += node_weighted_count * (impurity − w_left*imp_left − w_right*imp_right)`, summed per feature across all trees, normalized per tree to sum to 1, then averaged across trees (sklearn's exact recipe). **Every input is already in `ForestIR`** (`impurity`, `node_sample_count`/`node_weighted_count`, `feature_id`, children). No retraining, no extra GPU work.
**When to use:** Computed lazily/at fit on the assembled IR (`sylva-core::importance.rs`, device-neutral); exposed as `feature_importances_`.
**Anti-pattern:** Approximating with split-count frequency — that's not MDI and won't match sklearn distributionally.

### Anti-Patterns to Avoid
- **Float `atomicAdd` for weighted histograms or float sibling subtraction** — non-deterministic + non-subtractable. Use fixed-point integer weights (OQ3). (SC, PITFALLS 5.)
- **Per-node / per-wave `cudaMalloc`** — GPU-05 mandates a fit-scoped arena. Allocate once (pool or pre-sized slab), reuse across waves.
- **Re-transferring X per tree** — transfer the (binned + raw) matrix ONCE, keep resident for the whole fit (PITFALLS 1; the forest's natural amortization).
- **Logic in estimator `__init__`** — `check_estimator` fails. Store params verbatim; validate in `fit`.
- **Silent CPU fallback when `device="cuda"`** — raise a typed error. The whole project's differentiator (PITFALLS UX; DET-03 contract previewed here).
- **Reporting on-GPU kernel time as the speed claim** — the study times end-to-end `fit(X,y)` from numpy (PITFALLS 1, 13).
- **Comparing Sylva ET vs sklearn RF (or cuML RF) as if equivalent** — ET-vs-ET, RF-vs-RF only (PITFALLS 13; binding fairness note).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RF best-split algorithm | A new exact splitter | `sylva-core::cpu::split_rf::best_split` (the exact oracle) | The GPU must *match* it; it's the bit-exact target, already tested. |
| Bootstrap sampling | A GPU RNG resampler | `sylva-core::cpu::bootstrap::bootstrap_indices` (Philox, sentinel-keyed) | Already documents the GPU-repro contract; inline the same Philox call. |
| Impurity / proxy math | A GPU gini/entropy/mse | `cpu::criterion::{gini,entropy,mse,proxy_improvement}` op order | Bit-exactness needs identical f32 op sequence (host-side scoring locked Phase 4). |
| `get_params`/`set_params`/`clone` | Custom param plumbing | sklearn `BaseEstimator` (params = same-named `__init__` attrs) | Free, correct, `check_estimator`-clean. |
| numpy↔Rust array handoff | A manual buffer copy | `rust-numpy` (`numpy` crate 0.29) zero-copy views | Already the seam's pattern; version-locked to PyO3 0.29. |
| Forest IR assembly (global offsets) | A GPU tree merger | mirror `cpu::fit::assemble_forest` (child-id + leaf-offset global adjust) | Single shared representation (ENG-02); identical layout = bit-comparable. |
| MDI feature importances | A new tree-walk | `importance.rs` over IR arrays (15 lines) | All inputs already in the IR; matches sklearn's recipe. |
| Quantile binning | A GPU re-binner | `sylva-core::quantize` (Phase 3, bit-parity gated) | RF reads `BinnedMatrix`/`BinEdges`; produced + gated CPU-side. |

**Key insight:** Phase 5 hand-rolls **only** the new GPU kernels (RF binned scan+argmax, sibling-subtract, weighted histogram), the arena, the CPU cutover, and the Python class surface. The forest *algorithm* (bootstrap, best-split, assembly, MDI) already exists in `sylva-core` and is the bit-exact target — reimplement the GPU kernel from the algorithm, never copy cuML/XGBoost/LightGBM/sklearn source (Apache-2.0 discipline).

## Common Pitfalls

### Pitfall 1: RF binned-tree vs raw-value-tree divergence (THE Phase-5 fork)
**What goes wrong:** The CPU RF oracle (`split_rf.rs`) evaluates midpoint thresholds between *raw sorted distinct values* (`v_prev*0.5 + v_curr*0.5`). A GPU that splits on *bin boundaries* produces a different tree → the RF GPU-vs-CPU bit-exact gate fails.
**Why it happens:** Histogram tree builders split on bins for speed; the shipped oracle splits on raw midpoints.
**How to avoid:** Resolve OQ2 before planning. Two coherent options: (A) keep the CPU oracle authoritative and have the GPU evaluate raw-value candidates (slower, but bit-exact to the shipped oracle — likely infeasible at scale on GPU); (B) define the *binned* RF split as canonical and add a binned best-split path to the CPU oracle so both agree (a Phase-2 oracle change — must re-pass Phase-2 distributional gate). Phase 4 took the analogous fork for ET (Strategy A, raw-range) — Phase 5 must decide it explicitly for RF, where the cost asymmetry is far larger.
**Warning signs:** RF GPU tree differs from CPU at the first split; thresholds land exactly on bin edges.

### Pitfall 2: Sibling subtraction on the wrong (float) quantity
**What goes wrong:** Subtracting weighted/float histograms gives last-bit-different results from a direct build → divergent splits, broken determinism.
**How to avoid:** Subtract **integer** count histograms only. For `sample_weight`, accumulate fixed-point integer weights (OQ3) and subtract those; convert to f32 once at scoring. Add a test: direct-build child histogram == subtracted child histogram (exact integer equality).
**Warning signs:** Determinism test fails only when sibling subtraction is enabled; counts off by ±1 ULP after weighting.

### Pitfall 3: `check_estimator` `__init__` and validation gotchas
**What goes wrong:** `check_estimator` fails on: logic/validation in `__init__`; params not stored as same-named attributes; mutating params in `fit`; missing `n_features_in_`/`feature_names_in_`; not raising on unfitted predict; not handling `sample_weight`/`class_weight` per the API.
**Why it happens:** The contract is strict and easy to violate incrementally.
**How to avoid:** Inherit `BaseEstimator` + the right mixin (`ClassifierMixin`/`RegressorMixin`); store params verbatim; do all validation in `fit` via `self._validate_data` (sets the `*_in_` attrs); use `parametrize_with_checks([...])` in pytest and document any intentional exception via `expected_failed_checks={"check_name": "reason"}` (sklearn 1.6+; legacy=False for strict-API-only). Tags via `__sklearn_tags__` if a check needs opting out by capability.
**Warning signs:** `check_estimator` red on `check_parameters_default_constructible` or `check_no_attributes_set_in_init`.

### Pitfall 4: RF determinism under GPU parallelism + bootstrap
**What goes wrong:** Two same-seed RF runs produce different trees because bootstrap row order, histogram accumulation order, or scan order varies run-to-run.
**How to avoid:** Bootstrap indices are pure functions of `(seed, tree, i)` (already order-independent). Histograms are integer (order-free). The scan must use a FIXED reduction order. The argmax tie-break is a total order on `(feature, threshold_bits)`. Test: two same-seed RF fits → byte-identical serialized `ForestIR` (the `fit.rs::rf_seed_determinism_byte_identical` idiom, applied to the GPU path).
**Warning signs:** RF parity passes once then fails on rerun; only RF (not ET) is non-deterministic (points at scan/argmax order).

### Pitfall 5: max_samples / max_features default mismatch with sklearn
**What goes wrong:** Wrong defaults make the study's "identical hyperparameters" claim false and `check_estimator`/parity drift. sklearn defaults: classifier `max_features="sqrt"`, regressor `max_features=1.0` (all), `bootstrap=True` for RF / `False` for ET, `max_samples=None` (=n rows when bootstrapping).
**How to avoid:** Mirror sklearn defaults exactly in the Python `__init__`. `max_samples` (currently NOT in `TrainConfig`) must be added end-to-end: a float fraction or int count of bootstrap draws — extend `bootstrap_indices` to draw `m = round(max_samples * n)` rows (OQ5). `class_weight` reweights the per-class histogram contribution (folds into `sample_weight` — OQ6).
**Warning signs:** Sylva and sklearn report different tree depths/accuracy at "identical" params.

### Pitfall 6: Arena lifetime / stream-ordering hazards (GPU-05)
**What goes wrong:** A stream-ordered pool buffer freed/reused before the consuming kernel finishes → silent corruption or use-after-free; or the raw `driver::sys` FFI is misused (no `unsafe`/`SAFETY:` discipline) → UB.
**How to avoid:** If using `cuMemAllocAsync` (OQ1), free buffers with `cuMemFreeAsync` on the *same* stream after the last consumer, and `synchronize()` at fit end before D2H. If using the safe pre-sized slab fallback, lifetime is trivially the whole fit (no per-wave free). Confine all raw FFI to `arena.rs` with a checked safe wrapper + `// SAFETY:` comments; run `compute-sanitizer` (memcheck) on a forest fit.
**Warning signs:** memcheck flags after enabling the arena; intermittent corruption tied to wave boundaries.

### Pitfall 7: Comparative-study fairness violations (the credibility gate)
**What goes wrong:** Reporting a speed win that collapses under review — single-thread CPU baseline, "data already on GPU" timing, ET-vs-RF crossing, different hyperparameters, no accuracy beside speed, hidden OOM.
**How to avoid:** Bind the harness to the protocol (PITFALLS 1,2,13 + the binding note): time `fit(X,y)` from a host numpy array including dtype coercion + H2D + quantization; sklearn `n_jobs=-1`; cuML RF **labeled a different algorithm**; XGBoost rf-mode; cold AND warm separated; repeated runs; accuracy/probability calibration reported next to every speed cell; ET-vs-ET, RF-vs-RF only; pin all versions in the manifest (extend `conftest.py::VERSION_MANIFEST`); report OOM/failures honestly. This study is a per-phase data point that *feeds* Phase 7 — speed is reported with the crossover caveat, NOT gated. The gate is accuracy parity.
**Warning signs:** A speed number without an accuracy number beside it; the CPU baseline not using all cores; no cold/warm split.

## Code Examples

### check_estimator CI gate (sklearn 1.9 parametrize_with_checks + documented exceptions)
```python
# Source: scikit-learn 1.9 estimator_checks API.
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from sylva.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                            RandomForestClassifier, RandomForestRegressor)

# Document any intentional exception with a reason (1.6+). Empty dict = full parity.
EXPECTED_FAILED = {
    # e.g. "check_sample_weight_equivalence": "GPU fixed-point weights differ at 1 ULP — documented",
}

@parametrize_with_checks(
    [ExtraTreesClassifier(device="cpu"), ExtraTreesRegressor(device="cpu"),
     RandomForestClassifier(device="cpu"), RandomForestRegressor(device="cpu")],
    expected_failed_checks=lambda est: EXPECTED_FAILED,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)   # device="cpu" so GPU-less CI can run the API gate (PITFALLS 16)
```

### feature_importances_ MDI (device-neutral, from ForestIR)
```rust
// Source: original — sklearn MDI recipe over the shipped IR arrays. Device-neutral.
// importance[f] += weighted_count[node] * (imp[node] - wL*imp[L] - wR*imp[R]),
// summed over internal nodes, normalized per tree to sum 1, averaged over trees.
pub fn feature_importances(ir: &ForestIR) -> Vec<f32> {
    let mut imp = vec![0f64; ir.n_features];
    for t in 0..ir.n_trees {
        let mut tree_imp = vec![0f64; ir.n_features];
        for n in ir.tree_node_range(t) {
            if ir.is_leaf[n] { continue; }
            let (l, r) = (ir.left_child[n] as usize, ir.right_child[n] as usize);
            let (wn, wl, wr) = (ir.node_weighted_count[n] as f64,
                                ir.node_weighted_count[l] as f64,
                                ir.node_weighted_count[r] as f64);
            let decrease = wn*ir.impurity[n] as f64 - wl*ir.impurity[l] as f64 - wr*ir.impurity[r] as f64;
            tree_imp[ir.feature_id[n] as usize] += decrease;
        }
        let s: f64 = tree_imp.iter().sum();
        if s > 0.0 { for (a, v) in imp.iter_mut().zip(&tree_imp) { *a += v / s; } }
    }
    let n = ir.n_trees as f64;
    imp.iter().map(|v| (v / n) as f32).collect()  // averaged; sums ~1 across features
}
```

### RF GPU-vs-CPU-oracle bit-exact gate
```rust
// Source: extends fit.rs::rf_seed_determinism + the Phase-4 parity gate idiom.
#[test]
fn gpu_rf_forest_matches_cpu_oracle_bit_exact() {
    let (x, y) = fixed_seed_dataset();
    let cfg = TrainConfig { algo: Algo::RandomForest, bootstrap: true,
                            n_estimators: 8, seed: 42, /* .. */ };
    let cpu = CpuBackend.fit(x.view(), y.view(), &cfg).unwrap();
    let gpu = CudaBackend::new().unwrap().fit(x.view(), y.view(), &cfg).unwrap();
    assert_eq!(serde_json::to_string(&cpu).unwrap(),
               serde_json::to_string(&gpu).unwrap(),
               "GPU RF must equal CPU oracle byte-for-byte"); // gated on OQ2 resolution
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ET single random threshold (Phase 4) | RF best-of-all-bins via prefix-scan + argmax | this phase (GPU-04) | The `BinnedMatrix` becomes load-bearing; scan is the RF primitive. |
| Build both children's histograms | Sibling subtraction (compute smaller, derive larger) | this phase (GPU-03) | ~halves histogram work; only valid for integer counts. |
| Pre-sized per-fit node arrays (Phase 4) | Stream-ordered fit-scoped arena pool | this phase (GPU-05) | No per-node/per-wave alloc; cudarc safe API is synchronous → raw `sys` pool or slab. |
| `check_estimator` legacy boolean flags / `_get_tags` | `__sklearn_tags__` + `expected_failed_checks` + `legacy=False` | sklearn 1.6+ | Intentional exceptions are declared, not silently skipped. |
| sklearn `_get_tags()` dict | `Tags` dataclass via `__sklearn_tags__` | sklearn 1.6 | Tag opt-outs use the new dataclass API. |

**Deprecated/outdated:**
- sklearn `_get_tags`/`_more_tags` (pre-1.6) — use `__sklearn_tags__`.
- cudarc safe `alloc_zeros` is **not** stream-ordered — don't assume `cudaMallocAsync` semantics from the safe API; use `driver::sys` for a real pool.
- Don't reach for `--use_fast_math` / default FMA on the parity path (carried from Phase 3/4: `-fmad=false`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | cudarc 0.19.8 exposes `cuMemAllocAsync`/`cuMemPoolCreate` via `driver::sys` (raw FFI) even though the safe API is synchronous. | Standard Stack / OQ1 | If the raw symbols aren't bound, GPU-05 must use the safe pre-sized-slab fallback (still satisfies "no per-node alloc"). Low blast radius — fallback is documented. |
| A2 | The RF GPU path should split on BINS (scan+argmax over `BinnedMatrix`), accepting that this needs an OQ2 decision about whether the CPU oracle becomes binned too. | Pitfall 1 / OQ2 | If the user wants bit-exact-to-the-raw-value oracle, the binned GPU split won't match → either a slow raw-value GPU path or a Phase-2 oracle change. Single biggest design fork. |
| A3 | `max_samples` and `class_weight` are NOT yet in `TrainConfig` and must be added end-to-end for EST-03/parity. | Pitfall 5 / OQ5,6 | Verified absent in `config.rs`. If added wrong, sklearn parity + check_estimator drift. Must extend `TrainConfig`, `bootstrap_indices`, and the weighted path. |
| A4 | The Python user package is NEW (`python/sylva/`) — no estimator classes exist yet; only the test-only `pyseam` seam exists. | Project Structure | Verified: `find python` shows only `tests/`. The wheel's `[tool.maturin] module-name` and package layout are greenfield this phase. |
| A5 | cuML RF is the only first-class GPU baseline (no GPU ExtraTrees in cuML as of the PITFALLS June-2026 check) — so ET has no like-for-like GPU baseline; the study compares Sylva ET only to sklearn ET. | Pitfall 7 / Study | If cuML ships ET, add it (and note the kill-risk, PITFALLS 14). For now, ET's GPU baseline is "none — Sylva is the only GPU ET." |
| A6 | `sample_weight` determinism is achievable via fixed-point integer weight accumulation; pure-float weighted atomics would break DET-01 and sibling subtraction. | Pitfall 2 / OQ3 | If fixed-point precision is insufficient for sklearn parity, `check_sample_weight_equivalence` may need an `expected_failed_checks` entry with a documented 1-ULP reason. |
| A7 | MDI computed in f64 internally then cast to f32 matches sklearn distributionally (sklearn computes in double). | Code Examples | If exact f32-accumulation is required for some gate, recompute in f32 with fixed order; distributional parity is the realistic bar (ENG-04). |

## Open Questions (RESOLVED)

> All resolved by the orchestrator before planning; each is locked in the relevant PLAN.md `## Assumptions` with its rejected alternative:
> - **OQ1 RESOLVED:** safe pre-sized fit-scoped slab (`alloc_zeros`, zero `unsafe`); raw `cuMemAllocAsync` pool deferred. (Plan 05-04)
> - **OQ2 RESOLVED:** binned RF split is canonical — additive `best_split_binned` path added to the CPU oracle (raw path retained), re-validated vs Phase-2 distributional gate. (Plan 05-03)
> - **OQ3 RESOLVED:** `sample_weight` accumulates as fixed-point integers (keeps sibling-subtraction exact; no float histogram atomics). (Plan 05-03)
> - **OQ4 RESOLVED:** Python `BaseEstimator` subclasses over a Rust seam (not `#[pyclass]`). (Plan 05-05)
> - **OQ5 RESOLVED:** `max_samples` added end-to-end to `TrainConfig`. (Plan 05-01/05-05)
> - **OQ6 RESOLVED:** `class_weight` added end-to-end to `TrainConfig`. (Plan 05-01/05-05)
> - **OQ7 RESOLVED:** cuML/XGBoost availability gated behind a blocking `checkpoint:human-verify` (WSL2 / honest "unavailable" fallback). (Plan 05-06)
> - **OQ8 RESOLVED:** reuse `cpu::predict_forest` for the timing study; GPU predict kernel deferred. (Plan 05-06)

1. **GPU-05 arena: raw `cuMemAllocAsync` pool vs safe pre-sized slab?**
   - What we know: cudarc 0.19.8's *safe* alloc (`alloc_zeros`/`clone_htod` on `CudaStream`) is **synchronous**, not stream-ordered `[CITED: docs.rs/cudarc/0.19.8]`. The raw `driver::sys` module exposes `cuMemAllocAsync`/`cuMemPoolCreate` FFI `[CITED: docs.rs cudarc driver::sys]`.
   - What's unclear: whether GPU-05's "cudaMallocAsync pool" must be *literally* a stream-ordered pool, or whether its INTENT ("reuse histogram/row-index buffers across waves with NO per-node allocation") is satisfied by a fit-scoped pre-sized slab reused across waves via the safe API.
   - Recommendation: **Start with the safe pre-sized-slab arena** (zero `unsafe`, trivially correct lifetime, satisfies the no-per-node-alloc intent and is sanitizer-clean), and treat the raw `cuMemAllocAsync` pool as an optional optimization behind a feature flag if the slab's peak memory is too high for large datasets. Confirm the literal-pool requirement with the user — it changes the `unsafe`/sanitizer surface materially.

2. **RF bit-exact contract: binned GPU split (canonical) vs raw-value oracle (current)?**
   - What we know: `split_rf.rs` splits on raw sorted-midpoint thresholds; the GPU RF wants bin-boundary thresholds (scan+argmax) for tractable cost.
   - What's unclear: whether Phase 5 must produce a tree bit-identical to the *current* raw-value CPU oracle (→ a slow raw-value GPU path, or no bit-exact gate for RF — only distributional) or whether the *binned* RF split becomes canonical (→ add a binned best-split to the CPU oracle, re-pass Phase-2's distributional gate).
   - Recommendation: **Define the binned RF split as canonical** and add a binned best-split path to `sylva-core` (CPU) so CPU and GPU agree bit-exactly on bins — mirroring how ET resolved its analogous fork. This keeps the bit-exact gate meaningful AND tractable on GPU. Requires explicit user sign-off because it touches the Phase-2 oracle contract (and its distributional gate must be re-run).

3. **sample_weight accumulation: fixed-point integer vs float?**
   - Recommendation: fixed-point (scaled u64) integer accumulation so histograms stay subtractable and deterministic; convert to f32 once at scoring. Confirm the scale factor / precision is enough for `check_sample_weight_equivalence`; if not, document a 1-ULP `expected_failed_checks` exception.

4. **Estimator surface: Rust `#[pyclass]` vs Python class over a Rust fit/predict seam?**
   - Recommendation: a **Python class** (`sylva/ensemble.py`) inheriting `BaseEstimator`, wrapping a minimal Rust seam (promote `pyseam.rs` to expose fit/predict/predict_proba/the fitted-attr payloads). Python-native introspection makes `check_estimator` far easier than reproducing `get_params` semantics in PyO3. Confirm the seam's IR-handle vs in-memory-handle choice (JSON handle works but adds ser/de per call).

5. **`max_samples` semantics + `TrainConfig` extension.**
   - What we know: not in `config.rs`. sklearn: `None`=n rows, float in (0,1]=fraction, int=count, only meaningful when `bootstrap=True`.
   - Recommendation: extend `TrainConfig` with `max_samples: Option<MaxSamples>` and `bootstrap_indices(m, n, seed, tree)` to draw `m` rows; validate `max_samples` requires `bootstrap=True`.

6. **`class_weight` semantics.**
   - Recommendation: fold `class_weight` ("balanced" / dict / None) into the per-row `sample_weight` at fit (sklearn's approach), so a single weighted-histogram path serves both. Confirm "balanced_subsample" (per-bootstrap) is in scope or deferred.

7. **cuML / XGBoost availability on the native-Windows benchmark host.**
   - What we know: RAPIDS/cuML is Linux-first; native-Windows install is historically problematic. The benchmark host is Windows (CLAUDE.md).
   - Recommendation: gate the study-baseline install behind a `checkpoint:human-verify`. If cuML won't run on the Windows host, document it as an honest study limitation (run cuML under WSL2 with the GPU passthrough, OR report "cuML baseline: not available on the measurement host" rather than faking it). XGBoost has Windows wheels and should work natively.

8. **predict on GPU vs reuse `cpu::predict_forest` for the study?**
   - Recommendation: reuse `cpu::predict_forest` for correctness/parity (the gate is on the trained IR). A GPU predict/predict_proba kernel is optional this phase; if added, it must be sanitizer-clean. The end-to-end study times `fit` (training is the claim); state the predict path used.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU (sm_89) | All GPU kernels + study | ✓ | RTX 4060 Ti, driver 595.79 | none (CUDA-only MVP) |
| CUDA Toolkit + NVRTC | Runtime kernel compile + sanitizer | ✓ | 12.8 | none |
| compute-sanitizer | memcheck/racecheck gate on forest+RF kernels | ✓ | CUDA 12.8 | none — required for the gate |
| cudarc `driver::sys` (cuMemAllocAsync) | GPU-05 stream-ordered pool (OQ1) | ✓ (raw FFI) | 0.19.8 | safe pre-sized slab arena (no raw FFI) |
| Rust stable + MSVC v143 | Build sylva-cuda / abi3 wheel | ✓ | rustc 1.96.0, cl.exe 14.44 | none |
| maturin | Build `sylva` abi3 wheel | ✓ | ≥1.14,<2.0 | none |
| scikit-learn | check_estimator gate + study baseline | ✓ | 1.9.0 (.venv-parity) | none — the API gate needs it |
| XGBoost | Study rf-mode baseline | likely (PyPI wheel) | pin in study | report "unavailable" honestly if missing |
| cuML (RAPIDS) | Study RF GPU baseline | ✗/uncertain on native Windows | pin in study | WSL2 GPU passthrough OR documented "not available on host" (OQ7) |

**Missing dependencies with no fallback:** none that block the kernel/API gates (those need only the proven CUDA toolchain + sklearn, all present).
**Missing dependencies with fallback:** cuML on native Windows (fallback: WSL2, or honest "unavailable" in the study — the study gate is accuracy parity, not the cuML comparison).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `#[test]`/`tests/` (cargo-nextest) for kernels+IR; Python pytest for `check_estimator`, estimator API, MDI, and the study harness |
| Config file | none for Rust (cargo built-in); Python harness mirrors `python/tests/parity/` (conftest version manifest) |
| Quick run command | `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` (RF bit-exact gate) |
| Full suite command | `cargo test -p sylva-cuda` + `cargo test -p sylva-core` (oracle regression) + `pytest python/tests/` (check_estimator + API + MDI) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-03 | Full forest: many-tree breadth-first scheduler + per-tree RNG/bootstrap + sibling subtraction | integration | `cargo test -p sylva-cuda --test forest_cpu_gpu` | ❌ Wave 0 |
| GPU-03 | Sibling-subtracted child histogram == direct-build child histogram (integer-exact) | unit + sanitizer | `cargo test -p sylva-cuda sibling_subtract` + racecheck | ❌ Wave 0 |
| GPU-04 | RF best-split via inclusive prefix-scan + argmax over bins, sharing the ET histogram engine | integration | `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` | ❌ Wave 0 |
| GPU-04 | RF GPU forest == CPU oracle bit-exact (clf+reg), fixed seed (gated on OQ2) | integration (byte-compare) | `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` | ❌ Wave 0 |
| GPU-05 | Fit-scoped arena: no per-node/per-wave alloc; buffers reused across waves | unit + sanitizer | `cargo test -p sylva-cuda arena` + memcheck on a forest fit | ❌ Wave 0 |
| GPU-06 | Small/deep nodes cut over to CPU leaf-finishing; result identical to all-GPU build | integration | `cargo test -p sylva-cuda cpu_cutover` | ❌ Wave 0 |
| EST-05 | `sample_weight` end-to-end via weighted-histogram (fixed-point); GPU==CPU | integration | `cargo test -p sylva-cuda sample_weight` + `pytest test_sample_weight` | ❌ Wave 0 |
| EST-01/02 | Four classes expose fit/predict/predict_proba/predict_log_proba/score, get/set_params, clone-able, no `__init__` logic | Python | `pytest python/tests/test_estimator_api.py` | ❌ Wave 0 |
| EST-03 | Constructor params + sklearn-matching defaults (incl. max_samples, class_weight) | Python | `pytest python/tests/test_estimator_api.py::test_defaults` | ❌ Wave 0 |
| EST-04 | Fitted attrs: classes_, n_classes_, n_features_in_, feature_names_in_, estimators_, feature_importances_ (real MDI) | Python + Rust | `pytest test_feature_importances.py` + `cargo test importance` | ❌ Wave 0 |
| EST-06 | `check_estimator` passes in CI; intentional exceptions documented (`expected_failed_checks`) | Python (parametrized) | `pytest python/tests/test_check_estimator.py` | ❌ Wave 0 |
| SC-6/7 | Study: end-to-end-from-numpy ET/RF vs sklearn/cuML/XGBoost; accuracy beside speed; cold/warm; like-for-like; OOM honest | Python (report) | `python python/benchmarks/comparative_study.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the touched gate — `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` (RF) or the relevant unit + sanitizer target; for Python tasks, `pytest python/tests/test_check_estimator.py -x`.
- **Per wave merge:** full `cargo test -p sylva-cuda` + `cargo test -p sylva-core` (oracle regression) + `pytest python/tests/` + all four sanitizer tools on each new kernel.
- **Phase gate:** RF GPU==CPU-oracle bit-exact (clf+reg, gated on OQ2) AND all new kernels sanitizer-clean (memcheck+racecheck+synccheck+initcheck) AND `check_estimator` green (four classes, exceptions documented) AND the Comparative Baseline Study run with accuracy-parity met and speed+OOM reported, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/sylva-cuda/tests/parity_rf_cpu_gpu.rs` — RF GPU==CPU bit-exact gate (GPU-04)
- [ ] `crates/sylva-cuda/tests/forest_cpu_gpu.rs` — many-tree + sibling-subtraction forest gate (GPU-03)
- [ ] `crates/sylva-cuda/tests/sanitizer_rf_kernels.rs` — standalone sanitizer targets for the new kernels (RF scan/argmax, sibling-subtract, weighted-hist)
- [ ] `crates/sylva-core/src/importance.rs` + unit tests — MDI from IR (EST-04)
- [ ] `python/sylva/` package (`_base.py`, `ensemble.py`, `_dispatch.py`) — the four estimators (EST-01..06)
- [ ] `python/tests/test_check_estimator.py`, `test_estimator_api.py`, `test_feature_importances.py` — API gates
- [ ] `python/tests/gpu_forest/` — RF/forest GPU-vs-CPU + sample_weight Python parity (mirror Phase-4 `gpu_parity/`)
- [ ] `python/benchmarks/comparative_study.py` + study version manifest (extend `conftest.py::VERSION_MANIFEST`)
- [ ] A full estimator pyseam (promote `pyseam.rs`): fit/predict/predict_proba/fitted-attr payloads (or the chosen seam per OQ4)
- [ ] `TrainConfig` extension: `max_samples`, `class_weight`/`sample_weight` plumbing (OQ5,6)

*(No existing GPU-forest/RF or estimator-API test infrastructure — all Phase-5 validation files are new. The Phase-4 `gpu_parity/` harness and `python/tests/parity/` conftest are the templates.)*

## Security Domain

> `security_enforcement` is enabled (config `security_enforcement: true`, ASVS level 1). Local GPU compute library + a Python FFI surface; relevant controls are input validation at the Python↔Rust↔CUDA boundary, memory safety (esp. the new arena raw FFI), and license provenance.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (local library). |
| V3 Session Management | no | N/A. |
| V4 Access Control | no | N/A. |
| V5 Input Validation | yes | Estimator boundary validates X/y shape, dtype, label range, `sample_weight`/`max_samples`/`class_weight` ranges → typed PyErr before any device launch (the shipped `pyseam`/`fit_forest` V5 pattern). Kernel indices (bin<n_bins, feat<n_features, row ranges) bounds-guarded; out-of-range = OOB device read. |
| V6 Cryptography | no | Philox is a non-cryptographic statistical RNG; never used for secrets. |

### Known Threat Patterns for Rust core + CUDA-C via NVRTC + PyO3 boundary
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OOB device read (RF scan/argmax indexing hist[bin*n_classes+cls], bin_edges) | Tampering / Info-disclosure | Host-side V5 validation before launch; `if (bin<n_bins)` / `if (cls<n_classes)` guards; `compute-sanitizer memcheck` clean gate. |
| Use-after-free / premature reuse in the stream-ordered arena (GPU-05) | Tampering | Free on the same stream after last consumer; `synchronize()` before D2H; confine raw `driver::sys` FFI to `arena.rs` with `// SAFETY:` + checked wrapper; memcheck on a forest fit. |
| Shared-memory race in privatized/weighted histogram | Tampering | zero-init → `__syncthreads()` → integer atomicAdd into shared → `__syncthreads()` → flush (Phase-1/4 proven); `racecheck` clean. |
| Silent CUDA/dispatch error swallow → wrong model or silent CPU fallback | Tampering | Every cudarc call `?`-propagated → `CudaError`→`SylvaError`→typed PyErr; `device="cuda"` with unmet requirement RAISES (no silent fallback). |
| Untrusted `sample_weight`/`class_weight` (NaN/neg/len-mismatch) | Tampering | Validate length==n_rows, finite, non-negative at the estimator boundary → `ValueError` before training. |
| License contamination (copying cuML/XGBoost/LightGBM/Snap-ML/sklearn kernel or splitter source) | (Legal/IP) | Apache-2.0 reimplementation from the algorithm; document provenance (as `split_rf.rs` already cites sklearn BestSplitter by description, not copy). |

## Project Constraints (from CLAUDE.md)

- **cudarc 0.19.8 + hand-written CUDA-C via NVRTC** — the only sanctioned kernel path. No CubeCL (M2), no Rust-CUDA `cust`, no nvcc/cc AOT, no wgpu.
- **Native Windows / MSVC, no WSL** for the MVP build (WSL allowed only as a study-baseline lane for cuML if needed — OQ7).
- **NO nvcc-at-build-time** — kernels are `.cu` strings compiled by NVRTC; compile with `-fmad=false`, never `--use_fast_math`.
- **Apache-2.0 reimplementation discipline** — reimplement from papers/docs; never copy GPL / Snap ML / sklearn / cuML / XGBoost / LightGBM source.
- **No silent fallback** — every CUDA/dispatch call is a `Result`/typed PyErr; `device="cuda"` unmet → raises; no `.unwrap()`/`.expect()` on device calls; `unsafe` confined with `// SAFETY:` (esp. the arena FFI).
- **Stable Rust 1.83+** (on 1.96.0) — never nightly. **`-D warnings`** clippy bar; non-deprecated cudarc APIs.
- **float32 end-to-end (D-05)** + **integer/deterministic accumulation** (the DET-01 foundation; sibling subtraction and weighted histograms must stay integer/fixed-point).
- **PyO3 0.29 abi3-py310 + maturin** one-wheel; rust-numpy locked to PyO3 0.29.
- **Many small files** (200–400 lines, 800 max); organize by domain (mirror `cpu/` and the Phase-4 `cuda_backend/` shape).
- **Comparative-study fairness binding** — Phase 5 is the FIRST real speed claim: end-to-end-from-numpy, like-for-like (ET-vs-ET, RF-vs-RF), cold/warm separated, strongest baselines, versions pinned, accuracy beside speed, OOM honest; speed reported with the crossover caveat (Phase 7 is authoritative), accuracy parity is the gate.

## Sources

### Primary (HIGH confidence)
- `crates/sylva-core/src/cpu/{fit.rs,split_rf.rs,bootstrap.rs,criterion.rs,predict.rs}` — the exact RF best-split, Philox bootstrap (with documented GPU-repro contract), forest assembly (global offsets), and float op order the GPU must match.
- `crates/sylva-core/src/{ir.rs,backend.rs,config.rs,pyseam.rs}` — `ForestIR` SoA (MDI inputs already present), `Backend`/`HistogramBackend`, `TrainConfig` (no `max_samples`/`class_weight` yet), the test-only PyO3 seam to promote.
- `crates/sylva-core/src/quantize/binned_matrix.rs` — `BinnedMatrix`/`BinEdges` (col-major SoA, CSR edges) the RF scan reads.
- `crates/sylva-cuda/{Cargo.toml, src/cuda_backend/*}` (Phase-4 plan) — the single-tree kernels + breadth-first scheduler + `CudaError` pattern Phase 5 scales.
- `.planning/phases/04-single-gpu-extratree/04-RESEARCH.md` + `04-02-PLAN.md` — the locked Strategy A, host-side scoring, no-FMA, privatized-histogram shape carried forward.
- `.planning/{ROADMAP.md,REQUIREMENTS.md,STATE.md}` — Phase-5 goal/7 SC, GPU-03..06 / EST-01..06, the binding fairness note, the Comparative Baseline Study spec.
- `.planning/research/PITFALLS.md` — Pitfalls 1,2,3,4,5,13 (the binding fairness rules + breadth-first/atomics/determinism architecture).
- `.venv-parity/.../scikit_learn-1.9.0.dist-info` + `python/tests/parity/{conftest.py,datasets.py}` — pinned sklearn 1.9.0, the version manifest + hyperparameter sets the study extends.

### Secondary (MEDIUM confidence)
- scikit-learn 1.9 estimator-checks docs — `check_estimator`/`parametrize_with_checks`, `expected_failed_checks` (1.6+), `__sklearn_tags__`, `legacy=False`. https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html ; https://scikit-learn.org/stable/developers/develop.html
- scikit-learn MDI feature-importances docs — impurity-decrease recipe (inputs all present in `ForestIR`). https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
- cudarc 0.19.8 docs — safe alloc (`alloc_zeros`/`clone_htod` on `CudaStream`) is synchronous; `driver::sys` exposes raw `cuMemAllocAsync`/`cuMemPoolCreate` FFI. https://docs.rs/cudarc/latest/cudarc/driver/sys/index.html
- NVIDIA stream-ordered memory allocator docs — `cuMemAllocAsync`/`cuMemPoolCreate`/`cuMemFreeAsync` semantics for the arena. https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html

### Tertiary (LOW confidence)
- General level-synchronous histogram-tree + sibling-subtraction + prefix-scan-split algorithm knowledge (XGBoost/LightGBM/cuML hist builders) — used only as *algorithmic* reference; reimplemented clean, never copied. `[ASSUMED]` for any intrinsic-level detail beyond the Phase-1/4-proven shape.
- cuML/XGBoost exact PyPI package names + Windows availability — `[ASSUMED]`; gate study-baseline install behind `checkpoint:human-verify` (OQ7).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new Rust deps; all pinned/proven Phases 1–4; sklearn 1.9.0 confirmed present.
- Architecture (RF scan+argmax, sibling subtraction, per-tree schedule, MDI, estimator contract): HIGH — grounded in shipped `sylva-core` code + sklearn 1.9 docs.
- GPU-05 arena: MEDIUM — cudarc safe API is synchronous; the literal stream-ordered pool needs raw FFI (OQ1, with a safe-slab fallback).
- RF bit-exact contract: MEDIUM — the binned-vs-raw fork (OQ2) is a real design decision the planner/user must lock, mirroring Phase-4's ET Strategy A.
- Comparative Baseline Study: HIGH for the protocol (binding fairness rules); MEDIUM for cuML-on-Windows availability (OQ7).

**Research date:** 2026-06-27
**Valid until:** ~2026-07-27 (stable — toolchain pinned, sklearn 1.9 pinned; the volatile elements are the two locked-decision forks OQ1/OQ2 and the cuML availability check, which are decisions/setup, not moving external facts).
