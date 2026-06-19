# Requirements: Sylva — GPU-Native Forest Ensembles

**Defined:** 2026-06-20
**Core Value:** GPU-trained Extra Trees / Random Forest that match scikit-learn semantics, never silently fall back, and beat optimized CPU baselines on large dense workloads — validated by a pre-registered benchmark crossover before any broad build-out.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Toolchain (Gate 1)

- [ ] **TOOL-01**: A throwaway spike builds a hand-written CUDA C kernel via cudarc 0.19.8 + NVRTC and launches it on the local NVIDIA GPU, natively on Windows/MSVC (no WSL)
- [ ] **TOOL-02**: `compute-sanitizer` runs against the spike kernel and reports clean (toolchain is debuggable)
- [ ] **TOOL-03**: A minimal PyO3 + maturin `abi3` wheel builds and imports in a clean Python environment on Windows
- [ ] **TOOL-04**: Pinned, verified versions recorded for cudarc feature flags, rust-numpy↔PyO3 compatibility, and CUDA toolkit; documented kill-criteria result (proceed / WSL-fallback / stop)

### Engine & Contracts

- [ ] **ENG-01**: A device-neutral `trait Backend` defines all device ops (quantize, build_histograms, eval_splits, partition, predict); CUDA types never cross the trait boundary
- [ ] **ENG-02**: A SoA `ForestIR` (feature_id / threshold / left / right / default-child / leaf-value arrays) is the single shared representation written by training and read read-only by inference, SHAP, and export
- [ ] **ENG-03**: A pure-Rust `CpuBackend` (ndarray + rayon) trains and predicts ET + RF correctly, serving as the differential-test oracle and small-data / `device="cpu"` path
- [ ] **ENG-04**: The documented parity contract is defined: Sylva's own bit-identical CPU↔GPU RNG per seed and distributional equivalence to scikit-learn (NOT bit-identical reproduction of sklearn's serial PRNG)
- [ ] **ENG-05**: NaN / missing-value routing policy is defined and implemented consistently across CPU and GPU paths
- [ ] **ENG-06**: Stateless counter-based Philox-4×32-10 RNG is implemented identically in Rust and the CUDA kernel, keyed by (seed, tree, node, feature, draw)

### Quantizer

- [ ] **QUANT-01**: A feature quantizer produces a SoA `BinnedMatrix` (uint8/uint16) via per-feature quantile bins, on both CPU and GPU
- [ ] **QUANT-02**: CPU and GPU quantizers produce bit-identical bin assignments on a fixed seed (parity test in CI)

### Estimators (sklearn parity)

- [ ] **EST-01**: `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomForestClassifier`, `RandomForestRegressor` expose a sklearn drop-in API (dense float32, single GPU)
- [ ] **EST-02**: Full estimator contract — `fit` / `predict` / `predict_proba` / `predict_log_proba` / `score`, `get_params` / `set_params`, clone-able, `BaseEstimator` semantics (no logic in `__init__`)
- [ ] **EST-03**: Core constructor params with correct defaults — `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `bootstrap`, `max_samples`, `criterion` (gini / entropy / squared_error), `random_state`, `n_jobs`, `class_weight`, `sample_weight`
- [ ] **EST-04**: Core fitted attributes — `classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_`, `estimators_`, `feature_importances_` (real MDI — direct cuML gap)
- [ ] **EST-05**: `sample_weight` is supported end-to-end via a weighted-histogram kernel (full `check_estimator` parity)
- [ ] **EST-06**: `sklearn.utils.estimator_checks.check_estimator` passes in CI, with any intentional exceptions explicitly documented
- [ ] **EST-07**: Differential tests vs scikit-learn (accuracy/distribution within stated tolerance) and property-based invariants (child rows partition parent, leaf probabilities valid, seed determinism, serialization round-trip)

### GPU Training

- [ ] **GPU-01**: A single GPU ExtraTree (classifier + regressor) trains with a privatized shared-memory histogram kernel, a fused random-candidate split kernel, and a scatter-partition kernel — built breadth-first, level-at-a-time
- [ ] **GPU-02**: A single GPU ExtraTree matches the CPU oracle bit-exactly on a fixed seed and is `compute-sanitizer` clean
- [ ] **GPU-03**: Full forest training adds a breadth-first NodeScheduler (frontier waves), per-tree RNG schedule, and sibling-histogram subtraction
- [ ] **GPU-04**: RandomForest adds best-split evaluation via inclusive prefix-sum (scan) + argmax over bins, sharing the histogram engine with ExtraTrees
- [ ] **GPU-05**: A stream-ordered fit-scoped arena (cudaMallocAsync pool) reuses histogram/row-index buffers across waves with no per-node allocation
- [ ] **GPU-06**: Small/deep nodes cut over to a CPU leaf-finishing path to avoid GPU underutilization

### Determinism & Dispatch

- [ ] **DET-01**: `deterministic=True` yields byte-identical models across two same-seed runs (verified by exact binary comparison, not `allclose`), via integer/fixed-point histogram accumulation + canonical reduction order + fixed tie-breaking
- [ ] **DET-02**: The documented performance cost of deterministic mode is measured and reported
- [ ] **DET-03**: `device="auto" | "cuda" | "cpu"` dispatch with `fallback="error"` — no silent CPU fallback; unsupported configs raise
- [ ] **DET-04**: `execution_report_` reports selected backend + reason, every input conversion (dtype/layout/H2D), bytes transferred, and fallback status

### Benchmark (Gate 3)

- [ ] **BENCH-01**: A scripted, reproducible benchmark measures end-to-end training time from numpy (including H2D transfer + quantization), cold and warm, with pinned hardware/driver/CUDA/package versions
- [ ] **BENCH-02**: Baselines include scikit-learn ET/RF (`n_jobs=-1`), oneDAL/sklearnex, and cuML RF (labeled as a different algorithm); accuracy parity reported alongside speed
- [ ] **BENCH-03**: A published (n × d) crossover surface identifies where GPU beats the strongest CPU baseline and where it does not; pre-registered kill criteria evaluated

### Explainability (SHAP)

- [ ] **SHAP-01**: A SHAP feasibility spike (Gate 2) confirms scope = exact **attributions** and verifies GPUTreeSHAP licensing/integration path before implementation
- [ ] **SHAP-02**: `sylva-shap` computes exact per-feature SHAP **attributions** from `ForestIR` (GPUTreeSHAP approach, Apache-2.0), CPU-first then GPU
- [ ] **SHAP-03**: `.shap_values()` results validate against `shap.TreeExplainer` within float tolerance

### Export & Packaging

- [ ] **EXP-01**: `sylva-export` serializes `ForestIR` to a Treelite 4.x `import_from_json()`-compatible JSON
- [ ] **EXP-02**: An exported model round-trips through Treelite/FIL and produces matching predictions (CI test)
- [ ] **EXP-03**: The `abi3` Windows wheel is validated in a fresh environment with CUDA driver dynamic-loading; documented install path

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Explainability

- **SHAP-V2-01**: Exact high-depth SHAP **interaction** values via WOODELF-HD (O(2^D·D²) Strassen-like + UFDP), pending upstream license clearance — the novel research wedge, optional upgrade over attributions

### Data & Scale

- **DATA-V2-01**: Sparse / CSR input (changes missing-vs-zero semantics, histogram, partitioning — near-rewrite)
- **DATA-V2-02**: Native categorical feature handling
- **SCALE-V2-01**: Multi-GPU ensemble sharding (bagging's natural advantage) + Dask/NCCL gather
- **PORT-V2-01**: Vendor-neutral CubeCL backend behind the existing `trait Backend` (ROCm/SYCL reach)

### API surface

- **API-V2-01**: `oob_score` / `oob_score_` (direct cuML gap, issue #3361)
- **API-V2-02**: Zero-copy GPU input via `__cuda_array_interface__` / DLPack
- **API-V2-03**: `ccp_alpha`, `max_leaf_nodes` (best-first growth), `absolute_error` / `poisson` criteria

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Gradient boosting (GBDT/XGBoost-style) | Owned by XGBoost/LightGBM/CatBoost; sequential, saturated, no defensible wedge |
| AdaBoost | Low demand, sequential, tiny per-round work |
| Variance-based redundant-tree pruning (blueprint #5) | Method provenance UNVERIFIED in literature; would need its own empirical validation study before being built or advertised |
| SpMM Tensor-Core kernels (blueprint #1) | Different subfield (sparse linalg); contradicts the debunked matmul premise; trees don't do GEMM |
| CPU BLAS thread autotuning / ADSALA (blueprint #3) | CPU dense GEMM; unrelated to GPU forests |
| INT8/INT4 quantized inference + sparse residual (blueprint #4) | Deep-learning inference technique, not core to forests |
| sTiles selected matrix inversion (blueprint #6) | Spatial-stats / GMRF solver; unrelated to tree ML |
| Tensor-Core histogram reformulation | Research-only hypothesis; tree training is bandwidth/atomics-bound, not GEMM-bound |
| `warm_start=True` | Accept param, raise on True for v1 |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

> Note: Each phase additionally carries a **Comparative Baseline Study** (see ROADMAP.md) — an existing-library + baseline-implementation comparison with associated success criteria. These studies are success-criteria additions on top of the requirements below (foundational phases test correctness-parity / microbenchmarks; the benchmark requirements BENCH-01..03 own the authoritative crossover). They introduce **no new requirement IDs**, so coverage remains 38/38.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TOOL-01 | Phase 1 | Pending |
| TOOL-02 | Phase 1 | Pending |
| TOOL-03 | Phase 1 | Pending |
| TOOL-04 | Phase 1 | Pending |
| ENG-01 | Phase 2 | Pending |
| ENG-02 | Phase 2 | Pending |
| ENG-03 | Phase 2 | Pending |
| ENG-04 | Phase 2 | Pending |
| ENG-05 | Phase 2 | Pending |
| ENG-06 | Phase 2 | Pending |
| EST-07 | Phase 2 | Pending |
| QUANT-01 | Phase 3 | Pending |
| QUANT-02 | Phase 3 | Pending |
| GPU-01 | Phase 4 | Pending |
| GPU-02 | Phase 4 | Pending |
| GPU-03 | Phase 5 | Pending |
| GPU-04 | Phase 5 | Pending |
| GPU-05 | Phase 5 | Pending |
| GPU-06 | Phase 5 | Pending |
| EST-01 | Phase 5 | Pending |
| EST-02 | Phase 5 | Pending |
| EST-03 | Phase 5 | Pending |
| EST-04 | Phase 5 | Pending |
| EST-05 | Phase 5 | Pending |
| EST-06 | Phase 5 | Pending |
| DET-01 | Phase 6 | Pending |
| DET-02 | Phase 6 | Pending |
| DET-03 | Phase 6 | Pending |
| DET-04 | Phase 6 | Pending |
| BENCH-01 | Phase 7 | Pending |
| BENCH-02 | Phase 7 | Pending |
| BENCH-03 | Phase 7 | Pending |
| SHAP-01 | Phase 8 | Pending |
| SHAP-02 | Phase 8 | Pending |
| SHAP-03 | Phase 8 | Pending |
| EXP-01 | Phase 9 | Pending |
| EXP-02 | Phase 9 | Pending |
| EXP-03 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 38 total (enumerated IDs; supersedes the earlier header estimate of 33)
- Mapped to phases: 38 ✓
- Unmapped: 0 ✓

**Coverage by phase:**

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1 — Toolchain Spike (Gate 1) | TOOL-01, TOOL-02, TOOL-03, TOOL-04 | 4 |
| 2 — CPU Oracle, Contracts & Forest IR | ENG-01..06, EST-07 | 7 |
| 3 — Feature Quantizer | QUANT-01, QUANT-02 | 2 |
| 4 — Single GPU ExtraTree | GPU-01, GPU-02 | 2 |
| 5 — Full Forest, RandomForest & sklearn Estimators | GPU-03..06, EST-01..06 | 10 |
| 6 — Determinism & Honest Dispatch | DET-01..04 | 4 |
| 7 — Crossover Benchmark (Gate 3) | BENCH-01..03 | 3 |
| 8 — Exact Tree SHAP (Gate 2 + Implementation) | SHAP-01..03 | 3 |
| 9 — Treelite Export & Packaging | EXP-01..03 | 3 |

---
*Requirements defined: 2026-06-20*
*Last updated: 2026-06-20 — roadmap revised to add a per-phase Comparative Baseline Study + fairness note; no new requirement IDs, coverage unchanged at 38/38*
