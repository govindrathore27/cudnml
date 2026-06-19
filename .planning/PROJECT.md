# Sylva — GPU-Native Forest Ensembles (Rust core, Python API)

> Project codename **Sylva** is provisional; rename before first public release.

## What This Is

A GPU-native, scikit-learn-compatible library for the tree-ensemble algorithms the
current GPU ecosystem leaves underserved — **Extra Trees** and **Random Forest**
(classifier + regressor) — plus **exact, high-depth tree SHAP** explainability. The
performance core is written in **Rust** and runs on **NVIDIA CUDA**; the user-facing
layer is a **Python** package with a sklearn-parity API installable via `pip`
(PyO3 + maturin). Target user: ML practitioners who use bagging ensembles on large
tabular data and want GPU training without leaving the sklearn idiom, with a CPU
reference path and honest, *non-silent* CPU/GPU dispatch.

## Core Value

GPU-trained Extra Trees / Random Forest that match scikit-learn semantics, never
silently fall back, and beat optimized CPU baselines on large dense workloads —
validated by a pre-registered benchmark crossover before any broad build-out.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] `ExtraTreesClassifier` — GPU-native, dense float32, sklearn-parity API
- [ ] `ExtraTreesRegressor` — GPU-native, dense float32
- [ ] `RandomForestClassifier` — shared histogram/split engine with Extra Trees
- [ ] `RandomForestRegressor`
- [ ] Deterministic training mode (`deterministic=True`) — reproducible models, documented perf cost
- [ ] Explicit device dispatch (`device="auto" | "cuda" | "cpu"`) with **no silent fallback** (`fallback="error"`) and an `execution_report_` explaining every decision and input conversion
- [ ] CPU reference backend (correctness oracle + small-data path)
- [ ] Variance-based redundant-tree pruning option (blueprint fix #5), folded into training
- [ ] Prefix-sum (inclusive scan) split-evaluation kernels (blueprint fix #7)
- [ ] Exact high-depth tree SHAP — WOODELF-HD approach: vectorized Strassen-like scheme + UFDP path compression (blueprint fix #2)
- [ ] Model export to a Treelite-compatible representation for CPU/FIL serving
- [ ] Differential tests vs scikit-learn; property-based invariants; CUDA correctness tooling

### Out of Scope

- **Gradient boosting (GBDT/XGBoost-style)** — owned by XGBoost/LightGBM/CatBoost; sequential, saturated, no defensible wedge
- **AdaBoost** — low demand, sequential, tiny per-round work
- **Sparse / CSR input** — deferred to a post-MVP RFC; changes missing-vs-zero semantics, histogram, partitioning (near-rewrite). Dense-only first
- **Native categorical pipeline** — CatBoost-class target statistics out of reach for MVP
- **Multi-output / multi-class-heavy histogram compression** — later
- **Multi-GPU / multi-node** — defer until single-GPU engine proves value
- **SpMM Tensor-Core kernels (blueprint fix #1)** — different subfield (sparse linalg); contradicts the debunked matmul premise; not tree training
- **CPU BLAS thread autotuning / ADSALA (fix #3)** — CPU dense GEMM, unrelated to GPU forests
- **INT8/INT4 quantized inference + sparse residual (fix #4)** — deep-learning inference technique, not core to forests
- **sTiles selected matrix inversion (fix #6)** — spatial-stats/GMRF solver, unrelated to tree ML
- **Tensor-Core reformulation of histograms** — research-only hypothesis; tree training is bandwidth/atomics-bound, not GEMM-bound

## Context

- **Origin:** Two independent feasibility studies (markdown + HTML) concluded *narrow
  wedge only* — GPU Extra Trees + sparse RF + determinism, ideally upstreamable to
  RAPIDS cuML. A third "7 fixes" blueprint proposed a much broader, incoherent scope
  spanning 4+ subfields; the user explicitly narrowed to **wedge + interpretability**.
- **Debunked premise:** "GPUs help because matmul is fast" is wrong for trees. Tree
  training is feature quantization → histogram construction → split scoring →
  row partitioning — bandwidth- and atomic-contention-bound, **no GEMM in the hot path**.
  Tensor Cores are essentially idle. The GPU win comes from parallelism over
  (samples × features × bins) and HBM bandwidth.
- **Extra Trees is the seam:** random split thresholds delete RF's most divergence-
  and atomic-heavy step, leaving a near-pure histogram-evaluate workload — the most
  GPU-amenable classical ensemble, yet with the least dedicated GPU support.
- **Ecosystem (June 2026):** Boosting on GPU is closed (XGBoost 3.3 / LightGBM 4.6 /
  CatBoost 1.2.10). RF on GPU is closed (cuML 26.6 + cuml.accel). Inference is closed
  (FIL + Treelite). Daylight remains only in: GPU-native Extra Trees, sparse RF,
  determinism contract, generic bagging.
- **Top kill-risk:** cuML adds first-class Extra Trees and/or sparse RF before this
  ships. Mitigation: design kernels for upstreamability; pre-register benchmarks +
  kill criteria; differentiate on determinism + exact-SHAP + sklearn parity.

## Constraints

- **Tech stack — Rust core + Python API**: performance/orchestration in Rust, exposed via PyO3 + maturin as a `pip`-installable package — user's explicit choice
- **Tech stack — NVIDIA CUDA only (MVP)**: local NVIDIA GPU with CUDA toolkit installed; CUDA-first, vendor-neutral backends deferred
- **Tech risk — Rust↔CUDA maturity**: kernel-authoring path (Rust→PTX via rust-cuda/`cust`, vs `cudarc` loading hand-written CUDA C, vs alternative) is **unresolved** and is the single biggest technical risk — must be settled by research phase before kernel work
- **Platform — Windows 11**: development + benchmarking host is Windows; toolchain (Rust, CUDA, maturin, Python) must work on Windows or via documented WSL fallback
- **Performance — pre-registered crossover**: success is defined by a benchmark crossover surface (n × d) vs sklearn/oneDAL CPU + cuML RF, measured end-to-end including transfers; not an arbitrary speedup number
- **Correctness — sklearn semantics + determinism**: must match algorithmic semantics (not merely similar accuracy); deterministic mode must be bit-reproducible
- **License — Apache-2.0**: permissive, patent grant, compatible with cuML/XGBoost/Treelite; reuse algorithms from papers, never copy GPL/Snap ML source

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Narrow to Extra Trees + RF + exact SHAP; defer everything else | Both feasibility studies say narrow-or-don't; "7 fixes" blueprint was incoherent multi-subfield scope | — Pending |
| Rust core + Python (PyO3/maturin) API | User's explicit choice; Rust for safety/perf, Python for sklearn-idiom reach | — Pending |
| CUDA-first, single NVIDIA GPU, dense float32 MVP | Matches available hardware; smallest coherent path to prove the crossover | — Pending |
| No silent CPU fallback; explicit `execution_report_` | Differentiator vs cuml.accel / H2O4GPU silent fallback; audit-friendly | — Pending |
| Kernel-authoring path (Rust→PTX vs cudarc+CUDA C) left to research phase | Rust↔CUDA tooling maturity is the top technical risk; decide on evidence, not assumption | — Pending |
| Validation-first with pre-registered kill criteria | Studies stress benchmark-dependent advantage; avoid sunk cost in a broad build | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-19 after initialization*
