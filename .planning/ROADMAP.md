# Roadmap: Sylva — GPU-Native Forest Ensembles

## Overview

Sylva proves a single, falsifiable thesis: GPU-native Extra Trees / Random Forest can beat the strongest CPU baselines on large dense workloads, end-to-end from numpy, while matching scikit-learn semantics and never silently falling back. The roadmap is **validation-gated**: it front-loads the three pre-registered gates (toolchain feasibility, the crossover benchmark, the SHAP feasibility decision) so that broad build-out only happens behind proven assumptions. The path is foundation-first and dependency-strict — a debuggable Windows-native CUDA toolchain, then a pure-Rust CPU oracle that makes every later GPU result verifiable, then a bit-parity quantizer, then the simplest possible GPU hot path (a single ExtraTree with random splits), then the full forest and RandomForest, then determinism layered onto correct kernels, then the crossover gate that defines success — and only after that do the IR-only consumers (exact tree SHAP, Treelite export, packaging polish) get built. Four architecture decisions that would be near-rewrites if deferred (breadth-first level-at-a-time build, shared-memory privatized histograms, integer/deterministic accumulation, and the CPU↔GPU parity contract) are locked into their correct early phases.

> **Comparative-study fairness note (binding on every phase).** Each phase carries a **Comparative Baseline Study** that measures Sylva against an existing library / baseline implementation. These studies are NOT all "we are faster" claims — they are calibrated to what each phase can honestly measure. Foundational phases (1–4) have no full estimator yet, so their comparison is **CORRECTNESS-PARITY and/or a kernel/op-level MICROBENCHMARK** against the matching baseline, with **no end-to-end algorithm speed claim**. The first phase permitted to make a real end-to-end speed claim is **Phase 5**, and the authoritative speed study is **Phase 7's pre-registered (n×d) crossover** — the per-phase studies feed into it. ALL comparative studies MUST follow the fairness rules in `.planning/research/PITFALLS.md` (Pitfalls 1, 2, 13) and `.planning/research/SUMMARY.md`: compare **equivalent algorithms only** (ExtraTrees vs ExtraTrees, RF vs RF — never ExtraTrees vs RandomForest as if identical), time **end-to-end from numpy including H2D transfer + quantization** (never "data already on GPU" in reported numbers), separate **cold vs warm** runs, use the **strongest** baseline (sklearn `n_jobs=-1`, oneDAL/sklearnex, cuML RF labeled as a different algorithm), **pin** all hardware/driver/CUDA/package versions, take **repeated runs**, report **accuracy parity alongside speed**, and **report failures / OOM** honestly rather than hiding them.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Toolchain Spike (Gate 1)** - Prove cudarc+NVRTC builds a debuggable, packageable CUDA kernel natively on Windows (completed 2026-06-20)
- [ ] **Phase 2: CPU Oracle, Contracts & Forest IR** - Pure-Rust correctness oracle, Backend trait, SoA ForestIR, parity contract, differential tests
- [ ] **Phase 3: Feature Quantizer (CPU/GPU Bit-Parity)** - SoA BinnedMatrix with bit-identical CPU↔GPU bin assignments
- [ ] **Phase 4: Single GPU ExtraTree** - One GPU ExtraTree (privatized shared-mem histograms, breadth-first) bit-exact vs CPU oracle
- [ ] **Phase 5: Full Forest, RandomForest & sklearn Estimators** - All four estimators with full sklearn-parity API, arena memory, prefix-sum RF splits
- [ ] **Phase 6: Determinism & Honest Dispatch** - Bit-reproducible `deterministic=True`, `fallback="error"`, full `execution_report_`
- [ ] **Phase 7: Crossover Benchmark (Gate 3)** - Pre-registered end-to-end (n×d) crossover surface vs strongest CPU + cuML baselines
- [ ] **Phase 8: Exact Tree SHAP (Gate 2 + Implementation)** - SHAP feasibility decision, then exact attributions from ForestIR validated vs `shap.TreeExplainer`
- [ ] **Phase 9: Treelite Export & Packaging** - Treelite-compatible JSON export, FIL round-trip, clean-environment abi3 Windows wheel

## Phase Details

### Phase 1: Toolchain Spike (Gate 1)

**Goal**: Prove the entire kernel-authoring + packaging path works natively on Windows before any algorithm is built — resolving PROJECT.md's single biggest technical risk.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: TOOL-01, TOOL-02, TOOL-03, TOOL-04
**Success Criteria** (what must be TRUE):

  1. A throwaway hand-written CUDA C kernel compiles via cudarc 0.19.8 + NVRTC and launches on the local NVIDIA GPU, natively on Windows/MSVC with no WSL
  2. `compute-sanitizer` runs against the spike kernel and reports clean, proving the toolchain is debuggable
  3. A minimal PyO3 + maturin `abi3` wheel builds and imports in a clean Python environment on Windows
  4. Pinned, verified versions are recorded (cudarc feature flags, rust-numpy↔PyO3, CUDA toolkit) with a documented kill-criteria result: proceed / WSL-fallback / stop
  5. **KILL CRITERION:** If no kernel path (including the documented WSL fallback) yields a reproducible, debuggable, packageable result within the spike timebox — stop kernel work and reconsider the stack
  6. A kernel-launch / vector-op MICROBENCHMARK (the Comparative Baseline Study below) confirms cudarc+NVRTC launch + a trivial elementwise op is not pathologically slow versus the CuPy/raw-CUDA baseline — explicitly with **no algorithm speed claim**

**Comparative Baseline Study**

- **Baseline / library:** A trivial CuPy (or raw CUDA-C) elementwise/vector kernel on the same GPU — the simplest possible "is the toolchain alive" reference. No estimator exists at this phase.
- **Comparison type:** MICROBENCHMARK (kernel-launch overhead + a single vector op), **not** an end-to-end "we are faster" claim. Explicitly state: *no algorithm speed claim at this phase.*
- **Metric:** Per-launch overhead (µs) and elementwise throughput (GB/s) for the cudarc+NVRTC path vs the CuPy/raw-CUDA baseline; correctness of the vector op (bit-/tolerance-exact result).
- **Dataset / shape:** A fixed synthetic vector (e.g. 1e7-element float32 array) and a fixed launch-count loop; pinned GPU/driver/CUDA versions.
- **Pass bar:** cudarc+NVRTC launch overhead within a small constant factor of the CuPy/raw baseline (sanity threshold, e.g. ≤ ~2–3× per-launch overhead) and vector-op result correct — confirming the toolchain is not pathologically slow. This is a feasibility sanity check, not a speed gate.

**Plans**: 3/3 plans complete
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Toolchain prerequisites (install Rust ≥1.83, verify MSVC/NVRTC/maturin) + persisted Cargo workspace + sylva-cuda crate + maturin abi3 pyproject + CI + VERSIONS.md template (D-04)
- [x] 01-02-PLAN.md — NVRTC compile+launch of vector_add (TOOL-01) and a representative privatized histogram kernel, compute-sanitizer clean (TOOL-02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-03-PLAN.md — abi3 wheel (dynamic-loading) builds + imports in a clean venv (TOOL-03), fairness-encoded microbench vs CuPy/raw baseline, finalize VERSIONS.md kill-decision (TOOL-04)

**UI hint**: no

### Phase 2: CPU Oracle, Contracts & Forest IR

**Goal**: Stand up the device-neutral contracts and a trusted pure-Rust CPU backend that trains and predicts ET + RF correctly — the correctness oracle that makes every later GPU result verifiable and enables GPU-less cloud CI.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: ENG-01, ENG-02, ENG-03, ENG-04, ENG-05, ENG-06, EST-07
**Success Criteria** (what must be TRUE):

  1. A device-neutral `trait Backend` (quantize / build_histograms / eval_splits / partition / predict) exists with no CUDA types crossing the trait boundary, and a SoA `ForestIR` is the single shared representation written by training and read read-only by inference, SHAP, and export
  2. A pure-Rust `CpuBackend` (ndarray + rayon) trains and predicts ExtraTrees + RandomForest correctly and serves as the differential-test oracle and `device="cpu"` / small-data path
  3. The parity contract is documented (Sylva's own bit-identical CPU↔GPU RNG per seed + distributional equivalence to scikit-learn, NOT bit-identical replay of sklearn's serial PRNG), and a stateless Philox-4×32-10 RNG is implemented identically in Rust, keyed by (seed, tree, node, feature, draw)
  4. NaN / missing-value routing policy is defined and implemented consistently, with NaN fixtures in the test suite
  5. Differential tests vs scikit-learn (accuracy/distribution within tolerance) and property-based invariants pass — child rows partition parent, leaf probabilities valid, seed determinism, serialization round-trip
  6. The Comparative Baseline Study confirms the CpuBackend reaches **accuracy/distribution PARITY** with scikit-learn ExtraTrees/RandomForest (like-for-like algorithm) within the documented tolerance on the named dataset
  7. CpuBackend CPU training time vs the matching sklearn estimator is **reported as informational only** (not gated) on the same dataset, with cold/warm separated and versions pinned

**Comparative Baseline Study**

- **Baseline / library:** scikit-learn `ExtraTreesClassifier`/`RandomForestClassifier` (and the regressors) on CPU — the canonical like-for-like algorithm. ET compared to ET, RF compared to RF (never crossed).
- **Comparison type:** Exact accuracy/distribution **PARITY (primary)** + CPU-vs-CPU training-time comparison (informational only). No GPU and no end-to-end-from-numpy speed claim yet.
- **Metric:** (a) accuracy / predicted-probability agreement and split-distribution equivalence (e.g. accuracy within CI + KS test on split structure) vs sklearn; (b) wall-clock CPU `fit` time of CpuBackend vs sklearn (reported, not gated).
- **Dataset / shape:** A small public dataset — `make_classification` (e.g. 20k×50) and/or a Covertype subset — fixed seed, identical hyperparameters across both implementations.
- **Pass bar:** **Parity within stated tolerance is the gate** (accuracy within CI, split-distribution equivalent). Training-time difference is **reported, not gated** — speed is informational at this phase.

**Plans**: 1/5 plans executed
Plans:
**Wave 1**

- [x] 02-01-PLAN.md — sylva-core crate, device-neutral Backend trait, SoA ForestIR, Philox-4×32-10 + verified KAT vectors (ENG-01, ENG-02, ENG-06)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 02-02-PLAN.md — CpuBackend ExtraTrees slice: criterion, random-threshold splitter, recursive builder, NaN-safe predict (ENG-03, ENG-05)

**Wave 3** *(blocked on Wave 2 completion)*

- [ ] 02-03-PLAN.md — CpuBackend RandomForest slice: bootstrap + exact best-split, completing ET/RF × clf/reg (ENG-03)

**Wave 4** *(blocked on Wave 3 completion)*

- [ ] 02-04-PLAN.md — Property invariants + byte-determinism + parity contract doc + split_statistics extractor (EST-07, ENG-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [ ] 02-05-PLAN.md — Test-only PyO3 seam + sklearn calibration + distributional parity gate (ENG-04, EST-07)

**UI hint**: no

### Phase 3: Feature Quantizer (CPU/GPU Bit-Parity)

**Goal**: Build the quantizer that every GPU kernel reads from, and prove its bins are bit-identical across CPU and GPU before any histogram is ever built on them.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: QUANT-01, QUANT-02
**Success Criteria** (what must be TRUE):

  1. A feature quantizer produces a SoA `BinnedMatrix` (uint8/uint16) via per-feature quantile bins, on both CPU and GPU
  2. CPU and GPU quantizers produce bit-identical bin assignments on a fixed seed, enforced by a parity test in CI
  3. `execution_report_` hooks record dtype/contiguity handling and bytes transferred for the quantize step
  4. The Comparative Baseline Study confirms Sylva's binning is **correct vs an established binning baseline** (sklearn HistGradientBoosting quantile binning / numpy quantile binning) on the named dataset — same bin edges / assignments within the documented tolerance
  5. A **quantize-throughput MICROBENCHMARK** (rows/s, or bins/s) is reported for Sylva's GPU/CPU quantizer vs the baseline binning, explicitly as an op-level number with **no end-to-end algorithm speed claim**

**Comparative Baseline Study**

- **Baseline / library:** scikit-learn `HistGradientBoosting` binning (`_BinMapper`-style quantile binning) and/or `numpy.quantile` per-feature binning — the standard reference for quantile bin construction.
- **Comparison type:** **Binning-correctness PARITY (primary)** + quantize-throughput MICROBENCHMARK (op-level). No estimator-level speed claim.
- **Metric:** (a) bin-edge / bin-assignment agreement vs the baseline binning within tolerance (ties handled per documented policy); (b) quantize throughput (rows/s) of Sylva CPU and GPU quantizers vs the baseline (informational).
- **Dataset / shape:** A medium dense matrix — e.g. `make_classification` 100k×100 float32 with a fixed bin count (128/256) — pinned versions, fixed seed.
- **Pass bar:** **Binning correctness within tolerance is the gate.** Throughput is reported as an op-level microbench, **not** an end-to-end speed claim.

**Plans**: TBD
**UI hint**: no

### Phase 4: Single GPU ExtraTree

**Goal**: Train one GPU ExtraTree (classifier + regressor) that matches the CPU oracle bit-exactly — validating the kernel + toolchain in the simplest possible GPU hot path, with the breadth-first and shared-memory-privatization architecture decisions locked in.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: GPU-01, GPU-02
**Success Criteria** (what must be TRUE):

  1. A single GPU ExtraTree (classifier + regressor) trains with a privatized shared-memory histogram kernel, a fused random-candidate split kernel, and a scatter-partition kernel, built breadth-first / level-at-a-time (not depth-first per-node)
  2. The GPU ExtraTree matches the CPU oracle bit-exactly on a fixed seed
  3. `compute-sanitizer` (racecheck + memcheck) is clean against every kernel in the path
  4. Histograms are shared-memory-resident with privatized accumulation as the baseline — no global float-atomic accumulation on the hot path
  5. The Comparative Baseline Study confirms **correctness PARITY** of the single GPU ExtraTree against both the CPU oracle (bit-exact, same seed) and a sklearn single ExtraTree (distributional) on the named dataset
  6. Single-tree GPU-vs-CPU training time (transfer-inclusive) is **reported as informational only** on the named medium dataset, with the explicit caveat that this is a single-tree timing, **not** an end-to-end forest speed claim

**Comparative Baseline Study**

- **Baseline / library:** (a) the Phase 2 CPU oracle — **bit-exact** target for the same seed; (b) a sklearn single ExtraTree (`ExtraTreeClassifier`/`ExtraTreeRegressor`, or `ExtraTrees*` with `n_estimators=1`) — distributional like-for-like reference.
- **Comparison type:** **Correctness PARITY (primary)** + single-tree GPU-vs-CPU timing (informational, transfer-inclusive). Not a forest-level or end-to-end "we are faster" claim.
- **Metric:** (a) bit-exact match of GPU tree vs CPU oracle on a fixed seed; (b) distributional equivalence vs sklearn single tree (accuracy/split structure); (c) wall-clock single-tree `fit` time GPU (including H2D) vs CPU (reported).
- **Dataset / shape:** A medium dense dataset — e.g. Covertype subset or `make_classification` 200k×50 float32 — fixed seed, fixed `max_depth`, identical hyperparameters.
- **Pass bar:** **GPU == CPU oracle bit-exact (gate)** and distributional parity vs sklearn. Single-tree timing is **reported, not gated**, and explicitly labeled as a transfer-inclusive single-tree number, not a forest speed claim.

**Plans**: TBD
**UI hint**: no

### Phase 5: Full Forest, RandomForest & sklearn Estimators

**Goal**: Scale the single tree into the full forest and add RandomForest, exposing all four estimators through a strict sklearn-parity API backed by the arena memory model.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: GPU-03, GPU-04, GPU-05, GPU-06, EST-01, EST-02, EST-03, EST-04, EST-05, EST-06
**Success Criteria** (what must be TRUE):

  1. Full forest training works via a breadth-first NodeScheduler (frontier waves), per-tree RNG schedule, and sibling-histogram subtraction; RandomForest adds best-split evaluation via inclusive prefix-sum (scan) + argmax over bins, sharing the histogram engine with ExtraTrees
  2. A stream-ordered fit-scoped arena (cudaMallocAsync pool) reuses histogram/row-index buffers across waves with no per-node allocation, and small/deep nodes cut over to a CPU leaf-finishing path
  3. `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomForestClassifier`, `RandomForestRegressor` expose a sklearn drop-in API with the full estimator contract (`fit`/`predict`/`predict_proba`/`predict_log_proba`/`score`, `get_params`/`set_params`, clone-able, no logic in `__init__`)
  4. Core constructor params (`n_estimators`, `max_depth`, `max_features`, `min_samples_split/leaf`, `bootstrap`, `max_samples`, `criterion`, `random_state`, `n_jobs`, `class_weight`, `sample_weight`) and fitted attributes (`classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_`, `estimators_`, `feature_importances_` real MDI) are correct, with `sample_weight` supported end-to-end via a weighted-histogram kernel
  5. `sklearn.utils.estimator_checks.check_estimator` passes in CI with any intentional exceptions explicitly documented
  6. The Comparative Baseline Study reports **end-to-end-from-numpy training time** of Sylva ET/RF vs sklearn ET/RF (`n_jobs=-1`), cuML RF (labeled different algorithm), and XGBoost rf-mode on at least one large dense dataset, **with accuracy parity reported alongside** — this is the first phase permitted to make a real speed comparison
  7. Like-for-like is enforced in the study (Sylva ET vs sklearn ET; Sylva RF vs sklearn RF / cuML RF; never ET-vs-RF as if identical), cold and warm separated, transfers included, versions pinned, and any OOM/failure reported honestly

**Comparative Baseline Study**

- **Baseline / library:** scikit-learn `ExtraTrees`/`RandomForest` (`n_jobs=-1`), cuML `RandomForest` (**labeled as a different algorithm** — RF, not ET), and XGBoost in `rf`/`num_parallel_tree` random-forest mode. ET compared to ET, RF compared to RF.
- **Comparison type:** **End-to-end-from-numpy training-time comparison** + accuracy parity. This is the **first phase that may make a real speed claim** (a full estimator now exists).
- **Metric:** Wall-clock `fit(X, y)` from a host numpy array including dtype coercion + H2D transfer + quantization (cold and warm, repeated runs), reported next to test-set accuracy / probability calibration for each implementation.
- **Dataset / shape:** At least one large dense dataset — e.g. Higgs subset (~1M+ rows) and/or full Covertype (~581k×54) float32 — identical hyperparameters across implementations, pinned hardware/driver/CUDA/package versions.
- **Pass bar:** Accuracy **parity** with the like-for-like sklearn baseline (gate); end-to-end speed **reported with the crossover caveat** — this study establishes the per-phase data point that feeds the authoritative Phase 7 crossover, and any region where Sylva loses is reported, not hidden. OOM/failures reported honestly.

**Plans**: TBD
**UI hint**: no

### Phase 6: Determinism & Honest Dispatch

**Goal**: Layer bit-reproducible determinism onto the now-correct forest and enforce the no-silent-fallback dispatch contract that is a core differentiator.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: DET-01, DET-02, DET-03, DET-04
**Success Criteria** (what must be TRUE):

  1. `deterministic=True` yields byte-identical models across two same-seed runs (verified by exact binary comparison, not `allclose`) via integer/fixed-point histogram accumulation + canonical reduction order + fixed tie-breaking
  2. The performance cost of deterministic mode is measured and reported
  3. `device="auto" | "cuda" | "cpu"` dispatch with `fallback="error"` is enforced — no silent CPU fallback; unsupported configs raise
  4. `execution_report_` reports the selected backend + reason, every input conversion (dtype/layout/H2D), bytes transferred, and fallback status
  5. The Comparative Baseline Study quantifies the **determinism cost overhead (%)** of Sylva's `deterministic=True` vs Sylva's own non-deterministic path (the like-for-like internal baseline), on the named dataset
  6. The study **qualitatively confirms the determinism gap** in cuML RF / LightGBM-GPU (i.e. that their GPU training is not bit-reproducible under the same protocol), establishing the differentiator without overclaiming a speed win

**Comparative Baseline Study**

- **Baseline / library:** **Sylva's own non-deterministic training path** is the primary internal baseline (apples-to-apples: same kernels, determinism toggled). cuML RF and LightGBM-GPU serve as external references for the determinism-gap claim only.
- **Comparison type:** Determinism-cost overhead measurement (Sylva-vs-Sylva) + a qualitative external determinism-gap confirmation. **Not** a cross-library speed claim.
- **Metric:** (a) throughput / `fit`-time overhead (%) of `deterministic=True` vs Sylva non-deterministic on identical inputs and seed; (b) byte-identical reproducibility verified across two same-seed deterministic runs; (c) qualitative: do cuML RF / LightGBM-GPU produce byte-identical models across same-seed runs under the same protocol? (expected: no).
- **Dataset / shape:** The Phase 5 large dense dataset (e.g. Covertype / Higgs subset) at fixed hyperparameters, repeated runs, pinned versions.
- **Pass bar:** Deterministic mode is **byte-reproducible (gate)** and its overhead is measured and reported (target consistent with research: ~95–98% throughput retention, i.e. small overhead — reported, not a hard gate). The external determinism gap is documented qualitatively.

**Plans**: TBD
**UI hint**: no

### Phase 7: Crossover Benchmark (Gate 3)

**Goal**: Run the pre-registered, reproducible benchmark that defines success — measuring end-to-end training from numpy against the strongest CPU and cuML baselines and publishing where GPU wins and where it does not.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: BENCH-01, BENCH-02, BENCH-03
**Success Criteria** (what must be TRUE):

  1. A scripted, reproducible benchmark measures end-to-end training time from numpy (including H2D transfer + quantization), cold and warm, with pinned hardware/driver/CUDA/package versions
  2. Baselines include scikit-learn ET/RF (`n_jobs=-1`), oneDAL/sklearnex, and cuML RF (labeled as a different algorithm), with accuracy parity reported alongside speed (like-for-like algorithms only)
  3. A published (n×d) crossover surface identifies where GPU beats the strongest CPU baseline and where it does not
  4. **KILL CRITERION:** If no region of the (n×d) surface shows end-to-end GPU Extra Trees beating the strongest CPU baseline — the core premise is false; pivot to the SHAP/determinism layer (which can ride on cuML) rather than continuing broad build-out
  5. The Comparative Baseline Study **is** the authoritative pre-registered (n×d) crossover — every per-phase study (Phases 1–6, 8–9) feeds into and is reconciled against this surface, and any conflicts between a per-phase data point and the crossover are explained
  6. The crossover protocol is verified to satisfy all fairness rules (equivalent algorithms only, end-to-end-from-numpy with transfers, cold vs warm separated, strongest baselines, pinned versions, repeated runs, accuracy beside speed, failures/OOM reported) before the kill criterion is evaluated

**Comparative Baseline Study**

- **Baseline / library:** scikit-learn ET/RF (`n_jobs=-1`), oneDAL/sklearnex, cuML RF (labeled different algorithm) — the strongest available CPU and GPU baselines. This is the **authoritative study**; the per-phase studies feed it.
- **Comparison type:** Pre-registered end-to-end (n×d) **crossover surface** — the definition of project success. Real speed claims are made here, fully fenced by the fairness protocol.
- **Metric:** End-to-end `fit(X, y)` wall-clock from host numpy (dtype coercion + H2D + quantization included), cold and warm, repeated runs, with test accuracy/parity reported alongside every speed cell, across a 2D grid of n (samples) × d (features).
- **Dataset / shape:** A pre-registered grid spanning small→large n and low→high d (synthetic `make_classification` grid plus real anchors: Covertype, Higgs subset), pinned hardware/driver/CUDA/package versions, fixed hyperparameters identical across implementations.
- **Pass bar:** A published surface that **honestly identifies the win/loss boundary**. The KILL CRITERION applies: if **no** region shows end-to-end GPU ExtraTrees beating the strongest CPU baseline, the premise is false → pivot. There is no "must win everywhere" bar — honesty about where CPU wins is itself a pass.

**Plans**: TBD
**UI hint**: no

### Phase 8: Exact Tree SHAP (Gate 2 + Implementation)

**Goal**: Resolve the SHAP feasibility decision (attributions vs interactions, license-clean path) before building, then implement exact per-feature SHAP attributions from the ForestIR and validate them against the reference explainer.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: SHAP-01, SHAP-02, SHAP-03
**Success Criteria** (what must be TRUE):

  1. A SHAP feasibility spike (Gate 2) confirms scope = exact **attributions** and verifies the GPUTreeSHAP licensing/integration path before any implementation begins
  2. **KILL CRITERION:** If WoodelfHD is GPL/closed and interactions were the target — descope to GPUTreeSHAP attributions (Apache-2.0); do not let exact-SHAP block or balloon the MVP
  3. `sylva-shap` computes exact per-feature SHAP attributions from `ForestIR` (consuming the IR only, never a backend), CPU-first then GPU
  4. `.shap_values()` results validate against `shap.TreeExplainer` within float tolerance
  5. The Comparative Baseline Study confirms **attribution agreement** between `sylva-shap` and `shap.TreeExplainer` within float tolerance (correctness gate) on the named deep-tree dataset
  6. A SHAP-compute **speedup** of the GPU path vs `shap.TreeExplainer` (CPU) and vs GPUTreeSHAP is reported on the deep-tree dataset, with accuracy/agreement reported alongside and the same fairness protocol applied

**Comparative Baseline Study**

- **Baseline / library:** `shap.TreeExplainer` (Lundberg polynomial TreeSHAP) as the **correctness oracle**; `rapidsai/gputreeshap` (Apache-2.0) as the **GPU speed reference** for attributions.
- **Comparison type:** Attribution **correctness PARITY (primary)** + SHAP-compute speedup (GPU vs reference). Like-for-like = exact attributions vs exact attributions (not interactions).
- **Metric:** (a) per-feature SHAP attribution agreement vs `shap.TreeExplainer` within float tolerance (max abs / relative error); (b) wall-clock SHAP compute time of `sylva-shap` GPU vs `shap.TreeExplainer` CPU and vs GPUTreeSHAP, repeated runs.
- **Dataset / shape:** A deep-tree scenario — a forest trained on a medium dense dataset (e.g. Covertype subset) at high `max_depth` so the explanation cost is meaningful, plus a fixed explain-set of rows.
- **Pass bar:** **Attribution agreement within float tolerance is the gate.** Speedup vs the references is reported (with agreement beside it); a faster-but-wrong result is not a win.

**Plans**: TBD
**UI hint**: no

### Phase 9: Treelite Export & Packaging

**Goal**: Make trained models portable and the library installable — serialize the ForestIR to a Treelite-compatible representation, prove a FIL round-trip, and validate the distributable wheel in a clean environment.
**Mode:** mvp
**Depends on**: Phase 7 (IR stable after Phase 2; independent of Phase 8 — can run in parallel)
**Requirements**: EXP-01, EXP-02, EXP-03
**Success Criteria** (what must be TRUE):

  1. `sylva-export` serializes `ForestIR` to a Treelite 4.x `import_from_json()`-compatible JSON
  2. An exported model round-trips through Treelite/FIL and produces matching predictions (CI test)
  3. The `abi3` Windows wheel is validated in a fresh environment with CUDA driver dynamic-loading, with a documented install path
  4. The Comparative Baseline Study confirms **prediction round-trip PARITY** — Sylva predictions match the same model after export→import through Treelite/FIL (and TL2cgen) within tolerance on the named dataset
  5. **Inference throughput** of the exported model via FIL / TL2cgen-compiled path is reported vs Sylva's native predict (and vs sklearn predict) on the named dataset, with the same fairness protocol applied

**Comparative Baseline Study**

- **Baseline / library:** Treelite 4.x + FIL (GPU inference) and TL2cgen (compiled CPU inference) as the export/inference targets; Sylva's native `predict` and sklearn `predict` as reference predictors.
- **Comparison type:** Prediction **round-trip PARITY (primary)** + inference-throughput comparison.
- **Metric:** (a) prediction agreement of exported-then-reloaded model (Treelite/FIL and TL2cgen) vs Sylva native prediction within tolerance; (b) inference throughput (rows/s) of FIL / TL2cgen vs Sylva native predict vs sklearn predict, repeated runs.
- **Dataset / shape:** A held-out test split of a medium/large dense dataset (e.g. Covertype) on a model trained in Phase 5, fixed batch sizes, pinned versions.
- **Pass bar:** **Round-trip prediction parity within tolerance is the gate** (export is lossless). Inference-throughput numbers are reported with the fairness protocol; faster inference is a reported benefit, not the gate.

**Plans**: TBD
**UI hint**: no

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
(Phase 8 and Phase 9 are both IR-only consumers gated behind Phase 7 and may proceed in parallel.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Toolchain Spike (Gate 1) | 3/3 | Complete    | 2026-06-20 |
| 2. CPU Oracle, Contracts & Forest IR | 1/5 | In Progress|  |
| 3. Feature Quantizer (CPU/GPU Bit-Parity) | 0/TBD | Not started | - |
| 4. Single GPU ExtraTree | 0/TBD | Not started | - |
| 5. Full Forest, RandomForest & sklearn Estimators | 0/TBD | Not started | - |
| 6. Determinism & Honest Dispatch | 0/TBD | Not started | - |
| 7. Crossover Benchmark (Gate 3) | 0/TBD | Not started | - |
| 8. Exact Tree SHAP (Gate 2 + Implementation) | 0/TBD | Not started | - |
| 9. Treelite Export & Packaging | 0/TBD | Not started | - |
