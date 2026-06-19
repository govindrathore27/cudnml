# Project Research Summary

**Project:** Sylva — GPU-Native Forest Ensembles (Rust core, CUDA, Python API)
**Domain:** GPU-native tree-ensemble ML library (Extra Trees + Random Forest + exact tree SHAP)
**Researched:** 2026-06-19
**Confidence:** MEDIUM-HIGH overall (stack binding/interop: HIGH; kernel architecture: HIGH; SHAP exact algo: MEDIUM; competitive landscape: MEDIUM)

---

## Executive Summary

Sylva is a GPU-native scikit-learn-compatible library for Extra Trees and Random Forest (classifier + regressor) with exact high-depth tree SHAP. The central technical insight validated by research is that GPU tree training is **bandwidth- and atomic-contention-bound, not GEMM-bound** — no Tensor Cores in the hot path; the win comes from parallelism over (samples × features × bins) and HBM bandwidth. This determines every architecture choice: privatized shared-memory histograms, breadth-first level-at-a-time tree construction, scatter-partition with double-buffered row indices, and a stream-ordered fit-scoped arena. The recommended kernel path is **cudarc 0.19.8 + hand-written CUDA C compiled at runtime via NVRTC** — this resolves the project's single biggest stated technical risk: it builds natively on Windows/MSVC, needs no `nvcc` at build time, gives full access to warp intrinsics and shared memory, and works with the mature CUDA tooling (compute-sanitizer, Nsight). Rust-CUDA/`cust` and `cc`-driven `nvcc` are both ruled out on Windows. CubeCL is the deliberate deferred portability bet (Plan B behind a kernel trait).

The market wedge is real and currently open: cuML 26.6 has no Extra Trees and has known gaps in `feature_importances_`, `oob_score`, and guaranteed determinism. The durable differentiators — deterministic GPU training (bit-reproducible), exact high-depth tree SHAP (WoodelfHD, arXiv 2604.10569, verified real), non-silent device dispatch with `execution_report_`, and strict sklearn parity — are each individually unlikely to be matched by cuML on the same timeline. However, the top kill-risk is cuML shipping Extra Trees before Sylva proves its crossover benchmark; mitigation is designing for upstreamability and leading marketing with determinism + SHAP + honest dispatch rather than "GPU Extra Trees exists."

The recommended development discipline is **validation-gated with three mandatory pre-registered spikes before broad kernel build-out**: (1) a toolchain spike proving `cudarc + NVRTC` builds and runs on Windows with working compute-sanitizer, (2) a SHAP feasibility spike deciding attributions-vs-interactions and confirming WoodelfHD port licensing, and (3) a pre-registered end-to-end crossover benchmark measured from numpy, cold-start, against sklearn `n_jobs=-1` + oneDAL + cuML RF. If the crossover spike fails, the project pivots — not doubles down. The key sequencing insight is **ExtraTrees before RandomForest**: random split thresholds eliminate the best-split argmax/scan/atomics step, making ET the simplest possible GPU hot path to validate the toolchain, exactly as PROJECT.md identifies it as "the seam."

---

## Key Findings

### Recommended Stack

The kernel-authoring risk (explicitly flagged in PROJECT.md as "the single biggest technical risk") is resolved: **cudarc 0.19.8 + hand-written CUDA C via NVRTC** is the correct path. NVRTC compiles `.cu` source strings to PTX at runtime, sidestepping the `cc-rs`/MSVC incompatibility that makes AOT `nvcc` in `build.rs` broken on native Windows. cudarc provides the driver API, NVRTC, cuBLAS, and cuRAND bindings and was released 2026-06-19 (actively maintained, 300k+ downloads/version). The Python binding layer is **PyO3 0.29 + maturin 1.14.1 + rust-numpy ≈0.25.x** — the canonical, stable combination for an `abi3` pip-installable wheel on Windows. Determinism is achieved via **hand-rolled Philox-4×32-10** sharing the same `(key=seed, counter=(tree, node, feature, draw))` scheme between Rust and the CUDA kernel — giving bit-identical CPU↔GPU RNG streams. CubeCL 0.10.0 is named the explicit deferred fallback: hide all kernels behind a Rust `trait Backend` from day one so a CubeCL backend is purely additive in Milestone 2.

**Core technologies:**

| Technology | Version | Purpose |
|------------|---------|---------|
| Rust (stable) | 1.83+ | Performance core, orchestration, CPU oracle — PyO3 MSRV floor |
| cudarc | 0.19.8 | CUDA driver API + NVRTC + cuBLAS/cuRAND — native MSVC Windows builds |
| Hand-written CUDA C (via NVRTC) | CUDA Toolkit 12.6+ | Histogram / split-score / scatter-partition hot path |
| PyO3 | 0.29.0 | Rust↔Python FFI, abi3 stable ABI, CPython 3.7–3.14 |
| maturin | 1.14.1 | pip-installable wheel build, Windows-native |
| rust-numpy | ≈0.25.x | Zero-copy host numpy↔ndarray interop (must version-lock to PyO3 0.29) |
| ndarray + rayon | 0.16.x / 1.x | CPU reference backend (oracle + small-data path) |
| Philox-4×32-10 (hand-rolled) | — | Stateless counter-based RNG; bit-identical CPU↔GPU; enables `deterministic=True` |
| serde_json | 1.x | Treelite 4.x compatible JSON export for FIL serving |
| proptest / approx | 1.x / 0.5.x | Property-based invariants + float tolerance in differential tests |

**Critical version constraints:**
- rust-numpy version must exactly track PyO3 minor — verify against rust-numpy changelog before pinning.
- cudarc feature flag (`cuda-12060` etc.) must match installed CUDA toolkit; use `dynamic-loading` feature for distributable wheels.
- CubeCL 0.10 is alpha with breaking changes between minor versions — Milestone 2 only.
- Treelite 4.x JSON schema field names must be verified against live docs during the export phase (MEDIUM confidence only).

### Expected Features

Research confirms the sklearn parity surface, competitive gaps, and SHAP method. The WOODELF-HD method (arXiv 2604.10569, 2026) is verified real — authored by Wettenstein, Nadel, Boker at Reichman University + Technion. Key complexity correction: **O(mTL + nTLD + TL·2^D·D²)** (not `2^D·D` as originally stated); the 3^D → 2^D improvement comes from a Strassen-like block scheme and UFDP (Unique-Feature Decision Pattern) path compression. The WoodelfHD authors already have a GPU implementation; Sylva's differentiator is **port + integrate on Apache-2.0**, not invent. License diligence is mandatory before any code reuse.

**Must have (table stakes for v1):**
- `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomForestClassifier`, `RandomForestRegressor` — dense float32, single GPU, sklearn drop-in API
- Full sklearn estimator contract: `fit/predict/predict_proba/predict_log_proba/score`, `get_params/set_params`, clone-able, `BaseEstimator` semantics (no logic in `__init__`)
- Core constructor params with correct defaults: `n_estimators`, `max_depth`, `max_features`, `min_samples_split/leaf`, `bootstrap`, `max_samples`, `criterion` (gini/entropy/squared_error MVP), `random_state`, `n_jobs`, `class_weight`
- Core fitted attrs: `classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_`, `estimators_`, `feature_importances_` (real MDI — direct cuML gap)
- `deterministic=True` — bit-reproducible training; designed in from day one
- `device="auto"|"cuda"|"cpu"` with `fallback="error"` and `execution_report_` — no silent CPU fallback
- CPU reference backend — correctness oracle + dispatch target
- `check_estimator` CI gate with documented expected-failures
- Differential tests vs scikit-learn + property-based invariants

**Should have (competitive differentiators, v1.x after validation):**
- Exact high-depth tree SHAP (WoodelfHD) — gated: port + upstream license diligence first; GPUTreeSHAP (Apache-2.0) is the safe fallback for attributions
- Treelite-compatible model export → FIL/Triton serving
- `sample_weight` support (weighted histogram kernel)
- `oob_score` / `oob_score_` — direct cuML gap (issue #3361)
- Zero-copy GPU input via `__cuda_array_interface__` / DLPack

**Defer (v2+):**
- Sparse/CSR input — near-rewrite; changes histogram, partitioning, missing-vs-zero semantics
- Multi-GPU / multi-node
- `ccp_alpha`, `max_leaf_nodes` (best-first growth), `absolute_error`/`poisson` criteria
- Variance-based redundant-tree pruning — **method provenance UNVERIFIED in literature**; keep P3/optional; requires its own empirical validation study before advertising
- `warm_start=True` — accept param, raise error if True

**SHAP-specific decision required before scheduling:** Standard/interventional TreeSHAP is already polynomial O(T·L·D²). The O(3^D) blowup applies only to SHAP *interaction* values. If the target is attributions: GPUTreeSHAP (Apache-2.0) is the safe baseline. If interactions at high depth: WoodelfHD is the ambitious target but requires license-clean re-implementation. This must be resolved in the SHAP feasibility spike (Phase 2).

### Architecture Approach

The system is a **five-layer dispatch stack**: Python sklearn-parity API (L5) → PyO3 binding seam (L4) → Rust device-agnostic orchestration (L3) → Backend trait seam (L2) → CUDA kernels and GPU memory (L1/L0). The load-bearing design decision is the **`trait Backend`** at L2 — every device operation is expressed through device-neutral method signatures; CUDA types never cross the trait boundary. The CPU backend (`sylva-cpu`) is built first and carries most CI correctness weight.

The trained forest is a **SoA ForestIR** that training writes once and inference, SHAP, and export read without mutation. The `sylva-shap` and `sylva-export` crates depend only on the IR and can be developed in parallel with the GPU training path. Tree construction is **breadth-first, level-at-a-time** (retrofitting this from depth-first is a near-rewrite). Memory is managed via a **stream-ordered fit-scoped arena** (`cudaMallocAsync` pool) with histogram buffers reused across node waves.

**Major components:**
1. `sylva-core` — ForestIR (SoA), TableView, NodeScheduler (BF frontier), Philox RNG schedule, arena, ExecutionReport, estimator state machine
2. `sylva-backend` — `trait Backend` seam (quantize, build_histograms, eval_splits, partition, predict)
3. `sylva-cpu` — CpuBackend: pure Rust + rayon oracle; exact sklearn semantics; CI correctness workhorse
4. `sylva-cuda` — CudaBackend: cudarc host code + hand-written CUDA C kernels (histogram.cu, split_rf.cu, split_extratrees.cu, partition.cu, quantize.cu, predict.cu)
5. `sylva-shap` — exact tree SHAP; consumes ForestIR only; IR-consumer, never touches a backend
6. `sylva-export` — ForestIR → Treelite-compatible JSON; IR-consumer
7. Python layer (`python/sylva/`) — thin sklearn-parity wrappers over the PyO3 module

**Four architecture decisions that are near-rewrites if deferred:**
1. Breadth-first level-at-a-time tree construction (not depth-first per-node)
2. Shared-memory privatized histograms (not global-atomic accumulation)
3. Deterministic non-float (integer/fixed-point) accumulation + canonical reduction order for `deterministic=True`
4. The parity-contract definition: Sylva's own documented RNG (bit-identical CPU↔GPU per seed), distributional equivalence to sklearn — NOT bit-identical reproduction of sklearn's serial PRNG (infeasible on parallel GPU)

### Critical Pitfalls

1. **H2D transfer swamps the kernel win** — End-to-end `fit(numpy_array)` including PCIe transfer can cost more than the histogram pass for small/medium data. Clock the public API call cold and warm. Transfer the matrix once and keep it device-resident for the entire fit. Surface bytes-transferred in `execution_report_`.

2. **Shrinking-node underutilization + atomic contention** — Depth-first per-node launches collapse to near-serial at deep levels; naive global `atomicAdd` histograms contend on hot bins. Mitigation: breadth-first level-at-a-time build from day one; privatized shared-memory histograms as the baseline, not an optimization. Both are architecture decisions, not retrofits.

3. **Nondeterministic float atomics break `deterministic=True`** — Float `atomicAdd` is ordering-nondeterministic. Test determinism with byte-exact comparison, not `np.allclose`. Mitigation: integer/fixed-point accumulation + warp-synchronous binary-tree reduction on the deterministic path.

4. **SHAP is a gated spike, not a roadmap commitment** — Standard TreeSHAP is already polynomial; WoodelfHD is verified real but requires license-clean re-implementation; GPUTreeSHAP (Apache-2.0) is the safe fallback. The spike must answer attributions-vs-interactions before any SHAP implementation is scheduled.

5. **cuML ships Extra Trees first** — The top kill-risk. Mitigation: lead with determinism + exact SHAP + honest dispatch; design for Apache-2.0 upstreamability; pre-register kill criteria and monitor every cuML release (~2 month cadence).

---

## Implications for Roadmap

The overriding discipline: **three gating spikes before broad kernel build-out**. Each spike has pre-registered kill criteria; failing triggers a pivot, not more build.

### Phase 1: Toolchain Spike (Gate 1)
**Rationale:** All subsequent phases assume the kernel path works. Must be first.
**Delivers:** Proof that cudarc 0.19.8 + NVRTC + MSVC builds a CUDA kernel natively on Windows; compute-sanitizer introspects it; a minimal wheel runs on a clean Windows environment.
**Kill criteria:** If no kernel path (including WSL fallback) yields a reproducible, debuggable, packageable result within timebox — stop kernel work, reconsider stack.
**Research flag:** Standard patterns — STACK.md provides the exact approach. No additional research-phase needed.

### Phase 2: SHAP Feasibility Spike (Gate 2)
**Rationale:** Attributions vs interactions changes scope dramatically. Must be resolved before scheduling any SHAP work.
**Delivers:** Written decision — attributions (use GPUTreeSHAP as baseline, WoodelfHD as upgrade pending license) vs interactions at high depth (WoodelfHD ambitious path, set kill date). Upstream license verified.
**Kill criteria:** If WoodelfHD is GPL/closed and interactions are the target → descope to GPUTreeSHAP for attributions.
**Research flag:** Needs fresh license check on WoodelfHD authors' repository.

### Phase 3: CPU Reference Backend + Contracts (Build B0–B1)
**Rationale:** Without a trusted CPU oracle, GPU correctness is unverifiable. The ForestIR and Backend trait are the shared contracts everything plugs into.
**Delivers:** `trait Backend`; `ForestIR` (SoA); `CpuBackend` (exact sklearn semantics); full fit/predict on CPU; `check_estimator` CI gate; differential tests vs sklearn; property-based invariants; NaN policy defined; parity contract documented.
**Pitfalls avoided:** Pitfall 6 (parity contract), Pitfall 7 (NaN routing), Pitfall 11 (GPU-less CI), Pitfall 16 (CPU carries CI weight)

### Phase 4: Feature Quantizer (Build B2)
**Rationale:** Every GPU kernel reads bins. Bins must be bit-equal CPU↔GPU before building histograms on them.
**Delivers:** BinnedMatrix (SoA uint8/uint16); CPU quantizer; GPU quantize kernel (NVRTC); bit-parity test; `execution_report_` hooks for dtype/contiguity and transfer reporting.

### Phase 5: Single-Tree GPU Path — ExtraTrees First (Build B3)
**Rationale:** ET before RF: random split thresholds eliminate best-split argmax/scan/atomics. Validates kernel+toolchain in the simplest possible hot path.
**Delivers:** Single GPU ExtraTree (classifier + regressor); privatized shared-memory histogram kernel; partition-scatter kernel; ET-fused random-candidate split kernel; GPU tree vs CPU oracle bit-exact on fixed seed; compute-sanitizer clean.
**Pitfalls avoided:** Pitfall 3 (BF design locked in), Pitfall 4 (shared-mem privatization as baseline), Pitfall 12 (roofline framing)
**Research flag:** Well-documented patterns. No additional research-phase needed.

### Phase 6: Full Forest + RandomForest + Arena Memory (Build B4)
**Rationale:** Adds NodeScheduler (BF frontier waves), RNG seed schedule, sibling subtraction, RF prefix-sum split-eval, stream-ordered fit-scoped arena.
**Delivers:** All four estimator classes; complete sklearn constructor params and fitted attrs; arena memory model; `feature_importances_` (real MDI); `predict_proba` working.
**Pitfalls avoided:** Pitfall 3 (node-size cutover to CPU leaf-finishing), Pitfall 14 (all four classes shipped before cuML moves)

### Phase 7: Determinism Mode + Dispatch Contract (Build B5)
**Rationale:** Determinism must be layered onto correct kernels. After the forest works, audit and harden reduction order and tie-breaking.
**Delivers:** `deterministic=True` bit-reproducible (byte-identical two same-seed runs, verified with exact binary comparison); integer/fixed-point histogram accumulation on deterministic path; fixed tie-breaking; `fallback="error"` enforcement; complete `execution_report_`; documented perf cost.
**Pitfalls avoided:** Pitfall 5 (bit-exact determinism), Pitfall 4 (no silent fallback)

### Phase 8: Pre-Registered Crossover Benchmark (Gate 3)
**Rationale:** This is the definition of success. All prior phases exist to enable this gate.
**Delivers:** Published (n×d) crossover surface map; accuracy parity alongside speed; scripted reproducible benchmark; decision: proceed if crossover proven, pivot if not.
**Kill criteria:** No region of the (n×d) surface where end-to-end GPU Extra Trees beats strongest CPU baseline → pivot to SHAP/determinism layer atop cuML.
**Pitfalls avoided:** Pitfall 1 (end-to-end from numpy), Pitfall 2 (crossover surface published), Pitfall 13 (pre-registered protocol)

### Phase 9: Exact Tree SHAP (Build B6)
**Rationale:** After crossover proven and SHAP spike decision known. `sylva-shap` consumes ForestIR only — CPU first, GPU path added after.
**Delivers:** `sylva-shap` crate; exact tree SHAP via chosen path (WoodelfHD port if license-clean + attributions scope confirmed, or GPUTreeSHAP baseline); `.shap_values()` on estimators; validated against `shap.TreeExplainer` within float tolerance.
**Pitfalls avoided:** Pitfall 8 (gated spike preceded this; no novel unverified method)
**Research flag:** Needs research-phase for WoodelfHD UFDP + Strassen-like kernel implementation planning once Phase 2 decision is known.

### Phase 10: Treelite Export + Packaging Polish (Build B7)
**Rationale:** `sylva-export` depends only on ForestIR; can run in parallel with Phase 9 after IR stabilizes. Packaging validation must happen before any public release.
**Delivers:** `sylva-export` crate; Treelite 4.x `import_from_json()`-compatible JSON; FIL round-trip test in CI; abi3 Windows wheel validated in a fresh environment; CUDA driver API linkage (dynamic-loading).
**Pitfalls avoided:** Pitfall 10 (clean-env wheel validation)
**Research flag:** Treelite 4.x JSON schema field names must be verified against live docs during this phase (MEDIUM confidence currently).

### Phase Ordering Rationale
- Spikes before build-out: the three gates (toolchain, SHAP feasibility, crossover) must precede broad kernel investment.
- CPU oracle before GPU kernels: the only asset that makes GPU-less cloud CI viable.
- ExtraTrees before RandomForest: ET's random split thresholds remove argmax/scan/atomics complexity.
- Single tree before forest: isolates kernel correctness from scheduler, RNG, and memory complexity.
- Determinism after forest works: cannot make something reproducible before it is correct.
- SHAP + Export as IR consumers: both fork off early (B1 provides the IR) and can develop in parallel with GPU training phases 5–8.
- Crossover gate before SHAP/Export investment: if crossover fails, the project pivots.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2 (SHAP Spike):** Fresh license check on WoodelfHD authors' implementation; GPUTreeSHAP C++ integration path from Rust.
- **Phase 9 (SHAP Implementation):** WoodelfHD UFDP + Strassen-like kernel implementation; GPU path planning for the path-compression scheme.
- **Phase 10 (Export):** Treelite 4.x JSON schema field names must be verified against live Treelite 4.x docs before implementation.

**Phases with standard patterns (skip research-phase):**
- Phase 1 (Toolchain Spike), Phase 3 (CPU Backend), Phase 4 (Quantizer), Phase 5 (Single-Tree GPU), Phase 6 (Full Forest), Phase 7 (Determinism), Phase 8 (Crossover Benchmark) — all well-documented by research files.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack (binding/interop) | HIGH | cudarc 0.19.8 + PyO3 0.29 + maturin 1.14.1 verified. NVRTC-over-AOT-nvcc Windows decision is HIGH (cc-rs MSVC gap confirmed). |
| Stack (kernel-authoring path) | HIGH | cudarc + CUDA C is the clear choice; Rust-CUDA/cust rejected on documented grounds. |
| Features (sklearn parity surface) | HIGH | Stable sklearn contract. cuML gaps confirmed via issue tracker. |
| Features (SHAP — WoodelfHD) | MEDIUM | Method verified real (arXiv 2604.10569). License and port feasibility UNVERIFIED — requires spike. |
| Features (variance-based pruning) | LOW | Method provenance unverified in literature. Engineering heuristic only. |
| Architecture (histogram engine) | HIGH | Privatized shared-mem histogram + sibling subtraction + BF level-at-a-time are cuML/XGBoost standard. |
| Architecture (determinism) | HIGH | Integer/fixed-point accumulation + canonical reduction order confirmed by arXiv 2408.05148; ~95–98% throughput retention documented. |
| Architecture (Treelite export schema) | MEDIUM | Import path confirmed; exact field names must be verified against live 4.x docs. |
| Pitfalls (technical) | HIGH | All grounded in documented failure modes. |
| Competitive landscape | MEDIUM | cuML 26.6 API confirmed (no Extra Trees). Kill-risk timeline unknown — ongoing monitoring required. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **rust-numpy exact version for PyO3 0.29** — verify precise compatible release against rust-numpy changelog before pinning.
- **cudarc 0.19.8 exact feature-flag names** — capabilities HIGH-confidence; exact strings (`"cuda-12060"`, `"nvrtc"`, `"dynamic-loading"`) must be confirmed against docs.rs/cudarc/0.19.8.
- **WoodelfHD upstream license** — must be checked in Phase 2 (SHAP spike) before any code reuse.
- **Treelite 4.x JSON schema** — exact `task_param`, `model_param`, node field names must be pinned in Phase 10.
- **SHAP attributions vs interactions decision** — product decision with major scope implications; resolve in Phase 2, ratified by requirements owner before Phase 9 is planned.
- **`sample_weight` MVP tradeoff** — MVP recommendation is error-on-non-None, which causes some `check_estimator` invariance failures; requirements owner must ratify this scoped gap.
- **Variance-based tree pruning validation** — do not advertise before an internal empirical validation study.

---

## Sources

### Primary (HIGH confidence)
- crates.io API — cudarc 0.19.8 (2026-06-19), pyo3 0.29.0 (2026-06-11), cust 0.3.2 (stale 2022-02-16)
- pypi.org — maturin 1.14.1 confirmed latest
- docs.rs/cudarc — driver/NVRTC/cuBLAS/cuRAND layering, Windows MSVC native builds, CUDA 11.4–13.0
- PyO3 releases + MSRV — abi3, CPython 3.7–3.14, MSRV 1.83
- docs.rs/cc + github.com/narsil/bindgen_cuda — cc-rs CUDA = GNU/Clang only; MSVC gap confirmed
- sklearn developer guide — estimator contract, BaseEstimator, no-logic-in-`__init__`
- XGBoost GPU algorithm updates + TPDS'19 GPU GBDT — privatized shared-mem histograms, sibling subtraction
- arXiv 2408.05148 + Deterministic Atomic Buffering (MICRO'53) — atomics/reduction determinism, ~95–98% throughput retained
- Random123 paper + cuRAND docs — Philox-4×32-10 as standard counter-based parallel RNG

### Secondary (MEDIUM confidence)
- arXiv 2604.10569 (WoodelfHD, 2026) + arXiv 2511.09376 (Woodelf/AAAI 2026) — algorithm verified real, complexity confirmed, GPU impl by authors
- rapidsai/gputreeshap (Apache-2.0, arXiv 2010.13972) — exact GPU TreeSHAP baseline
- cuML stable API reference (docs.rapids.ai, June 2026) — no Extra Trees confirmed; feature_importances_ NaN gap; oob_score gap (issue #3361)
- github.com/tracel-ai/cubecl v0.10.0 — alpha, breaking changes between minor versions
- Rust-GPU project blog (2025 reboot) — nightly-pinned, LLVM 7.x, Windows PATH fragility
- treelite.readthedocs.io 4.x — import_from_json() path confirmed (schema details MEDIUM)
- NVIDIA RMM blog — stream-ordered pool / arena design patterns

### Tertiary (LOW confidence / needs validation)
- "Blueprint fix #5" (variance-based redundant-tree pruning) — provenance unverified in peer-reviewed literature; treat as unvalidated engineering heuristic
- Treelite 4.x exact JSON schema field names — inferred from docs/tutorial examples, not pinned against 4.x spec

---
*Research completed: 2026-06-19*
*Ready for roadmap: yes*
