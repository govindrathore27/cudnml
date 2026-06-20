# Phase 2: CPU Oracle, Contracts & Forest IR - Research

**Researched:** 2026-06-20
**Domain:** Pure-Rust correctness oracle for tree ensembles (ExtraTrees + RandomForest, clf+reg), device-neutral `trait Backend`, SoA `ForestIR`, stateless Philox-4×32-10 RNG, differential + distributional parity testing vs scikit-learn
**Confidence:** HIGH (Philox algorithm, sklearn split algorithms, trait shape, IR field set, determinism strategy); MEDIUM (exact KS/CI thresholds — defensible defaults proposed, must be empirically calibrated during the parity phase; Treelite v4 field names — verified against v4 serialization doc)

## Summary

Phase 2 builds the **bit-level correctness oracle** that every later GPU result is verified against. The work is *not* GPU code — it is the device-neutral contract layer plus a trusted pure-Rust CPU backend. Three things dominate the risk surface and must be designed exactly right because they are the "near-rewrite-if-deferred" CPU↔GPU parity contract (per STATE.md): (1) the `trait Backend` seam must let `CpuBackend` train ET/RF *today* via recursive exact splitting while leaving GPU-histogram methods (`quantize`/`build_histograms`/`eval_splits`/`partition`) as part of the contract for Phase 4's `CudaBackend` — with **no CUDA types crossing the boundary** (ENG-01); (2) the SoA `ForestIR` must be designed now for *all* known consumers (train/predict + tree-SHAP per-node sample counts + Treelite-export fields) so it is never rewritten (D-03); (3) the stateless **Philox-4×32-10** RNG must be implemented with **documented test vectors** so Phase-4's CUDA copy is bit-verifiable (ENG-06).

The single highest-leverage finding (confirmed against sklearn source): **sklearn's tree splitters use a serial, stateful 32-bit xorshift PRNG (`our_rand_r`) whose draw order cannot be replayed by a parallel counter-based RNG.** This is *why* the parity contract is distributional (D-04), not bit-exact-to-sklearn (ENG-04, Pitfall 6). Sylva uses its **own** Philox stream for bit-identical CPU↔GPU reproducibility, and proves *statistical* equivalence to sklearn via accuracy/probability CIs plus a KS test on aggregate split statistics. The ExtraTrees random-threshold algorithm and the RandomForest best-split (midpoint candidate) algorithm are both fully documented in sklearn's `_splitter.pyx` and reproducible from the algorithm (Apache-2.0 clean — reimplement, never copy).

**Primary recommendation:** Create a new device-neutral crate `crates/sylva-core` (no CUDA deps) holding `trait Backend`, `ForestIR` (SoA), `CpuBackend` (ndarray+rayon), and `philox` (hand-rolled, with KAT test vectors). Build single-tree ET → forest ET → single-tree RF → forest RF, each differential-tested. Lock the parity contract as: **Sylva-internal bit-determinism (exact) + sklearn distributional equivalence (CI + KS)**, with f32 end-to-end (D-05). Per-tree Philox keying makes rayon tree-parallelism order-independent; the only FP-reduction hazard is parallel leaf/impurity accumulation within a tree, which must use a fixed (sequential or canonically-ordered) reduction to stay f32-bit-reproducible.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — NaN/missing-value routing = simple deterministic default-direction.** At each split, missing/NaN rows route to a deterministic default child — the **higher-sample-count child** — recorded in the `ForestIR` `default-child` array; **tie → left child** (deterministic). Cheap, deterministic, CPU/GPU bit-matchable (ENG-05). scikit-learn's trees have no missing-value story aligned with this, so differential tests run on **clean (non-NaN) data** and the default-direction path is exercised by dedicated NaN fixtures (SC-4). Learned/impurity-optimal default-direction (XGBoost-style) is **deferred**.
- **D-02 — CpuBackend scope this phase = both ExtraTrees AND RandomForest** (classifier + regressor for each). RF brings bootstrap resampling + best-split (impurity search over candidate thresholds) vs ET's random thresholds; both get full differential + property-based coverage. "Single tree before forest" *within* each, but RF is **not** sliced to a later phase — SC-2 met in full this phase.
- **D-03 — Design the SoA `ForestIR` now for all known downstream consumers**, not minimal-for-training. Arrays carry, beyond training/predict essentials (feature_id / threshold / left / right / default-child / leaf-value), the fields **tree-SHAP (Phase 8)** needs (per-node **sample/cover counts**) and **Treelite-export (Phase 6/9)** compatibility. Rationale: the IR is a near-rewrite risk if under-designed (STATE.md).
- **D-04 — Strict distributional parity.** The differential-test gate (EST-07 / SC-6) requires BOTH (a) accuracy / predicted-probability agreement within a **tight CI**, AND (b) a **KS test on aggregate split statistics** (feature-selection frequency, threshold distribution) across many trees. ExtraTrees splits are random → **distributional** equivalence, never per-tree structural match (consistent with ENG-04: Sylva's own bit-identical CPU↔GPU RNG + *distributional* equivalence to sklearn, NOT bit-replay of sklearn's serial PRNG). Bar is strict.
- **D-05 — float32 end-to-end.** `CpuBackend` and `ForestIR` compute and store in **f32**, matching the future GPU path so the **Phase-4 `GPU == CPU oracle` bit-exact gate** is achievable. Parity to scikit-learn (f64 thresholds) stays **distributional** — last-bit f32/f64 differences absorbed by the D-04 CI/KS bar.

### Claude's Discretion

- **RNG / determinism:** implement **Philox-4×32-10** in Rust now with **documented test vectors** that Phase-4's CUDA copy must bit-match. Per-tree RNG keyed by `(seed, tree, node, feature, draw)` makes `rayon` tree-parallelism order-independent.
- **`device="cpu"` dispatch:** this phase exposes `CpuBackend` as the **explicit `device="cpu"`** path + the differential oracle only. Auto small-data dispatch, `fallback="error"`, `execution_report_` are **Phase 6 (DET)** — not built here; no silent fallback.
- **RF split-finding** must mirror scikit-learn's candidate-threshold best-split search closely enough to satisfy D-04 (researcher confirms algorithm alignment; RF here is exact/sort-based — no quantizer until Phase 3).
- **Exact parity thresholds** (CI width, KS p-value, number of trees) — researcher picks defensible statistical values.
- **Exact `ForestIR` field set** for SHAP/export forward-design — researcher pins the array set from ARCHITECTURE.md + Treelite 4.x JSON schema + tree-SHAP requirements.

### Deferred Ideas (OUT OF SCOPE)

- Learned (impurity-optimal) default-direction for missing values (XGBoost-style) — Phase later.
- `sample_weight` end-to-end (EST-05) + full estimator API (EST-02, `fit`/`predict_proba`/`check_estimator`) — **Phase 5**.
- Quantizer / binning (QUANT-01/02) — **Phase 3**; RF best-split in Phase 2 operates without a quantizer.
- Auto small-data CPU dispatch, `fallback="error"`, `execution_report_` (DET-*) — **Phase 6**.
- All GPU/CUDA work (GPU-*), incl. the privatized-histogram kernel the IR stays compatible with — **Phase 4**.
- SHAP (Phase 8) and Treelite export (Phase 6/9) — IR is *designed for* them now (D-03) but neither implemented this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENG-01 | Device-neutral `trait Backend` (quantize, build_histograms, eval_splits, partition, predict); CUDA types never cross the boundary | Trait shape in §Architecture Pattern 1; associated-type/handle design keeps GPU methods in the contract while `CpuBackend` implements train/predict via a *direct* path. Anti-pattern: leaking CUDA types (PITFALLS AP-2). |
| ENG-02 | SoA `ForestIR` (feature_id/threshold/left/right/default-child/leaf-value) — single shared representation, write-once-by-train, read-only by inference/SHAP/export | §SoA ForestIR Field Set — full array list incl. SHAP cover counts + Treelite fields; ARCHITECTURE.md Pattern 3. |
| ENG-03 | Pure-Rust `CpuBackend` (ndarray+rayon) trains+predicts ET+RF correctly; differential oracle + `device="cpu"` path | §Standard Stack (ndarray 0.16, rayon 1.x); §sklearn Algorithm Specifics (ET random-threshold, RF best-split midpoint); §Determinism. |
| ENG-04 | Documented parity contract: Sylva-own bit-identical CPU↔GPU RNG per seed + distributional equivalence to sklearn (NOT bit-replay of sklearn serial PRNG) | §The Parity Contract; sklearn `our_rand_r` serial PRNG confirmed un-replayable in parallel (Pitfall 6). |
| ENG-05 | NaN/missing routing policy defined + implemented consistently across CPU and (future) GPU | §NaN Routing (D-01 default-direction); §ForestIR `default_child` array; NaN fixtures in §Validation Architecture. |
| ENG-06 | Stateless counter-based Philox-4×32-10, identical in Rust + (future) CUDA, keyed by (seed,tree,node,feature,draw) | §Philox-4×32-10 — verified constants, round fn, uint32→f32 conversion, KAT test vectors, counter packing. |
| EST-07 | Differential tests vs sklearn + property invariants (child rows partition parent, leaf probs valid, seed determinism, serialization round-trip) | §Parity Test Design; §Validation Architecture maps each invariant to a test. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Device-neutral contract (`trait Backend`) | L2 Backend trait seam (`sylva-core`/`sylva-backend`) | — | The abstraction boundary; defined device-neutral so CPU now + CUDA later are both additive (ENG-01). |
| ForestIR (SoA node arrays) | L3 orchestration / device-agnostic (`sylva-core::ir`) | read by L5/SHAP/export | Single shared representation owned by the device-neutral crate so no backend leaks into it (ENG-02). |
| CpuBackend train/predict (ET+RF) | L2 CpuBackend impl (pure Rust + rayon) | — | The correctness oracle; exact recursive splitting on CPU, no histograms/quantizer this phase (ENG-03). |
| Philox-4×32-10 RNG + seed schedule | L3 orchestration (`sylva-core::rng`) | inlined in L1 CUDA later | Stateless counter-based RNG shared bit-identically by CPU now and CUDA kernel later (ENG-06). |
| NaN/default-direction routing | L2 CpuBackend (split + predict) + L3 IR `default_child` | future L1 GPU | Policy is device-neutral data in the IR; CPU and GPU read the same `default_child` array (ENG-05). |
| Differential + property tests | Test tier (Rust `#[cfg(test)]`/`tests/` + Python harness vs sklearn) | — | CPU-only CI carries correctness (Pitfall 16); the oracle is what makes GPU-less CI viable. |
| Parity statistics harness (KS/CI) | Python script invoking the wheel + sklearn | Rust fixtures | sklearn is a Python distributional oracle; the harness compares like-for-like (ET-vs-ET, RF-vs-RF). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust (stable) | 1.83+ | Pure-Rust core | `[VERIFIED: Cargo.toml workspace]` MSRV floor set by PyO3 0.29; already pinned in `rust-toolchain.toml`. Stable, not nightly. |
| ndarray | 0.16.x | CPU backend data structures (`X` matrix, row views, leaf stats) | `[CITED: STACK.md]` The CPU correctness-oracle data structure; pairs with rust-numpy for zero-copy host transfer later. f32 dense matrix is `Array2<f32>` / `ArrayView2<f32>`. |
| rayon | 1.x | Data-parallel CPU backend (parallelize over trees) | `[CITED: STACK.md]` Trees are embarrassingly parallel; `par_iter` over the `n_estimators` tree-build closures. Per-tree Philox keying makes this order-independent. |
| thiserror | 1.x (workspace) | Typed library error enum | `[VERIFIED: Cargo.toml workspace.dependencies]` Already the project's pinned error-enum crate; `sylva-cuda::CudaError` is the template. Map to Python exceptions later via PyO3. |
| serde + serde_json | 1.x | ForestIR (de)serialization + round-trip test (EST-07) + future Treelite export | `[CITED: STACK.md]` `#[derive(Serialize, Deserialize)]` on the SoA IR; round-trip is a required property invariant this phase. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| proptest | 1.x | Property-based invariants | `[CITED: STACK.md / rules/rust/testing.md]` child-rows-partition-parent, leaf-prob validity, seed determinism, serialization round-trip — all property tests. |
| approx | 0.5.x | Float tolerance assertions | `[CITED: STACK.md]` `assert_abs_diff_eq!` / `assert_relative_eq!` for CPU-vs-sklearn distributional checks (the f32/f64 tolerance band). |
| rstest | latest | Parameterized test cases (clf/reg × ET/RF × fixtures) | `[CITED: rules/rust/testing.md]` Optional convenience for the 4-estimator matrix; not load-bearing. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled Philox | cuRAND host API (via cudarc) / `rand` crate PRNGs | `[CITED: STACK.md / CLAUDE.md]` Rejected: must bit-match CPU↔GPU and ship documented KAT vectors; cuRAND host Philox is harder to bit-match across CPU/GPU; `rand` PRNGs are stateful, not counter-based. Hand-roll is ~20 lines and stateless. |
| ndarray | raw `Vec<f32>` + manual strides | ndarray gives `ArrayView2` row-slicing and is the rust-numpy bridge; raw Vec re-invents it. |
| proptest | quickcheck | proptest has better shrinking + is the project-pinned choice. |
| Trait in separate `sylva-backend` crate | Trait inside `sylva-core` | ARCHITECTURE.md suggests a tiny `sylva-backend` crate so `sylva-cpu`/`sylva-cuda` depend on the seam only. For Phase 2 (CPU only), keeping the trait + CpuBackend + IR in one `sylva-core` crate is simpler and still CUDA-free; split into `sylva-backend`/`sylva-cpu` later if the dependency graph demands it. **Recommend: single `sylva-core` crate now** (fewer files, no CUDA dep, matches the "near-rewrite risk is the IR design, not the crate count" framing). |

**Installation (add to a new `crates/sylva-core/Cargo.toml`):**
```toml
[dependencies]
ndarray = "0.16"
rayon = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = { workspace = true }

[dev-dependencies]
proptest = "1"
approx = "0.5"
# rstest = "0.x"   # optional, for the clf/reg × ET/RF matrix
```
Then add `"crates/sylva-core"` to the root `Cargo.toml` `[workspace] members`. **No CUDA deps in this crate** (ENG-01).

**Version verification:** `cargo search` returned empty in this sandboxed session (offline). Versions above are carried from the project's own STACK.md (researched 2026-06-19, HIGH confidence) and the major-version pins (ndarray 0.16, rayon 1, serde 1, proptest 1, approx 0.5) are stable, slow-moving lines. `[ASSUMED]` for the exact patch — the planner should add a `checkpoint:human-verify` or a `cargo add` step that lets Cargo resolve the latest compatible patch within these majors, then record the resolved versions in a VERSIONS note.

## Package Legitimacy Audit

> All packages are mature, well-known crates already named in the project's STACK.md / CLAUDE.md. Registry verification via `cargo search` was unavailable (offline sandbox); legitimacy is established by these being the canonical, multi-million-download crates in the Rust ecosystem and the project's own pre-vetted stack.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| ndarray | crates.io | ~9 yrs | very high | github.com/rust-ndarray/ndarray | OK (assumed — offline) | Approved (project-pinned) |
| rayon | crates.io | ~9 yrs | very high | github.com/rayon-rs/rayon | OK (assumed — offline) | Approved (project-pinned) |
| serde / serde_json | crates.io | ~9 yrs | highest in ecosystem | github.com/serde-rs/serde | OK (assumed — offline) | Approved (project-pinned) |
| thiserror | crates.io | ~6 yrs | very high | github.com/dtolnay/thiserror | OK (already in workspace) | Approved (in use) |
| proptest | crates.io | ~8 yrs | high | github.com/proptest-rs/proptest | OK (assumed — offline) | Approved (project-pinned) |
| approx | crates.io | ~8 yrs | high | github.com/brendanzab/approx | OK (assumed — offline) | Approved (project-pinned) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none
**Note:** Because registry checks were offline this session, the planner should run `cargo add <crate>` (which validates against crates.io) rather than hand-pinning, and confirm each resolves to a github.com-hosted, high-download crate before first build. This is low-risk — all six are foundational, decade-old ecosystem crates.

## Architecture Patterns

### System Architecture Diagram

```
                        Phase 2 scope (device-neutral, CPU-only)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                            │
  X: ndarray Array2<f32>  +  y: Array1                                         │
  │      │                                                                     │
  │      ▼                                                                     │
  │  TrainConfig (n_estimators, max_depth, max_features, min_samples_*,        │
  │   bootstrap, criterion, seed, algo=ET|RF)  ──► validated at boundary       │
  │      │                                                                     │
  │      ▼                                                                     │
  │  ┌─────────────────── CpuBackend::fit (rayon par over trees) ──────────┐   │
  │  │  for tree t in 0..n_estimators (PARALLEL, order-independent):       │   │
  │  │    Philox(seed, t, ...)                                             │   │
  │  │      ├─ RF: bootstrap row sample  | ET: use all rows                │   │
  │  │      └─ recursive build_node(rows, depth):                          │   │
  │  │           ├─ feature subset draw  (Philox draw=feature-select)      │   │
  │  │           ├─ ET: per-feature 1 random threshold ∈ [min,max]         │   │
  │  │           │       (Philox draw=threshold)  → proxy impurity         │   │
  │  │           ├─ RF: sort feature, midpoint candidates → best gain      │   │
  │  │           ├─ enforce min_samples_split / min_samples_leaf           │   │
  │  │           ├─ pick best split (fixed tie-break: lowest feature,thr)  │   │
  │  │           ├─ record node → ForestIR (feature,threshold,default_child│   │
  │  │           │       = higher-count child; tie→left; sample_count)     │   │
  │  │           └─ recurse left/right                                     │   │
  │  └─────────────────────────────────────────────────────────────────────┘  │
  │      │                                                                     │
  │      ▼                                                                     │
  │  ForestIR (SoA: per-tree offsets + node arrays)  ── write-once             │
  │      │                                                                     │
  │      ├──[CpuBackend::predict]──► Predictions (clf: class probs / reg: mean)│
  │      ├──[serde round-trip]─────► JSON ⇄ ForestIR (property test)           │
  │      └──(future) SHAP / Treelite export read the SAME arrays               │
  └──────────────────────────────────────────────────────────────────────────┘
            ▲                                          ▲
            │ trait Backend (the seam)                 │ Philox stream (bit-identical)
            │  quantize/build_histograms/eval_splits/  │  shared by future CudaBackend
            │  partition/predict  (CUDA-free signatures)│  (Phase 4 reproduces from KATs)
            └──────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/sylva-core/              # NEW device-neutral crate — NO cuda deps (ENG-01)
├── Cargo.toml
└── src/
    ├── lib.rs                  # re-exports; crate error enum (thiserror)
    ├── backend.rs              # trait Backend (the seam) + neutral handle types
    ├── ir.rs                   # ForestIR: SoA node arrays + per-tree offsets (ENG-02)
    ├── rng/
    │   ├── mod.rs              # Philox-4x32-10 + uint32->f32 + counter packing (ENG-06)
    │   └── kat.rs              # known-answer test vectors (documented, Phase-4-verifiable)
    ├── config.rs              # TrainConfig, Criterion enum, MaxFeatures enum, validation
    ├── cpu/
    │   ├── mod.rs              # CpuBackend struct impl Backend
    │   ├── fit.rs              # rayon tree loop, recursive node builder
    │   ├── split_et.rs         # ExtraTrees random-threshold splitter
    │   ├── split_rf.rs         # RandomForest best-split (sorted midpoint) splitter
    │   ├── criterion.rs        # Gini / entropy / MSE impurity (f32)
    │   ├── predict.rs          # tree/forest traversal, NaN default-child routing
    │   └── bootstrap.rs        # RF bootstrap row sampling (Philox)
    └── parity.rs               # (optional) split-statistic extractors for the KS harness

python/tests/parity/            # the Comparative Baseline Study harness (vs sklearn)
├── conftest.py
├── test_distributional_parity.py   # accuracy/proba CI + KS on split stats
└── datasets.py                 # make_classification 20k×50, Covertype subset loaders
```

**Structure rationale:** Many small files (CLAUDE.md coding-style: 200–400 lines, 800 max). `split_et.rs` and `split_rf.rs` are separate because they are genuinely different algorithms (random single threshold vs exhaustive midpoint search). `rng/kat.rs` isolates the documented test vectors so Phase 4 has a single file to bit-match against. `ir.rs` is the highest-leverage file (the near-rewrite risk) — keep it device-agnostic and consumer-complete.

### Pattern 1: `trait Backend` — CUDA-free seam with a CPU-direct path

**What:** The trait lists the GPU-histogram-oriented ops (`quantize`/`build_histograms`/`eval_splits`/`partition`/`predict`) AND `fit`/`predict` at the right granularity. The CPU oracle does **not** route its training through `build_histograms` (there is no quantizer until Phase 3 and no histograms until Phase 4) — it implements a coarse `fit`/`predict` entrypoint directly via recursive exact splitting. The histogram-oriented methods exist on the trait so the future `CudaBackend` satisfies the same contract; `CpuBackend` may implement them as a CPU reference (useful later for Phase-3 quantizer parity) or leave them behind a sub-trait.

**Recommended concrete signature (ENG-01):** Keep the trait coarse-grained (per-fit / per-predict), with device-neutral types only. Two viable shapes:

**Option A (recommended for Phase 2) — high-level trait, histogram ops as a separate future trait:**
```rust
// sylva-core/src/backend.rs  — NO cuda types anywhere
pub trait Backend {
    /// Train a forest from a host-side dense f32 table + labels, producing the IR.
    fn fit(&self, x: ArrayView2<f32>, y: ArrayView1<f32>, cfg: &TrainConfig)
        -> Result<ForestIR, SylvaError>;
    /// Predict from a completed IR (clf: class probabilities; reg: mean response).
    fn predict(&self, ir: &ForestIR, x: ArrayView2<f32>)
        -> Result<Predictions, SylvaError>;
}

// The GPU-histogram ops live in a SEPARATE trait that CudaBackend (Phase 4)
// implements; CpuBackend need not until it serves the Phase-3 quantizer oracle.
// Defining it now documents the contract without forcing CUDA types into Backend.
pub trait HistogramBackend {
    type Bins;                  // opaque, device-describable (NO CudaSlice in the signature)
    fn quantize(&self, x: ArrayView2<f32>, edges: &BinEdges) -> Result<Self::Bins, SylvaError>;
    fn build_histograms(&self, bins: &Self::Bins, frontier: &Frontier, rows: &RowIndex)
        -> Result<Histograms, SylvaError>;
    fn eval_splits(&self, h: &Histograms, mode: SplitMode) -> Result<Vec<SplitDecision>, SylvaError>;
    fn partition(&self, bins: &Self::Bins, splits: &[SplitDecision], rows: &mut RowIndex)
        -> Result<ChildRanges, SylvaError>;
}
```
**Option B — single trait with associated `Buffer` type (matches ARCHITECTURE.md Pattern 1 literally):** one `trait Backend` with `type Buffer;` and all six methods. `CpuBackend::Buffer = ndarray`-backed; `CudaBackend::Buffer` = device buffer. Risk: `build_histograms`/`partition` signatures are awkward for the CPU recursive path and tempt CUDA-shaped leakage.

**Recommendation:** **Option A.** It satisfies ENG-01 literally (the words "device-neutral `trait Backend` defines all device ops" — keep `quantize/build_histograms/eval_splits/partition/predict` named in a trait, CUDA-free, via the associated `type Bins`), keeps the CPU oracle's `fit`/`predict` clean, and avoids forcing the recursive CPU builder through a histogram-shaped API it doesn't use. The associated type (not a concrete CUDA type) is the mechanism that keeps CUDA out of the boundary (`[CITED: ARCHITECTURE.md Pattern 1, Anti-Pattern 2]`). The planner should confirm with the user whether ENG-01's wording requires all five op-names on a *single* trait (Option B) or accepts the split (Option A) — flagged in §Open Questions.

**When to use:** Always — it is the spine of the additive-backend requirement.
**Trade-offs:** Slight indirection. Forces discipline: anything CUDA-shaped escaping the trait is a design smell.

### Pattern 2: SoA ForestIR — write-once-by-train, read-many

**What:** Structure-of-Arrays of node records, parallel arrays indexed by global node id, plus per-tree offset table. Training *appends* nodes; inference/SHAP/export *read*. Stored f32 (D-05). See §SoA ForestIR Field Set for the exact arrays.

**Source pattern:** `[CITED: ARCHITECTURE.md Pattern 3]` SoA is coalescing-friendly on GPU and cache-friendly on CPU; one representation eliminates train→infer translation bugs.

### Pattern 3: Per-tree counter-based RNG → order-independent parallelism

**What:** Every random draw is `Philox(key=seed, counter=pack(tree, node, feature, draw))`. No sequential PRNG state. Because the draw is a pure function of its coordinates, rayon may build trees in any order and produce an identical ForestIR. This is what makes `deterministic` reachable and CPU↔GPU bit-identical. `[CITED: ARCHITECTURE.md Pattern 4; STACK.md RNG row]`

### Anti-Patterns to Avoid

- **Leaking CUDA types through the trait** `[CITED: ARCHITECTURE.md AP-2]`: never put device pointers/streams in trait signatures. Use associated types / neutral handles. (ENG-01 is literally this.)
- **Treating sklearn as a bit oracle** `[CITED: PITFALLS Pitfall 6]`: sklearn's serial `our_rand_r` PRNG draw order cannot be replayed in parallel. Use sklearn as a *distributional* oracle; the bit oracle is Sylva's own CPU path for a given seed.
- **Float `atomicAdd`-style unordered FP reduction** `[CITED: PITFALLS Pitfall 5]`: even on CPU, a rayon parallel-sum of f32 impurity/leaf statistics is non-associative and breaks bit-reproducibility. Use a fixed reduction order (sequential within a node, or canonical tree-reduction).
- **Skipping NaN fixtures** `[CITED: PITFALLS Tech-debt table]`: NaN comparisons are always false → silent misrouting. NaN fixtures required from the first differential test (SC-4).
- **Copying sklearn / GPL source** `[CITED: PITFALLS Security; CLAUDE.md]`: reimplement the algorithm from the paper/docs (Apache-2.0). Provenance documented.

## sklearn Algorithm Specifics (for distributional parity)

> Source: `[CITED: github.com/scikit-learn/scikit-learn /sklearn/tree/_splitter.pyx]` (RandomSplitter, BestSplitter) and `[CITED: scikit-learn ExtraTreesClassifier docs 1.7/1.8]`. These are the algorithms Sylva must *reproduce from* (clean-room, Apache-2.0), and the distribution Sylva's split statistics are KS-tested against (D-04).

### ExtraTrees (RandomSplitter) — what Sylva must match distributionally
- **Per node:** draw `max_features` candidate features (without replacement until a valid split is found — sklearn keeps drawing past `max_features` if no valid partition yet). For **each** candidate feature, draw **one** threshold uniformly in `(min_feature_value, max_feature_value)` of the *non-constant* feature *within the current node's samples*: `threshold = rand_uniform(min, max, rng)`. `[VERIFIED: _splitter.pyx RandomSplitter — `current_split.threshold = rand_uniform(min_feature_value, max_feature_value, random_state)`]`
- **Selection:** evaluate each candidate via `proxy_impurity_improvement()`; keep the **best** across the `max_features` candidates. (So ET is "1 random threshold per feature, best across features" — not fully random unless `max_features=1`.)
- **Constant-feature guard:** if `max_feature_value <= min_feature_value + FEATURE_THRESHOLD` (≈1e-7), the feature is constant → skipped. `[VERIFIED: _splitter.pyx]`
- **Partition:** `X[:, f] <= threshold` → left, else right. sklearn uses `<=` (left if value ≤ threshold). `[VERIFIED: tree splitter comparison convention]`

### RandomForest (BestSplitter) — what Sylva must match distributionally
- **Per node:** draw `max_features` candidate features. For each, **sort** the feature's values among node samples, then evaluate candidate thresholds at **midpoints between consecutive distinct sorted values**: `threshold = feature_values[p_prev]/2.0 + feature_values[p]/2.0`. `[VERIFIED: _splitter.pyx BestSplitter midpoint]`
- **Selection:** choose the threshold maximizing impurity improvement across all candidates and features (greedy exact best split). Skip pairs closer than `FEATURE_THRESHOLD`.
- **Bootstrap:** RF default `bootstrap=True` draws `n` rows **with replacement** per tree (sample weights = multiplicity). ET default `bootstrap=False` uses all rows. `[CITED: RandomForest docs; ET docs]`

### Defaults Sylva must replicate (EST-03 lands in Phase 5, but parity needs them now)
| Param | Classifier default | Regressor default | Note |
|-------|-------------------|-------------------|------|
| `max_features` | `"sqrt"` (⌊√d⌋) | `1.0` (all features) | `[CITED: ExtraTrees/RandomForest docs]` Different per task — must branch on clf vs reg. |
| `criterion` | `"gini"` (also `"entropy"`/`"log_loss"`) | `"squared_error"` (MSE) | `[CITED: docs]` Phase 2 implements gini+entropy (clf), MSE (reg). |
| `min_samples_split` | 2 | 2 | node must have ≥2 samples to split. |
| `min_samples_leaf` | 1 | 1 | each child ≥1 sample. |
| `max_depth` | `None` (grow until pure / min_samples) | same | Phase-2 tests pin a finite `max_depth` for tractable distributions. |
| `bootstrap` | ET `False`, RF `True` | same | governs row sampling. |

### Leaf computation
- **Classifier:** leaf stores **class probabilities** = normalized class counts of training samples reaching the leaf (∑=1, each ∈[0,1]). Forest `predict_proba` = mean of per-tree leaf probabilities; `predict` = argmax. `[CITED: sklearn forest averaging — soft-vote mean of tree probabilities]`
- **Regressor:** leaf stores **mean** of training target values reaching the leaf; forest predict = mean across trees.

### What Sylva is FREE to differ on (does NOT break D-04)
- **Exact threshold values & per-tree structure** — driven by Sylva's *own* Philox stream, which differs from sklearn's `our_rand_r` order. D-04 only requires the *distribution* of (feature-selection frequency, threshold positions, accuracy, probabilities) to match — not per-tree identity. `[CITED: ENG-04; PITFALLS Pitfall 6]`
- **f32 vs f64 last-bit threshold differences** (D-05) — absorbed by the tolerance band.
- **RNG internals** — sklearn's PRNG is not part of the parity contract; Sylva documents its own.

### CRITICAL un-replayable detail (drives the whole contract)
sklearn's splitters draw randomness from `our_rand_r` — a **stateful 32-bit xorshift** seeded per-node from the tree's `random_state`, advanced **serially** as features/thresholds are drawn. `[CITED: sklearn/utils/_random.pyx / _random.pxd `our_rand_r`, `rand_int`, `rand_uniform`]` A parallel counter-based RNG **cannot** reproduce this serial draw order. **Therefore bit-identical-to-sklearn is infeasible and is not attempted** — this is the documented justification for ENG-04 / D-04. `[VERIFIED: confirmed in source + PITFALLS Pitfall 6 + ARCHITECTURE.md]`

## Philox-4×32-10 (ENG-06)

> Algorithm: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (Random123). Constants verified against the DEShawResearch Random123 reference header. `[VERIFIED: github.com/DEShawResearch/random123 include/Random123/philox.h]`

### Constants (hex)
- `PHILOX_M4x32_0 = 0xD2511F53` (multiplier for word 0)
- `PHILOX_M4x32_1 = 0xCD9E8D57` (multiplier for word 2)
- `PHILOX_W32_0   = 0x9E3779B9` (key bump for key[0], the golden-ratio Weyl constant)
- `PHILOX_W32_1   = 0xBB67AE85` (key bump for key[1], √3 fractional)
- **Rounds = 10**, state = 4×u32 counter, key = 2×u32.

### `mulhilo32(a, b) -> (hi, lo)`
Compute `p = (a as u64) * (b as u64)`; `hi = (p >> 32) as u32`, `lo = p as u32`. (Single 64-bit multiply; no overflow concerns.) `[VERIFIED: philox.h mulhilo32]`

### Single round (`philox4x32round(ctr, key)`)
```
(hi0, lo0) = mulhilo32(M0, ctr[0])
(hi1, lo1) = mulhilo32(M1, ctr[2])
out[0] = hi1 ^ ctr[1] ^ key[0]
out[1] = lo1
out[2] = hi0 ^ ctr[3] ^ key[1]
out[3] = lo0
```
`[VERIFIED: philox.h _philox4x32round]`

### Key schedule (per round, applied between rounds)
After each round, bump the key: `key[0] += PHILOX_W32_0`, `key[1] += PHILOX_W32_1` (wrapping u32 add). The 10-round generator applies round → bumpkey ten times (the reference applies the key for round 0 first, then bumps; implement to match the KAT vectors — see below). `[VERIFIED: philox.h Philox4x32_R bumpkey loop]`

### uint32 → uniform f32 in [0,1)
Use the high 24 bits (f32 has 24-bit mantissa) for a clean dyadic uniform:
`u32_to_f32_unit(x) = (x >> 8) as f32 * (1.0 / 16777216.0)`  → range `[0, 1)`, never reaching 1.0.
For ET threshold `∈ (min, max)`: `min + u * (max - min)`. **Do this identically in Rust and (Phase 4) CUDA** so the streams bit-match. (Document this conversion as part of the parity contract; it is as load-bearing as the round function.) `[CITED: standard Random123 → unit-float convention; f32 24-bit mantissa]`

### Counter packing `(seed, tree, node, feature, draw)` → (key, counter)
- `key = [seed_lo, seed_hi]` (split the u64 seed into two u32), **or** `key = [seed, stream_id]` if a 32-bit seed + stream is preferred. Recommend: `key = [seed as u32, (seed >> 32) as u32]`.
- `counter = [tree, node, (feature << K) | draw_kind, draw_index]` — pack the four coordinates into the 4×u32 counter. Exact bit-allocation is Sylva's choice but **must be documented and frozen** (Phase 4 reproduces it). A clean default: `ctr = [tree_id, node_id, feature_id, draw_counter]` with `draw_counter` incremented per draw within a (tree,node,feature) context. This gives a unique counter per draw → independent stream.
- **Each Philox call yields 4×u32** → 4 independent uniforms per call; use a sub-index (0..3) to pick the word, so one call serves up to 4 draws before incrementing the counter.

### Documented test vectors (REQUIRED deliverable — `rng/kat.rs`)
Ship the canonical Random123 known-answer tests so Phase-4 CUDA is bit-verifiable. The two anchor KATs from the Random123 `kat_vectors` for `philox4x32 10`:
- **All-zero input:** `ctr = {0,0,0,0}`, `key = {0,0}` → output `{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}`.
- **All-ones input:** `ctr = {0xffffffff,0xffffffff,0xffffffff,0xffffffff}`, `key = {0xffffffff,0xffffffff}` → output `{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}`.
- **Mixed input:** `ctr = {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}`, `key = {0xa4093822, 0x299f31d0}` → output `{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}`.

`[ASSUMED]` — these specific output triples are the well-known published Random123 `philox4x32 10` KAT values from training knowledge; the canonical `kat_vectors.txt` fetch 404'd this session. **The planner MUST add a `checkpoint:human-verify` task** to confirm these three output vectors against the official Random123 `kat_vectors.txt` (or by running a reference implementation) before they are committed as the frozen Phase-4 oracle. If any output word differs, the Sylva implementation is correct only if it reproduces whatever the *verified* reference emits — so verify the reference first, then assert Sylva matches it. (The constants and round function above are VERIFIED from the header; only the literal KAT outputs are pending confirmation.)

## SoA ForestIR Field Set (ENG-02, D-03)

> Design-for-all-consumers. Each array is parallel, indexed by global node id; a per-tree offset table maps tree→node range. Stored f32 (D-05). Field names/types chosen to map cleanly onto Treelite v4 (Phase 6/9) and tree-SHAP (Phase 8).

### Per-node arrays
| Field | Type | Consumer(s) | Purpose / Maps to |
|-------|------|-------------|-------------------|
| `feature_id` | `Vec<i32>` (-1 for leaf) | train, predict, SHAP, export | Split feature. Treelite `split_index`. |
| `threshold` | `Vec<f32>` | train, predict, SHAP, export | Split threshold (f32, D-05). Treelite `threshold` (note Treelite is f64 — widen on export). |
| `left_child` | `Vec<i32>` (-1 if leaf) | all | Left child node id. Treelite `cleft`. |
| `right_child` | `Vec<i32>` (-1 if leaf) | all | Right child node id. Treelite `cright`. |
| `default_child` | `Vec<i32>` | predict (NaN routing), SHAP, export | **D-01**: missing/NaN routes here (higher-sample-count child; tie→left). Maps to Treelite `default_left` (bool) on export: `default_left = (default_child == left_child)`. |
| `is_leaf` | `Vec<bool>` | all | Leaf marker (or derive from `feature_id == -1`). Treelite `node_type`. |
| `node_sample_count` | `Vec<u64>` | **SHAP (cover)**, export, D-01 tie-break | **Per-node training sample count reaching the node.** Required by path-dependent TreeSHAP as the background-distribution weight; also the data used to compute `default_child`. Maps to Treelite `data_count`. `[VERIFIED: SHAP path-dependent uses node sample counts as background; Treelite `data_count`]` |
| `node_weighted_count` | `Vec<f32>` | SHAP, export, future sample_weight | Sum of sample weights at node (= count when unweighted). Maps to Treelite `sum_hess` analogue (cover). Carry now so Phase-5 `sample_weight` and SHAP don't force an IR rewrite (D-03). |
| `impurity` / `gain` | `Vec<f32>` | export (`gain`), diagnostics | Node impurity or split gain. Maps to Treelite `gain` (optional). Cheap to record during build. |

### Leaf-value storage (classifier vs regressor)
- **Regressor:** `leaf_value: Vec<f32>` — one value per leaf node (the mean). Treelite `leaf_value`.
- **Classifier:** `leaf_proba: Vec<f32>` flattened `n_classes` per leaf + `n_classes: usize` + per-leaf offset (or a fixed-stride `[num_leaves * n_classes]` row-major block). Maps to Treelite `leaf_vector` + `leaf_vector_begin/end`. Each leaf's slice sums to 1, each ∈[0,1] (a property invariant).

### Per-tree / per-forest fields
| Field | Type | Purpose |
|-------|------|---------|
| `tree_offsets` | `Vec<usize>` (len `n_trees+1`) | node range `[tree_offsets[t]..tree_offsets[t+1])` per tree. |
| `tree_root` | `Vec<i32>` | root node id per tree (usually `tree_offsets[t]`). |
| `n_trees` | `usize` | forest size. |
| `n_features` | `usize` | input dimensionality (Treelite `num_feature`). |
| `task` | enum `{Classification{n_classes}, Regression}` | governs leaf storage + predict aggregation. Treelite `task_type` / `leaf_vector_shape`. |
| `criterion` | enum `{Gini, Entropy, Mse}` | recorded for reproducibility/report. |
| `seed` | `u64` | the Philox key — enables exact reproduction (seed-determinism invariant). |

**Consumer mapping summary:** train *writes* every array; `predict` reads `feature_id/threshold/left/right/default_child/is_leaf/leaf_*`; **SHAP (Phase 8)** additionally reads `node_sample_count`/`node_weighted_count` (cover) + the tree structure; **export (Phase 6/9)** maps the whole set onto Treelite v4's SoA arrays (`split_index`, `threshold`, `cleft`, `cright`, `default_left`, `leaf_value`/`leaf_vector`, `data_count`, `sum_hess`, `gain`). Carrying `node_sample_count`, `node_weighted_count`, and `gain` **now** is the entire point of D-03 — they are cheap at train time and prevent the cross-phase IR rewrite flagged by STATE.md. `[VERIFIED: Treelite v4 serialization field list; SHAP cover requirement]`

## NaN / Missing-Value Routing (ENG-05, D-01)

- **Policy:** at predict time, if `X[i, feature_id[node]]` is NaN (or the feature is missing), route row `i` to `default_child[node]`. `default_child` is set **at train time** to the child with the **higher `node_sample_count`**; on a tie, the **left child** (deterministic).
- **Implementation guard:** because `NaN <= threshold` is always `false` in IEEE-754, a naive `if x <= threshold { left } else { right }` silently sends NaN right. **Must** branch on `x.is_nan()` *first* → `default_child`, before the threshold comparison. `[CITED: PITFALLS Pitfall 7 — NaN comparisons always false → silent misrouting]`
- **Training data:** sklearn's ET/RF reject NaN (or route randomly in recent versions, which Sylva does not match) — so **differential tests use clean data** (D-01). The default-direction path is covered by **dedicated NaN fixtures** (SC-4), asserting CPU routing is deterministic and matches the IR's `default_child`.
- **CPU/GPU consistency (ENG-05):** because `default_child` is *data in the IR* (not code), the future GPU predict kernel reads the same array → identical routing by construction. This is the cheap-and-bit-matchable rationale for D-01.

## The Parity Contract (ENG-04) — document this explicitly

A standalone `PARITY.md` (or a section in the crate docs) is a phase deliverable. It must state:

1. **Sylva-internal reproducibility (EXACT / bit-level):** for a fixed `seed`, two `CpuBackend::fit` runs produce a **byte-identical** serialized `ForestIR`. (Property test: serialize both → assert equal bytes.) This is the contract the Phase-4 GPU path will be held to (`GPU == CPU` bit-exact).
2. **CPU↔GPU RNG identity (future, enabled now):** Sylva's Philox stream is defined by documented constants + counter packing + uint32→f32 conversion + KAT vectors. Phase 4 reproduces it bit-for-bit in CUDA. Phase 2 ships the Rust reference + KATs.
3. **sklearn equivalence (DISTRIBUTIONAL, NOT bit-level):** Sylva does **not** reproduce sklearn's per-tree structure (sklearn's serial `our_rand_r` PRNG is un-replayable in parallel — §sklearn Specifics). Equivalence is asserted statistically: accuracy + predicted-probability within a tight CI, and a KS test on aggregate split statistics (feature-selection frequency, threshold distribution) across many trees. Like-for-like only (ET-vs-ET, RF-vs-RF).
4. **f32 precision (D-05):** all compute/storage is f32; sklearn (f64) last-bit differences are inside the distributional tolerance.

## Parity Test Design (D-04, EST-07)

### Distributional tests vs sklearn (the gate)
| Statistic | Test | Defensible threshold (calibrate empirically) | Rationale |
|-----------|------|----------------------------------------------|-----------|
| **Test-set accuracy** (clf) | Sylva vs sklearn, same dataset/hyperparams, many seeds | `|acc_sylva − acc_sklearn|` within a 95% CI of the seed-to-seed spread (e.g. ≤ ~0.5–1.0 pp, OR within ±2σ of sklearn's own seed variance) | ET/RF are stochastic; the right bar is "indistinguishable from sklearn's own run-to-run variance," not a fixed epsilon. |
| **Predicted probabilities** (clf) | mean abs / max abs diff of `predict_proba` over test rows | mean abs diff ≤ ~1e-2; max abs diff reported (not gated) | Probabilities are smoother than accuracy; tighter band. |
| **R² / MSE** (reg) | Sylva vs sklearn | within seed-variance CI | Regressor analogue of accuracy. |
| **Feature-selection frequency** | KS test: distribution of chosen split-features across all nodes of all trees, Sylva vs sklearn | KS p-value **> 0.05** (fail to reject "same distribution") | Confirms `max_features` sampling + split preference match in aggregate. |
| **Threshold distribution per feature** | KS test: distribution of chosen thresholds (normalized to feature range) per feature | KS p-value **> 0.05** | Confirms ET uniform-in-[min,max] / RF midpoint behavior matches in aggregate. |

**Stability requirements for stable distributions:** use **enough trees × seeds** that the KS sample is large. Recommend `n_estimators ≥ 200` per implementation (more if cheap) and **≥ 10 seeds**, pooling split statistics across all trees and seeds → thousands of split observations per feature. `make_classification(20_000, 50)` (ROADMAP dataset) at `n_estimators=200`, `max_depth≈12` yields a large split population. **The exact tree/seed counts and KS/CI thresholds are `[ASSUMED]` defensible starting points — they MUST be calibrated empirically** (run sklearn-vs-sklearn across seeds first to measure the *null* spread, then set Sylva's bar to that null band). Flag for the planner: add a calibration task that establishes the sklearn-vs-sklearn baseline before gating Sylva.

### Property-based invariants (proptest — CPU-only, fast, in CI)
| Invariant | Assertion |
|-----------|-----------|
| Child rows partition parent **exactly** | `rows(left) ∪ rows(right) == rows(parent)` and `rows(left) ∩ rows(right) == ∅` (on clean data; NaN rows go to `default_child`, still a partition). |
| Leaf probabilities valid (clf) | every leaf's `leaf_proba` slice: each ∈ [0,1], sum == 1 (within f32 epsilon). |
| Leaf value finite (reg) | every `leaf_value` is finite (no NaN/Inf). |
| Seed determinism | same `(seed, cfg, data)` → **byte-identical** serialized ForestIR (twice). |
| Serialization round-trip | `deserialize(serialize(ir)) == ir` (structural equality + predictions equal). |
| Sample-count consistency | `node_sample_count[node] == node_sample_count[left] + node_sample_count[right]` for internal nodes (cover invariant — pre-validates SHAP). |
| `default_child` correctness | `default_child` points to the higher-`node_sample_count` child; tie→left. |
| Tree-parallel order-independence | building with rayon (parallel) vs sequential → identical ForestIR (validates Philox keying). |

### NaN fixtures (SC-4, ENG-05)
Small hand-built matrices with NaN in known cells + known `default_child` directions → assert routing is deterministic and matches the IR. (Not compared to sklearn — sklearn has no matching policy.)

## Comparative Baseline Study (SC-6, SC-7)

> Fairness protocol is **binding** (STATE.md / PITFALLS 1,2,13). Foundational phase → **correctness/distribution PARITY is the gate; training time is reported, not gated** (SC-7). Like-for-like only.

### Harness structure (recommended)
A **Python script** (`python/tests/parity/`) is the right tool: it imports the built Sylva wheel (the `device="cpu"` path) and sklearn in the same process, runs both on identical data/hyperparameters, and computes the CI + KS statistics. Rationale: sklearn is Python-native; the parity oracle naturally lives where sklearn does; and the wheel/PyO3 seam already exists (Phase 1). A Rust-side fixture set complements it for the proptest invariants (which need no sklearn). **Note:** the full estimator API is Phase 5 — for Phase 2 the wheel may expose a minimal `fit_cpu(X, y, cfg) -> ir_handle` + `predict_cpu` + a `split_statistics()` accessor purely for the harness, *without* the sklearn-drop-in surface. Confirm this minimal Python seam is acceptable (it is below EST-02's bar, which is deferred). Flagged in §Open Questions.

### Protocol
- **Datasets:** `make_classification(n_samples=20_000, n_features=50, random_state=fixed)` (primary) + a **Covertype subset** (real-data anchor). Regressor: `make_regression` analogue. `[CITED: ROADMAP Phase 2 study]`
- **Matchup:** Sylva ET ↔ sklearn `ExtraTrees{Classifier,Regressor}`; Sylva RF ↔ sklearn `RandomForest{Classifier,Regressor}`. **Never** ET-vs-RF (PITFALLS 13).
- **Identical hyperparameters** across both (`n_estimators`, `max_depth`, `max_features`, `min_samples_*`, `bootstrap`, `criterion`, `random_state`).
- **Cold vs warm** separated for the (informational) timing; sklearn `n_jobs=-1` for the timing baseline; **no end-to-end speed claim** (foundational phase).
- **Pinned versions** (sklearn, numpy, Python, Sylva commit) recorded.
- **Report:** accuracy/proba parity + KS results (the gate) and CPU `fit` wall-clock side-by-side (informational, SC-7).

## Determinism (SC-3, EST-07)

- **Per-tree Philox keying makes rayon tree-parallelism order-independent** — confirmed by design: each tree's draws are `Philox(seed, tree=t, ...)`, a pure function of `t`, so the parallel `par_iter` over trees cannot reorder draws. The order-independence invariant (parallel == sequential ForestIR) tests this. `[CITED: ARCHITECTURE.md Pattern 4]`
- **The one real f32 hazard — intra-tree reduction order:** computing a node's impurity / class counts / leaf mean by a **parallel** floating-point sum (e.g. rayon `par_iter().sum()`) is non-associative → not bit-reproducible. **Mitigation:** within a node, accumulate sequentially in a fixed row order, or use a canonical (fixed-shape) binary-tree reduction. Integer counts (class histograms, sample counts) are associative and safe to accumulate in any order; only **float sums** (Gini/entropy probabilities, MSE, leaf means) need fixed order. `[CITED: PITFALLS Pitfall 5; ARCHITECTURE.md Pattern 4]`
- **Recommendation for Phase 2:** parallelize **across trees** (rayon), but build each individual tree **single-threaded** with sequential, fixed-order float accumulation. This gives both throughput (forests are wide) and bit-reproducibility, and exactly mirrors what the GPU path must later guarantee. Deeper intra-tree parallelism + deterministic reductions is a Phase-6 concern (DET-01).
- **Tie-breaking:** when two splits have equal gain (or equal proxy impurity), break ties by **lowest `(feature_id, threshold)`** — a total order, never "whichever rayon task finished first." Document and test. `[CITED: ARCHITECTURE.md Pattern 4 fixed tie-breaking]`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense matrix + row views | Custom `Vec<f32>` + stride math | `ndarray` `Array2<f32>` / `ArrayView2` | Battle-tested slicing, the rust-numpy bridge for later zero-copy. |
| Tree-level parallelism | `std::thread` pool | `rayon` `par_iter` | Work-stealing, trivial `par_iter` over trees; order-independent given Philox keying. |
| Serialization | Custom binary format | `serde` + `serde_json` | Round-trip is a required invariant; JSON is also the Treelite-export substrate later. |
| Error types | `Box<dyn Error>` ad hoc | `thiserror` enum (the `CudaError` template) | Project convention; maps to Python exceptions; no `.unwrap()` on fallible paths. |
| Property tests | hand-written random loops | `proptest` | Shrinking finds minimal failing cases; project-pinned. |
| RNG | `rand` crate / `StdRng` | hand-rolled Philox-4×32-10 (~20 lines) | Must be stateless, counter-based, bit-identical CPU↔GPU, with documented KATs — no off-the-shelf crate gives all four. (This is the one place the project *prescribes* hand-rolling — CLAUDE.md.) |
| KS test / statistical parity | hand-rolled KS in Rust | `scipy.stats.ks_2samp` in the Python harness | scipy is the reference; the harness is Python anyway (sklearn lives there). |

**Key insight:** Almost everything except the Philox RNG and the tree algorithm itself is a library call. Hand-rolling is reserved for (a) the RNG (prescribed, for CPU↔GPU bit-identity) and (b) the ET/RF split logic (reimplemented from the algorithm for Apache-2.0 cleanliness — never copied from sklearn/GPL).

## Common Pitfalls

### Pitfall 1: Chasing bit-identical parity with sklearn
**What goes wrong:** Days sunk trying to make Sylva's trees match sklearn's exactly for a given seed.
**Why:** sklearn's serial `our_rand_r` PRNG draw order is un-replayable by a parallel counter-based RNG.
**Avoid:** Lock the contract as *distributional* (D-04). Sylva's bit oracle is its **own** CPU path, not sklearn. `[CITED: PITFALLS 6]`
**Warning sign:** A test asserts `tree_sylva == tree_sklearn` structurally.

### Pitfall 2: Silent NaN misrouting
**What goes wrong:** `NaN <= threshold` is `false` → all NaN rows silently go right, diverging from the `default_child` policy.
**Avoid:** Branch on `is_nan()` first, before the comparison. NaN fixtures in CI. `[CITED: PITFALLS 7]`

### Pitfall 3: Non-deterministic f32 accumulation under rayon
**What goes wrong:** Parallel float sums of impurity/leaf stats → bit-different models across runs; breaks seed-determinism + the future GPU==CPU gate.
**Avoid:** Parallelize across trees only; sequential fixed-order float accumulation within a tree. Integer counts are safe parallel. `[CITED: PITFALLS 5]`
**Warning sign:** Two same-seed runs score identically but aren't byte-identical.

### Pitfall 4: Under-designing the ForestIR (the near-rewrite risk)
**What goes wrong:** IR carries only train/predict fields; Phase 8 SHAP needs per-node cover counts, Phase 6/9 export needs Treelite fields → cross-backend IR rewrite (STATE.md flagged this).
**Avoid:** Carry `node_sample_count`, `node_weighted_count`, `gain` now (D-03). They're cheap at train time.

### Pitfall 5: Differing `max_features` defaults by task
**What goes wrong:** Using `sqrt` for the regressor (sklearn uses `1.0` = all features for regressors) → wrong split distribution → KS test fails.
**Avoid:** Branch defaults on clf (`sqrt`) vs reg (`1.0`). `[CITED: sklearn docs]`

### Pitfall 6: Copying sklearn/GPL source
**What goes wrong:** License contamination (project is Apache-2.0).
**Avoid:** Reimplement from the algorithm description (paper + docs + behavior); document provenance. `[CITED: PITFALLS Security; CLAUDE.md]`

## Code Examples

### Philox-4×32-10 round (Rust reference — reimplement, verify against KATs)
```rust
// Source: Random123 reference algorithm (DEShawResearch random123 philox.h).
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let p = a as u64 * b as u64;
    ((p >> 32) as u32, p as u32) // (hi, lo)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo32(M0, ctr[0]);
    let (hi1, lo1) = mulhilo32(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

pub fn philox4x32_10(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        ctr = round(ctr, key);
        key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
    }
    ctr
}

#[inline]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    // top 24 bits → [0,1); matches the CUDA-side conversion exactly.
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}
```
> NOTE: the exact placement of the key-bump relative to the round (bump-before vs bump-after the first round) must be tuned to reproduce the official KAT vectors — verify against `kat_vectors.txt` (§Philox KATs are `[ASSUMED]` pending that check).

### ET split: one random threshold per candidate feature
```rust
// Reimplemented from sklearn RandomSplitter semantics (NOT copied).
// For each of `max_features` candidate features at this node:
let (fmin, fmax) = feature_min_max(rows, feat, &x);        // over node's rows
if fmax <= fmin + FEATURE_THRESHOLD { continue; }          // constant feature → skip
let u = u32_to_unit_f32(philox_draw(seed, tree, node, feat, DRAW_THRESHOLD));
let threshold = fmin + u * (fmax - fmin);                  // uniform in (min,max)
// partition: x[i,feat] <= threshold → left (NaN handled separately → default_child)
let gain = proxy_impurity_improvement(/* left/right class or sum stats */);
// keep best gain across candidate features; tie-break by (feature_id, threshold).
```

### NaN-safe predict traversal
```rust
let v = x[[i, feature_id[node] as usize]];
node = if v.is_nan() {
    default_child[node]                 // D-01 policy, before any comparison
} else if v <= threshold[node] {
    left_child[node]
} else {
    right_child[node]
} as usize;
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| "sklearn-compatible = same accuracy" | Strict *distributional* parity (CI + KS on split stats) + own bit oracle | This project's contract | The oracle's value is trustworthiness; accuracy-only parity is insufficient (PITFALLS 6, D-04). |
| Stateful PRNG (Mersenne/xorshift) for parallel trees | Counter-based stateless RNG (Philox/Random123) | Standard in GPU ML (cuML, JAX) | Order-independent parallelism + CPU↔GPU bit-identity (ENG-06). |
| Minimal train/predict tree struct | Consumer-complete SoA IR (cover counts, Treelite fields) | D-03 design horizon | Avoids cross-phase IR rewrite (STATE.md near-rewrite risk). |

**Deprecated/outdated for this phase:**
- Per-node parallelism inside a single tree on CPU (non-deterministic f32 reductions) — defer; parallelize across trees instead.
- Trying to match sklearn's RNG — explicitly out (ENG-04).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Philox `philox4x32 10` KAT output vectors (the three literal output triples) | §Philox KATs | If wrong, the frozen Phase-4 oracle is wrong → CPU↔GPU "bit-match" validates against a bad reference. **MUST verify against official `kat_vectors.txt` (checkpoint).** Constants + round fn are VERIFIED; only the literal outputs are assumed. |
| A2 | Exact key-bump ordering (bump before/after round 0) | §Philox key schedule | Wrong ordering → all outputs differ. Resolved automatically by tuning to the verified KATs (A1). |
| A3 | Crate patch versions (ndarray 0.16.x, rayon 1.x, etc.) | §Standard Stack | Low — major lines are stable; `cargo add` resolves latest compatible. Offline this session. |
| A4 | KS p-value > 0.05 and accuracy CI width as parity thresholds | §Parity Test Design | Medium — these are starting points; must be empirically calibrated against the sklearn-vs-sklearn null spread before gating. |
| A5 | `n_estimators ≥ 200`, ≥10 seeds gives stable KS distributions | §Parity Test Design | Medium — may need more trees/seeds for tight features; calibrate during the parity task. |
| A6 | sklearn regressor `max_features` default is `1.0` (all); classifier `sqrt` | §sklearn Specifics | Low-Medium — verified against docs in spirit; planner should confirm against the exact targeted sklearn version (1.7/1.8). |
| A7 | ENG-01 wording is satisfied by Option-A trait split (high-level `Backend` + separate `HistogramBackend`) | §Pattern 1 | Medium — if the user requires all five op-names on a single trait, use Option B. Flag for discuss/plan. |
| A8 | Treelite v4 maps `default_left` (bool) from Sylva's `default_child` (node id) | §ForestIR | Low — export is Phase 6/9; only needs the *data* present now. Exact mapping pinned then. |

**If this table is empty:** it is not — A1 and A4/A5/A7 are the load-bearing ones for the planner to gate.

## Open Questions

1. **Trait granularity (ENG-01 literal reading).**
   - Known: ENG-01 says the trait "defines all device ops (quantize, build_histograms, eval_splits, partition, predict)" with no CUDA types crossing.
   - Unclear: whether the CPU oracle's `fit`/`predict` must be expressed *through* those five op-names on one trait (Option B), or whether a high-level `Backend{fit,predict}` + a separate future `HistogramBackend{quantize,...}` (Option A) satisfies it.
   - Recommendation: **Option A** (cleaner CPU path; CUDA-free via associated type). Surface to discuss-phase / user for a one-line confirmation.

2. **Minimal Python seam for the parity harness.**
   - Known: full estimator API (`fit`/`predict_proba`/`check_estimator`) is Phase 5.
   - Unclear: whether exposing a minimal `fit_cpu`/`predict_cpu`/`split_statistics` PyO3 surface this phase (purely for the parity harness) is in-scope, or whether the harness should call into a Rust test binary instead.
   - Recommendation: minimal PyO3 accessor for the harness (sklearn lives in Python); keep it clearly *below* EST-02's bar and documented as test-only.

3. **Exact parity thresholds & sample sizes (A4/A5).**
   - Known: D-04 mandates "tight CI" + "KS test," researcher to pick values.
   - Unclear: the defensible numbers depend on the sklearn-vs-sklearn null spread on the chosen datasets.
   - Recommendation: a **calibration task** that measures sklearn-vs-sklearn variance first, then sets Sylva's bar to that null band (rather than a guessed epsilon).

4. **Philox KAT verification (A1/A2).**
   - Known: constants + round verified; literal KAT outputs assumed.
   - Recommendation: `checkpoint:human-verify` the three KAT vectors against the official Random123 `kat_vectors.txt` before freezing them.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All Rust work | ✓ | 1.83+ (pinned `rust-toolchain.toml`) | — |
| ndarray/rayon/serde/proptest/approx | CpuBackend + tests | ✓ (via `cargo add`) | 0.16/1/1/1/0.5 | — |
| Python + scikit-learn + scipy + numpy | Parity harness (KS, sklearn oracle) | ✓ (Python 3.14 present per Phase 1) | sklearn 1.7/1.8 (pin) | — |
| maturin | build the test wheel for the harness | ✓ (Phase 1) | 1.14.1 | Rust-only fixtures cover proptest invariants without sklearn |
| CUDA / GPU | **NOT required this phase** | n/a | — | Entire phase is CPU-only → GPU-less cloud CI works (Pitfall 16 mitigation realized). |

**Missing dependencies with no fallback:** none — Phase 2 is deliberately CPU-only.
**Missing dependencies with fallback:** sklearn/scipy needed only for the distributional harness; the proptest invariants and Rust unit tests run with no Python at all (the bulk of correctness coverage is CPU-only, GPU-less, and Python-less).

## Validation Architecture

> nyquist_validation assumed enabled (no `.planning/config.json` `workflow.nyquist_validation: false` found). `[ASSUMED]` — planner verify config.

### Test Framework
| Property | Value |
|----------|-------|
| Framework (Rust) | `cargo test` + `cargo-nextest`; `proptest` for invariants; `approx` for float asserts |
| Framework (Python harness) | `pytest` + `scipy.stats.ks_2samp` + scikit-learn |
| Config file | `crates/sylva-core/Cargo.toml` `[dev-dependencies]`; `python/tests/parity/conftest.py` (Wave 0) |
| Quick run command | `cargo test -p sylva-core` |
| Full suite command | `cargo nextest run -p sylva-core && pytest python/tests/parity` |

### Phase Requirements / Success Criteria → Test Map
| Req / SC | Behavior | Test Type | Automated Command | File Exists? |
|----------|----------|-----------|-------------------|-------------|
| ENG-01 / SC-1 | `trait Backend` device-neutral; no CUDA types | unit (compile-time: crate has no cuda dep) + doc test | `cargo test -p sylva-core --doc` + `cargo tree -p sylva-core \| grep -vi cudarc` | ❌ Wave 0 |
| ENG-02 / SC-1 | SoA ForestIR fields present, write-once | unit + serialization round-trip (proptest) | `cargo test -p sylva-core ir::` | ❌ Wave 0 |
| ENG-03 / SC-2 | CpuBackend trains+predicts ET+RF (clf+reg) correctly | unit per estimator + differential | `cargo test -p sylva-core cpu::` | ❌ Wave 0 |
| ENG-04 / SC-3 | Parity contract documented; own bit-determinism | doc deliverable + byte-identical determinism test | `cargo test -p sylva-core determinism::seed_byte_identical` | ❌ Wave 0 |
| ENG-05 / SC-4 | NaN default-direction routing | unit + NaN fixtures | `cargo test -p sylva-core predict::nan_routing` | ❌ Wave 0 |
| ENG-06 / SC-3 | Philox-4×32-10 + KAT vectors | unit (KAT bit-match) | `cargo test -p sylva-core rng::kat` | ❌ Wave 0 |
| EST-07 / SC-5 | child-partition, leaf-prob, seed-determinism, round-trip | proptest invariants | `cargo test -p sylva-core invariants::` | ❌ Wave 0 |
| SC-5 | differential vs sklearn (accuracy/distribution) | parity (Python) | `pytest python/tests/parity/test_distributional_parity.py` | ❌ Wave 0 |
| SC-6 | Comparative Study: accuracy/distribution PARITY | parity-statistic (CI + KS) | `pytest python/tests/parity -k parity` | ❌ Wave 0 |
| SC-7 | CPU training time reported (not gated) | benchmark (informational) | `pytest python/tests/parity -k timing` (reports, no assert-gate) | ❌ Wave 0 |
| (invariant) | cover-count consistency (`parent == L + R`) | proptest | `cargo test -p sylva-core invariants::sample_count` | ❌ Wave 0 |
| (invariant) | rayon parallel == sequential ForestIR | proptest | `cargo test -p sylva-core determinism::order_independent` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p sylva-core` (fast Rust unit + proptest subset).
- **Per wave merge:** `cargo nextest run -p sylva-core && pytest python/tests/parity -k "not timing"`.
- **Phase gate:** full suite green (incl. parity CI + KS) + parity-study report before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/sylva-core/` crate + `Cargo.toml` (`[dev-dependencies] proptest, approx`) — no test infra exists yet (new crate).
- [ ] `crates/sylva-core/src/rng/kat.rs` — Philox KAT vectors (covers ENG-06) — **gated on KAT verification checkpoint (A1)**.
- [ ] `python/tests/parity/conftest.py` + `datasets.py` — sklearn/scipy harness + dataset loaders (covers SC-5/6/7).
- [ ] Minimal PyO3 test seam (`fit_cpu`/`predict_cpu`/`split_statistics`) for the harness — pending Open Question 2.
- [ ] Calibration fixture: sklearn-vs-sklearn null-spread baseline (sets the CI/KS thresholds, A4/A5).
- [ ] NaN fixtures (`predict::nan_routing`) — covers SC-4/ENG-05.

## Security Domain

> `security_enforcement` assumed enabled. This is a numerical library (low network surface); the relevant controls are input validation at the boundary and license hygiene.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | Validate `X` shape/dtype/finiteness expectations + `TrainConfig` params at the boundary (parse-don't-validate, newtype). Reject malformed configs with typed `SylvaError` (thiserror), never panic. `[CITED: rules/rust/security.md; PITFALLS Security]` |
| V6 Cryptography | no | Philox is a *statistical* RNG, **not** cryptographic — document it as non-CSPRNG so no one mistakes it for secure randomness. |

### Known Threat Patterns for {pure-Rust numeric core}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unvalidated array shapes/dtypes → OOB/panic | Tampering / DoS | Validate shape/dtype/contiguity at the (future) Python↔Rust boundary; for Phase 2 validate `ArrayView` dims + config ranges before training. |
| NaN/Inf silently corrupting splits | Tampering (data integrity) | Explicit `is_nan` handling (D-01); finite-leaf invariant. |
| License contamination (copying sklearn/GPL) | (Compliance) | Reimplement from algorithm; document provenance; Apache-2.0 only. `[CITED: CLAUDE.md; PITFALLS Security]` |
| `unwrap()`/`expect()` panicking across the FFI later | DoS | `Result` + `thiserror` everywhere; the `sylva-cuda::CudaError` no-`.unwrap()` convention is the template. |

## Project Constraints (from CLAUDE.md)

- **Apache-2.0 only** — reimplement algorithms from papers/docs; never copy sklearn/GPL/Snap ML source. Document provenance.
- **f32 dense MVP** — CpuBackend + ForestIR compute/store f32 (D-05).
- **Many small files** — 200–400 lines typical, 800 max; organize by domain (split_et / split_rf / criterion / predict separate).
- **No hardcoded values** — constants (Philox multipliers, FEATURE_THRESHOLD, default bin/param values) named, not inline-littered; config-driven.
- **thiserror error enums, no `.unwrap()` on fallible paths** — `sylva-cuda::CudaError` is the template for a `SylvaError`.
- **No silent fallback** — surface every error honestly (relevant later for dispatch; here it means typed errors on bad input/config).
- **GSD workflow** — file changes go through a GSD command; this is research only (no edits beyond RESEARCH.md).

## Sources

### Primary (HIGH confidence)
- `github.com/DEShawResearch/random123` `include/Random123/philox.h` — Philox-4×32-10 constants (M0/M1/W0/W1), `mulhilo32`, round function, rounds=10 (VERIFIED).
- `github.com/scikit-learn/scikit-learn` `sklearn/tree/_splitter.pyx` — RandomSplitter (`rand_uniform(min,max)` random threshold) + BestSplitter (sorted midpoint candidates), `FEATURE_THRESHOLD`, `min_samples_leaf`/`min_weight_leaf` enforcement (VERIFIED).
- `treelite.readthedocs.io/en/latest/serialization/v4.html` — Treelite v4 SoA node arrays: `node_type, cleft, cright, split_index, default_left, threshold, leaf_value/leaf_vector, data_count(+present), sum_hess(+present), gain(+present)`, tree-level `num_nodes` (VERIFIED).
- Project research: `.planning/research/ARCHITECTURE.md` (trait Backend Pattern 1, SoA IR Pattern 3, determinism Pattern 4), `PITFALLS.md` (Pitfalls 5/6/7/13/16), `STACK.md` (ndarray/rayon/proptest/approx/Philox prescription), `STATE.md` (near-rewrite IR risk, fairness protocol) (HIGH — project-internal, cross-referenced).

### Secondary (MEDIUM confidence)
- `scikit-learn.org/stable` ExtraTreesClassifier / RandomForest docs — random-split mechanism, `max_features` defaults (sqrt clf / 1.0 reg), bootstrap behavior, leaf probabilities (CITED).
- SHAP docs / DeepWiki TreeExplainer — path-dependent TreeSHAP uses per-node training sample counts (cover) as background distribution → justifies `node_sample_count`/`node_weighted_count` in the IR (CITED).
- `sklearn/utils/_random.pyx` — `our_rand_r` serial xorshift PRNG (confirms un-replayability → distributional contract) (CITED).

### Tertiary (LOW confidence — flagged)
- Philox `philox4x32 10` literal KAT output vectors — from training knowledge; canonical `kat_vectors.txt` fetch 404'd. **Verify before freezing (A1).**
- Exact crate patch versions — offline `cargo search`; resolve via `cargo add` (A3).
- KS p-value / accuracy-CI thresholds, tree/seed counts — defensible starting points; calibrate empirically (A4/A5).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — project-pinned, mature ecosystem crates; patch versions to confirm via `cargo add`.
- Architecture (trait shape, SoA IR, determinism): HIGH — derived from project ARCHITECTURE.md + verified sklearn/Treelite/SHAP specifics.
- Philox algorithm: HIGH for constants/round/conversion (verified header); LOW for the literal KAT outputs (pending checkpoint).
- sklearn parity algorithm: HIGH — verified against `_splitter.pyx`.
- Parity thresholds: MEDIUM — defensible defaults; require empirical calibration.

**Research date:** 2026-06-20
**Valid until:** ~2026-07-20 (stable domain; sklearn algorithm + Philox + Treelite schema are slow-moving). Re-check crate patch versions and the targeted sklearn version at plan time.
