# Phase 3: Feature Quantizer (CPU/GPU Bit-Parity) - Research

**Researched:** 2026-06-26
**Domain:** Quantile feature binning (SoA `BinnedMatrix`) with a CPU↔GPU bit-identical-by-construction parity contract; Rust core (`sylva-core`), no CUDA this phase
**Confidence:** HIGH (parity mechanism, dtype/layout, numpy baseline); MEDIUM (sklearn `_BinMapper` exact schema — version-pinned but distributional-only by design)

## Summary

This phase builds a standalone, separately-testable **quantile quantizer** in `sylva-core` that turns the dense f32 matrix into a SoA `BinnedMatrix` (uint8 for ≤256 bins, uint16 above), and freezes a **bit-parity contract** (golden bin-assignment fixtures on a fixed seed) that the Phase-4 GPU assignment kernel must reproduce exactly. The discussion (CONTEXT.md) already locked the load-bearing decisions; research's job is to *pin the exact algorithm* so the binning is correct against the `numpy.quantile` baseline and the parity contract is bit-precise.

The single most important finding: the **edges-on-host / GPU-assigns-only** design (D-04) reduces the hard parity problem to a pure IEEE-754 `<=` comparison of an f32 value against an f32 edge. Such a comparison has **no rounding, no FMA, no accumulation/reduction** — the only operations that cause CPU↔GPU float divergence — so bin assignment is **bit-identical by construction**, not by chasing. This is verified against NVIDIA's IEEE-754 GPU documentation. The real (and only) landmines are: (a) computing the edges deterministically on the host, (b) getting the boundary/`searchsorted` side exactly right and *identical* on both devices, and (c) NaN handling, which must be checked *before* any comparison (NaN compares are always false → silent misrouting).

The second important finding is a **baseline-comparison subtlety**: D-03 chose "edges = quantile values directly," which anchors correctness to `numpy.quantile`. sklearn's `_BinMapper` does NOT use raw quantiles — it uses **midpoints between consecutive distinct values when distinct ≤ max_bins**, raw `np.percentile(method="averaged_inverted_cdf")` otherwise, then `np.unique` dedupe. It also uses the **opposite boundary convention** to D-06. Therefore the sklearn comparison is **distributional-only** (agreement %, not edge-exact), exactly as CONTEXT.md anticipated, and `numpy.quantile` is the edge-exact oracle.

**Primary recommendation:** Implement a per-feature exact-quantile edge builder using `numpy.quantile`-equivalent `method="linear"` interpolation, dedupe with sorted-unique, store edges as `Vec<f32>` per feature (jagged, since dedupe yields variable lengths); assign bins with `partition_point`/`searchsorted side='right'` semantics (`bin = count of edges ≤ x`); reserve the highest bin index for NaN; and ship golden-vector fixtures `(seed, X-fixture) → bin_u8[]` as the Phase-4 contract. Make `numpy.quantile` the edge-exact gate and `_BinMapper` a distributional check.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Quantile edge construction (sort/quantile) | Rust core / CPU host (L2/L3) | — | D-04: edges computed ONCE on host; never on GPU. Deterministic sequential sort. |
| Bin assignment (`searchsorted`) | Rust core / CPU (this phase) → CUDA kernel (Phase 4) | — | Pure compare; the only op that crosses to GPU later. Bit-identical by construction. |
| `BinnedMatrix` SoA storage | Rust core data structure | GPU device buffer (Phase 4) | SoA column-major for GPU coalescing; written once (quantize-once). |
| Real-valued threshold table (export) | Rust core data structure | — | Stored alongside binned edges now; hand-off to Phase 9 Treelite export. |
| `execution_report_` quantize record | Rust core struct | — | Minimal dtype/contiguity/bytes; Phase 6 extends into full report. |
| Baseline parity test + microbench | Python test harness (pytest) | — | numpy/sklearn are test-only oracles, never runtime deps. |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Quantile edge construction**
- **D-01 (Claude's discretion):** Exact full-data quantiles. Edges computed from the entire feature column via exact percentile/sort — no subsampling, no RNG this phase. Subsampling (with Philox) deferred as a profiling-driven optimization.
- **D-02 (USER-DECIDED):** Collapse duplicate edges to unique (sklearn `_BinMapper`-style `np.unique` dedupe). Low-cardinality feature gets fewer effective bins than nominal.
- **D-03 (USER-DECIDED):** Edges = quantile values directly (NOT midpoints). Anchors binning to `numpy.quantile`; the sklearn `_BinMapper` comparison (which uses midpoints) is therefore distributional-only, not edge-exact.

**CPU↔GPU bit-parity mechanism**
- **D-04 (USER-DECIDED):** Edges-on-host, GPU assigns-only. Quantile edge construction runs only on CPU/host; edges uploaded to GPU; GPU performs only the searchsorted comparison. Both devices do the identical IEEE-754 compare against the same f32 edges → bit-identical by construction. Eliminates the parallel-GPU-sort parity risk.
- **D-05 (USER-DECIDED — scope narrowing, flagged):** GPU quantize kernel deferred to Phase 4. Phase 3 delivers the CPU quantizer + a bit-parity test harness (golden bin assignments on a fixed seed) that the Phase-4 GPU kernel must reproduce exactly.
  - ⚠ ROADMAP SC-2 implication: For THIS phase, SC-2 is satisfied as "CPU quantizer + enforced bit-parity contract/golden vectors"; the live CPU↔GPU CI proof moves to Phase 4. The verifier/planner must NOT read SC-2 as fully unmet, nor as fully met with a live GPU run — it is met as a **contract** here. QUANT-02's live half lands Phase 4.
- **D-06 (Claude's discretion):** Boundary rule = `side='right'` — bin index = number of edges ≤ x (a value exactly equal to an edge goes to the **higher** bin). Single documented rule used identically on CPU and GPU; required to match the Phase-4 histogram split comparison. Out-of-range values clamp into the first/last bin via searchsorted.

**Bin count, dtype & missing-value bin**
- **D-07 (Claude's discretion):** 255 data bins + 1 reserved missing bin = 256 total (matches sklearn HGBT `max_bins=255`, fits uint8, within the 128–256 shared-mem budget). Internal quantizer config field (default 255); public sklearn `max_bins` param waits for Phase 5 (EST-03).
- **D-08 (USER-DECIDED):** Auto dtype selection — `n_bins ≤ 256 → uint8`, `> 256 → uint16`. Both code paths exist and are tested; default exercises uint8.
- **D-09 (USER-DECIDED):** Reserved missing bin (sklearn-style, highest index — the `+1` of `255+1`). NaN/missing → that bin. Split-time routing reuses the Phase-2 D-01 default-child rule (higher-sample-count child; tie → left). Out-of-range clamps as in D-06.

**execution_report_ hook scope**
- **D-10 (Claude's discretion):** Minimal quantize-step record only — input dtype, contiguity/layout handling, byte size — into a small struct that Phase 6 (DET) extends. No backend-selection/fallback surface this phase.
- **D-11 (Claude's discretion):** Record computed payload size (byte size of `BinnedMatrix` + edges = the payload that *will* transfer H2D) and explicitly mark H2D as not-executed / N/A this phase.

**Comparative Baseline Study scope**
- **D-12 (Claude's discretion):** CPU-vs-baseline this phase. Correctness parity + quantize-throughput microbench (rows/s) of Sylva's CPU quantizer vs `numpy.quantile` and sklearn `_BinMapper`; GPU quantize throughput reported in Phase 4. Op-level number, no end-to-end algorithm speed claim.

### Claude's Discretion (delegated / flagged for research)
- Exact percentile interpolation method for D-01 (numpy `linear` vs `lower`) and the precise `searchsorted`-side numpy alignment for D-06 — pin against the `numpy.quantile` baseline so the correctness comparison is apples-to-apples. **(RESOLVED below — see Pitfall 1 & Code Examples.)**
- Exact correctness tolerance for the binning-parity gate (bin-assignment agreement %) and the throughput-microbench dataset details. **(RESOLVED — see Validation Architecture.)**
- `BinnedMatrix` concrete SoA layout + how the binned-threshold table and the parallel real-valued threshold table are stored together. **(RESOLVED — see Architecture Patterns.)**
- The concrete shape of the bit-parity golden-vector fixtures (D-05) that Phase-4 will assert against. **(RESOLVED — see Architecture Patterns / Validation.)**

### Deferred Ideas (OUT OF SCOPE)
- Live GPU quantize kernel + live CPU↔GPU CI parity run + GPU throughput microbench → Phase 4 (validated against this phase's golden-vector contract).
- Subsampled quantile edges (with Philox) → profiling-driven optimization later (D-01).
- Public sklearn `max_bins` constructor param → Phase 5 (EST-03); internal config field only here.
- Full `execution_report_` surface, `device`/`fallback="error"` dispatch → Phase 6 (DET-*).
- Learned/optimal missing-value direction → still deferred from Phase 2 D-01.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUANT-01 | A feature quantizer produces a SoA `BinnedMatrix` (uint8/uint16) via per-feature quantile bins, on both CPU and GPU | CPU half built this phase: exact-quantile edge builder + SoA `BinnedMatrix` with auto dtype (D-08). GPU half is Phase 4 — but the parity contract (golden vectors) is the binding hand-off so the GPU kernel is testable on arrival. See Standard Stack, Architecture Patterns, Code Examples. |
| QUANT-02 | CPU and GPU quantizers produce bit-identical bin assignments on a fixed seed (parity test in CI) | Satisfied this phase as a **contract**: edges-on-host/GPU-assigns-only (D-04) makes assignment bit-identical by construction (IEEE-754 pure compare — verified vs NVIDIA FP docs); golden-vector fixtures encode the expected bins; CI asserts CPU output == golden. Live CPU↔GPU run lands Phase 4. See Pitfall 2 & Validation Architecture. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ndarray` | 0.16.x (already pinned) | f32 matrix views (`ArrayView2<f32>`) input to `quantize` | Established Phase-2 host array type; the `HistogramBackend::quantize` seam already takes `ArrayView2<f32>`. |
| `rayon` | 1.x (already pinned) | Parallelize per-feature edge construction across columns | Per-column quantile build is embarrassingly parallel; rayon is the established Phase-2 CPU-parallel tool. **Order-independence note:** parallelize *across features* (each column independent), never *within* a column's sort/quantile (keep that sequential for determinism). |
| `thiserror` (`SylvaError`) | workspace | Typed quantizer errors (all-NaN column, empty matrix, bin count out of range) | Established convention — no `.unwrap()` on fallible paths, no panics across the boundary. |
| `serde` / `serde_json` | 1.x (already pinned) | Serialize the golden-vector parity fixtures and (optionally) the quantizer model | Same path used by `ForestIR` and the Phase-5 parity harness. |

### Supporting (test/dev only — never runtime deps)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `proptest` | 1.x (already a dev-dep) | Property invariants: bins ∈ [0, n_bins); monotonic (x1 ≤ x2 ⇒ bin1 ≤ bin2); NaN → missing bin; round-trip edges | Rust unit/property tests. |
| `approx` | 0.5.x (already a dev-dep) | f32 tolerance assertions where exactness isn't required (e.g. edge-value distributional checks vs sklearn) | Rust tests. |
| `numpy` (Python) | 2.4.2 (installed; **pin in test harness**) | The **edge-exact** baseline oracle (`numpy.quantile`, `np.searchsorted`) | Python pytest baseline-parity test (D-12). |
| `scikit-learn` (Python) | 1.8.0 (installed; **pin in test harness**) | The **distributional** baseline (`_BinMapper`) | Python pytest distributional check (D-12). |
| `pytest` + `pytest-benchmark` (or `time.perf_counter`) | latest | Throughput microbench harness (rows/s) | D-12 op-level microbench. `pytest-benchmark` gives repeated-run statistics for free; plain `perf_counter` is sufficient if avoiding a new dep. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled quantile in Rust | A quantile crate (e.g. `quantiles`, `tdigest`) | **Reject.** Approximate-quantile crates (t-digest, CKMS) deliberately trade exactness for streaming memory — incompatible with D-01's "exact full-data quantiles" and bit-parity. The exact algorithm (sort + linear interpolation) is ~30 lines and must match `numpy.quantile` precisely; a third-party crate that doesn't replicate numpy's `linear` interpolation would *break the baseline gate*. |
| `Vec<Vec<f32>>` jagged edges | Flat `Vec<f32>` + `offsets: Vec<usize>` (CSR-style) | Flat-with-offsets is more GPU-upload-friendly (one contiguous buffer for Phase 4) and avoids per-feature allocation. **Recommended** — see Architecture Patterns. |
| `partition_point` (std) | manual binary search | `slice::partition_point` IS a binary search returning the `side='right'` count directly; use it. No need to hand-roll. |

**Installation:** No new Rust crates required — all core deps already pinned in `crates/sylva-core/Cargo.toml`. Python baseline deps (`numpy`, `scikit-learn`, optionally `pytest-benchmark`) live in the test harness only (e.g. a `requirements-dev.txt` or the existing parity-harness environment), pinned to the versions below.

**Version verification (performed this session):**
```
numpy        2.4.2   (installed, confirmed via python -c import)
scikit-learn 1.8.0   (installed, confirmed via python -c import)
rustc        1.96.0  (installed; MSRV floor 1.83 satisfied)
ndarray 0.16, rayon 1, serde 1, thiserror, proptest 1, approx 0.5  (already in Cargo.toml)
```

## Package Legitimacy Audit

> No NEW external packages are installed by this phase. All Rust deps are already vendored/pinned in `crates/sylva-core/Cargo.toml` and were legitimacy-audited in Phase 2. Python baseline libraries are test-only oracles, not shipped dependencies.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| ndarray | crates.io | mature (8+ yrs) | very high | github.com/rust-ndarray/ndarray | OK | Already pinned (Phase 2) |
| rayon | crates.io | mature | very high | github.com/rayon-rs/rayon | OK | Already pinned (Phase 2) |
| numpy (py) | PyPI | mature | very high | github.com/numpy/numpy | OK | Test-only oracle |
| scikit-learn (py) | PyPI | mature | very high | github.com/scikit-learn/scikit-learn | OK | Test-only oracle |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram (quantize data flow)

```
   X: ArrayView2<f32>  (n_rows × n_features, dense)
         │
         │  [boundary validation: shape, dtype f32, contiguity → record in QuantizeReport]
         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  EDGE CONSTRUCTION  (host only, D-04)                          │
   │  per feature f (rayon-parallel across features):              │
   │    1. gather finite values of column f (skip NaN)             │
   │    2. sort ascending (sequential — determinism)               │
   │    3. quantile cut points q = linspace(0,1,n_data_bins)[1:-1] │
   │       via numpy-'linear' interpolation                        │
   │    4. dedupe to sorted-unique  → edges_f  (≤ n_data_bins-1)   │
   └──────────────────────────────┬───────────────────────────────┘
                                   │  BinEdges  { flat: Vec<f32>, offsets: Vec<usize>,
                                   │              real_values: Vec<f32> (export table) }
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  BIN ASSIGNMENT  (the ONLY op the GPU will replicate, Phase 4)│
   │  per cell (row r, feature f):                                 │
   │    v = X[r,f]                                                 │
   │    if v.is_nan() → bin = MISSING_BIN_IDX (highest)  [check 1st]│
   │    else bin = partition_point(edges_f, |e| e <= v)            │
   │              = searchsorted(edges_f, v, side='right')  (D-06) │
   │              (clamps to [0, n_edges] → in-range by construction)│
   └──────────────────────────────┬───────────────────────────────┘
                                   ▼
   BinnedMatrix { data: Vec<u8|u16> (SoA, column-major),
                  n_rows, n_features, n_bins, dtype, missing_bin_idx }
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
   Phase-4 histogram kernels                         Golden-vector fixtures
   (read bins, never f32 again)                      (seed, X) → expected bin_u8[]
                                                      = the QUANT-02 parity contract
```

*The diagram shows data flow, not files. File mapping is in the structure block below.*

### Recommended Project Structure

```
crates/sylva-core/src/
└── quantize/                    # new module (many small files, per coding-style)
    ├── mod.rs                   # public surface: Quantizer, QuantizeConfig, re-exports
    ├── edges.rs                 # exact-quantile edge construction (numpy-linear), dedupe
    ├── binned_matrix.rs         # SoA BinnedMatrix (uint8/uint16), Dtype enum, accessors
    ├── assign.rs                # searchsorted side='right' bin assignment + NaN routing
    ├── report.rs                # QuantizeReport (minimal execution_report_ record, D-10/11)
    └── parity.rs                # golden-vector fixture struct + (de)serialize (D-05 contract)
```

- Replace the `backend.rs` placeholder `BinEdges` (currently `struct BinEdges;`) with a real type, OR define a concrete `BinEdges` in `quantize/edges.rs` and have the CPU `HistogramBackend::Bins` impl bind to `BinnedMatrix`. **Note:** the `HistogramBackend` trait's `quantize(&self, x) -> (Self::Bins, BinEdges)` returns the *placeholder* `BinEdges` from `backend.rs` — decide whether to (a) flesh out that struct or (b) keep `quantize` standalone (not via the trait) this phase, since CpuBackend does NOT implement `HistogramBackend` (it trains by exact splitting). **Recommendation:** build the quantizer as a standalone `Quantizer` component (matching the Phase-2 note that "binning is a standalone, separately-testable component"); wire it into the trait's `Bins` seam in Phase 4 when the GPU `HistogramBackend` impl appears. This avoids forcing CpuBackend to implement a trait it doesn't need.

### Pattern 1: Flat-with-offsets `BinEdges` (CSR-style jagged edges)

**What:** Because dedupe (D-02) makes per-feature edge counts variable, store edges as one flat `Vec<f32>` plus a `Vec<usize>` offset array (length `n_features + 1`), CSR-style. Feature `f`'s edges are `flat[offsets[f]..offsets[f+1]]`.
**When to use:** Always here — it is one contiguous buffer (single H2D upload in Phase 4), avoids `n_features` heap allocations, and `partition_point` works directly on each sub-slice.
**Example:**
```rust
// Source: derived from ARCHITECTURE.md SoA principle + numpy searchsorted semantics
pub struct BinEdges {
    /// All per-feature edges concatenated, each sub-range sorted ascending, deduped.
    pub flat: Vec<f32>,
    /// Length n_features+1; feature f's edges = flat[offsets[f]..offsets[f+1]].
    pub offsets: Vec<usize>,
    /// Parallel real-valued threshold table for Treelite export (Phase 9).
    /// Same layout; here identical to `flat` because D-03 uses quantile values
    /// directly as edges, but kept separate so a future binned/real split is clean.
    pub real_values: Vec<f32>,
    pub n_features: usize,
}
impl BinEdges {
    #[inline]
    pub fn feature(&self, f: usize) -> &[f32] { &self.flat[self.offsets[f]..self.offsets[f + 1]] }
}
```

### Pattern 2: SoA column-major `BinnedMatrix` for GPU coalescing

**What:** Store bins **column-major** (feature-contiguous): all of feature 0's rows, then feature 1's, etc. A histogram kernel processes one feature at a time across rows, so column-major makes consecutive threads read consecutive addresses → coalesced loads on GPU.
**When to use:** Always — ARCHITECTURE.md's quantize-once + shared-mem-residency design assumes the histogram reads bins coalesced. (Row-major would scatter a feature's values across the buffer and de-coalesce the GPU read in Phase 4.)
**Trade-off:** The input `X` is row-major (C-contiguous ndarray); the quantizer transposes during assignment. That transpose is the natural place to also record contiguity handling in `QuantizeReport`.
**Example:**
```rust
// Source: ARCHITECTURE.md "BinnedMatrix(SoA uint8/16)" + coalescing rationale
pub enum BinDtype { U8, U16 }
pub struct BinnedMatrix {
    /// Column-major: index(row, feat) = feat * n_rows + row.
    pub data_u8:  Option<Vec<u8>>,   // populated when n_bins <= 256 (D-08)
    pub data_u16: Option<Vec<u16>>,  // populated when n_bins  > 256 (D-08)
    pub n_rows: usize,
    pub n_features: usize,
    pub n_bins: usize,            // total incl. missing bin (e.g. 256)
    pub dtype: BinDtype,
    pub missing_bin_idx: u32,     // = n_bins - 1 (D-09)
}
```

### Pattern 3: Golden-vector parity fixture as the QUANT-02 contract (D-05)

**What:** A serialized fixture: `{ seed, n_bins, X (small, with NaNs), expected_bins (column-major uint8), edges }`. Phase 3 CI asserts the CPU quantizer reproduces `expected_bins` exactly. Phase 4 CI asserts the GPU kernel reproduces the *same* `expected_bins`. The fixture is the literal hand-off.
**When to use:** This is the deliverable that satisfies SC-2 "as a contract."
**Shape recommendation:** Include at least: (1) a generic dense block, (2) a low-cardinality column (exercises dedupe → fewer bins), (3) a column with values exactly equal to edge values (exercises the D-06 boundary), (4) a column with NaNs (exercises the reserved missing bin), (5) out-of-range values (predict-time clamp). Store as JSON via `serde` (consistent with the Phase-5 harness pattern in `parity.rs`).

### Anti-Patterns to Avoid

- **Recomputing edges on the GPU (D-04 violation):** Any plan that sorts/quantiles on the device reintroduces the parallel-reduction-order parity risk this design was chosen to eliminate. Edges are host-only, always.
- **Comparing NaN against an edge:** `NaN <= edge` is `false` in IEEE-754 → NaN silently lands in bin 0. **Always branch `is_nan()` first** (mirrors the established `predict.rs` convention exactly).
- **Float `BinnedMatrix` (storing f32 bins):** defeats the quantize-once bandwidth win; bins MUST be uint8/uint16 (QUANT-01).
- **Parallelizing the per-column sort/quantile with a non-deterministic reduction:** keep each column's quantile computation sequential; only parallelize *across* columns.
- **Using sklearn `_BinMapper` as the edge-exact oracle:** its midpoint + averaged-inverted-cdf + opposite-side semantics will *never* match D-03/D-06 edges exactly. It is a *distributional* check only (see Pitfalls).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| `searchsorted side='right'` | Custom branchy binary search | `slice::partition_point(\|&e\| e <= v)` (std) | Returns the exact `side='right'` count; correct boundary handling; no off-by-one. |
| Quantile interpolation | Ad-hoc "pick the k-th sorted element" | Replicate numpy `method="linear"`: `j,g = floor/frac of q*(n-1)`; `out = a[j] + g*(a[j+1]-a[j])` | The baseline gate compares against `numpy.quantile`; only the *exact* numpy formula passes it. A naive nearest-rank quantile silently fails the parity gate. |
| Sorted-unique dedupe | Manual loop with epsilon | sort then `Vec::dedup()` (exact bit equality, NOT epsilon) | sklearn/numpy dedupe is exact-value; an epsilon dedupe diverges and is non-deterministic. |
| Parallel-across-columns | Manual thread pool | `rayon` `par_iter` over features | Established; each column is independent so no shared-state hazard. |

**Key insight:** Quantile binning *looks* trivial but every "obvious" shortcut (nearest-rank quantile, epsilon dedupe, comparison-before-NaN-check, midpoint-vs-raw confusion) silently breaks either the numpy baseline gate or the CPU↔GPU parity contract. Pin to the exact numpy formulas and pure-compare assignment.

## Runtime State Inventory

> Not a rename/refactor/migration phase — this is greenfield (new `quantize/` module). The only "inventory" concern is the existing `backend.rs` placeholder `BinEdges`/`type Bins` seam, addressed under Recommended Project Structure. No stored data, live-service config, OS-registered state, secrets, or stale build artifacts are touched.

**Nothing found in any runtime-state category** — verified: this phase adds a new module and test fixtures only; it does not rename or migrate any persisted state, service config, or installed artifact.

## Common Pitfalls

### Pitfall 1: Quantile interpolation / boundary semantics that don't match the numpy baseline (the apples-to-oranges trap)

**What goes wrong:** The CPU quantizer "works" but the Comparative Baseline Study reports <100% agreement with `numpy.quantile` because the interpolation method or the `searchsorted` side differs by one element at boundaries — and you can't tell whether it's a real bug or a definitional mismatch.
**Why it happens:** numpy has multiple quantile methods; `np.quantile` default is `'linear'`, but `np.percentile(method="averaged_inverted_cdf")` (what sklearn `_BinMapper` uses) is *different*. And `searchsorted side='right'` (D-06) vs `side='left'` differ exactly on values equal to an edge.
**How to avoid:**
- Pin the edge math to **`numpy.quantile` default `method='linear'`** [VERIFIED: numpy docs]: cut points are `q = np.linspace(0, 1, n_data_bins + 1)[1:-1]` (the `n_data_bins - 1` interior quantiles); interpolation `j, g = divmod(q*(n-1), 1)`, `edge = a[j] + g*(a[j+1]-a[j])`.
- Pin assignment to **`np.searchsorted(edges, v, side='right')`** = `partition_point(|e| e <= v)` [VERIFIED: numpy docs] — this is exactly D-06 ("count of edges ≤ x", value-equal-to-edge → higher bin).
- Write the baseline test to compute the numpy oracle with **these same** functions so the gate is definitional-exact, not approximate.
**Warning signs:** Agreement is "99.9%" with mismatches only at exact edge values (boundary-side mismatch) or only on features with ties (interpolation-method mismatch).

### Pitfall 2: Assuming CPU↔GPU parity needs effort — it's by construction here, but only if you keep it a pure compare

**What goes wrong:** A future Phase-4 implementer "optimizes" the assignment kernel with an FMA, a normalized index computation, or a fused reduction — and bins diverge by one on a handful of cells, breaking QUANT-02 mysteriously.
**Why it happens:** FP divergence between CPU and GPU comes from FMA contraction, transcendental rounding, and non-associative reductions — NOT from a bare `<=` compare [VERIFIED: NVIDIA CUDA Floating Point & IEEE 754 doc]. The moment assignment stops being a pure compare-against-stored-edges, the by-construction guarantee is lost.
**How to avoid:**
- Document in the golden-vector contract that the GPU kernel MUST do a literal `v <= edge` compare against the **uploaded host edges** — no recomputation, no FMA, no fast-math.
- The Phase-4 kernel should be compiled WITHOUT `--use_fast_math` for the assignment path (note this in the contract for the planner to carry forward).
- On the Rust/CPU side, x86_64 uses SSE2 by default (IEEE-754 f32 compares, not x87 80-bit), so the host side is already clean [VERIFIED: NVIDIA FP doc CPU-comparison guidance].
**Warning signs:** Phase-4 parity test fails only on values *near* an edge; `--use_fast_math` in the quantize kernel build flags.

### Pitfall 3: sklearn `_BinMapper` mistaken for an edge-exact oracle

**What goes wrong:** The baseline test asserts Sylva's edges equal `_BinMapper.bin_thresholds_` and fails everywhere, prompting a wrong "fix."
**Why it happens:** `_BinMapper` uses **midpoints between consecutive distinct values when distinct ≤ max_bins**, else `np.percentile(method="averaged_inverted_cdf")`; then `np.unique` + clip to `ALMOST_INF`; missing bin at `n_bins-1`; and its transform uses `threshold[i-1] < x <= threshold[i]` (value-equal-to-edge → **lower** bin) — the *opposite* side to D-06 [CITED: github.com/scikit-learn .../binning.py]. D-03 deliberately chose raw quantile values, so edges will NOT match `_BinMapper`.
**How to avoid:** Make the `_BinMapper` comparison **distributional only** (CONTEXT.md D-03/D-12): compare *bin-assignment agreement %* and *per-bin population balance*, with a documented tolerance — NOT edge equality. Reserve edge-exactness for the `numpy.quantile` oracle.
**Warning signs:** A test named "edges match sklearn" — rename/rescope it to "bin populations distributionally match sklearn within tolerance."

### Pitfall 4: NaN compares silently misroute (mirror Phase-2 Pitfall 7)

**What goes wrong:** A NaN cell lands in bin 0 instead of the reserved missing bin, because `NaN <= edge` is `false`.
**Why it happens:** IEEE-754 NaN comparisons are always false; a naive `partition_point` returns 0 for NaN.
**How to avoid:** Branch `v.is_nan()` FIRST → `missing_bin_idx` (= `n_bins - 1`, D-09), before any `partition_point`. This mirrors the established `predict.rs` convention exactly. Carry NaN fixtures from Phase-2 SC-4 into the golden vectors.
**Warning signs:** No NaN fixture in the parity golden vectors; bin-0 population spikes on dirty data.

### Pitfall 5: Degenerate columns (all-NaN, all-constant, fewer distinct than bins)

**What goes wrong:** Edge construction panics (empty slice quantile), or produces zero edges → every value maps to bin 0, which is *correct* but untested.
**Why it happens:** Dedupe (D-02) on a constant column yields a single distinct value → zero interior edges. An all-NaN column has no finite values to quantile.
**How to avoid:** Define and test: (a) constant column → 0 edges → all non-NaN rows in bin 0 (1 effective data bin); (b) all-NaN column → 0 edges, all rows in missing bin; (c) `< n_data_bins` distinct values → fewer effective bins (D-02 expected behavior). Return a typed `SylvaError` only for truly invalid input (empty matrix, n_bins out of `[2,256]`/`[2,65536]`), NOT for degenerate-but-valid columns.
**Warning signs:** `unwrap()` on a quantile of an empty/constant slice; no degenerate-column test.

## Code Examples

### Exact-quantile edge construction (numpy `linear` match)
```rust
// Source: numpy.quantile method='linear' formula [VERIFIED: numpy v2 docs]
// Build the n_data_bins-1 interior quantile edges for one feature, deduped.
fn feature_edges(sorted_finite: &[f32], n_data_bins: usize) -> Vec<f32> {
    let n = sorted_finite.len();
    if n == 0 { return Vec::new(); }                // all-NaN column → no edges
    let mut edges = Vec::with_capacity(n_data_bins.saturating_sub(1));
    // interior cut points q in (0,1): linspace(0,1,n_data_bins+1)[1:-1]
    for k in 1..n_data_bins {
        let q = k as f64 / n_data_bins as f64;      // f64 for the index math (matches numpy)
        let pos = q * (n - 1) as f64;
        let j = pos.floor() as usize;
        let g = (pos - j as f64) as f32;
        let lo = sorted_finite[j];
        let hi = sorted_finite[(j + 1).min(n - 1)];
        edges.push(lo + g * (hi - lo));             // f32 result (D-05 f32 end-to-end)
    }
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap()); // already ~sorted; ensure monotone
    edges.dedup();                                   // exact-value dedupe (D-02)
    edges
}
```
> Note: numpy computes the interpolation index in float64 then the gathered values are f32. Pin the *index math* in f64 to match numpy bit-for-bit; the final edge is f32 (D-05). The Python oracle must use the identical `q` grid.

### Bin assignment (searchsorted side='right' + NaN-first)
```rust
// Source: numpy searchsorted side='right' == partition_point [VERIFIED: numpy docs];
// NaN-first routing mirrors crates/sylva-core/src/cpu/predict.rs (D-09).
#[inline]
fn assign_bin(v: f32, edges: &[f32], missing_bin_idx: u32) -> u32 {
    if v.is_nan() { return missing_bin_idx; }                 // check NaN FIRST (Pitfall 4)
    edges.partition_point(|&e| e <= v) as u32                 // = #edges <= v  (D-06)
    // result is in [0, edges.len()]; with n_data_bins-1 edges that is [0, n_data_bins-1],
    // and out-of-range values clamp naturally to the first/last data bin (D-06).
}
```

### Python baseline oracle (edge-exact gate vs numpy)
```python
# Source: numpy.quantile linear + searchsorted right; the definitional oracle for D-01/D-03/D-06.
import numpy as np
def numpy_oracle_bins(X, n_data_bins, missing_idx):           # X: (n,d) float32
    out = np.empty(X.shape, dtype=np.uint8)                   # column-major fill below
    q = np.linspace(0.0, 1.0, n_data_bins + 1)[1:-1]          # interior cut points
    for f in range(X.shape[1]):
        col = X[:, f]
        finite = np.sort(col[~np.isnan(col)])
        edges = np.unique(np.quantile(finite, q, method="linear")) if finite.size else np.array([])
        b = np.searchsorted(edges, col, side="right")         # value==edge -> higher bin (D-06)
        b[np.isnan(col)] = missing_idx                        # NaN -> reserved missing bin
        out[:, f] = b
    return out
# Assert: 100% bit-equality vs Sylva's BinnedMatrix on the fixture (this is the SC-4 gate).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| numpy `interpolation=` kwarg | `method=` kwarg (default still `'linear'`) | numpy 1.22 | Use `method="linear"` in the oracle; `interpolation=` is deprecated. [VERIFIED: numpy docs] |
| sklearn `_BinMapper` raw percentile | midpoints-when-distinct≤max_bins + `averaged_inverted_cdf` percentile | sklearn ≥1.x | Reinforces that `_BinMapper` is distributional-only vs D-03's raw-quantile edges. [CITED: scikit-learn binning.py] |

**Deprecated/outdated:**
- numpy `np.quantile(..., interpolation=...)` — replaced by `method=`. The test oracle must use `method="linear"`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `numpy.quantile` `method="linear"` index math should be computed in **f64** (then gathered values are f32) to match numpy bit-for-bit | Code Examples / Pitfall 1 | If numpy internally casts differently for an f32 input array, the last-ULP edge could differ, dropping baseline agreement below 100% on tie-heavy columns. **Mitigation:** the planner should add a Wave-0 spike test comparing one column's Rust edges to `np.quantile` on the *same* f32 input and adjust the cast point until bit-equal. |
| A2 | sklearn `_BinMapper` transform uses `threshold[i-1] < x <= threshold[i]` (value==edge → lower bin), opposite to D-06 | Pitfall 3 / State of the Art | If sklearn's actual `_map_to_bins` side differs, the *distributional* comparison framing is unaffected (it's not edge-exact anyway), so risk is low. Tagged because the exact side came from a docs/source read, not an executed check. |
| A3 | Column-major SoA is the right layout for the Phase-4 histogram kernel's coalesced reads | Architecture Pattern 2 | If Phase-4 chooses a different access pattern, the layout would need transposing. Low risk — ARCHITECTURE.md explicitly states quantize-once + coalesced histogram reads, which implies feature-contiguous. |
| A4 | Reusing `_BinMapper`'s subsample is unnecessary because D-01 mandates exact full-data quantiles | Standard Stack | None functionally (D-01 is locked); noted only so the planner doesn't accidentally add subsampling to "match sklearn." |

## Open Questions

1. **f64-vs-f32 cast point in the quantile interpolation (A1).**
   - What we know: numpy default is `method='linear'`; numpy computes positions in float64.
   - What's unclear: the exact ULP behavior when the input array is float32 — whether to interpolate in f32 or f64 to be bit-equal to `np.quantile(float32_array, ..., method='linear')`.
   - Recommendation: planner adds a tiny Wave-0 calibration test (one column, compare to numpy) and pins the cast point empirically before locking the golden vectors. Cheap, removes the only HIGH-risk assumption.

2. **Whether to flesh out `backend.rs::BinEdges` now or keep `Quantizer` standalone.**
   - What we know: CpuBackend does not implement `HistogramBackend`; the quantizer is a standalone component (Phase-2 note).
   - What's unclear: whether the planner prefers to make the trait's `Bins`/`BinEdges` real this phase or defer the trait wiring to Phase 4.
   - Recommendation: build standalone `Quantizer` (re-exported from `lib.rs`); leave the `HistogramBackend::Bins = BinnedMatrix` binding for Phase-4's CUDA impl. Lower coupling, matches the established seam discipline.

## Environment Availability

> Phase is CPU/Rust-only (no CUDA, no GPU). External dependency surface is the Python baseline oracle.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable) | Quantizer core + tests | ✓ | 1.96.0 (≥1.83 MSRV) | — |
| numpy (Python) | D-12 edge-exact baseline | ✓ | 2.4.2 | — |
| scikit-learn (Python) | D-12 distributional baseline | ✓ | 1.8.0 | — |
| pytest | Baseline-parity + microbench harness | ✓ (used in Phase-5 harness) | (pin in dev env) | plain `time.perf_counter` if pytest-benchmark absent |
| CUDA / GPU | — (deferred to Phase 4) | N/A | — | N/A this phase (D-05) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `pytest-benchmark` optional — `time.perf_counter` with manual repeated runs suffices for the op-level microbench (D-12).

## Validation Architecture

> `workflow.nyquist_validation` is `true` in config.json — this section is included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust: `cargo test` + `proptest` 1.x (unit/property) ; Python: `pytest` (baseline-parity + microbench) |
| Config file | Cargo workspace (`crates/sylva-core/Cargo.toml`); Python harness env reused from Phase-5 `pyseam` parity harness |
| Quick run command | `cargo test -p sylva-core quantize` |
| Full suite command | `cargo test -p sylva-core` then `pytest tests/quantize_parity` (Python baseline) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QUANT-01 | SoA `BinnedMatrix` uint8 for ≤256 bins | unit | `cargo test -p sylva-core quantize::binned_matrix` | ❌ Wave 0 |
| QUANT-01 | uint16 path for >256 bins (D-08) | unit | `cargo test -p sylva-core quantize::dtype_u16` | ❌ Wave 0 |
| QUANT-01 | Exact-quantile edges, deduped (D-01/D-02) | unit | `cargo test -p sylva-core quantize::edges` | ❌ Wave 0 |
| QUANT-01 | Bins ∈ [0,n_bins); monotone; NaN→missing | property | `cargo test -p sylva-core quantize::props` (proptest) | ❌ Wave 0 |
| QUANT-02 | CPU bins == golden vectors on fixed seed (parity contract) | integration | `cargo test -p sylva-core quantize::parity_golden` | ❌ Wave 0 |
| QUANT-02 | Pure-compare assignment documented for Phase-4 GPU reuse | (contract doc) | n/a (asserted in golden fixture metadata) | ❌ Wave 0 |
| SC-4 | Edge-exact bin agreement vs `numpy.quantile` = 100% | integration | `pytest tests/quantize_parity/test_numpy_oracle.py` | ❌ Wave 0 |
| SC-4 | Distributional bin-population agreement vs `_BinMapper` within tolerance | integration | `pytest tests/quantize_parity/test_sklearn_distributional.py` | ❌ Wave 0 |
| SC-3 | `QuantizeReport` records dtype/contiguity/bytes; H2D marked N/A | unit | `cargo test -p sylva-core quantize::report` | ❌ Wave 0 |
| SC-5 | Quantize throughput microbench (rows/s) reported, not gated | benchmark | `pytest tests/quantize_parity/test_throughput.py` (informational) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p sylva-core quantize` (fast Rust unit/property subset).
- **Per wave merge:** full `cargo test -p sylva-core` + `pytest tests/quantize_parity`.
- **Phase gate:** full suite green; edge-exact numpy agreement = 100%; golden-vector parity test green; throughput number recorded (informational, not gated) before `/gsd-verify-work`.

### Tolerances (resolving the CONTEXT.md discretion item)
- **numpy edge-exact gate:** **100% bit-equality** of bin assignments (after the A1 f64/f32 cast calibration). This is the GATE.
- **sklearn distributional check:** bin-assignment agreement **≥ 99%** on `make_classification` 100k×100 f32 (different by design due to midpoint vs raw-quantile + opposite side), reported with the actual number; treat <95% as a signal to investigate, not an automatic fail (documented divergence). **Informational, not a hard gate.**
- **Throughput microbench:** report rows/s for Sylva-CPU vs `numpy.quantile` vs `_BinMapper` on the pinned dataset (100k×100 f32, bins 128 and 256, fixed seed, pinned numpy 2.4.2 / sklearn 1.8.0). Op-level only; **no end-to-end speed claim** (foundational-phase fairness rule). Report cold/warm separately; repeated runs.

### Wave 0 Gaps
- [ ] `crates/sylva-core/src/quantize/*.rs` — the module itself (no quantizer exists yet)
- [ ] `crates/sylva-core/tests/` or `#[cfg(test)]` — Rust unit/property tests for edges/assign/dtype/report
- [ ] Golden-vector fixture file (JSON) + loader — the QUANT-02 contract
- [ ] `tests/quantize_parity/` Python harness — numpy edge-exact + sklearn distributional + throughput (reuse the Phase-5 `pyseam` env pattern)
- [ ] A1 calibration micro-test — pin the f64/f32 cast point against `np.quantile` before locking golden vectors

## Security Domain

> `security_enforcement: true`, ASVS level 1. This phase is a pure numerical/in-process library component (no network, no auth, no persistence of user data, no secrets). Most ASVS categories are N/A.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (in-process library). |
| V3 Session Management | no | No sessions. |
| V4 Access Control | no | No access control surface. |
| V5 Input Validation | **yes** | Validate `X` shape (non-empty, 2-D), dtype (f32), and `n_bins` range at the boundary → typed `SylvaError` (no panic). Established Phase-2 convention. |
| V6 Cryptography | no | No crypto. (Philox is a non-cryptographic statistical RNG and is not even used this phase per D-01.) |

### Known Threat Patterns for {Rust numerical core}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds index from malformed `X` / bin overflow | Tampering / DoS | Validate shape + dtype + n_bins at the boundary; `partition_point` is bounds-safe; bins fit the chosen uint by construction (n_bins ≤ 256 → u8, ≤ 65536 → u16). No `unsafe`. |
| NaN/Inf causing silent misrouting | Tampering (data integrity) | NaN-first routing to the reserved missing bin; Inf clamps via searchsorted into the last bin (in-range by construction). Test with NaN/Inf fixtures. |
| Panic on degenerate column crossing the boundary | DoS | Return typed `SylvaError` for invalid input; handle degenerate-but-valid columns (constant/all-NaN) without panic (Pitfall 5). No `.unwrap()` on fallible paths. |
| License contamination (copying sklearn/GPL binning source) | (compliance) | Reimplement from the numpy/sklearn *algorithm description*, never copy source — same Apache-2.0 discipline used in Phase-2 `split_et.rs`. Cite numpy formula, not code. |

## Sources

### Primary (HIGH confidence)
- NVIDIA "Floating Point and IEEE 754" (CUDA docs, 12.x/13.x) — GPU IEEE-754 compliance; divergence sources are FMA/transcendentals/reductions, not bare compares; SSE-vs-x87 CPU guidance. Confirms D-04 bit-identical-by-construction. https://docs.nvidia.com/cuda/floating-point/
- numpy `quantile`/`digitize`/`searchsorted` manuals (v2.x) — `method='linear'` default + formula; `searchsorted side='right'` = count of edges ≤ x; `digitize`↔`searchsorted` mapping. https://numpy.org/doc/stable/reference/generated/numpy.quantile.html , https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
- Existing codebase: `crates/sylva-core/src/cpu/predict.rs` (NaN-first routing convention), `src/cpu/split_et.rs` (`FEATURE_THRESHOLD`, `to_bits()` tie-break, fixed-order accumulation), `src/ir.rs` (SoA + f32 + serde), `src/backend.rs` (the `quantize`/`BinEdges`/`Bins` seam), `src/error.rs`, `src/config.rs`.

### Secondary (MEDIUM confidence)
- scikit-learn `_hist_gradient_boosting/binning.py` (`_BinMapper`, `_find_binning_thresholds`) — subsample=2e5, percentile `averaged_inverted_cdf`, midpoint-when-distinct≤max_bins, `np.unique` dedupe, `missing_values_bin_idx_ = n_bins-1`, `n_bins` default 256 / `max_bins=255`. Establishes that `_BinMapper` is distributional-only vs D-03 raw-quantile edges. https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/binning.py
- `.planning/research/ARCHITECTURE.md` — SoA `BinnedMatrix(uint8/16)`, quantize-once, binned + parallel real-valued threshold table, "GPU bins must match CPU bins bit-for-bit."
- `.planning/research/PITFALLS.md` — Pitfalls 1/2/5/7/13 (transfer framing, crossover honesty, float determinism, NaN routing, benchmark fairness).

### Tertiary (LOW confidence)
- General WebSearch summaries of `_BinMapper` behavior (cross-checked against the source file above before use).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps already pinned; numpy/sklearn versions verified installed.
- Parity mechanism (D-04 bit-identical-by-construction): HIGH — verified against NVIDIA IEEE-754 doc; reduces to pure compare.
- numpy edge-exact baseline: HIGH — formula + side verified against numpy docs; one residual f64/f32 cast calibration (A1) flagged for a Wave-0 spike.
- sklearn `_BinMapper` exact schema: MEDIUM — read from source/docs, not executed; but only needed for a *distributional* check, so exactness is non-load-bearing.
- Architecture (SoA layout, fixture shape): HIGH — derived from ARCHITECTURE.md + established `ir.rs`/`parity.rs` patterns.

**Research date:** 2026-06-26
**Valid until:** ~2026-07-26 (30 days — numpy/sklearn binning semantics are stable; re-verify only if numpy/sklearn major versions change before the test harness is pinned).
