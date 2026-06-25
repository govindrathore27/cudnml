# Phase 3: Feature Quantizer (CPU/GPU Bit-Parity) - Context

**Gathered:** 2026-06-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the **feature quantizer** that every later GPU histogram kernel reads from, and lock the **bit-parity contract** that proves its bin assignments are identical across CPU and GPU. Concretely (QUANT-01, QUANT-02):

1. A per-feature **quantile** quantizer that produces a SoA `BinnedMatrix` (uint8/uint16) from the f32 matrix — the `HistogramBackend::quantize` seam already declared in `crates/sylva-core/src/backend.rs` (`fn quantize(&self, x) -> (Self::Bins, BinEdges)`).
2. A **bit-parity test harness / contract** (golden bin vectors on a fixed seed) that the Phase-4 GPU quantize kernel must reproduce bit-for-bit.
3. `execution_report_` hooks (minimal, quantize-step scope) recording dtype/contiguity handling and the computed transfer payload size.
4. The Comparative Baseline Study: binning **correctness** vs `numpy.quantile` / sklearn `_BinMapper`, plus a CPU-vs-baseline quantize-throughput **microbench** (op-level, no end-to-end speed claim).

**MVP mode.** This is the binning + parity-contract layer.

**Explicit scope narrowing decided in discussion (see D-05):** the **live GPU quantize kernel is deferred to Phase 4**. This phase ships the CPU quantizer plus the enforced parity contract the GPU kernel will be validated against. No histograms, no estimator API (Phase 5), no full `execution_report_`/dispatch (Phase 6), no SHAP/export.

</domain>

<decisions>
## Implementation Decisions

### Quantile edge construction
- **D-01 (Claude's discretion):** **Exact full-data quantiles.** Edges are computed from the entire feature column via exact percentile/sort — **no subsampling, no RNG this phase**. Cleanest path to deterministic CPU↔GPU parity. Subsampling (with Sylva's own Philox stream) is **deferred** as a profiling-driven optimization, not built now.
- **D-02 (USER-DECIDED):** **Collapse duplicate edges to unique** (sklearn `_BinMapper`-style `np.unique` dedupe). A low-cardinality feature therefore gets **fewer effective bins** than the nominal count.
- **D-03 (USER-DECIDED):** **Edges = quantile values directly** (NOT midpoints between consecutive quantiles). Consequence: the binning anchor aligns with **`numpy.quantile`**; the sklearn `_BinMapper` comparison (which uses midpoints) is therefore **distributional-only, not edge-exact** — this is consistent with the D-04 parity philosophy from Phase 2 (distributional equivalence to sklearn, never bit-replay).

### CPU↔GPU bit-parity mechanism
- **D-04 (USER-DECIDED):** **Edges-on-host, GPU assigns-only.** Quantile edge construction (sort/quantile) runs **only on the CPU/host**; edges are uploaded to the GPU, and the GPU performs **only the searchsorted comparison**. Since both devices do the identical IEEE-754 comparison against the same f32 edges (pure compare, no accumulation/reduction), bin assignments are **bit-identical by construction**. This eliminates the hardest parity risk (parallel GPU sort/reduction order).
- **D-05 (USER-DECIDED — scope narrowing, flagged):** **GPU quantize kernel deferred to Phase 4.** Phase 3 delivers the CPU quantizer + a **bit-parity test harness** (golden bin assignments on a fixed seed) that the Phase-4 GPU assignment kernel must reproduce exactly.
  - ⚠ **ROADMAP SC-2 implication (must be honored downstream):** SC-2 ("CPU and GPU quantizers produce bit-identical bin assignments, enforced by a parity test in CI") is, for **this phase**, satisfied as "**CPU quantizer + enforced bit-parity contract/golden vectors**"; the **live CPU↔GPU CI proof moves to Phase 4** when the kernel exists. The verifier/planner must NOT read SC-2 as fully unmet, nor as fully met with a live GPU run — it is met as a **contract** here. QUANT-02's live half lands Phase 4.
- **D-06 (Claude's discretion):** **Boundary rule = `side='right'`** — bin index = number of edges ≤ x (a value exactly equal to an edge goes to the **higher** bin). Single documented rule used identically on CPU and GPU, and **required to match the Phase-4 histogram split comparison**. Out-of-range values (predict-time values beyond training edges) **clamp** into the first/last bin via searchsorted.

### Bin count, dtype & missing-value bin
- **D-07 (Claude's discretion):** **255 data bins + 1 reserved missing bin = 256 total** (matches sklearn HGBT `max_bins=255`, fits uint8 exactly, within the 128–256 shared-memory budget from PITFALLS.md). Held in an **internal quantizer config field** (default 255); the **public sklearn `max_bins` param waits for Phase 5** (EST-03).
- **D-08 (USER-DECIDED):** **Auto dtype selection** — `n_bins ≤ 256 → uint8`, `> 256 → uint16`. Both code paths exist and are tested; the default exercises uint8. Honors QUANT-01's "uint8/uint16".
- **D-09 (USER-DECIDED):** **Reserved missing bin** (sklearn-style, highest index — the `+1` of `255+1`). NaN/missing → that bin. **Split-time routing reuses the Phase-2 D-01 default-child rule** (higher-sample-count child; tie → left). Out-of-range clamps as in D-06.

### execution_report_ hook scope
- **D-10 (Claude's discretion):** **Minimal quantize-step record only** — input dtype, contiguity/layout handling, and byte size — written into a small struct that **Phase 6 (DET) extends** into the full `execution_report_`. No backend-selection/fallback surface this phase.
- **D-11 (Claude's discretion):** **Record computed payload size** (byte size of `BinnedMatrix` + edges = the payload that *will* transfer H2D) and **explicitly mark H2D as not-executed / N/A this phase** (no device path yet). Honest, non-silent; satisfies SC-3 without fabricating a transfer.

### Comparative Baseline Study scope
- **D-12 (Claude's discretion):** **CPU-vs-baseline this phase.** Correctness parity + quantize-throughput microbench (rows/s) of Sylva's **CPU** quantizer vs `numpy.quantile` and sklearn `_BinMapper`; **GPU quantize throughput is reported in Phase 4** alongside the kernel. Op-level number, **no end-to-end algorithm speed claim** (per ROADMAP fairness rules — foundational phase).

### Claude's Discretion (delegated / flagged for research)
- Exact percentile interpolation method for D-01 (e.g. numpy `linear` vs `lower`) and the precise `searchsorted`-side numpy alignment for D-06 — researcher to pin against the `numpy.quantile` baseline so the correctness comparison is apples-to-apples.
- Exact correctness tolerance for the binning-parity gate (bin-assignment agreement %) and the throughput-microbench dataset details — researcher picks defensible values within the ROADMAP shape (`make_classification` 100k×100 f32, bins 128/256, pinned versions, fixed seed).
- `BinnedMatrix` concrete SoA layout + how the binned-threshold table and the parallel **real-valued threshold table** (for export, per ARCHITECTURE.md) are stored together.
- The concrete shape of the bit-parity golden-vector fixtures (D-05) that Phase-4 will assert against.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 3: Feature Quantizer (CPU/GPU Bit-Parity)" — goal, 5 success criteria, the Comparative Baseline Study (sklearn `HistGradientBoosting` `_BinMapper` / `numpy.quantile`, `make_classification` 100k×100, bins 128/256), Mode: mvp, and the binding comparative-study fairness note.
- `.planning/REQUIREMENTS.md` — QUANT-01 (SoA `BinnedMatrix` uint8/uint16 via per-feature quantile bins, CPU + GPU), QUANT-02 (bit-identical CPU↔GPU bin assignments, CI parity test).
- `.planning/PROJECT.md` — constraints: sklearn semantics + determinism, no silent fallback, dense float32 MVP, Apache-2.0 (reimplement from algorithm; never copy GPL/sklearn source).
- `.planning/STATE.md` — the four near-rewrite-if-deferred architecture decisions (the CPU↔GPU parity contract is the one this phase operationalizes for binning); binding fairness protocol.

### Architecture & technical research (the contracts this phase fills)
- `.planning/research/ARCHITECTURE.md` — §`BinnedMatrix(SoA uint8/16)`, the `Backend::quantize → BinnedMatrix` flow (lines ~47, ~120, ~182, ~210, ~230), the **quantize-once** principle, the **binned-threshold + parallel real-valued table** representation, and "GPU bins must match CPU bins bit-for-bit".
- `.planning/research/PITFALLS.md` — bin-count tuning (128–256 for shared-mem budget), determinism/accumulation pitfalls, and comparative-study fairness (Pitfalls 1, 2, 13).
- `.planning/research/SUMMARY.md` — Phase-3 deliverable summary (BinnedMatrix, CPU quantizer, GPU quantize kernel [now deferred to Ph4 per D-05], bit-parity test, `execution_report_` hooks); honesty/fairness calibration.
- `.planning/research/FEATURES.md` — sklearn param surface (e.g. `max_bins` is a Phase-5 estimator concern; internal config only here).

### Code seam this builds on
- `crates/sylva-core/src/backend.rs` — `HistogramBackend::quantize` signature + the placeholder `BinEdges` / `type Bins` seam that Phase 3 makes real (CPU side).
- `.planning/phases/02-cpu-oracle-contracts-forest-ir/02-CONTEXT.md` — Phase-2 decisions this depends on: f32 end-to-end (D-05), NaN default-child routing (D-01), Philox-4×32-10 keyed RNG (available for any future subsampling), distributional-not-bit-replay parity philosophy (D-04).
- `.claude/CLAUDE.md` — Technology Stack (ndarray/rayon for CPU; thiserror error enums), coding style (many small files, no hardcoded values, f32 MVP).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`HistogramBackend::quantize` seam** (`crates/sylva-core/src/backend.rs:79`) — device-neutral signature `quantize(&self, x: ArrayView2<f32>) -> Result<(Self::Bins, BinEdges), SylvaError>` already declared with placeholder `BinEdges` / `type Bins`. Phase 3 makes the CPU `Bins` (the SoA `BinnedMatrix`) and `BinEdges` real; the GPU `Bins` binding is the Phase-4 seam.
- **`CpuBackend`** (`crates/sylva-core/src/cpu/mod.rs`) — the trusted oracle; the CPU quantizer slots in as the binning front-end (note: Phase-2 CpuBackend trains by exact splitting and does NOT implement `HistogramBackend`; binning here is a standalone, separately-testable component).
- **Philox-4×32-10** (Phase 2) — reproducible keyed stream, available if/when subsampling is added (not used in D-01's exact-quantile path).
- **`SylvaError` (thiserror)** + Results-not-`.unwrap()` convention — template for quantizer error variants (e.g. degenerate column, all-NaN feature).

### Established Patterns
- f32 numerics (D-05 from Phase 2); many small files (200–400 lines); no hardcoded values (bin count is a config field, D-07); honest/non-silent error surfacing; deterministic-by-construction.

### Integration Points
- `BinnedMatrix` is written by the quantizer and read by the Phase-4 histogram kernels (uint8/16 bins, never the f32 matrix again — the quantize-once bandwidth win).
- The **bit-parity contract / golden vectors** (D-05) are the explicit hand-off to Phase 4: the GPU assignment kernel must reproduce them exactly.
- The parallel **real-valued threshold table** is the hand-off to Phase 9 (Treelite export) — store it alongside binned thresholds now (ARCHITECTURE.md).

</code_context>

<specifics>
## Specific Ideas

- **Edges-on-host / GPU-assigns-only** (D-04) is the load-bearing parity design — it makes bit-identical bins a *construction* property, not a thing to chase.
- **Edges = quantile values directly** (D-03) intentionally anchors correctness to `numpy.quantile` rather than sklearn `_BinMapper` midpoints — researcher should write the baseline comparison around that anchor.
- The ROADMAP Comparative Baseline dataset: `make_classification` 100k×100 f32, bin counts 128 and 256, fixed seed, pinned versions.
- Reserved missing bin = highest index (the `+1` of 255+1), exercised by NaN fixtures carried over from Phase 2's SC-4 fixtures.

</specifics>

<deferred>
## Deferred Ideas

- **Live GPU quantize kernel + live CPU↔GPU CI parity run** (the GPU half of QUANT-01/QUANT-02) — **Phase 4**, validated against this phase's golden-vector contract (D-05).
- **GPU quantize-throughput microbench** — **Phase 4**, reported alongside the kernel (D-12).
- **Subsampled quantile edges** (with Sylva's Philox stream) — profiling-driven optimization, only if exact full-data quantiles are too slow on large `n` (D-01).
- **Public sklearn `max_bins` constructor param** — **Phase 5** (EST-03); internal config field only here (D-07).
- **Full `execution_report_` surface, `device`/`fallback="error"` dispatch** — **Phase 6** (DET-*); minimal quantize-step record only here (D-10/D-11).
- **Learned/optimal missing-value direction** — still deferred from Phase 2 D-01; the reserved missing bin + default-child rule is the shipped behavior.

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 3-Feature Quantizer (CPU/GPU Bit-Parity)*
*Context gathered: 2026-06-26*
</content>
</invoke>
