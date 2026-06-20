---
phase: 02-cpu-oracle-contracts-forest-ir
plan: 04
subsystem: oracle-contracts
tags: [rust, sylva-core, proptest, determinism, parity, split-statistics, est-07, eng-04]

requires:
  - phase: 02-cpu-oracle-contracts-forest-ir
    plan: 02
    provides: CpuBackend, ForestIR, Backend trait, NaN-safe predict
  - phase: 02-cpu-oracle-contracts-forest-ir
    plan: 03
    provides: RandomForest (bootstrap + exact best-split), four-estimator matrix

provides:
  - proptest property-based invariants over ET/RF clf+reg (tests/invariants.rs)
  - byte-identical seed determinism + rayon order-independence proofs (tests/determinism.rs)
  - ENG-04 parity contract document (docs/PARITY.md)
  - split_statistics extractor for Phase-05 KS harness (src/parity.rs + SplitStats re-export)

affects:
  - 02-05 (parity harness consumes split_statistics / SplitStats)
  - phase-4 (CUDA backend is held to byte-identical serialized ForestIR + Philox KAT contract)
  - phase-8 (cover invariant proven; SHAP pre-validated)
  - phase-6/9 (serde round-trip proven; export substrate validated)

tech-stack:
  added:
    - proptest integration tests (tests/invariants.rs, tests/determinism.rs)
    - parity.rs module with SplitObservation, SplitStats, split_statistics
  patterns:
    - proptest strategies over (n, d, trees, seed) tuples; 32 cases/test for CI budget
    - 1-thread rayon pool (ThreadPoolBuilder::num_threads(1)) to force sequential order
    - per-feature threshold normalization: (thr - min) / range, clamped to [0, 1]

key-files:
  created:
    - crates/sylva-core/tests/invariants.rs
    - crates/sylva-core/tests/determinism.rs
    - crates/sylva-core/src/parity.rs
    - docs/PARITY.md
  modified:
    - crates/sylva-core/src/lib.rs

key-decisions:
  - "Proptest covers all four estimators (ET/RF x clf/reg) with bounded shapes (n<=100, d<=8, trees<=6, depth<=5) and 32 cases/test — stays within 90s CI budget"
  - "Parallel==sequential proven via rayon::ThreadPoolBuilder::num_threads(1) — the 1-thread pool forces strict index order; Philox keying makes result identical to multi-thread rayon"
  - "Threshold normalization is per-feature (min/max of recorded split thresholds), not global — makes thresholds from different features comparable in the KS test"
  - "PARITY.md explicitly states sklearn equivalence is DISTRIBUTIONAL not bit-level — cites our_rand_r serial PRNG as the reason bit-replay is infeasible"

metrics:
  duration: ~22 min
  completed: 2026-06-20
  tasks: 3
  files_created: 4
  files_modified: 1

status: complete
---

# Phase 2 / Plan 02-04: Property Invariants, Determinism, Parity Contract & split_statistics Summary

**Proves CpuBackend trustworthiness via proptest property-based invariants (EST-07), byte-exact seed determinism tests (ENG-04 exact side), a documented parity contract distinguishing Sylva-internal EXACT reproducibility from sklearn DISTRIBUTIONAL equivalence, and a `split_statistics` extractor that feeds the Phase-05 KS harness.**

## Performance

- **Duration:** ~22 min
- **Completed:** 2026-06-20
- **Tasks:** 3 (all auto)
- **Files created:** 4 (+1 modified)

## Accomplishments

### Task 1: Property invariants (`tests/invariants.rs`) — EST-07 property side

9 tests using `proptest` (32 generated cases each) covering all four estimators:

| Invariant | How asserted |
|---|---|
| Child rows partition parent (cover) | `node_sample_count[node] == L + R` for all internal nodes — algebraic form of disjoint-union on clean data |
| Leaf probabilities valid (clf) | each ∈ [0,1] and sum == 1 ± 1e-4 (f32 tolerance) |
| Leaf values finite (reg) | `v.is_finite()` for all `leaf_value` entries |
| Sample count positive | every node has `node_sample_count >= 1` |
| `default_child` correctness | points to higher-count child; tie → left child |
| Serde round-trip | `deserialize(serialize(ir)) == ir` (structural equality) |
| Leaf sentinels | `feature_id == LEAF_FEATURE`, `left/right == NO_CHILD` for all leaves |
| `validate_structure` | passes for every generated ForestIR |
| Seed determinism (proptest) | proptest sweep over 4 algo combos × seeds → byte-identical |

Proptest strategies span `n ∈ [20, 100]`, `d ∈ [2, 8]`, `trees ∈ [2, 6]`, `seed ∈ [0, 9999]`. Suite completes in ~70ms — well within the 90s budget.

### Task 2: Byte-identical determinism (`tests/determinism.rs`) — ENG-04 exact contract

12 tests asserting **exact string equality** (never `allclose`/approx) for all four estimators:

- **Same-seed byte-identity:** two `CpuBackend::fit` calls with the same `(seed, cfg, data)` → identical `serde_json::to_string` output for ET clf, ET reg, RF clf, RF reg, and ET entropy clf.
- **Parallel == sequential:** normal rayon parallel build vs. `rayon::ThreadPoolBuilder::new().num_threads(1).build().install(|| ...)` → byte-identical ForestIR for all four estimators. This proves Philox counter-keying makes tree scheduling order irrelevant.
- **Multi-seed sweep:** 7 seeds for ET clf (including 0, `u64::MAX / 2`), 4 seeds for RF reg.
- **Anti-regression guard:** different seeds produce different IRs (guards the assertion logic itself).

### Task 3: Parity contract + `split_statistics` — ENG-04 documentation + D-04 Phase-05 seam

**`docs/PARITY.md`** — four-point contract:

1. **Sylva-internal EXACT:** same seed → byte-identical serialized ForestIR (the Phase-4 `GPU == CPU` bar).
2. **CPU-to-GPU RNG identity (Phase 4):** Philox constants + counter layout + uint32→f32 conversion frozen; KAT vectors in `rng/kat.rs` are the Phase-4 bit-match target.
3. **sklearn equivalence is DISTRIBUTIONAL, NOT bit-level:** sklearn's `our_rand_r` serial PRNG is un-replayable in parallel; equivalence is asserted via accuracy/R² CI + KS test on aggregate split statistics.
4. **f32 precision (D-05):** f32/f64 last-bit differences from sklearn are inside the distributional tolerance band. No speed claims are made in this document.

**`src/parity.rs`** — `split_statistics(&ForestIR) -> SplitStats`:
- Two-pass implementation: collect raw `(feature_id, threshold)` for all internal nodes; compute per-feature `(min, max)` of observed thresholds; normalize each to `[0, 1]`.
- Leaves contribute zero observations (by construction: the first pass only visits `feature_id != LEAF_FEATURE`).
- `SplitStats` + `SplitObservation` derive `Serialize, Deserialize` — the Phase-05 Python harness reads them via `serde_json`.
- Doctest included (passes in `cargo test --doc`).
- 8 unit tests: leaves excluded, `[0,1]` range, feature bounds, metadata matches IR, reg forest, serde round-trip, depth-0 all-leaf → empty stats, RF clf.

**`src/lib.rs`** — added `pub mod parity; pub use parity::{split_statistics, SplitObservation, SplitStats};`

## Test Outcomes (103 / 103 pass)

| Test category | Tests | Key invariants |
|---|---|---|
| `tests/invariants.rs` | 9 | cover-partition, leaf-proba, finite-reg, sample-count, default_child, serde, sentinels, validate_structure, seed-det proptest |
| `tests/determinism.rs` | 12 | byte-identical same-seed (all 4 estimators + entropy), parallel==sequential (all 4), multi-seed sweep, anti-regression guard |
| `parity::tests` | 8 | leaves excluded, [0,1] range, feature bounds, metadata, reg, serde round-trip, depth-0 empty, RF clf |
| `doc-tests` | 1 | split_statistics example in doc comment |
| Prior unit tests | 82 | all prior plans unbroken (ir, rng, cpu::criterion/split_et/split_rf/bootstrap/fit/predict) |

All 103 tests pass. `cargo clippy -p sylva-core --all-targets -- -D warnings` clean.

## Files Created/Modified

- `crates/sylva-core/tests/invariants.rs` — 9 proptest + fixed invariant tests across all 4 estimators (EST-07)
- `crates/sylva-core/tests/determinism.rs` — 12 byte-identical determinism + parallel==sequential tests (ENG-04 exact)
- `crates/sylva-core/src/parity.rs` — `split_statistics`, `SplitStats`, `SplitObservation` + 8 unit tests + doctest
- `docs/PARITY.md` — four-point parity contract (ENG-04 deliverable)
- `crates/sylva-core/src/lib.rs` — `pub mod parity` + re-exports

## Task Commits

1. **Task 1** (proptest invariants) — `25c3f73`
2. **Task 2** (byte-identical determinism) — `0311172`
3. **Task 3** (PARITY.md + split_statistics) — `877874d`

## Decisions Made

- `proptest` strategies bounded to `n ∈ [20, 100]` and 32 cases/test to stay within the 90s CI budget. Shrinking still finds minimal failing cases on any regression.
- `rayon::ThreadPoolBuilder::num_threads(1)` chosen for the parallel==sequential proof over alternatives (single-tree manual loop, single-thread rayon scope). The 1-thread pool exercises the same `par_iter` code path as the production build — it does not bypass rayon.
- Threshold normalization is per-feature (min/max of split thresholds observed in the IR for that feature), not per-training-data feature range. This is intentional: the KS test in Phase 05 compares Sylva's threshold distribution to sklearn's for the same forest, so using the same observed range makes sense. The training-data range is not stored in the IR.
- `PARITY.md` placed in `docs/` (repo root) rather than crate docs, so it is accessible as a project-level contract independent of the Rust crate boundary.

## Deviations from Plan

None — plan executed exactly as written. The proptest strategy choices (case count = 32, shape bounds) are implementation details that fall within the plan's "bounded shapes + case counts" instruction.

## Known Stubs

None — `split_statistics` is fully wired: reads `feature_id` and `threshold` from the live ForestIR, normalizes, and returns populated `SplitStats`. No placeholder values flow to any output.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundaries introduced. `split_statistics` is a pure read-only extractor over the IR — no mutation, no external I/O. `SplitStats` serialization is JSON (serde), not a security boundary. No additional threat flags found.

## Next Phase Readiness

- **02-05 (parity harness):** `split_statistics(&ForestIR) -> SplitStats` is the Rust seam the Phase-05 PyO3 accessor exposes. `SplitStats` serializes to JSON; the Python harness reads it and feeds `scipy.stats.ks_2samp`. The Phase-05 plan can wrap this in a `#[pyfunction]` without changes to `parity.rs`.
- **Phase 4 (CUDA):** the Phase-4 GPU path inherits the byte-identical contract. The `determinism.rs` tests define exactly what "byte-identical" means (exact serialized JSON equality). The Philox constants + counter layout + KAT vectors are frozen in `rng/kat.rs` and `docs/PARITY.md` Point 2.
- **Phase 8 (SHAP):** the cover-invariant proptest (`assert_cover_partition`) confirms `node_sample_count[node] == L + R` for all trees/configs — the structural pre-condition path-dependent TreeSHAP requires.

---
*Phase: 02-cpu-oracle-contracts-forest-ir*
*Completed: 2026-06-20*

## Self-Check: PASSED
