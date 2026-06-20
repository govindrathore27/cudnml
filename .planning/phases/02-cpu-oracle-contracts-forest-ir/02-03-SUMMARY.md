---
phase: 02-cpu-oracle-contracts-forest-ir
plan: 03
subsystem: cpu-backend
tags: [rust, sylva-core, cpu-backend, random-forest, bootstrap, best-split, determinism, rayon]

requires:
  - phase: 02-cpu-oracle-contracts-forest-ir
    plan: 02
    provides: BuildCtx, build_node, criterion, ForestIR, NaN-safe predict, CpuBackend, EtSplitCtx

provides:
  - bootstrap_indices (Philox-keyed with-replacement row resampling, BOOTSTRAP_NODE_SENTINEL)
  - RfSplitCtx / RfSplitResult / best_split (sorted-midpoint BestSplitter, exact best-split)
  - NodeSplit struct unifying ET+RF split results in build_node dispatch
  - CpuBackend trains RandomForest (clf+reg) via bootstrap + exact best-split (SC-2 complete)
  - RF parallel==sequential determinism invariant (T-02-10) passing
  - Four-estimator matrix (ET/RF x clf/reg) all train+predict correctly (D-02 / SC-2)

affects:
  - 02-04 (property-based invariant tests now cover ET+RF x clf/reg)
  - 02-05 (parity harness can now compare RF clf+reg to sklearn RF)
  - phase-4 (CUDA backend reproduces bootstrap keying from BOOTSTRAP_NODE_SENTINEL counter)

tech-stack:
  added:
    - bootstrap.rs module with Philox-based with-replacement sampling
    - split_rf.rs module with sorted-midpoint BestSplitter algorithm
  patterns:
    - NodeSplit struct centralises split-result fields shared by ET and RF splitters
    - BOOTSTRAP_NODE_SENTINEL = u32::MAX as sentinel counter namespace (T-02-12)
    - Algorithm dispatch via match cfg.algo in build_node (single extension point)
    - Bootstrap keyed by (seed, tree) — order-independent under rayon (T-02-10)

key-files:
  created:
    - crates/sylva-core/src/cpu/bootstrap.rs
    - crates/sylva-core/src/cpu/split_rf.rs
  modified:
    - crates/sylva-core/src/cpu/fit.rs
    - crates/sylva-core/src/cpu/predict.rs
    - crates/sylva-core/src/cpu/mod.rs

key-decisions:
  - "NodeSplit struct (not a tuple) used in build_node to avoid clippy::type_complexity on the match arm"
  - "Bootstrap counter = [tree, u32::MAX, 0, draw_i] — BOOTSTRAP_NODE_SENTINEL=u32::MAX is provably distinct from any valid node_id counter (T-02-12 stream collision mitigation)"
  - "prev_val advances on every iteration (including skipped pairs) so consecutive-distinct-values logic matches BestSplitter — duplicate values simply produce no candidate"

metrics:
  duration: ~21 min
  completed: 2026-06-20
  tasks: 3
  files_created: 2
  files_modified: 3

status: complete
---

# Phase 2 / Plan 02-03: RandomForest (clf+reg) — Bootstrap + Exact Best-Split Summary

**Extends `CpuBackend` to train RandomForest classifier and regressor via Philox-keyed bootstrap resampling and a sorted-midpoint exhaustive best-split, reusing the shared recursive builder and NaN-safe predict path — completing SC-2 (both ET and RF, clf+reg) and D-02.**

## Performance

- **Duration:** ~21 min
- **Completed:** 2026-06-20
- **Tasks:** 3 (all auto + TDD)
- **Files created:** 2 (+3 modified)

## Accomplishments

### Task 1: Bootstrap row resampling (`bootstrap.rs`)

- `bootstrap_indices(n, seed, tree) -> Vec<usize>`: draws `n` indices from `0..n` with replacement using Philox counter `[tree, BOOTSTRAP_NODE_SENTINEL, 0, i]`.
- `BOOTSTRAP_NODE_SENTINEL = u32::MAX` — provably distinct from all valid node ids (T-02-12); both the constant and the non-collision property are unit-tested.
- T-02-09 OOB guard: every drawn index is clamped to `n-1`.
- 11 unit tests: count, range, determinism, cross-tree independence, cross-seed independence, roughly-uniform distribution, with-replacement (duplicate) verification, empty/n=1 edge cases, sentinel-is-u32::MAX invariant, counter-namespace non-collision with split draws.

### Task 2: RF exact best-split (`split_rf.rs`)

- `best_split(ctx: &RfSplitCtx)` reimplements sklearn's BestSplitter midpoint algorithm (Apache-2.0; no GPL/sklearn source copied; provenance comment included).
- Algorithm: for each of `max_features` candidate features (Fisher-Yates Philox prefix, same as ET): sort node's feature values, walk consecutive DISTINCT pairs, compute midpoint `v_prev*0.5 + v_curr*0.5`, skip pairs within `FEATURE_THRESHOLD`, evaluate impurity improvement via shared `criterion.rs` helpers, keep the global best with `(feature_id, threshold_bits)` tie-break.
- `RfSplitCtx<'x,'y,'r>` mirrors `EtSplitCtx` shape (3-lifetime params for recursive reuse).
- `RfSplitResult` shape identical to ET's `SplitResult` — consumed unchanged by `build_node`.
- D-01: `default_left = n_left >= n_right` (tie → left), same as ET.
- `<=` left convention consistent with ET and `predict.rs`.
- 11 unit tests: separating midpoint, distinct-values-only candidates, constant-feature skip, near-equal FEATURE_THRESHOLD skip, min_samples_leaf, tie-break correctness, default_child policy, <= convention, determinism, regression task, partition-covers-all-rows.

### Task 3: Dispatch + bootstrap wiring — SC-2 complete (`fit.rs`, `predict.rs`, `mod.rs`)

- `NodeSplit` struct introduced to carry `(feature_id, threshold, left_rows, right_rows, default_left)` from either splitter — avoids clippy `type_complexity` on the match arm.
- `build_node` gains a single `match ctx.cfg.algo` block dispatching to `best_random_split` (ET) or `rf_best_split` (RF); the recursion, leaf emission, `assemble_forest`, and `predict.rs` are all **unchanged**.
- `fit_forest` gains the per-tree bootstrap branch: when `cfg.bootstrap == true`, each tree receives `bootstrap_indices(n, seed, tree_id)` as its row set; otherwise `all_rows` (unchanged ET path).
- 9 new tests in `fit.rs`: RF clf/reg build+validate, leaf proba sums, reg finite values, seed determinism, parallel==sequential (T-02-10), cover invariant over bootstrap sample, four-estimator matrix (ET/RF × clf/reg).
- 2 new tests in `predict.rs`: RF clf proba shape+sum, RF reg finite values.
- `predict.rs` traversal is unchanged — RF reuses it directly.

## Test Outcomes (82 / 82 pass)

| Test category | Tests | Key invariants |
|---|---|---|
| `cpu::bootstrap` | 11 | count, range, determinism, with-replacement, cross-tree independence, T-02-12 |
| `cpu::split_rf` | 11 | exact midpoint separating split, distinct-only candidates, FEATURE_THRESHOLD, min_leaf, tie-break, D-01, <=, determinism, regression, cover |
| `cpu::fit` | 17 | ET+RF clf+reg build+validate, leaf proba, cover invariant, ET/RF seed det., ET/RF parallel==seq, four-estimator matrix (SC-2) |
| `cpu::predict` | 10 | shape, proba sum, reg finite, NaN routing, NaN det., feature mismatch, RF clf/reg |
| `cpu::criterion` | 14 | prior plan tests unbroken |
| `cpu::split_et` | 7 | prior plan tests unbroken |
| `ir`, `rng` | 12 | prior plan tests unbroken |

### Critical invariants confirmed

- **RF seed determinism:** `same (seed, cfg, data) -> byte-identical ForestIR` ✓
- **RF parallel == sequential (T-02-10):** rayon build twice produces identical JSON ✓
- **Cover invariant (RF):** `node_sample_count[node] == L + R` for all internal nodes ✓ (over bootstrap sample)
- **RF clf proba sums to 1:** each row's probability vector sums to 1 ± 1e-5 ✓
- **RF reg finite values:** all leaf_value entries are finite ✓
- **Four-estimator matrix (SC-2 / D-02):** ET clf, ET reg, RF clf, RF reg — all train and predict via `CpuBackend` ✓
- **No predict path change:** RF traversal goes through unchanged `predict.rs` ✓
- **No quantizer/histogram:** RF best-split is exact sort-based (plan prohibition respected) ✓
- **No copied sklearn source:** provenance comment in `split_rf.rs` documents clean-room reimplementation ✓

## Files Created/Modified

- `src/cpu/bootstrap.rs` — Philox-keyed bootstrap resampling, BOOTSTRAP_NODE_SENTINEL, 11 tests
- `src/cpu/split_rf.rs` — RF BestSplitter sorted-midpoint algorithm, RfSplitCtx, 11 tests
- `src/cpu/fit.rs` — NodeSplit dispatch, bootstrap branch in fit_forest, 9 new RF tests
- `src/cpu/predict.rs` — 2 new RF predict tests (predict path itself unchanged)
- `src/cpu/mod.rs` — `pub mod bootstrap; pub mod split_rf;` declarations

## Task Commits

1. **Task 1** (bootstrap) — `5b42c41`
2. **Task 2** (RF best-split) — `8664cde`
3. **Task 3** (wiring + SC-2) — `c9491ff`

## Decisions Made

- `NodeSplit` struct (not a tuple) in `build_node` to satisfy `clippy::type_complexity` without suppressing the lint.
- `BOOTSTRAP_NODE_SENTINEL = u32::MAX` chosen because no valid node id in any tree can reach `u32::MAX` (tree depth is bounded; node_ids are sequential usize capped by the tree's node count). Documented, unit-tested, and noted for Phase-4 CUDA reproduction.
- `prev_val` advances on every iteration in the sorted-midpoint walk (including skipped pairs). This matches BestSplitter: duplicate values simply yield no candidate threshold; the walk continues to the next distinct value rather than re-anchoring to the last distinct one.

## Deviations from Plan

None — plan executed exactly as written. The `NodeSplit` struct is an internal implementation detail (clippy-driven refactor during Task 3 commit preparation) that does not change the algorithm.

## Known Stubs

None — all ForestIR fields are fully populated for both ET and RF, for both clf and reg. No placeholder data flows to predict output.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundaries introduced. Bootstrap index clamping (T-02-09) and stream-namespace separation (T-02-12) were both implemented per the threat register and verified by unit tests. The RF split provenance (T-02-11) is documented in `split_rf.rs` header. No additional threat flags found.

## Next Phase Readiness

- **Wave 4 (02-04 property tests):** `fit_forest` + `predict_forest` are callable from proptest closures for all four estimators; `ForestIR` is serde-serializable.
- **Wave 5 (02-05 parity):** `CpuBackend` implements `Backend` for ET+RF × clf/reg — the Python harness can drive it against sklearn.
- **Phase 4 (CUDA):** `BOOTSTRAP_NODE_SENTINEL` is documented for CUDA kernel reproduction; Philox counter layout is frozen from Phase 01.

---
*Phase: 02-cpu-oracle-contracts-forest-ir*
*Completed: 2026-06-20*

## Self-Check: PASSED
