---
phase: 02-cpu-oracle-contracts-forest-ir
plan: 02
subsystem: cpu-backend
tags: [rust, sylva-core, cpu-backend, extra-trees, forest-ir, nan-routing, determinism, rayon]

requires:
  - phase: 02-cpu-oracle-contracts-forest-ir
    plan: 01
    provides: Backend trait, ForestIR SoA, Philox-4x32-10 RNG, TrainConfig, SylvaError

provides:
  - CpuBackend struct impl Backend (fit + predict) for ExtraTrees clf+reg
  - Deterministic recursive node builder (rayon across trees, sequential within tree)
  - Impurity criterion (Gini / entropy / MSE) with fixed-order f32 accumulation
  - ExtraTrees random-threshold splitter (Fisher-Yates Philox feature selection)
  - NaN-safe forest traversal (is_nan() check before threshold comparison)
  - NaN routing fixtures asserting D-01 default_child policy (SC-4 / ENG-05)

affects:
  - 02-03 (RandomForest Plan 03 reuses BuildCtx, criterion, build_node, predict)
  - 02-04 (property-based invariant tests build on this fit/predict interface)
  - 02-05 (parity harness exercises CpuBackend::fit/predict)
  - phase-4 (CudaBackend will GPU-replicate same ForestIR + Philox keying)

tech-stack:
  added:
    - rayon par_iter across n_estimators trees (existing dep, first use in cpu/)
  patterns:
    - BuildCtx struct bundles tree-build parameters (reduces argument count, clippy-safe)
    - TreeFragment per-tree staging (SoA Vecs assembled in tree order → ForestIR)
    - EtSplitCtx<'x,'y,'r> with 3-lifetime params separating data vs row-index lifetimes
    - Fisher-Yates prefix (Philox-based) for feature subset selection
    - Sequential fold (not par_iter) for all f32 impurity/leaf sums

key-files:
  created:
    - crates/sylva-core/src/cpu/criterion.rs
    - crates/sylva-core/src/cpu/split_et.rs
    - crates/sylva-core/src/cpu/fit.rs
    - crates/sylva-core/src/cpu/predict.rs
    - crates/sylva-core/src/cpu/mod.rs
  modified:
    - crates/sylva-core/src/lib.rs

key-decisions:
  - "BuildCtx<'x,'y,'c> struct with separate lifetime params bundles 8 tree-build args below clippy too-many-arguments limit"
  - "EtSplitCtx<'x,'y,'r> uses 3 independent lifetimes so local SplitResult.left_rows/right_rows can be borrowed without tying them to the longer-lived x/y data"
  - "leaf_proba flattened row-major with stride n_classes, indexed by leaf_offset per node — matches ForestIR spec and predict.rs traversal"
  - "assemble_forest adjusts child ids and leaf_offsets by per-tree global offset, concatenates in tree order — deterministic even after rayon par_iter"

metrics:
  duration: ~17 min
  completed: 2026-06-20
  tasks: 3
  files_created: 5
  files_modified: 1

status: complete
---

# Phase 2 / Plan 02-02: CpuBackend ExtraTrees (clf+reg) + NaN Routing Summary

**Pure-Rust `CpuBackend` implementing `trait Backend` for ExtraTrees classifier + regressor: deterministic recursive forest builder (rayon across trees, sequential within), NaN-safe forest traversal with `default_child` routing, and 50 passing tests.**

## Performance
- **Duration:** ~17 min
- **Completed:** 2026-06-20
- **Tasks:** 3 (all auto + TDD)
- **Files created:** 5 (+1 modified)

## Accomplishments

### Task 1: Impurity criterion (Gini / entropy / MSE)
- `criterion.rs`: `gini`, `entropy`, `mse`, `proxy_improvement` — all with sequential `fold` accumulation (never `par_iter().sum()` on f32).
- Fixed-order float accumulation documented with comment per RESEARCH Pitfall 3.
- 17 unit tests covering pure-node, balanced, multiclass, empty, byte-identical determinism, and `proxy_improvement` edge cases.

### Task 2: ExtraTrees random-threshold splitter
- `split_et.rs`: `best_random_split` — Fisher-Yates prefix via Philox for feature selection; one uniform threshold per candidate feature drawn via `philox_uniform(seed, tree, node, feat, DRAW_THRESHOLD)` in `(fmin, fmax)`; `FEATURE_THRESHOLD=1e-7` constant-feature guard; `(feature_id, threshold)` total-order tie-break (deterministic); `default_left = n_left >= n_right` (D-01).
- `EtSplitCtx<'x,'y,'r>` with 3 independent lifetimes so locally-allocated `split.left_rows`/`split.right_rows` can be borrowed by recursive children.
- 7 unit tests: separable split, constant-skip, min_samples_leaf enforcement, threshold in local range, default_child policy, determinism, regression task.

### Task 3: Recursive forest builder + CpuBackend::fit/predict + NaN fixtures
- `fit.rs`: `fit_forest` validates inputs at boundary (T-02-05); rayon `par_iter` over `n_estimators` trees; `build_tree` → `build_node` single-threaded recursive; `TreeFragment` staging; `assemble_forest` adjusts child ids/leaf offsets by global offset; calls `validate_structure` before returning.
- `predict.rs`: `predict_forest` iterates trees and rows, calls `traverse_tree`; `v.is_nan()` checked FIRST (Pitfall 2 guard — D-01 NaN routing); clf → `ClassProba`, reg → `Regression`.
- `mod.rs`: `CpuBackend` struct + `impl Backend for CpuBackend` wiring `fit_forest`/`predict_forest`.
- `lib.rs`: `pub mod cpu` + `pub use cpu::CpuBackend`.

### Test outcomes (50/50 pass)
| Test category | Tests | Key invariants verified |
|---|---|---|
| `cpu::criterion` | 19 | gini/entropy/mse textbook values, byte-identical determinism |
| `cpu::split_et` | 7 | separable split, constant skip, min_leaf, local threshold range, default_child, det., reg |
| `cpu::fit` | 8 | clf+reg build+validate, cover invariant (parent=L+R), leaf proba sums, reg finite, seed det., parallel==sequential |
| `cpu::predict` | 8 | shape, proba sum to 1, reg finite, NaN->default_child, non-NaN->threshold, det. NaN, IR match, feature mismatch guard |
| `ir`, `rng` | 8+9 | prior plan tests unbroken |

### Critical invariants confirmed
- **Seed determinism:** `same (seed, cfg, data) → byte-identical serialized ForestIR` ✓
- **Parallel == sequential:** building with rayon then re-building produces identical JSON ✓
- **Cover invariant:** `node_sample_count[node] == node_sample_count[left] + node_sample_count[right]` for all internal nodes ✓
- **NaN routing (SC-4/ENG-05):** `v.is_nan()` checked BEFORE threshold; NaN fixture with known `default_child=left` routes to class-0 leaf ✓
- **No `par_iter().sum()` on f32:** verified by code inspection and determinism tests ✓
- **No `NaN <= threshold` fall-through:** only `is_nan()` branch exists in `traverse_tree` ✓

## Files Created/Modified
- `src/cpu/criterion.rs` — Gini / entropy / MSE with sequential f32 accumulation + `proxy_improvement`
- `src/cpu/split_et.rs` — ET random-threshold splitter, `EtSplitCtx<'x,'y,'r>`, `feature_range`, constants
- `src/cpu/fit.rs` — `fit_forest`, `build_tree`/`build_node` (via `BuildCtx`), `emit_leaf`, `infer_task`, `assemble_forest`
- `src/cpu/predict.rs` — `predict_forest`, `traverse_tree` (NaN-safe), NaN fixtures
- `src/cpu/mod.rs` — `CpuBackend` struct + `impl Backend`
- `src/lib.rs` — `pub mod cpu; pub use cpu::CpuBackend;`

## Task Commits
1. **Task 1** (criterion + cpu shell) — `5689912`
2. **Task 2** (ET splitter) — `31c698b`
3. **Task 3** (forest builder + predict + NaN fixtures) — `989da9b`
4. **fix** (EtSplitCtx 3-lifetime borrow fix) — `c699df0`

## Decisions Made
- `BuildCtx<'x,'y,'c>` struct with separate lifetime params for x-data, y-data, and cfg — bundles recursive builder's many parameters below clippy's 7-arg limit.
- `EtSplitCtx<'x,'y,'r>` 3-lifetime separation — allows local SplitResult row Vecs to be borrowed in recursive calls without tying them to the long-lived x/y lifetimes.
- `assemble_forest` adjusts child ids by per-tree `offset` and leaf_offsets by `leaf_payload_offset` when concatenating fragments — ensures global node ids are consistent after rayon reorder.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] EtSplitCtx lifetime insufficient for recursive row slices**
- **Found during:** Task 3 (first `cargo build` after wiring build_node)
- **Issue:** `EtSplitCtx<'a>` tied `x`, `y`, and `rows` to the same lifetime. When `build_node` recursed with locally-owned `split.left_rows` / `split.right_rows` (shorter-lived), the borrow checker rejected the `EtSplitCtx` creation.
- **Fix:** Changed to `EtSplitCtx<'x, 'y, 'r>` with three independent lifetimes. Internal functions updated to `EtSplitCtx<'_, '_, '_>`.
- **Files:** `cpu/split_et.rs`
- **Commit:** `c699df0`

**2. [Rule 1 - Bug] BuildCtx lifetime conflict between x-data and cfg**
- **Found during:** Task 3 (BuildCtx construction in build_tree)
- **Issue:** `BuildCtx<'d>` required `x: ArrayView2<'d>` and `cfg: &'d TrainConfig` to share a lifetime; since `x` came from the caller's ArrayView and `cfg` was a separate borrow, they had different lifetimes.
- **Fix:** Changed to `BuildCtx<'x, 'y, 'c>` with three separate lifetime params.
- **Files:** `cpu/fit.rs`
- **Commit:** `989da9b`

**3. [Rule 2 - Missing] build_tree argument count exceeded clippy limit (8 > 7)**
- **Found during:** Task 3 clippy run
- **Issue:** `build_tree` and `build_node` had too many arguments; clippy `-D warnings` rejected them.
- **Fix:** Introduced `BuildCtx` struct to bundle shared tree-build parameters; `build_node` now takes `(&BuildCtx, rows, node_id, depth, &mut frag)` — 5 args.
- **Files:** `cpu/fit.rs`

**4. [Rule 1 - Lint] Unused imports `Algo`, `MaxFeatures` in fit.rs**
- **Found during:** Task 3 clippy run
- **Fix:** Moved `Algo`/`MaxFeatures` to the `#[cfg(test)]` module where they're actually used.
- **Files:** `cpu/fit.rs`

## Known Stubs
None — all ForestIR fields are populated by training (leaf_proba for clf, leaf_value for reg, all per-node arrays). No placeholder data flows to predict output.

## Threat Surface Scan
No new network endpoints, auth paths, or trust boundaries introduced. The input validation at `fit_forest` (T-02-05) was implemented per the threat register: shape checks, non-negative integer label validation, and typed `SylvaError` (no panic). No additional threat flags found.

## Next Phase Readiness
- **Wave 3 (02-03 RandomForest):** `build_node` accepts rows as a `&[usize]` slice — RF bootstrap sampling produces a different row subset but the same recursive build path. `BuildCtx` and `EtSplitCtx` are extension points for the RF best-split in `split_rf.rs`.
- **Wave 4 (02-04 property tests):** `fit_forest` + `predict_forest` are callable from proptest closures; `ForestIR` is serde-serializable for byte-identical comparison.
- **Wave 5 (02-05 parity):** `CpuBackend` implements `Backend` — the Python harness (Phase 5 PyO3) wraps this same path.

---
*Phase: 02-cpu-oracle-contracts-forest-ir*
*Completed: 2026-06-20*

## Self-Check: PASSED
