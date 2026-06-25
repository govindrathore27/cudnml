---
phase: 02-cpu-oracle-contracts-forest-ir
verified: 2026-06-21T00:00:00Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 02: CPU Oracle, Contracts & Forest IR — Verification Report

**Phase Goal:** Stand up the device-neutral contracts and a trusted pure-Rust CPU backend that trains and predicts ET + RF correctly — the correctness oracle that makes every later GPU result verifiable and enables GPU-less cloud CI.
**Verified:** 2026-06-21
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| #  | Truth                                                                                                                                                                                       | Status     | Evidence                                                                                                                                                                                                                  |
|----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | A device-neutral `trait Backend` (quantize/build_histograms/eval_splits/partition/predict) exists with no CUDA types crossing the trait boundary, and a SoA `ForestIR` is the single shared representation written by training and read read-only by inference, SHAP, and export | VERIFIED   | `backend.rs`: `trait Backend` (fit+predict) + `trait HistogramBackend` (quantize/build_histograms/eval_splits/partition). All 5 Phase-4 op names present. `ir.rs`: full SoA ForestIR with all consumer arrays. `cargo tree -p sylva-core` default build: 0 cudarc, 0 pyo3 hits. Clippy clean. |
| 2  | A pure-Rust `CpuBackend` (ndarray + rayon) trains and predicts ExtraTrees + RandomForest correctly and serves as the differential-test oracle and `device="cpu"` / small-data path        | VERIFIED   | `cpu/mod.rs`: `CpuBackend` implements `Backend`. `cpu/fit.rs`: both `Algo::ExtraTrees` and `Algo::RandomForest` dispatched. `cpu/predict.rs`: NaN-safe traversal. 90 unit tests + 12 determinism integration tests + 9 proptest invariants all pass (112 total, 0 failures). |
| 3  | The parity contract is documented (bit-identical CPU↔GPU RNG per seed + distributional equivalence to sklearn, NOT bit-identical replay of sklearn's serial PRNG), and a stateless Philox-4×32-10 RNG is implemented in Rust keyed by (seed, tree, node, feature, draw)             | VERIFIED   | `docs/PARITY.md` exists, ~160 lines. Explicitly distinguishes exact vs distributional guarantees. Philox constants (PHILOX_M0/M1/W0/W1, PHILOX_ROUNDS=10) in `rng/mod.rs`. Counter layout `[tree, node, feature, draw]` in `pack_counter`. KAT vectors in `rng/kat.rs` match authoritative Random123 `kat_vectors.txt` rows; `philox_matches_kat_vectors` test passes for all 3 vectors. |
| 4  | NaN / missing-value routing policy is defined and implemented consistently, with NaN fixtures in the test suite                                                                               | VERIFIED   | `predict.rs` line 93: `v.is_nan()` checked BEFORE threshold comparison; NaN → `default_child[node]`. D-01 comment present at the exact branch. 4 NaN-routing tests in `predict.rs`: `nan_row_routes_to_default_child`, `nan_row_not_nan_goes_threshold_path`, `nan_routing_is_deterministic`, `nan_default_direction_matches_ir`. All pass. |
| 5  | Differential tests vs scikit-learn and property-based invariants pass — child rows partition parent, leaf probabilities valid, seed determinism, serialization round-trip                   | VERIFIED   | `tests/invariants.rs`: 9 proptest tests covering cover-partition, leaf-proba-valid, leaf-values-finite, default-child, serde-round-trip, seed-determinism across ET/RF×clf/reg. `tests/determinism.rs`: 12 tests including parallel-vs-sequential rayon order-independence, multi-seed coverage. All 21 integration tests pass.                         |
| 6  | Comparative Baseline Study: CpuBackend reaches accuracy/distribution PARITY with sklearn ET/RF within documented tolerance on the named dataset                                              | VERIFIED   | `python/tests/parity/test_distributional_parity.py` ran via orchestrator. All 4 estimators pass: ET-clf accuracy parity (tol=0.010), RF-clf accuracy parity (tol=0.010), ET-reg R2 parity (tol=0.011), RF-reg R2 parity (tol=0.010). KS feature-freq gate passes for 3/4; regression KS is INFORMATIONAL (see Parity Gate section below). Thresholds in `thresholds.json` calibrated from 12-seed sklearn-vs-sklearn null spread, 200 estimators. |
| 7  | CpuBackend CPU training time vs sklearn is reported as informational only (not gated) on same dataset, cold/warm separated, versions pinned                                                  | VERIFIED   | `test_distributional_parity.py` lines 312–318 and 409–415: cold/warm timing printed in each per-estimator test block with "INFORMATIONAL ONLY - no speed claim" label. Version manifest in `conftest.py` (sklearn 1.9.0, numpy 2.4.6, scipy 1.18.0, Python 3.12.8). Timing never asserted against any threshold. |

**Score: 7/7 truths verified (0 present-but-behavior-unverified)**

---

## Requirement Coverage

| Requirement | Phase 2 Plan     | Description                                                                    | Codebase Status                | Evidence                                        |
|-------------|-----------------|--------------------------------------------------------------------------------|--------------------------------|-------------------------------------------------|
| ENG-01      | 02-01           | Device-neutral `trait Backend` with all device ops, no CUDA types              | ACHIEVED (Phase 2 scope)       | `backend.rs`: 5 trait fn names present, zero CUDA imports in entire crate |
| ENG-02      | 02-01           | SoA `ForestIR` single shared representation                                    | ACHIEVED                       | `ir.rs`: 15+ SoA arrays, serde, `validate_structure`, SHAP cover fields |
| ENG-03      | 02-02, 02-03    | CpuBackend ET+RF clf+reg correct oracle                                        | ACHIEVED                       | 112 tests pass; all 4 estimators build, validate, predict correctly |
| ENG-04      | 02-04, 02-05    | Parity contract documented + Philox CPU bit-identical per seed                 | ACHIEVED                       | `docs/PARITY.md` + byte-identical determinism tests + parity gate |
| ENG-05      | 02-02           | NaN/missing routing defined and implemented consistently                        | ACHIEVED                       | `predict.rs` NaN branch + 4 NaN tests pass |
| ENG-06      | 02-01           | Philox-4×32-10 implemented in Rust, keyed by (seed,tree,node,feature,draw) — CUDA identity is Phase 4 | ACHIEVED (Rust portion) | `rng/mod.rs`: correct constants, `pack_counter`, KAT test passes; CUDA portion deferred to Phase 4 per ROADMAP |
| EST-07      | 02-04, 02-05    | Differential tests + property invariants                                       | ACHIEVED                       | `tests/invariants.rs` (proptest, 9 tests) + parity gate (Python) |

**Note on ENG-01 and ENG-06 REQUIREMENTS.md checkboxes:** Both are marked `[ ]` in REQUIREMENTS.md, which reflects the full cross-phase requirement text ("CUDA types never cross" and "in Rust AND the CUDA kernel"). The CUDA kernel portion of ENG-06 is a Phase 4 deliverable by ROADMAP. Phase 2's ROADMAP success criteria address only the Rust-side obligations, which are fully met.

---

## Required Artifacts

| Artifact                                          | Expected                                           | Status     | Details                                               |
|---------------------------------------------------|----------------------------------------------------|------------|-------------------------------------------------------|
| `crates/sylva-core/src/backend.rs`                | `trait Backend` + `trait HistogramBackend`, CUDA-free | VERIFIED | 98 lines; both traits present; no CUDA imports        |
| `crates/sylva-core/src/ir.rs`                     | SoA `ForestIR` struct + serde round-trip            | VERIFIED   | 219 lines; all arrays; serde; `validate_structure`     |
| `crates/sylva-core/src/config.rs`                 | `TrainConfig`, `Criterion`, `MaxFeatures`, `Task`   | VERIFIED   | 111 lines; `validate()` returns typed error           |
| `crates/sylva-core/src/error.rs`                  | Typed `SylvaError` via thiserror                    | VERIFIED   | 3 variants; no `.unwrap()` in production paths        |
| `crates/sylva-core/src/rng/mod.rs`                | Philox-4×32-10 + conversion + counter packing        | VERIFIED   | 145 lines; correct constants; KAT test; `philox_uniform` |
| `crates/sylva-core/src/rng/kat.rs`                | Frozen human-verified KAT vectors                   | VERIFIED   | 3 vectors matching canonical Random123 rows           |
| `crates/sylva-core/src/cpu/fit.rs`                | ET+RF recursive builder, rayon                       | VERIFIED   | 991 lines; dispatches both `Algo` variants; boundary validation |
| `crates/sylva-core/src/cpu/predict.rs`            | NaN-safe traversal + aggregation                     | VERIFIED   | 421 lines; NaN-first check at line 93; 4 NaN tests    |
| `crates/sylva-core/src/cpu/split_et.rs`           | ET random-threshold splitter with Philox              | VERIFIED   | 522 lines; `best_random_split` with Philox draws       |
| `crates/sylva-core/src/cpu/split_rf.rs`           | RF exact best-split sorted-midpoint search            | VERIFIED   | 550 lines; `best_split` with midpoint candidates       |
| `crates/sylva-core/src/cpu/criterion.rs`          | Gini / entropy / MSE, f32 sequential accumulation     | VERIFIED   | 231 lines; all three criteria; `proxy_improvement`     |
| `crates/sylva-core/src/cpu/bootstrap.rs`          | Philox-keyed with-replacement bootstrap                | VERIFIED   | 229 lines; sentinel namespace separation              |
| `crates/sylva-core/src/parity.rs`                 | `split_statistics` / `SplitStats` extractor           | VERIFIED   | 352 lines; serde; doctest passes                      |
| `crates/sylva-core/src/pyseam.rs`                 | Test-only PyO3 seam (fit_cpu/predict_cpu/split_statistics) | VERIFIED | 396 lines; feature-gated; no `.unwrap()` across FFI   |
| `docs/PARITY.md`                                  | Parity contract document (ENG-04)                    | VERIFIED   | 163 lines; 4 numbered points; deferred items listed   |
| `python/tests/parity/test_distributional_parity.py` | Parity gate harness                                | VERIFIED   | 592 lines; 4 estimator test classes + fairness report |
| `python/tests/parity/thresholds.json`             | Calibrated thresholds from null spread               | VERIFIED   | 12-seed, 200-estimator calibration; provenance metadata |
| `python/tests/parity/test_shallow_depth_proof.py` | Shallow-depth KS proof for regression               | VERIFIED   | Proves deep KS divergence is RNG artifact, not split bug |
| `crates/sylva-core/tests/determinism.rs`          | 12 byte-identical determinism + rayon independence tests | VERIFIED | All 12 pass; covers ET/RF×clf/reg + entropy + multi-seed |
| `crates/sylva-core/tests/invariants.rs`           | 9 proptest structural invariant tests                | VERIFIED   | All 9 pass; cover-partition, proba-valid, serde, determinism |

---

## Key Link Verification

| From                          | To                              | Via                                                     | Status  |
|-------------------------------|---------------------------------|---------------------------------------------------------|---------|
| `cpu/mod.rs`                  | `backend.rs`                    | `impl Backend for CpuBackend` in `cpu/mod.rs`           | WIRED   |
| `cpu/fit.rs`                  | `rng/mod.rs`                    | `use crate::rng::philox_uniform` (split_et, split_rf, bootstrap) | WIRED |
| `cpu/fit.rs`                  | `cpu/split_et.rs`               | `best_random_split(&et_ctx)` at line 251                | WIRED   |
| `cpu/fit.rs`                  | `cpu/split_rf.rs`               | `rf_best_split(&rf_ctx)` at line 273                    | WIRED   |
| `cpu/predict.rs`              | `ir.rs`                         | `ir.default_child[node]` NaN path at line 97            | WIRED   |
| `pyseam.rs`                   | `cpu/mod.rs`                    | `crate::cpu::CpuBackend` → `backend.fit` / `backend.predict` | WIRED |
| `pyseam.rs`                   | `parity.rs`                     | `parity::split_statistics(&ir)` at line 371             | WIRED   |
| `parity.rs`                   | `ir.rs`                         | Reads `ir.feature_id`, `ir.threshold`, `ir.is_leaf`     | WIRED   |
| `tests/determinism.rs`        | `cpu/mod.rs`                    | `use sylva_core::{cpu::CpuBackend, Backend}`            | WIRED   |
| `tests/invariants.rs`         | `cpu/mod.rs`                    | `use sylva_core::{cpu::CpuBackend, Backend}`            | WIRED   |
| `test_distributional_parity.py` | `pyseam.rs` (via `sylva_core_pyseam`) | `sylva.fit_cpu`, `sylva.predict_cpu`, `sylva.split_statistics` | WIRED |

---

## Data-Flow Trace (Level 4)

Not applicable for this phase — there are no frontend/dashboard components. The dynamic data path is: Python numpy arrays → pyseam FFI → `CpuBackend::fit` → `ForestIR` JSON → `predict_cpu` → numpy output. The orchestrator confirmed this full chain runs (parity harness passed), which constitutes Level 4 evidence for the parity pipeline.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 90 unit tests pass | `cargo test -p sylva-core` (unit) | 90/90 ok | PASS |
| 12 determinism integration tests pass | `cargo test -p sylva-core` (tests/determinism.rs) | 12/12 ok | PASS |
| 9 proptest invariant tests pass | `cargo test -p sylva-core` (tests/invariants.rs) | 9/9 ok | PASS |
| 1 doctest passes | `cargo test -p sylva-core` (doc-tests) | 1/1 ok | PASS |
| Clippy -D warnings clean | `cargo clippy -p sylva-core` | Finished, 0 warnings | PASS |
| `cargo tree` default: no cudarc/pyo3 | `cargo tree -p sylva-core \| grep cudarc\|pyo3` | 0 hits | PASS |

---

## Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for Phase 2. The Python parity harness is the functional probe and was run end-to-end by the orchestrator (RF ~27 min; ET faster). The shallow-depth proof test was also run by the orchestrator. Verifier corroborated the full Rust test suite independently.

---

## Parity Gate Assessment (SC-6 / D-04)

The regression KS INFORMATIONAL decision deserves explicit scrutiny, as it is the only place a gate was weakened from blocking to informational.

**The claim:** Deep (max_depth=12) ET-regression feature-frequency KS p~1e-4 (fails p>0.05 floor), but this is an RNG-stream compounding artifact, not a split-logic bug.

**The evidence:**

1. `test_shallow_depth_proof.py` runs at max_depth=4 (shallow) and max_depth=12 (deep) with identical seeds and hyperparameters (including `max_features=sqrt` for both, confirmed in `datasets.py` `REG_HYPERPARAMS`).
2. At shallow depth the KS p=0.2719 (passes well above 0.05 — confirmed by orchestrator run).
3. At deep depth p~1e-4 (fails).
4. Both Sylva and sklearn use `max_features=sqrt` for regression in `REG_HYPERPARAMS` — the old "max_features=all" rationale cited by the executor was incorrect; the actual code confirms `max_features: "sqrt"` in `datasets.py` line 60.
5. The logical chain: shallow=non-significant → RNG can't have compounded yet → algorithm is faithfully equivalent. Deep=significant → two independent RNG streams (Philox stateless vs sklearn's `our_rand_r` serial advance) diverge at 12 levels of draws. This is predicted by RESEARCH Pitfall #1 and explicitly acknowledged in `PARITY.md` Point 3.

**Verdict:** The INFORMATIONAL designation is correct and evidence-backed. The shallow-depth proof closes the D-04 concern. The regression R2 parity gate passes (tol=0.011, measured null-spread-derived) and is the substantive correctness check.

---

## Requirements Coverage (Prohibitions Audit)

| Prohibition                                                             | Status   | Evidence                                                                              |
|-------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------|
| NO cudarc / CUDA / device-pointer types in any trait signature or field | HELD     | `cargo tree -p sylva-core` default: 0 cudarc. Grep for "cudarc\|cuda" in `src/`: 0 hits in production code |
| NO `.unwrap()` / `.expect()` on fallible boundary/config paths         | HELD     | Grep of production (non-test) files: only `pyseam.rs` uses `.unwrap_or_else` for string parsing, not on fallible I/O; no `.unwrap()` in `backend.rs`, `ir.rs`, `config.rs`, `error.rs`, `rng/`, `cpu/mod.rs`, `cpu/predict.rs`, `cpu/criterion.rs`, `parity.rs`. Test-only code uses `.expect()` appropriately. |
| NO copied scikit-learn / GPL source                                     | HELD     | All files contain provenance comments ("NOT copied from sklearn source or any GPL code", "reimplemented from standard definitions", "reimplemented from the published algorithm"). Philox comment cites Apache-2.0 Random123. |
| NO hardcoded magic numbers                                              | HELD     | `PHILOX_M0`, `PHILOX_M1`, `PHILOX_W0`, `PHILOX_W1`, `PHILOX_ROUNDS`, `FEATURE_THRESHOLD`, `DRAW_THRESHOLD`, `DRAW_FEATURE_SELECT`, `BOOTSTRAP_NODE_SENTINEL`, `LEAF_FEATURE`, `NO_CHILD` — all named constants with documentation. |
| NO f64 in IR or model numeric arrays (D-05)                             | HELD     | `ir.rs` has no `f64` fields. `config.rs` uses `f64` only as a transient intermediate inside `MaxFeatures::resolve` (`(n_features as f64).sqrt().floor() as usize`) — never stored. All IR arrays (`threshold`, `leaf_value`, `leaf_proba`, `impurity`, `node_weighted_count`) are `f32` or integer. |
| NO estimator API surface (EST-02), quantizer, GPU, SHAP, Treelite export leaked | HELD | `pyseam.rs` exposes only `fit_cpu`, `predict_cpu`, `split_statistics`. Module comment explicitly lists deferred items. Pyseam is feature-gated (`#[cfg(feature = "pyseam")]`). |

---

## Anti-Patterns Found

| File                                    | Pattern         | Severity    | Notes                                                     |
|-----------------------------------------|-----------------|-------------|-----------------------------------------------------------|
| None                                    | —               | —           | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER found in any Phase-2 source file. No unreferenced debt markers. |

---

## Human Verification Required

None. All must-haves are substantively verifiable from the codebase. The parity gate was run by the orchestrator and the corroborating shallow-depth proof provides programmatically-inspectable evidence for the INFORMATIONAL designation.

---

## Gaps Summary

No gaps. All 7 ROADMAP success criteria are met with direct codebase evidence:

- SC-1 (Backend + ForestIR): structural code present, cudarc-clean, all consumer arrays in IR.
- SC-2 (CpuBackend oracle): 112 tests pass, all 4 estimators (ET/RF × clf/reg) build, validate, and predict.
- SC-3 (Parity contract + Philox): `docs/PARITY.md` written, Philox KAT passes, parity gate ran.
- SC-4 (NaN routing): code present + 4 NaN tests pass.
- SC-5 (Differential + property tests): 21 integration tests (12 determinism + 9 proptest) pass.
- SC-6 (Baseline study accuracy/distribution parity): all 4 estimators pass accuracy/R2 gates; regression KS INFORMATIONAL designation is evidence-backed.
- SC-7 (Timing informational): timing printed in each test, never asserted, versions pinned.

**Phase 2 is cleared to proceed to Phase 3 (Feature Quantizer).**

---

## Risks and Notes for Phase 3

1. **ENG-06 CUDA portion remains:** The CUDA kernel Philox reimplementation and KAT-match is a Phase 4 gate. The KAT vectors are frozen in `rng/kat.rs` — Phase 4 must reproduce all 3 exactly before any GPU training result is accepted.

2. **f64 transient in `MaxFeatures::resolve`:** The `(n_features as f64).sqrt().floor()` cast in `config.rs` line 54 uses `f64` for accuracy in the sqrt, then converts back to `usize`. This is intentional and does not violate D-05 (which prohibits f64 storage in IR/model arrays). Phase 3 should use the same pattern for any intermediate resolution math.

3. **Regression KS ongoing obligation:** The shallow-depth proof must be re-run against any changes to the ET-regression split path to confirm the INFORMATIONAL status remains valid. If future work narrows the deep KS gap (via improved RNG matching), the informational designation can be upgraded to a gate.

4. **REQUIREMENTS.md checkbox state:** ENG-01 (`[ ]`) and ENG-06 (`[ ]`) are deliberately left unchecked in REQUIREMENTS.md because their full cross-phase requirements (CUDA type never crosses, CUDA kernel identical) are not yet satisfied. This is correct — they should be checked only when Phase 4 completes the CUDA side.

---

_Verified: 2026-06-21_
_Verifier: Claude (gsd-verifier)_
