---
phase: 5
slug: full-forest-randomforest-sklearn-estimators
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-27
---

# Phase 5 — Validation Strategy

> Per-phase validation contract. Load-bearing gates: (1) `sklearn.utils.estimator_checks.check_estimator`
> green in CI for all four estimators (documented exceptions explicit); (2) GPU RandomForest matches the
> CPU oracle on the **binned** best-split contract (per the OQ2 decision: binned canonical); (3) full-forest
> ExtraTrees still bit-exact vs the Phase-4 path; (4) the end-to-end Comparative Baseline Study reports
> **accuracy parity (gate)** with speed reported under the Phase-7 crossover caveat.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: `cargo test` / `cargo-nextest`; Python: `pytest` (sklearn `parametrize_with_checks`); CUDA: `compute-sanitizer` |
| **Config file** | `Cargo.toml` workspace + `pyproject.toml` (maturin abi3); `.venv-parity` (sklearn 1.9) |
| **Quick run command** | `cargo test -p sylva-cuda forest` |
| **Full suite command** | `cargo nextest run` + `pytest python/tests/ -q` + per-tool `compute-sanitizer` on RF/forest kernels |
| **Estimated runtime** | ~3–8 min (check_estimator + study setup dominate) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p sylva-cuda forest`
- **After every plan wave:** full Rust suite + `pytest python/tests/` + racecheck/memcheck on new kernels
- **Before `/gsd-verify-work`:** `check_estimator` green for all 4 estimators; GPU-vs-CPU forest parity green; sanitizer clean
- **Max feedback latency:** ~180 s (study/check_estimator excluded from the quick loop)

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` during planning. Non-negotiable rows:
> `check_estimator` per estimator, GPU-vs-CPU forest/RF parity, sanitizer-clean per new kernel.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior (must_haves truth) | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------|-----------|-------------------|-------------|--------|
| 05-01-T1 | 01 | 1 | EST-03 | T-05-01 | TrainConfig gains max_samples/class_weight; validate() rejects bad/bootstrap-less max_samples with typed SylvaError::InvalidConfig | unit | `cargo test -p sylva-core config` ; `cargo build -p sylva-core` | ⬜ pending | ⬜ pending |
| 05-01-T2 | 01 | 1 | EST-04 | T-05-02 | feature_importances(&ForestIR) = real MDI (f64, per-tree-normalized→averaged), length n_features, sums ~1.0; leaves skipped, zero-decrease safe | unit | `cargo test -p sylva-core importance` ; `cargo clippy -p sylva-core -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-02-T1 | 02 | 1 | GPU-03 | T-05-03, T-05-04 | SIBLING_SUBTRACT_SRC integer u32 child_large=parent−child_small; index-guarded; no float atomics; runtime unit asserts parent−smaller==larger exactly | unit (device-guarded) | `cargo build -p sylva-cuda` ; `cargo test -p sylva-cuda sibling_subtract` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-02-T2 | 02 | 1 | GPU-03 | T-05-05, T-05-06 | fit_forest_gpu builds multi-tree ET breadth-first; per-tree bootstrap via shipped bootstrap_indices (Philox); multi-tree ForestIR with correct global offsets, validate_structure Ok; same-seed byte-identical | unit | `cargo test -p sylva-cuda cuda_backend` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-02-T3 | 02 | 1 | GPU-03 | T-05-04, T-05-06 | multi-tree GPU ET forest == CPU oracle byte-for-byte (serde_json string eq, clf+reg) on fixed seed; sibling subtraction integer-exact; graceful skip with no device | integration (bit-exact gate, device-guarded) | `cargo test -p sylva-cuda --test forest_cpu_gpu` ; `cargo test -p sylva-core` | ⬜ pending | ⬜ pending |
| 05-03-T1 | 03 | 2 | GPU-04 | T-05-11 | additive best_split_binned on CPU oracle (BinEdges boundaries, exact criterion op order + (feature,threshold_bits) tie-break); raw best_split + its tests untouched/green | unit | `cargo test -p sylva-core split_rf` ; `cargo test -p sylva-core` | ⬜ pending | ⬜ pending |
| 05-03-T2 | 03 | 2 | GPU-04, EST-05 | T-05-07, T-05-08, T-05-09, T-05-10 | RF_BINNED_HIST/RF_SCAN_ARGMAX/WEIGHTED_HIST kernels (integer/fixed-point only, -fmad=false); runtime smoke asserts scale_weights round-trip + RF_BINNED_HIST_SRC expected counts | unit (device-guarded) | `cargo build -p sylva-cuda` ; `cargo test -p sylva-cuda rf_kernels_smoke` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-03-T3 | 03 | 2 | GPU-04, EST-05 | T-05-07, T-05-08, T-05-09, T-05-10 | RF GPU forest == CPU binned oracle byte-for-byte (clf+reg) fixed seed; sample_weight GPU==CPU weighted counts (fixed-point); invalid weight → typed error; graceful skip no device | integration (bit-exact + parity gate, device-guarded) | `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` ; `cargo test -p sylva-core` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-04-T1 | 04 | 3 | GPU-05 | T-05-12, T-05-13 | FitArena pre-sizes device buffers ONCE via safe alloc_zeros, reused across waves/trees (no per-node/per-wave alloc); zero unsafe in arena.rs | unit (device-guarded) | `cargo test -p sylva-cuda arena` ; `grep -c 'unsafe' arena.rs == 0` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-04-T2 | 04 | 3 | GPU-06 | T-05-14 | small/deep nodes finish on CPU via shipped sylva_core::cpu builder; cutover-on forest byte-identical to all-GPU (ET+RF); scheduler draws from arena; Plan-02/03 gates still pass | unit + integration (device-guarded) | `cargo test -p sylva-cuda cutover` ; `cargo test -p sylva-cuda --test forest_cpu_gpu` ; `cargo test -p sylva-cuda --test parity_rf_cpu_gpu` ; `cargo clippy -p sylva-cuda -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-04-T3 | 04 | 3 | GPU-05, GPU-06 | T-05-15 | all new kernels (sibling-subtract, RF binned hist, scan/argmax, weighted hist) launch once for compute-sanitizer; four-tool sanitizer = 0 errors (manual dev-host gate); valid ForestIR | unit (device-guarded) + manual sanitizer | `cargo test -p sylva-cuda --test sanitizer_rf_kernels` ; `cargo clippy -p sylva-cuda -- -D warnings` ; (manual) `compute-sanitizer --tool {memcheck,racecheck,initcheck,synccheck}` | ⬜ pending | ⬜ pending |
| 05-05-T1 | 05 | 3 | EST-02, EST-04 | T-05-16 | pyseam.rs promoted: py_fit/py_predict_proba/py_get_fitted_attrs (real MDI) + parse_config max_samples/class_weight; typed errors + GIL release; Rust seam integration test on tiny synthetic | unit/integration (Rust seam) | `cargo build -p sylva-core --features pyseam` ; `cargo test -p sylva-core --features pyseam seam` ; `cargo clippy -p sylva-core --features pyseam -- -D warnings` | ⬜ pending | ⬜ pending |
| 05-05-T2 | 05 | 3 | EST-01, EST-02, EST-03 | T-05-17, T-05-18 | four sklearn-parity estimator classes (no __init__ logic, correct ET/RF clf/reg defaults); fit/predict family + fitted attrs; device='cuda' unmet + fallback='error' raises (no silent fallback) | unit (pytest API) | `cd python && python -m pytest tests/test_estimator_api.py -q` | ⬜ pending | ⬜ pending |
| 05-05-T3 | 05 | 3 | EST-06, EST-04 | T-05-18, T-05-19 | parametrize_with_checks green for all 4 estimators (device='cpu') with documented EXPECTED_FAILED; feature_importances_ length/non-neg/sums~1.0 + ranks informative > noise | integration (check_estimator CI gate) | `cd python && python -m pytest tests/test_check_estimator.py tests/test_feature_importances.py -q` | ⬜ pending | ⬜ pending |
| 05-06-CK | 06 | 4 | GPU-03, GPU-04 | T-05-22 | cuML/XGBoost baseline availability resolved + recorded honestly (cuML-native / cuML-WSL2 / unavailable); not faked | checkpoint:human-verify (blocking) | manual: `python -c "import xgboost"` / `python -c "import cuml"` → record in MANIFEST | ⬜ pending | ⬜ pending |
| 05-06-T1 | 06 | 4 | GPU-03, GPU-04 | T-05-21, T-05-23 | end-to-end-from-numpy study (cold/warm, transfers+quantization timed), accuracy beside every speed cell, like-for-like, n_jobs=-1, OOM-honest, pinned MANIFEST, crossover caveat | script (parse + import) | `cd python && python -c "import ast; ast.parse(open('benchmarks/comparative_study.py').read()); ast.parse(open('benchmarks/study_manifest.py').read())"` ; `python -c "from benchmarks.study_manifest import MANIFEST; assert len(MANIFEST)>0"` | ⬜ pending | ⬜ pending |
| 05-06-T2 | 06 | 4 | GPU-03, GPU-04 | T-05-21 | accuracy-parity GATE: Sylva ET vs sklearn ET, Sylva RF vs sklearn RF (clf accuracy / reg R2) within documented tolerance, like-for-like, device='cpu'; no timing gated | unit (pytest parity gate) | `cd python && python -m pytest tests/test_study_accuracy_parity.py -q` | ⬜ pending | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · File Exists: ⬜ pending until Wave 0 / executor creates the test file*

---

## Wave 0 Requirements

- [ ] GPU RandomForest best-split + full-forest parity tests vs CPU oracle (binned contract)
- [ ] `python/tests/estimator_checks/` — `parametrize_with_checks` over the 4 estimators with `expected_failed_checks`
- [ ] `python/tests/benchmark/` — end-to-end study harness (sklearn / cuML-labeled / XGBoost-rf), accuracy-beside-speed, cold/warm, pinned versions
- [ ] compute-sanitizer targets for the new RF/scan/weighted-histogram/sibling-subtraction kernels

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Comparative study run on the dev GPU (transfer-inclusive, cold/warm) | GPU-03/04 (study) | Needs the physical GPU + large dataset + pinned baselines (cuML may be Windows-unavailable → WSL2 or honest "unavailable") | Run the study harness on the dev host; record accuracy parity + speed cells + versions; report OOM/failures honestly |
| compute-sanitizer clean on all new kernels | GPU-03/04 | Requires physical CUDA device | `compute-sanitizer --tool {memcheck,racecheck,synccheck,initcheck}` against each kernel target |

---

## Nyquist Compliance

- [ ] Every Phase-5 success criterion maps to a verification above
- [ ] GPU-03..06 and EST-01..06 each have at least one gating check
- [ ] `check_estimator` is automated in CI (CPU path runs GPU-less; GPU path on dev host)
- Set `nyquist_compliant: true` only after the planner/plan-checker confirm every `must_haves` entry has a row.
