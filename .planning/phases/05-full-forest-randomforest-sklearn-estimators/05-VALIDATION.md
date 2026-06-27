---
phase: 5
slug: full-forest-randomforest-sklearn-estimators
status: draft
nyquist_compliant: false
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

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 1.. | GPU-03..06 / EST-01..06 | T-05-xx | typed errors; no OOB/race; honest dispatch | unit + sanitizer + check_estimator | `cargo nextest run` ; `pytest python/tests/` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

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
