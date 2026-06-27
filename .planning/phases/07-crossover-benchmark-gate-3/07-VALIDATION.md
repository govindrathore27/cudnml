---
phase: 7
slug: crossover-benchmark-gate-3
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 7 — Validation Strategy

> This is a measurement + documentation phase (Gate 3). Load-bearing gates: (1) the
> pre-registration doc is committed BEFORE any measurement cell runs (git timestamp =
> tamper evidence); (2) the harness encodes every PITFALLS.md fairness rule structurally
> (end-to-end-from-float64-numpy timing incl. dtype+H2D+quantization, cold/warm separated,
> like-for-like ET-vs-ET / RF-vs-RF, cuML labeled different algorithm, accuracy beside
> every speed cell, pinned MANIFEST, OOM/failures honest); (3) the (n×d) crossover surface
> is published; (4) the KILL CRITERION is adjudicated at a blocking human-verify checkpoint.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python `pytest` (harness/fairness unit tests) + the study scripts themselves; Rust unchanged |
| **Config file** | `pyproject.toml`; `.venv-parity`; `07-PRE-REGISTRATION.md` (frozen protocol); `crossover_manifest.py` (pinned versions) |
| **Quick run command** | `cd python && python -m pytest tests/test_crossover_fairness.py -q` |
| **Full suite command** | the fairness unit tests + a tiny smoke grid run (`crossover_study.py --smoke`) |
| **Estimated runtime** | fairness tests ~30 s; full measurement grid is long-running (hours) and dev-host/manual |

---

## Sampling Rate

- **After every task commit:** `pytest tests/test_crossover_fairness.py`
- **After every plan wave:** fairness tests + a smoke grid cell (tiny n×d) to prove the harness end-to-end
- **Before `/gsd-verify-work`:** pre-registration committed; fairness tests green; surface artifact produced; kill-criterion checkpoint adjudicated
- **Max feedback latency:** ~30 s for the fairness gate (the full grid is intentionally long/manual)

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` during planning. Non-negotiable rows:
> pre-registration-before-measurement, fairness-rule assertions, surface artifact, kill-criterion checkpoint.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 0.. | BENCH-01..03 | T-07-xx | timing region honest; no p-hacking; OOM reported | unit (fairness) + checkpoint | `pytest tests/test_crossover_fairness.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `07-PRE-REGISTRATION.md` — frozen grid, datasets, hyperparameters, baselines, kill-criterion wording, pivot path (committed before measurement)
- [ ] `tests/test_crossover_fairness.py` — asserts the timed region includes dtype coercion + H2D + quantization; cold/warm separated; cuML labeled different algorithm; accuracy recorded beside speed
- [ ] `benchmarks/crossover_study.py` + `crossover_manifest.py` — the (n×d) grid loop + pinned version manifest

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full (n×d) crossover measurement | BENCH-01/02/03 | Long-running; needs the dev GPU + large datasets + pinned baselines | Run `crossover_study.py` on the dev host per the pre-registration; record every cell + versions |
| Baseline availability (sklearnex / cuML / XGBoost) | BENCH-02 | Windows availability varies (cuML likely WSL2 or unavailable) | `checkpoint:human-verify`: record which baselines ran where, honestly |
| KILL-CRITERION adjudication (proceed vs pivot) | BENCH-03 | A human decision on the published surface | `checkpoint:human-verify`: if no cell shows GPU ET beating the strongest CPU baseline (speedup > observed IQR, accuracy within tol) → pivot path |

---

## Nyquist Compliance

- [ ] BENCH-01..03 + the kill criterion each map to a verification above
- [ ] The fairness-rule assertions are automated unit tests (run GPU-less)
- [ ] Pre-registration is committed before measurement (enforced as Task 0 / a blocking gate)
- Set `nyquist_compliant: true` once the planner/plan-checker confirm every `must_haves` entry has a row.
