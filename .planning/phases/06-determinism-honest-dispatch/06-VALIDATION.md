---
phase: 6
slug: determinism-honest-dispatch
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 6 — Validation Strategy

> Load-bearing gates: (1) DET-01 — two same-seed `deterministic=True` runs produce a
> BYTE-identical model (exact binary/serde_json compare, never `allclose`); (2) DET-03 —
> `device="cuda"` when CUDA is unavailable RAISES (no silent CPU fallback), every unsupported
> dispatch config raises a typed error; (3) DET-04 — `execution_report_` records backend+reason,
> conversions, bytes H2D/D2H, fallback status. DET-02 (overhead %) is measured + reported.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: `cargo test` / `cargo-nextest`; Python: `pytest`; CUDA: `compute-sanitizer racecheck` (determinism-relevant) |
| **Config file** | `Cargo.toml` workspace + `pyproject.toml`; `.venv-parity` |
| **Quick run command** | `cargo test -p sylva-core dispatch determinism execution_report` |
| **Full suite command** | `cargo nextest run` + `pytest python/tests/ -q` + `racecheck` on the GPU train path |
| **Estimated runtime** | ~2–5 min |

---

## Sampling Rate

- **After every task commit:** `cargo test -p sylva-core dispatch determinism`
- **After every plan wave:** full Rust suite + `pytest` + `racecheck` on the train path
- **Before `/gsd-verify-work`:** two-run byte-identical gate green; dispatch error tests green; execution_report assertions green
- **Max feedback latency:** ~120 s

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` during planning. Non-negotiable rows:
> two-run byte-compare (DET-01), dispatch-raises (DET-03), execution_report fields (DET-04).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 1.. | DET-01..04 | T-06-xx | no silent fallback; typed errors; no float reduction on det path | unit + byte-compare + dispatch | `cargo nextest run` ; `pytest python/tests/` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Two-run byte-identical determinism gate (extends the shipped `tests/determinism.rs` idiom to the GPU forest)
- [ ] Dispatch decision unit tests (GPU-less, monkeypatched CUDA probe) covering auto/cuda/cpu × available/unavailable × fallback
- [ ] `execution_report_` field assertions (backend+reason, conversions, bytes, fallback status, deterministic flag)
- [ ] Determinism-overhead micro-measurement (deterministic vs fast mode) — reported, not gated

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `racecheck` clean on the GPU determinism path | DET-01 | Physical CUDA device | `compute-sanitizer --tool racecheck` on the two-run train test |
| External determinism-gap (cuML/LightGBM-GPU not byte-reproducible) | DET (study) | Needs cuML (WSL2) / LightGBM-GPU; may be Windows-unavailable | Behind `checkpoint:human-verify`; record honestly; gates do not depend on it |

---

## Nyquist Compliance

- [ ] Every Phase-6 success criterion maps to a verification above
- [ ] DET-01..04 each have at least one gating check
- [ ] The two-run byte-identical gate is automated on the dev host
- Set `nyquist_compliant: true` only after the planner/plan-checker confirm every `must_haves` entry has a row.
