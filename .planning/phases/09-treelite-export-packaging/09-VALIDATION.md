---
phase: 9
slug: treelite-export-packaging
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 9 — Validation Strategy

> Load-bearing gates: (1) EXP-01 — ForestIR → a valid Treelite 4.x model (via `ModelBuilder`;
> a Wave-0 spike confirms whether a 3.x `import_from_json` JSON shim also exists); (2) EXP-02 —
> round-trip parity: exported model predictions via `treelite.gtil.predict()` (Windows-portable,
> pure Python) match Sylva native `predict`/`predict_proba` within tolerance, as a CI test
> (TL2cgen compiled-CPU secondary; GPU FIL is a WSL2/manual benchmark); (3) EXP-03 — the abi3
> Windows wheel imports in a fresh venv with CUDA dynamic-loading. Inference throughput is reported, not gated.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: `cargo test`; Python: `pytest` (treelite gtil round-trip, clean-venv import); maturin (wheel build) |
| **Config file** | `Cargo.toml` (+ `sylva-export` if a crate); `pyproject.toml` (maturin abi3, dynamic-loading); `.venv-parity` (treelite/tl2cgen) |
| **Quick run command** | `cd python && pytest tests/test_treelite_roundtrip.py -q` |
| **Full suite command** | `cargo nextest run` + `pytest tests/ -q` + a clean-venv `pip install dist/*.whl && python -c "import sylva"` |
| **Estimated runtime** | ~2–6 min (wheel build + clean-venv import dominate) |

---

## Sampling Rate

- **After every task commit:** `pytest tests/test_treelite_roundtrip.py`
- **After every plan wave:** full suite + a wheel build/import smoke
- **Before `/gsd-verify-work`:** gtil round-trip parity green; clean-venv import green; (manual) FIL benchmark recorded
- **Max feedback latency:** ~120 s (the wheel/clean-venv steps are heavier, run per-wave)

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` during planning. Non-negotiable rows:
> Treelite-4.x-shim spike, gtil round-trip parity, clean-venv wheel import.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 0.. | EXP-01..03 | T-09-xx | IR read-only; no untrusted-model exec; pinned deps | spike + roundtrip + wheel-import | `pytest tests/test_treelite_roundtrip.py` ; clean-venv import | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Treelite-4.x spike result: ModelBuilder vs `import_from_json` shim status; classifier `task_type`+`postprocessor` for predict_proba parity (committed decision)
- [ ] `forest_ir_to_treelite()` (ModelBuilder) — global→tree-local node reindex; classifier `leaf([p0..pn])` + `postprocessor="identity"`
- [ ] `python/tests/test_treelite_roundtrip.py` — gtil.predict round-trip parity vs Sylva native (clf+reg) within tolerance
- [ ] clean-venv wheel import test (abi3 + CUDA dynamic-loading), generalizing the Phase-1 proof

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU FIL inference round-trip + throughput | EXP-02 (study) | FIL/RAPIDS is Linux/WSL2 only (not native Windows) | `checkpoint:human-verify`: run FIL round-trip + throughput under WSL2; record agreement beside speed; honest "Windows-unavailable" note |
| Clean-environment wheel install on a fresh machine/venv | EXP-03 | Validates the documented install path + CUDA runtime prereq | Fresh venv: `pip install dist/sylva-*.whl`; `import sylva`; run a tiny fit/predict; record the install path |

---

## Nyquist Compliance

- [ ] EXP-01..03 each map to a verification above
- [ ] The gtil round-trip parity + clean-venv import gates are automated (gtil is pure-Python, Windows-portable)
- [ ] The Treelite-4.x API spike resolves before the export implementation (Wave-0 blocking)
- Set `nyquist_compliant: true` once the planner/plan-checker confirm every `must_haves` entry has a row.
