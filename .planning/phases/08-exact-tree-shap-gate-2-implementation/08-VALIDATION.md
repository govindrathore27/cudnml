---
phase: 8
slug: exact-tree-shap-gate-2-implementation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 8 — Validation Strategy

> Load-bearing gates: (1) SHAP-01 / Gate-2 — a feasibility spike confirms scope = exact
> attributions + a license-clean GPUTreeSHAP/WoodelfHD path BEFORE implementation (a
> blocking checkpoint with a documented descope path); (2) SHAP-03 — `sylva-shap`
> attributions agree with `shap.TreeExplainer` within float tolerance (atol 1e-4, 5e-4 for
> deep trees) on a deep-tree dataset — the correctness gate; plus the additivity check
> (sum of attributions + base value == model output). The speedup study is reported, not gated.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: `cargo test` / `cargo-nextest` (new `sylva-shap` crate); Python: `pytest` (shap.TreeExplainer agreement) |
| **Config file** | `Cargo.toml` workspace (+ `crates/sylva-shap`); `pyproject.toml`; `.venv-parity` (shap installed) |
| **Quick run command** | `cargo test -p sylva-shap` |
| **Full suite command** | `cargo nextest run` + `cd python && pytest tests/test_shap_agreement.py -q` |
| **Estimated runtime** | ~1–4 min (shap reference compute on a deep forest dominates) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p sylva-shap`
- **After every plan wave:** full Rust suite + `pytest tests/test_shap_agreement.py`
- **Before `/gsd-verify-work`:** shap.TreeExplainer agreement green (within documented tolerance) + additivity gate green
- **Max feedback latency:** ~120 s

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` during planning. Non-negotiable rows:
> Gate-2 feasibility checkpoint, shap.TreeExplainer agreement, additivity.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 0.. | SHAP-01..03 | T-08-xx | IR consumed read-only; no GPL source copied; typed errors | spike + unit + agreement | `cargo test -p sylva-shap` ; `pytest tests/test_shap_agreement.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Gate-2 feasibility spike result (WoodelfHD LICENSE confirmed; GPU CUDA-C expressibility decided; scope = attributions) — a committed checkpoint artifact
- [ ] `crates/sylva-shap/` crate scaffold + `compute_shap_cpu(&ForestIR, X)`
- [ ] `python/tests/test_shap_agreement.py` — ForestIR→shap-custom-dict + `shap.TreeExplainer` agreement (atol 1e-4 / 5e-4 deep) + additivity check
- [ ] deep-tree fixture forest for the agreement + speedup study

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU SHAP speedup vs shap.TreeExplainer (CPU) and GPUTreeSHAP | SHAP (study) | Needs the dev GPU + (optionally) GPUTreeSHAP build | Run the SHAP speedup study on the dev host; agreement reported beside speed |
| Gate-2 feasibility decision (proceed-with-GPU / CPU-only / descope) | SHAP-01 | A human decision after the license + CUDA-C spike | `checkpoint:human-verify`: record the WoodelfHD license + GPU-path verdict; descope path documented |

---

## Nyquist Compliance

- [ ] SHAP-01..03 + the Gate-2 decision each map to a verification above
- [ ] The shap.TreeExplainer agreement + additivity gates are automated (CPU path runs GPU-less)
- [ ] Gate-2 feasibility resolved before implementation (enforced as a Wave-0 blocking checkpoint)
- Set `nyquist_compliant: true` once the planner/plan-checker confirm every `must_haves` entry has a row.
