---
phase: 2
slug: cpu-oracle-contracts-forest-ir
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-20
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `02-RESEARCH.md` §"Validation Architecture". The per-task map is
> finalized by the planner (each task's `<verify><automated>`); the Nyquist
> auditor backfills any gaps.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (+ `cargo nextest` optional); `proptest` for invariants; `approx` for f32 tolerance; a Python differential/parity harness vs scikit-learn |
| **Config file** | `Cargo.toml` workspace; new `crates/sylva-core` member; `scripts/` for the sklearn parity harness |
| **Quick run command** | `cargo test -p sylva-core` |
| **Full suite command** | `cargo test --workspace` + the sklearn parity script (`scripts/parity_*.py` in a venv with scikit-learn pinned) |
| **Estimated runtime** | ~30–90 s (Rust unit + proptest); parity harness separate (trains many trees) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p sylva-core`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite green AND the sklearn parity harness passes (accuracy within CI + KS on split statistics)
- **Max feedback latency:** ~90 s (Rust suite)

---

## Per-Task Verification Map

> Scaffold — the planner fills exact Task IDs / commands per PLAN.md; rows below
> map each requirement to its test type from `02-RESEARCH.md` §Validation Architecture.

| Requirement | Test Type | Verification (target) | Status |
|-------------|-----------|-----------------------|--------|
| ENG-01 `trait Backend` (no CUDA types cross) | unit / compile | Trait defined; `cargo test` compiles `CpuBackend: Backend`; no `cudarc`/CUDA types in the trait crate | ⬜ pending |
| ENG-02 SoA `ForestIR` | unit + proptest | IR round-trips; SoA arrays incl. SHAP cover + Treelite-compat fields present; serialization round-trip invariant | ⬜ pending |
| ENG-03 `CpuBackend` trains/predicts ET+RF | differential-vs-sklearn | Accuracy/proba parity on `make_classification` / Covertype subset (ET-vs-ET, RF-vs-RF) | ⬜ pending |
| ENG-04 parity contract + Philox | unit (Philox KAT) + doc | Philox-4×32-10 matches verified test vectors (KAT checkpoint); parity contract documented | ⬜ pending |
| ENG-05 NaN/missing routing | unit (NaN fixtures) | Missing → higher-sample-count default-child (tie→left); CPU result deterministic; NaN fixtures in suite | ⬜ pending |
| ENG-06 Philox keyed (seed,tree,node,feature,draw) | unit + determinism | Same seed → identical ForestIR; documented KAT vectors for Phase-4 bit-match | ⬜ pending |
| EST-07 differential + property invariants | proptest + differential | child rows partition parent; leaf probs ∈[0,1] sum 1; seed determinism; serialization round-trip; KS on split stats | ⬜ pending |
| SC-6 Comparative Baseline Study | parity harness | CpuBackend accuracy/distribution PARITY with sklearn ET/RF within tolerance | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/sylva-core` crate + `cargo test` wired (ndarray + rayon + proptest + approx deps)
- [ ] sklearn parity harness scaffold (`scripts/`, a pinned-scikit-learn venv)
- [ ] Philox KAT-vector fixtures (verified before freezing — research flagged the canonical vectors as ASSUMED)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Parity-threshold calibration (KS p-value, CI width, tree/seed counts) | ENG-04 / EST-07 | Needs an empirical sklearn-vs-sklearn null-spread measurement before thresholds can be frozen | Run the calibration task: measure split-statistic distribution across sklearn seeds, set thresholds above the observed null spread |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
