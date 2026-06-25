---
phase: 3
slug: feature-quantizer-cpu-gpu-bit-parity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-26
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `03-RESEARCH.md` §"Validation Architecture". The per-task map is
> finalized by the planner (each task's `<verify><automated>`); the Nyquist
> auditor backfills any gaps.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` + `proptest` 1.x (unit/property); Python `pytest` for the baseline-parity + throughput microbench (reuses the Phase-5 `pyseam` parity-harness env pattern) |
| **Config file** | Cargo workspace — `crates/sylva-core/Cargo.toml`; Python harness env reused from the Phase-2/5 sklearn parity scripts |
| **Quick run command** | `cargo test -p sylva-core quantize` |
| **Full suite command** | `cargo test -p sylva-core` then `pytest tests/quantize_parity` |
| **Estimated runtime** | ~30–60 s (Rust unit + proptest); Python baseline harness separate |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p sylva-core quantize` (fast Rust unit/property subset)
- **After every plan wave:** Run `cargo test -p sylva-core` + `pytest tests/quantize_parity`
- **Before `/gsd-verify-work`:** Full suite green AND edge-exact numpy agreement = 100% AND golden-vector parity test green AND throughput number recorded (informational)
- **Max feedback latency:** ~60 s (Rust suite)

---

## Per-Task Verification Map

> Scaffold — the planner fills exact Task IDs / commands per PLAN.md; rows below
> map each requirement to its test type from `03-RESEARCH.md` §Validation Architecture.
> `Threat Ref` ties to the §Security Domain table (V5 input validation, ASVS L1).

| Requirement | Test Type | Verification (target) | Threat Ref | Status |
|-------------|-----------|-----------------------|------------|--------|
| QUANT-01 SoA `BinnedMatrix` uint8 (≤256 bins) | unit | `cargo test -p sylva-core quantize::binned_matrix` — column-major SoA, u8 path | — | ⬜ pending |
| QUANT-01 uint16 path (>256 bins, D-08) | unit | `cargo test -p sylva-core quantize::dtype_u16` | bin-overflow guard | ⬜ pending |
| QUANT-01 exact-quantile edges, deduped (D-01/D-02) | unit | `cargo test -p sylva-core quantize::edges` — raw `numpy.quantile method='linear'` edges, `np.unique` dedupe | — | ⬜ pending |
| QUANT-01 bins ∈ [0,n_bins); monotone; NaN→missing | property | `cargo test -p sylva-core quantize::props` (proptest) | NaN/Inf routing | ⬜ pending |
| QUANT-02 CPU bins == golden vectors on fixed seed | integration | `cargo test -p sylva-core quantize::parity_golden` — the bit-parity CONTRACT (golden fixtures, not a live GPU run this phase) | — | ⬜ pending |
| QUANT-02 pure-compare (`v <= edge`) assignment documented for Phase-4 GPU reuse | contract doc | asserted in golden-fixture metadata; no FMA / `--use_fast_math` in the future assign kernel | — | ⬜ pending |
| SC-4 edge-exact bin agreement vs `numpy.quantile` = 100% (GATE) | integration | `pytest tests/quantize_parity/test_numpy_oracle.py` | input validation (V5) | ⬜ pending |
| SC-4 distributional agreement vs sklearn `_BinMapper` (informational) | integration | `pytest tests/quantize_parity/test_sklearn_distributional.py` — ≥99% expected, documented divergence | — | ⬜ pending |
| SC-3 `QuantizeReport` records dtype/contiguity/bytes (H2D = N/A this phase) | unit | `cargo test -p sylva-core quantize::report` | — | ⬜ pending |
| SC-5 quantize-throughput microbench (rows/s), reported not gated | benchmark | `pytest tests/quantize_parity/test_throughput.py` (op-level only, cold/warm separated, pinned versions — NO end-to-end speed claim) | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/sylva-core/src/quantize/*.rs` — the quantizer module itself (none exists yet)
- [ ] `crates/sylva-core/tests/` or `#[cfg(test)]` — Rust unit/property tests for edges/assign/dtype/report
- [ ] Golden-vector fixture file (JSON) + loader — the QUANT-02 bit-parity contract
- [ ] `tests/quantize_parity/` Python harness — numpy edge-exact + sklearn distributional + throughput (reuse the Phase-2/5 pinned-sklearn venv pattern)
- [ ] **A1 calibration micro-test** — pin the f64/f32 cast point in quantile interpolation against `np.quantile` on f32 input BEFORE locking golden vectors (research-flagged HIGH-risk assumption; gates the edge-exact GATE)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| sklearn distributional tolerance calibration (the ≥99% agreement threshold) | SC-4 | `_BinMapper` differs from raw-quantile edges by design (midpoint-when-distinct + opposite boundary side); the acceptance threshold needs a one-time empirical read, not a hard-coded constant | Run the distributional harness on `make_classification` 100k×100 f32, record the actual agreement %, set the documented floor above the observed divergence; `<95%` is a signal to investigate, not an auto-fail |
| Throughput microbench numbers | SC-5 | Op-level informational measurement (foundational-phase fairness rule — NO end-to-end speed claim); reported, not gated | Run `test_throughput.py`, report rows/s for Sylva-CPU vs `numpy.quantile` vs `_BinMapper`, cold/warm separated, repeated runs, pinned numpy 2.4.2 / sklearn 1.8.0 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (esp. A1 calibration before golden vectors)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
