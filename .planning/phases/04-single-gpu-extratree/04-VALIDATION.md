---
phase: 4
slug: single-gpu-extratree
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. The
> authoritative, load-bearing gate for this phase is **bit-exact equality of the
> GPU ExtraTree's ForestIR against `CpuBackend.fit(n_estimators=1)` on a fixed
> seed** (byte-compare, reusing the Phase-2 determinism-test idiom), plus
> **compute-sanitizer clean (all 4 tools)** on every kernel.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: `cargo test` / `cargo-nextest`; CUDA correctness: `compute-sanitizer` (memcheck/racecheck/initcheck/synccheck) |
| **Config file** | `Cargo.toml` workspace (`crates/sylva-core`, `crates/sylva-cuda`) |
| **Quick run command** | `cargo test -p sylva-core cuda_backend` |
| **Full suite command** | `cargo nextest run` then `compute-sanitizer --tool memcheck cargo test -p sylva-core parity_cpu_gpu` (repeat per tool) |
| **Estimated runtime** | ~60–180 s (sanitizer passes dominate) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p sylva-core cuda_backend`
- **After every plan wave:** Run the full suite + at least `racecheck` + `memcheck` on the kernel parity test
- **Before `/gsd-verify-work`:** Full suite green AND all 4 compute-sanitizer tools clean on every kernel in the path
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

> Populated from each PLAN.md `must_haves` + `<verify>` blocks during planning.
> The non-negotiable rows are the bit-exact differential gate and the sanitizer gates.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| (from plans) | 01.. | 1.. | GPU-01 / GPU-02 | T-04-xx | no OOB / no data race in kernels | differential + sanitizer | `cargo test -p sylva-core parity_cpu_gpu` ; `compute-sanitizer ...` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/sylva-core/tests/parity_cpu_gpu.rs` — bit-exact GPU-vs-CPU-oracle differential test (the phase gate)
- [ ] `crates/sylva-core/tests/philox_device_kat.rs` — on-device Philox KAT vs the frozen `kat.rs` vectors (proves the shared RNG stream before any tree is built)
- [ ] `crates/sylva-core/tests/sanitizer_et_kernels.rs` (or scripted harness) — compute-sanitizer invocation across histogram / split-score / partition kernels

*If none of the above exist yet, Wave 0 of the plans installs them.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| compute-sanitizer racecheck/memcheck clean on local NVIDIA GPU | GPU-01 | Requires a physical CUDA device (RTX 4060 Ti, sm_89); not runnable in GPU-less CI | Run `compute-sanitizer --tool racecheck` and `--tool memcheck` against the kernel parity test on the dev host; expect 0 errors |

---

## Nyquist Compliance

- [ ] Every Phase-4 success criterion maps to an automated or manual verification above
- [ ] GPU-01 and GPU-02 each have at least one gating check
- [ ] The bit-exact differential gate is automated and runs on the dev host (the CPU-oracle half runs in GPU-less CI)
- Set `nyquist_compliant: true` only after the gsd-planner / plan-checker confirm every `must_haves.truth` and `must_haves.prohibition` has a corresponding row.
