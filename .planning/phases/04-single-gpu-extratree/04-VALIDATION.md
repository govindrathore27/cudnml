---
phase: 4
slug: single-gpu-extratree
status: planned
nyquist_compliant: true
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
| 04-01-T2/T3 | 01 | 1 | GPU-02 (ENG-06 reuse) | T-04-01/02/03 | on-device Philox stream == frozen KAT; no compile-error swallow | device unit/integration | `cargo test -p sylva-cuda --test philox_device_kat` | ❌ W0 | ⬜ pending |
| 04-02-T1 | 02 | 2 | GPU-01 | T-04-04/06 | no OOB / no shared-mem race; integer atomics only | unit + (sanitizer in P03) | `cargo test -p sylva-cuda cuda_backend` | ❌ W0 | ⬜ pending |
| 04-02-T3 | 02 | 2 | GPU-01 | T-04-05/07/08 | breadth-first build; valid ForestIR; typed errors | integration | `cargo test -p sylva-cuda cuda_backend` | ❌ W0 | ⬜ pending |
| 04-03-T1 | 03 | 3 | GPU-02 | T-04-10 | GPU==CPU oracle byte-exact (clf+reg), no tolerance | differential (byte-compare) | `cargo test -p sylva-cuda --test parity_cpu_gpu` | ❌ W0 | ⬜ pending |
| 04-03-T2/T4 | 03 | 3 | GPU-01 / GPU-02 | T-04-09 | no OOB/race/uninit in every kernel | sanitizer (4 tools) | `cargo test -p sylva-cuda --test sanitizer_et_kernels` + `compute-sanitizer --tool {memcheck,racecheck,synccheck,initcheck} <exe>` | ❌ W0 | ⬜ pending |
| 04-03-T3 | 03 | 3 | GPU-01 (study) | T-04-11 | no end-to-end speed claim; transfer-inclusive | Python (reported) | `pytest python/tests/gpu_parity/` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

> **Location reconciliation:** these gate tests live under `crates/sylva-cuda/tests/` (the
> crate where `CudaBackend` lives), not `crates/sylva-core/tests/` — the original draft path.
> RESEARCH.md + PATTERNS.md both place them in sylva-cuda; this is the corrected location.

- [ ] `crates/sylva-cuda/tests/philox_device_kat.rs` — on-device Philox KAT vs the frozen `kat.rs` vectors (Plan 04-01, proves the shared RNG stream before any tree is built)
- [ ] `crates/sylva-cuda/tests/parity_cpu_gpu.rs` — bit-exact GPU-vs-CPU-oracle differential test, clf + reg (Plan 04-03, the phase gate)
- [ ] `crates/sylva-cuda/tests/sanitizer_et_kernels.rs` — standalone single-launch compute-sanitizer targets per ET kernel (Plan 04-03)
- [ ] `python/tests/gpu_parity/` — sklearn distributional + transfer-inclusive timing study (Plan 04-03, informational)

*Wave 0 of the plans installs these: Plan 04-01 creates the device-KAT test; Plan 04-03 creates the parity gate, sanitizer targets, and the Python study.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| compute-sanitizer racecheck/memcheck clean on local NVIDIA GPU | GPU-01 | Requires a physical CUDA device (RTX 4060 Ti, sm_89); not runnable in GPU-less CI | Run `compute-sanitizer --tool racecheck` and `--tool memcheck` against the kernel parity test on the dev host; expect 0 errors |

---

## Nyquist Compliance

- [x] Every Phase-4 success criterion maps to an automated or manual verification above
- [x] GPU-01 and GPU-02 each have at least one gating check (GPU-01: `cargo test -p sylva-cuda cuda_backend` + sanitizer; GPU-02: `parity_cpu_gpu` byte-compare + four-tool sanitizer)
- [x] The bit-exact differential gate is automated and runs on the dev host (the CPU-oracle half runs in GPU-less CI)
- Every `must_haves.truth` and `must_haves.prohibition` across the three plans has a corresponding row in the Per-Task Verification Map above (bit-exact byte-compare and four-tool sanitizer-clean are the non-negotiable rows; "no global float atomics" and "CPU oracle unmodified" are enforced by the integer-atomic kernel design + the `git diff` prohibition check).
