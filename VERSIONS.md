# VERSIONS.md — Sylva Phase 1 Toolchain Spike (TOOL-04 durable pin + kill-decision)

Durable record of the exact toolchain, dependency pins, and hardware the Phase 1
walking-skeleton spike was proven against, plus the **kill-decision** that gates
the rest of the project. Static, already-verified values are filled now (Plan
01-01); runtime-determined fields are marked `TBD (Plan 0X)` with the plan that
fills them.

Captured on the native-Windows dev/benchmark host on 2026-06-20.

## Toolchain (verified at the Plan 01-01 Task 1 checkpoint)

| Component | Pin / Version | Notes |
|-----------|---------------|-------|
| Rust toolchain | `rustc 1.96.0 (ac68faa20 2026-05-25)`, **stable** | MSRV floor is 1.83 (PyO3 0.29). Stable only — never nightly. |
| cargo | `cargo 1.96.0 (30a34c682 2026-05-25)` | |
| Target triple | `x86_64-pc-windows-msvc` | The only installed target. |
| MSVC | v143, `cl.exe` **14.44.35207** (VS 2022 Build Tools) | Linking only — NVRTC compiles kernels. |
| Windows SDK | `10.0.26100.0` | |
| CUDA toolkit | **12.8**, `nvcc` release 12.8 `V12.8.93` | `nvrtc64_120_0.dll` present. |
| NVRTC | CUDA 12.8 (`nvrtc64_120_0.dll`) | Runtime kernel compilation path. |
| compute-sanitizer | present (`...\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`) | Not on PATH — invoked by full path (Plan 02 TOOL-02). |
| maturin | **1.14.1** | `>=1.14,<2.0`. |
| Python (primary) | **3.14.3** | Local interpreter; abi3-py310 wheel covers it. |
| Python (baseline) | 3.12 (`py -V:3.12`) | For the Plan 03 CuPy baseline (D-06: CuPy has no cp314 Windows wheel). |

## GPU / Driver (D-03 environment pins)

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce **RTX 4060 Ti** |
| Compute capability | **sm_89** (Ada Lovelace) |
| Driver | **595.79** |

## Dependency pins (committed in `Cargo.toml` / `Cargo.lock`)

| Crate | Version | Features | Notes |
|-------|---------|----------|-------|
| cudarc | **0.19.8** | launch-proof (committed default): `["driver", "nvrtc", "cuda-12080", "dynamic-linking"]` | See **Deviation** below — `dynamic-linking` is required; the plan's 3-feature list panics. |
| cudarc | **0.19.8** | shipping wheel (Plan 03 / D-02): `["driver", "nvrtc", "dynamic-loading"]` | CUDA resolved at **runtime** via the driver — one wheel for any compatible CUDA. Swapped in for the maturin wheel build only. |
| pyo3 | **0.29.0** | `["extension-module", "abi3-py310"]` | One cp310-abi3 wheel runs on any CPython ≥ 3.10. |
| rust-numpy (`numpy`) | **0.29.0** | — | NOT used in Phase 1 (no host array marshalling yet, per D-03). Pin recorded now: rust-numpy tracks the PyO3 minor exactly (0.29.0 ↔ PyO3 0.29.0). Supersedes the earlier CLAUDE.md "≈0.25.x" guess (resolved in 01-RESEARCH.md). Enters in Phase 2. |
| maturin (build req) | `>=1.14,<2.0` | — | `[build-system].requires` in `pyproject.toml`. |

`Cargo.lock` is committed at the repo root and pins the full transitive
dependency graph (supply-chain integrity per the plan threat model T-01-01).

### Deviation — cudarc requires an explicit link-MODE feature

The plan specified `cudarc = { ..., features = ["driver", "nvrtc", "cuda-12080"] }`.
That set **panics at build time**: cudarc 0.19.8's `build.rs` requires exactly one
link-mode feature from `{dynamic-loading, fallback-dynamic-loading,
dynamic-linking, static-linking}`. `cuda-12080` only selects the CUDA *version*
bindings — it is **not** a link mode. `static-linking` is unusable on native
Windows/MSVC (it needs the GNU/Clang static lib `stdc++`, which the MSVC toolchain
lacks: `could not find native static library stdc++`). The committed launch-proof
build therefore uses **`dynamic-linking`** (link against the installed CUDA 12.8
toolkit import libs at build time), which is exactly D-02's intent for the launch
proof and is distinct from the wheel's runtime `dynamic-loading`.

## Microbench numbers — *feasibility sanity check, no algorithm speed claim*

Phase 1 is a toolchain feasibility spike. Any kernel-launch / vector-op timing
recorded here is a **feasibility sanity check, no algorithm speed claim**. No
end-to-end speed comparison is made until Phase 5; the authoritative (n×d)
crossover is Phase 7 (per ROADMAP comparative-study fairness note).

| Metric | Value |
|--------|-------|
| Kernel-launch / vector-op microbench (Sylva NVRTC) | TBD (Plan 03, SC-6) |
| CuPy (or raw-CUDA-C) baseline | TBD (Plan 03, D-06) |
| Comparison verdict | TBD (Plan 03) — feasibility only, no algorithm speed claim |

## Wheel / abi3 runtime fields

| Field | Value |
|-------|-------|
| `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` needed? | TBD (Plan 03 — determined when the maturin abi3 wheel is built against Python 3.14) |
| Wheel import in clean venv (TOOL-03) | TBD (Plan 03) |
| CuPy baseline env (Python 3.12 venv) | TBD (Plan 03, D-06) |

## Kill-criteria result

**Kill-criteria result:** TBD — pending the Plan 01-03 phase gate. One of:

- **proceed** — native-Windows cudarc + NVRTC launch (TOOL-01) and sanitizer-clean
  histogram (TOOL-02) and abi3 wheel import (TOOL-03) all pass → continue to Phase 2.
- **WSL-fallback** — native Windows blocks on a toolchain wall that WSL2 (nvcc +
  GNU/Clang) clears → re-home the build profile to WSL2, keep benchmarks native.
- **stop** — neither native Windows nor WSL2 can launch a correct NVRTC kernel
  from the Rust core → the cudarc+NVRTC premise fails; halt and re-evaluate (D-05).

Status as of Plan 01-01: the workspace **builds and links** with the pinned
cudarc 0.19.8 + PyO3 0.29 set under native Windows/MSVC (first skeleton link
proven). Launch (TOOL-01), sanitizer (TOOL-02), and wheel import (TOOL-03) are
proven in Plans 01-02 / 01-03 before the verdict is written.
