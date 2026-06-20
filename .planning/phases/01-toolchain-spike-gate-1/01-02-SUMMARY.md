---
phase: 01-toolchain-spike-gate-1
plan: 02
subsystem: cuda-backend
tags: [rust, cudarc, nvrtc, cuda, sm_89, compute-sanitizer, histogram, atomics, windows-msvc, thiserror]

requires:
  - phase: 01-01
    provides: buildable workspace + sylva-cuda crate (cudarc 0.19.8 dynamic-linking + PyO3 abi3 linking on MSVC); ignored nvrtc_launch_vector_add placeholder
provides:
  - "Proven NVRTC compile->load->launch path: hand-written CUDA-C compiles for sm_89 and launches on the RTX 4060 Ti (TOOL-01)"
  - "run_vector_add / run_histogram entrypoints in nvrtc_launch.rs, re-exported from lib.rs for Plan 03's PyO3 wrapper"
  - "Representative 256-bin shared-mem privatized histogram (integer atomicAdd, two-level reduction) — the Phase-4 hot-path primitive de-risked"
  - "compute-sanitizer-clean proof (memcheck/racecheck/synccheck/initcheck all 0) — toolchain is debuggable (TOOL-02)"
  - "Typed CudaError (thiserror) wrapping cudarc CompileError/DriverError — no .unwrap() on device calls (no silent fallback)"
affects: [01-03 wheel + microbench + gate (reuses run_vector_add/run_histogram via PyO3), phase-4 CudaBackend histogram.cu]

tech-stack:
  added: [thiserror 1.0.69]
  patterns: [NVRTC compile_ptx_with_opts(arch=sm_89, -lineinfo), cudarc 'std' marker feature for std::error::Error impls, integer-only atomicAdd, privatized shared-mem histogram + two-level reduction, standalone compute-sanitizer test target]

key-files:
  created:
    - crates/sylva-cuda/src/kernels.rs
    - crates/sylva-cuda/src/nvrtc_launch.rs
    - crates/sylva-cuda/tests/sanitizer_histogram.rs
  modified:
    - crates/sylva-cuda/src/lib.rs
    - crates/sylva-cuda/tests/toolchain_smoke.rs
    - crates/sylva-cuda/Cargo.toml
    - Cargo.toml
    - Cargo.lock

key-decisions:
  - "Added cudarc 'std' feature (pure marker `std = []`, no cublas/curand) — required because cudarc gates `impl std::error::Error for DriverError/CompileError` behind #[cfg(feature='std')]; without it thiserror's #[from] cannot find the Error impl"
  - "Switched memcpy_stod/memcpy_dtov -> clone_htod/clone_dtoh (the non-deprecated 0.19.8 API) so clippy -D warnings stays clean"
  - "Added thiserror 1.x (CLAUDE.md-pinned) as a workspace dep to build the typed CudaError enum (no .unwrap on device calls)"
  - "Explicit stream.synchronize() after each launch so launch/exec errors surface before D2H readback (no silent fallback), even though same-stream async memcpy is already ordered"

metrics:
  duration: ~30min
  completed: 2026-06-20

requirements-completed: [TOOL-01, TOOL-02]

status: complete
---

# Phase 1 Plan 02: NVRTC Kernel Launch + Sanitizer-Clean Histogram Summary

**The single biggest technical risk in PROJECT.md is resolved: a hand-written CUDA-C kernel compiles via cudarc 0.19.8 + NVRTC for `sm_89`, launches bit-exactly on the local RTX 4060 Ti (TOOL-01), and a representative shared-memory privatized histogram (integer `atomicAdd`, two-level reduction) is `compute-sanitizer`-clean across all four tools (TOOL-02) — natively on Windows/MSVC, no WSL, with every cudarc call surfaced as a `Result`.**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-06-20
- **Tasks:** 3 (Task 1 + Task 2 auto; Task 3 human-verify gate, work completed)
- **Files created:** 3 / modified: 5

## Accomplishments

- **TOOL-01:** `nvrtc_launch_vector_add` passes — `vector_add` CUDA-C compiled by NVRTC (`arch=sm_89`, `-lineinfo`) launches on the RTX 4060 Ti and returns `a[i]+b[i]` with **max-abs-error exactly 0** over **N=10,000,000** f32 elements. Native Windows/MSVC.
- **TOOL-01 (primitive):** `histogram_privatized_matches_cpu` passes — the 256-bin shared-memory privatized histogram launches and matches a CPU reference count **exactly** across all 256 bins over N=1,000,000 uint8 indices.
- **TOOL-02:** All four `compute-sanitizer` tools report clean against the isolated histogram launch — the toolchain is debuggable (verbatim evidence below).
- Confirmed the cudarc 0.19.8 launch API against the installed crate source (not just research): `compile_ptx_with_opts` + `CompileOptions{arch, options}` -> `CudaContext::new(0)` -> `load_module` -> `load_function` -> `clone_htod` / `alloc_zeros` -> `launch_builder(&f).arg(..).launch(cfg)` (unsafe) -> `synchronize` -> `clone_dtoh`. Scalars (`u64` for `size_t n`, `i32` for `int n`) passed by ref via `DeviceRepr`.
- Typed `CudaError` (thiserror) wraps `CompileError` + `DriverError` + boundary `InvalidInput`; **no `.unwrap()`/`.expect()` on any device call** — every cudarc Result propagates via `?` (no silent fallback). `unsafe` confined to the single `launch` call with a `// SAFETY:` justification.

## Task Commits

1. **Task 1: NVRTC compile + launch vector_add on sm_89 (TOOL-01)** — `14951a0` (feat)
2. **Task 2: histogram_privatized matches CPU reference exactly** — `9088daa` (test)
3. **Task 3: standalone compute-sanitizer histogram target (TOOL-02)** — `0057c15` (test)

## TOOL-02 compute-sanitizer Evidence (verbatim)

**Sanitizer target exe:** `target/debug/deps/sanitizer_histogram-2fb1927c236ea5f1.exe`
(built via `cargo test -p sylva-cuda --test sanitizer_histogram --no-run`)

**compute-sanitizer (full path):** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`

The histogram kernel was NVRTC-compiled with `-lineinfo`, so any hazard would have carried source-line attribution. Exact commands run (Git Bash form; `<EXE>` = the path above):

```
compute-sanitizer.exe --tool memcheck  <EXE> --test-threads=1
compute-sanitizer.exe --tool racecheck <EXE> --test-threads=1
compute-sanitizer.exe --tool synccheck <EXE> --test-threads=1
compute-sanitizer.exe --tool initcheck <EXE> --test-threads=1
```

Verbatim trailing summary line from each tool (all four clean — exit 0):

| Tool | Verbatim summary line |
|------|-----------------------|
| memcheck  | `========= ERROR SUMMARY: 0 errors` |
| racecheck | `========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| synccheck | `========= ERROR SUMMARY: 0 errors` |
| initcheck | `========= ERROR SUMMARY: 0 errors` |

Per D-05, no hazard was found, so no kernel fix-and-rerun iteration was needed — the privatized shared-mem histogram (cooperative zero-init + `__syncthreads()` before use, integer `atomicAdd` into shared, `__syncthreads()` before the global flush) is clean as authored.

## Test Outcomes

| Test | Result |
|------|--------|
| `nvrtc_launch_vector_add` (TOOL-01, N=10,000,000, exact a+b) | **ok** |
| `histogram_privatized_matches_cpu` (256 bins exact vs CPU) | **ok** |
| `smoke_crate_links` (Plan 01 link proof, still green) | **ok** |
| `sanitizer_histogram_single_launch` (isolated target) | **ok** |

Full suite: `cargo test -p sylva-cuda --test toolchain_smoke -- --include-ignored` → `3 passed; 0 failed`.

## Files Created/Modified

- `crates/sylva-cuda/src/kernels.rs` — CUDA-C source strings: `VECTOR_ADD_SRC` (with `if (i < n)` bounds guard), `HISTOGRAM_PRIVATIZED_SRC` (`__shared__ unsigned int sh[256]`, cooperative zero-init, integer `atomicAdd`, two-level reduction), `BIN_COUNT = 256`.
- `crates/sylva-cuda/src/nvrtc_launch.rs` — `CudaError` (thiserror), `compile_for_sm89` (`arch=sm_89`, `-lineinfo`), `run_vector_add`, `run_histogram`; named consts `ARCH_SM_89`, `HISTOGRAM_BLOCK=256`; V5 boundary validation (length-match for vector_add, `bins[i] < BIN_COUNT` for histogram).
- `crates/sylva-cuda/tests/sanitizer_histogram.rs` — standalone single-launch compute-sanitizer target.
- `crates/sylva-cuda/src/lib.rs` — `pub mod kernels; pub mod nvrtc_launch;` + re-export `run_vector_add`, `run_histogram`, `CudaError`.
- `crates/sylva-cuda/tests/toolchain_smoke.rs` — de-ignored `nvrtc_launch_vector_add`; added `histogram_privatized_matches_cpu`; named consts `VECTOR_ADD_N`, `HISTOGRAM_N`.
- `crates/sylva-cuda/Cargo.toml` — added cudarc `std` feature + `thiserror` workspace dep.
- `Cargo.toml` — `[workspace.dependencies] thiserror = "1"`.
- `Cargo.lock` — thiserror 1.0.69 + thiserror-impl pinned.

## Decisions Made

- **cudarc `std` feature is mandatory for typed errors.** cudarc 0.19.8 gates `impl std::error::Error for DriverError` / `CompileError` behind `#[cfg(feature = "std")]`. With `default-features = false` (Pitfall 3, required to drop cublas/curand), `std` is off, so thiserror's `#[from]`/`{0}` could not find the `Error` impl (E0599 `as_dyn_error`/`as_display`). Fix: add `"std"` — a pure marker feature (`std = []`, no extra deps), so it does **not** reintroduce cublas/curand/runtime. D-03 / Pitfall 3 stay honoured.
- **Non-deprecated copy API.** `memcpy_stod`/`memcpy_dtov` are `#[deprecated]` in 0.19.8 (→ `clone_htod`/`clone_dtoh`); used the non-deprecated names so `clippy -D warnings` passes.
- **Explicit `synchronize()` after launch** to surface launch/exec errors before D2H readback (no-silent-fallback), beyond the same-stream async ordering.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added cudarc `std` feature to restore `std::error::Error` impls**
- **Found during:** Task 1 (building the `CudaError` enum)
- **Issue:** `#[from] CompileError` / `#[from] DriverError` failed to compile (E0599 `as_dyn_error`/`as_display`) because both `impl std::error::Error` are `#[cfg(feature = "std")]`-gated and `default-features = false` dropped `std`.
- **Fix:** Added `"std"` to the cudarc feature list. Verified `std = []` is a pure marker (no transitive deps; does not pull cublas/curand/runtime — Pitfall 3 / D-03 intact).
- **Files modified:** crates/sylva-cuda/Cargo.toml, Cargo.lock
- **Commit:** 14951a0

**2. [Rule 3 - Blocking] Added thiserror (workspace dep) for the typed crate error enum**
- **Found during:** Task 1
- **Issue:** Plan calls for a "thiserror-style crate error enum"; thiserror was not yet a dependency.
- **Fix:** Added `thiserror = "1"` as a `[workspace.dependencies]` entry (CLAUDE.md-pinned `thiserror 1.x`) and referenced it via `{ workspace = true }`.
- **Files modified:** Cargo.toml, crates/sylva-cuda/Cargo.toml, Cargo.lock
- **Commit:** 14951a0

**3. [Rule 1 - Deprecation/lint] Used clone_htod/clone_dtoh instead of deprecated memcpy_stod/memcpy_dtov**
- **Found during:** Task 1 (build warnings)
- **Issue:** `memcpy_stod`/`memcpy_dtov` are `#[deprecated]` in cudarc 0.19.8; under `clippy -D warnings` (project lint bar) the deprecation would fail the build/CI.
- **Fix:** Switched to the non-deprecated `clone_htod` / `clone_dtoh` (identical arg shapes).
- **Files modified:** crates/sylva-cuda/src/nvrtc_launch.rs
- **Commit:** 14951a0

**Total deviations:** 3 auto-fixed (2 blocking dependency/feature fixes, 1 deprecation fix). All required for a clean `-D warnings` build with typed errors. No scope creep — no estimator/Philox/multi-feature logic; no nvcc/cc/build.rs; integer atomics only; no float atomics anywhere.

## Notes on the Atomic Commit Split

`kernels.rs` and `nvrtc_launch.rs` are shared-infrastructure modules: they carry **both** the `vector_add` and `histogram_privatized` source + launch code (one `CudaError`, one `compile_for_sm89` helper, one `lib.rs` re-export). To keep each commit's **test signal** atomic, Task 1 committed the full launch infra + the `vector_add` correctness test; Task 2 committed the histogram **correctness test** (the kernel/launch infra it exercises was already present from Task 1); Task 3 committed the standalone sanitizer target. The histogram kernel source physically lands in the Task 1 commit but is first exercised by Task 2's test — the honest split for co-located shared-infra modules.

## Known Stubs

None. No placeholder/hardcoded-empty values; both kernels are fully wired to real device launches with exact-equality correctness assertions.

## Issues Encountered

- Session-process PATH was stale (predates the Rust install) — `cargo`/`git commit` hooks invoked with `$HOME/.cargo/bin` prepended per command, per the environment note.
- `compute-sanitizer` is not on PATH — invoked by full path (`...\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe`), as flagged in 01-01-SUMMARY.

## Next Phase Readiness

- **Plan 03 ready:** `run_vector_add` / `run_histogram` are `pub` and re-exported from `lib.rs` — Plan 03's PyO3 `#[pyfunction]` entrypoint can call them directly. The microbench can reuse `run_vector_add` on the fixed 1e7 array.
- **Kernel-authoring decision validated:** cudarc + NVRTC + hand-written CUDA C compiles, launches, and sanitizes clean on native Windows/MSVC — the GPU link of the walking-skeleton chain is proven. Phase-4's `histogram.cu` privatized layout is de-risked.
- **Remaining for the gate (Plan 03):** abi3 wheel build + clean-venv import (TOOL-03), microbench vs baseline (SC-6), and the `VERSIONS.md` kill-decision (TOOL-04) — record the four sanitizer "0 errors" outcomes there.

## Self-Check: PASSED

- Created files verified on disk: kernels.rs, nvrtc_launch.rs, tests/sanitizer_histogram.rs, 01-02-SUMMARY.md — all FOUND.
- Task commits verified in git log: 14951a0 (Task 1), 9088daa (Task 2), 0057c15 (Task 3) — all FOUND.

---
*Phase: 01-toolchain-spike-gate-1*
*Completed: 2026-06-20*
