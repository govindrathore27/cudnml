---
phase: 01-toolchain-spike-gate-1
plan: 01
subsystem: infra
tags: [rust, cargo, cudarc, nvrtc, pyo3, maturin, abi3, cuda, windows-msvc]

requires:
  - phase: none
    provides: greenfield — first buildable artifact in the repo
provides:
  - Persisted Cargo [workspace] root + crates/sylva-cuda crate (cdylib+rlib)
  - Pinned dependency set proven to build+link on native Windows/MSVC (cudarc 0.19.8 + PyO3 0.29.0 abi3)
  - maturin abi3 pyproject.toml scaffold (wheel build deferred to Plan 03)
  - Toolchain smoke test (MSVC link proof) + ignored TOOL-01 launch placeholder
  - Honest Windows CI scaffold (hosted lint + self-hosted GPU build/test)
  - VERSIONS.md TOOL-04 durable pin record + kill-decision template
affects: [01-02 nvrtc launch, 01-03 wheel + microbench + gate, phase-2 cpu-backend]

tech-stack:
  added: [cudarc 0.19.8, pyo3 0.29.0, maturin 1.14.1, rust 1.96.0 stable]
  patterns: [workspace.package inheritance, default-features=false on cudarc, abi3 single-wheel, NVRTC-runtime-compile (no nvcc/cc/build.rs)]

key-files:
  created:
    - Cargo.toml
    - crates/sylva-cuda/Cargo.toml
    - crates/sylva-cuda/src/lib.rs
    - crates/sylva-cuda/tests/toolchain_smoke.rs
    - pyproject.toml
    - rust-toolchain.toml
    - .gitignore
    - .github/workflows/ci.yml
    - VERSIONS.md
    - Cargo.lock
  modified: []

key-decisions:
  - "cudarc launch-proof features = [driver, nvrtc, cuda-12080, dynamic-linking] — the plan's 3-feature set panics (build.rs requires a link-mode feature); static-linking is unusable on MSVC (needs GNU/Clang stdc++)"
  - "CI split into hosted lint (fmt, no CUDA) + self-hosted GPU build/test — hosted runners cannot build (dynamic-linking needs the CUDA toolkit) and have no GPU; never falsely green"
  - "version() made pub so the integration-test binary can call it as the MSVC link signal"

patterns-established:
  - "Pattern: cudarc always default-features=false; link MODE is an explicit feature (dynamic-linking for the toolkit-linked launch proof, dynamic-loading for the shipping wheel)"
  - "Pattern: GPU/wheel CI steps are gated to self-hosted runners and never marked passing on hosted runners (honest, non-silent dispatch — mirrors the project's no-silent-fallback value)"

requirements-completed: [TOOL-01, TOOL-03, TOOL-04]

duration: ~25min
completed: 2026-06-20
status: complete
---

# Phase 1 / Plan 01-01: Toolchain Foundation Summary

**Buildable, pinned Cargo workspace + `sylva-cuda` crate that compiles and links cudarc 0.19.8 + PyO3 0.29 abi3 under native Windows/MSVC — the first link of the walking-skeleton chain proven, with a TOOL-04 VERSIONS.md pin/kill-decision record.**

## Performance

- **Duration:** ~25 min (excludes user-side MSVC Build Tools download)
- **Completed:** 2026-06-20
- **Tasks:** 4 (Task 1 checkpoint + Tasks 2–4 build)
- **Files created:** 10

## Accomplishments
- Resolved the prerequisite blocker: installed Rust stable 1.96.0 (rustup, x86_64-pc-windows-msvc) + maturin 1.14.1; user installed MSVC v143 Build Tools (cl.exe 14.44.35207 + Windows SDK 10.0.26100). All six Task-1 acceptance checks verified green.
- Persisted Cargo `[workspace]` + `crates/sylva-cuda` (cdylib+rlib) with the EXACT pinned dep set (`cudarc 0.19.8` default-features=false, `pyo3 0.29.0` extension-module+abi3-py310). `cargo build` and `cargo clippy -D warnings` clean.
- Proved the MSVC link path end-to-end: `cargo test --test toolchain_smoke` → `smoke_crate_links` passes; `nvrtc_launch_vector_add` exists as the ignored TOOL-01 placeholder.
- Created the maturin abi3 `pyproject.toml` (TOOL-03 scaffold) and the TOOL-04 `VERSIONS.md` durable pin + kill-decision template.

## Task Commits

1. **Task 2: workspace + crate + maturin pyproject** — `ec5f2c2` (chore)
2. **Task 3: toolchain smoke test + honest Windows CI** — `76f84a4` (test)
3. **Task 4: VERSIONS.md TOOL-04 pins + kill-decision** — `38eb0f4` (docs)

_Task 1 (checkpoint:human-verify) was a toolchain-install gate — no code commit; satisfied by the install + verification above._

## Files Created/Modified
- `Cargo.toml` — workspace root (resolver 2, member crates/sylva-cuda, shared Apache-2.0 + MSRV 1.83)
- `crates/sylva-cuda/Cargo.toml` — pinned cudarc + pyo3; default-features=false; documents the dynamic-linking deviation + the Plan-03 wheel feature swap
- `crates/sylva-cuda/src/lib.rs` — `#[pymodule] sylva_cuda` exposing `pub fn version()` (CARGO_PKG_VERSION probe; PyO3 seam for Plan 03)
- `crates/sylva-cuda/tests/toolchain_smoke.rs` — `smoke_crate_links` (MSVC link proof) + `#[ignore]` `nvrtc_launch_vector_add` placeholder
- `pyproject.toml` — maturin build-backend + `[tool.maturin]` abi3 config (module-name sylva_cuda)
- `rust-toolchain.toml` — channel stable, target x86_64-pc-windows-msvc (no nightly)
- `.gitignore` — /target, *.pyd, __pycache__/, .venv*/, dist/
- `.github/workflows/ci.yml` — hosted `lint` (fmt) + self-hosted GPU `build-test`; honest GPU/wheel comment
- `VERSIONS.md` — TOOL-04 pins (toolchain, deps, GPU/driver) + microbench label + kill-decision line
- `Cargo.lock` — full transitive pin (supply-chain integrity)

## Decisions Made
- **cudarc link mode:** used `dynamic-linking` for the committed launch-proof build (toolkit-linked at build time = D-02 intent); the shipping wheel (Plan 03) swaps to `dynamic-loading` (runtime CUDA resolution). `static-linking` rejected on MSVC (needs GNU/Clang `stdc++`).
- **CI honesty:** hosted runners cannot build this crate (dynamic-linking needs the CUDA toolkit; no GPU for launch tests), so build/test is pinned to a self-hosted `[windows, cuda, gpu]` runner and is skipped — never falsely green — when none is registered.

## Deviations from Plan

### Auto-fixed Issues

**1. [cudarc build.rs requires a link mode] Added `dynamic-linking` to the cudarc feature set**
- **Found during:** Task 2 (workspace/crate scaffold)
- **Issue:** The plan's `features = ["driver","nvrtc","cuda-12080"]` panics — cudarc 0.19.8 requires exactly one of `{dynamic-loading, fallback-dynamic-loading, dynamic-linking, static-linking}`; `cuda-12080` is a version selector, not a link mode.
- **Fix:** Added `dynamic-linking` (toolkit-linked launch proof). `static-linking` is unusable on MSVC (`could not find native static library stdc++`).
- **Files modified:** crates/sylva-cuda/Cargo.toml
- **Verification:** `cargo build -p sylva-cuda` exits 0; clippy clean.
- **Committed in:** ec5f2c2 (Task 2 commit)

**2. [Honest CI runner split] windows-latest build/test cannot run on hosted runners**
- **Found during:** Task 3 (CI scaffold)
- **Issue:** The plan said run `cargo build`+`cargo test` on `windows-latest`, but the committed `dynamic-linking` default needs the CUDA toolkit at build time and the launch test needs a GPU — neither exists on GitHub-hosted runners. A green hosted "build" would be a silent false-positive (violates the project's no-silent-fallback value).
- **Fix:** Split CI into a hosted `lint` job (rustfmt, no CUDA) + a self-hosted `[windows, cuda, gpu]` `build-test` job, with an explicit honest comment. The self-hosted job is skipped (not green) absent a matching runner.
- **Files modified:** .github/workflows/ci.yml
- **Verification:** YAML structure + comment reviewed; lint job is genuinely runnable on hosted.
- **Committed in:** 76f84a4 (Task 3 commit)

**3. [Test-enablement] Made `version()` pub**
- **Found during:** Task 3 (smoke test)
- **Issue:** The integration test links the crate as an external rlib and can only call `pub` items; `version()` was private.
- **Fix:** `pub fn version()`. The PyO3 `#[pyfunction]` wrapper is unaffected.
- **Files modified:** crates/sylva-cuda/src/lib.rs
- **Verification:** `cargo test --test toolchain_smoke` → `smoke_crate_links` passes.
- **Committed in:** ec5f2c2 (Task 2 commit, with the file)

---

**Total deviations:** 3 auto-fixed (1 build-blocking feature fix, 1 CI-honesty restructure, 1 test-enablement visibility change)
**Impact on plan:** All necessary for correctness/honesty. No scope creep — no kernel/estimator/Philox logic; no nvcc/cc/build.rs; cudarc default features stay disabled.

## Issues Encountered
- Prior session left the Task-2 skeleton files uncommitted (no SUMMARY) and the run was interrupted by a session limit; this session verified the existing files against the plan, completed Tasks 3–4, ran the full build/clippy/test, and committed all three tasks atomically.
- Session-process PATH was stale (predated the Rust install) — cargo invoked with `$HOME/.cargo/bin` prepended per command.

## User Setup Required
None outstanding — the one-time toolchain install (Rust, MSVC v143 Build Tools, maturin) is complete and verified. CUDA 12.8 + RTX 4060 Ti (driver 595.79) were already present (D-03).

## Next Phase Readiness
- **Wave 1 (01-02) ready:** the crate builds/links; `nvrtc_launch_vector_add` placeholder is in place for the real NVRTC launch on sm_89. `compute-sanitizer` located at `...\CUDA\v12.8\compute-sanitizer\` (invoke by full path).
- The skeleton is shaped for Phase 2 `trait Backend` / `CpuBackend` / `ForestIR` drop-in (D-04).

---
*Phase: 01-toolchain-spike-gate-1*
*Completed: 2026-06-20*
