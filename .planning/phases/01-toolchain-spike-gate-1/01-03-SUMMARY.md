---
phase: 01-toolchain-spike-gate-1
plan: 03
subsystem: packaging-gate
tags: [pyo3, maturin, abi3, cudarc, nvrtc, dynamic-loading, microbench, cupy, versions, kill-decision, windows-msvc]

requires:
  - phase: 01-02
    provides: run_vector_add / run_histogram NVRTC launch core (re-exported from lib.rs); compute-sanitizer-clean histogram; CudaError
provides:
  - "PyO3 #[pyfunction] entrypoints (run_vector_add / run_histogram) mapping CudaError -> PyErr (no silent fallback across FFI)"
  - "Cargo feature toggle: default=cuda-static (launch proof) vs wheel=dynamic-loading (D-02 shipping wheel) — both link modes provably exercised"
  - "Built cp310-abi3-win_amd64 wheel (dynamic-loading); clean-venv import + entrypoint call proven (TOOL-03) — 5-link skeleton chain closed"
  - "Fairness-encoded microbench (benches/microbench.rs) + CuPy baseline (scripts/cupy_baseline.py); SC-6 PASS, no algorithm speed claim"
  - "Finalized VERSIONS.md with all runtime fields + Kill-criteria result: proceed (TOOL-04)"
affects: [phase-2 cpu-oracle (Backend trait drops into this skeleton), phase-9 packaging (abi3 wheel path proven)]

tech-stack:
  added: []
  patterns: [cargo-feature-toggle-for-mutually-exclusive-link-modes, pyo3-error-mapping-to-pyerr, harness-false-microbench-with-cuda-events, warmup+median-fairness-protocol, clean-venv-import-smoke-test]

key-files:
  created:
    - scripts/import_smoke.py
    - scripts/cupy_baseline.py
    - crates/sylva-cuda/benches/microbench.rs
  modified:
    - crates/sylva-cuda/src/lib.rs
    - crates/sylva-cuda/Cargo.toml
    - VERSIONS.md

key-decisions:
  - "cuda-12080 (CUDA version selector) is required in BOTH link modes — cudarc's build.rs panics without a cuda-1xxxx feature even under dynamic-loading; the link MODE (dynamic-linking vs dynamic-loading) is orthogonal to the version selector"
  - "Link modes implemented as a Cargo feature toggle (default=cuda-static, wheel=dynamic-loading) because they are mutually exclusive and both D-02 modes must be exercised; maturin passes --no-default-features --features wheel straight through to cargo"
  - "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 was pre-armed (Pitfall 1) but NOT required — maturin's abi3 build auto-generated the import library and succeeded against Py3.14 with no interpreter fallback"
  - "harness=false microbench with a plain main() + --test smoke mode keeps the cargo bench -- --test verify gate fast while the default run does full warmup+median timing"

metrics:
  duration: ~10min
  completed: 2026-06-20

requirements-completed: [TOOL-03, TOOL-04]

status: complete
---

# Phase 1 Plan 03: abi3 Wheel + Microbench + Gate Decision Summary

**The walking-skeleton chain is closed and Gate 1 is decided `proceed`: a PyO3 0.29 + maturin `abi3` wheel built with cudarc `dynamic-loading` (`sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl`) imports in a clean Python 3.14.3 venv and calls `run_vector_add` into the Rust NVRTC core (TOOL-03), the fairness-encoded microbench shows cudarc+NVRTC at 0.61× CuPy's per-launch overhead and 185 vs 237 GB/s with an exact vector op (SC-6 PASS, no algorithm speed claim), and `VERSIONS.md` is finalized with every runtime pin and a `proceed` kill-decision (TOOL-04) — all natively on Windows/MSVC, no WSL.**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-06-20
- **Tasks:** 3 (Task 1 + Task 3 auto; Task 2 human-verify gate, work completed)
- **Files created:** 3 / modified: 3

## Accomplishments

- **TOOL-03 (build):** Added `#[pyfunction]` `run_vector_add` / `run_histogram` thin wrappers in `lib.rs` that validate input at the boundary and map `CudaError` → `PyErr` (`ValueError` for boundary/input, `RuntimeError` for NVRTC/driver) — no `.unwrap()` across FFI, no silent fallback. Implemented a Cargo **feature toggle** (`default = ["cuda-static"]` vs `wheel`) so both D-02 link modes are provably exercised; `maturin build --release --no-default-features --features wheel` produced **`sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl`** with `dynamic-loading`.
- **TOOL-03 (import gate — Task 2):** Fresh `py -3.14 -m venv .venv-smoke` (Python **3.14.3**), `pip install` the wheel ONLY (no repo source on PYTHONPATH), ran `scripts/import_smoke.py`:
  - `sylva_cuda.__file__ = C:\Users\PC\OneDrive\Documents\Code Implementation\gpu_classical_ml\.venv-smoke\Lib\site-packages\sylva_cuda\__init__.py` — **inside the venv**, not the source tree.
  - `run_vector_add([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) = [11.0, 22.0, 33.0, 44.0]` — correct.
  - `OK: run_vector_add correct`, exit 0. The 5-link skeleton chain (Rust → CUDA/NVRTC → Python/abi3) is **closed**.
- **SC-6 (microbench):** `benches/microbench.rs` (warmup 50, median of 10, CUDA-event timing, fixed 1e7 f32 vector, GB/s = 3·4·N/median, per-launch overhead via a 1000-launch loop, correctness asserted first) vs `scripts/cupy_baseline.py` (same protocol, `cupy-cuda12x 14.1.1` in the Py3.12 `.venv-cupy`, same RTX 4060 Ti). Both vector ops exact.
- **TOOL-04:** Finalized `VERSIONS.md` — wheel filename, build command, the PyO3 forward-compat outcome (not needed), the four compute-sanitizer 0-errors (from 01-02), the CuPy baseline env/pin, both microbench paths labeled "feasibility sanity check, no algorithm speed claim", and the definitive **`Kill-criteria result: proceed`** with a decision-tree walk and one-line rationale. References the committed `Cargo.lock`.

## Microbench Numbers (RTX 4060 Ti — feasibility sanity check, NO algorithm speed claim)

| Metric | Sylva (cudarc + NVRTC) | CuPy 14.1.1 baseline |
|--------|------------------------|----------------------|
| Per-launch overhead (µs/launch) | **4.85** | **7.98** |
| Throughput (GB/s, vector_add 1e7 f32) | **185.42** | **237.48** |
| Vector-op correctness | exact (max_abs_err = 0) | exact (max_abs_err = 0) |

**Per-launch overhead ratio (Sylva / CuPy) = 4.85 / 7.98 ≈ 0.61×** — Sylva's launch overhead is *lower* than CuPy's, far inside the ≤ ~2–3× pass bar; throughput is the same order of magnitude (~78% of CuPy on a ~288 GB/s-peak GPU). **PASS.**

## Kill-Decision Verdict

**`proceed`** — TOOL-01 (NVRTC launch), TOOL-02 (sanitizer clean), TOOL-03 (abi3 wheel import) all green natively on Windows/MSVC with no WSL, and the SC-6 microbench is well within the pass bar. Not `WSL-fallback` (no native wheel/link wall) and not `stop` (every path sanitizes + launches natively). The cudarc + NVRTC + PyO3/maturin premise is proven; the project continues to Phase 2.

## Task Commits

1. **Task 1: PyO3 entrypoint + dynamic-loading wheel feature toggle (TOOL-03)** — `c99158e` (feat)
2. **Task 2: clean-venv import smoke test for the abi3 wheel (TOOL-03)** — `17c74de` (test)
3. **Task 3: microbench + CuPy baseline + finalize VERSIONS.md (SC-6, TOOL-04)** — `ac62c4d` (feat)

## Files Created/Modified

- `crates/sylva-cuda/src/lib.rs` — `#[pyfunction]` `py_run_vector_add` / `py_run_histogram` (exposed as `run_vector_add`/`run_histogram`), `cuda_error_to_pyerr` mapper, registered in `#[pymodule] sylva_cuda`.
- `crates/sylva-cuda/Cargo.toml` — cudarc line trimmed to `["std","driver","nvrtc"]`; `[features]` toggle (`default=["cuda-static"]`, `cuda-static`, `wheel`); `[[bench]] microbench` with `harness=false`.
- `scripts/import_smoke.py` — clean-venv import + entrypoint-call smoke test (asserts `__file__` in site-packages, asserts vector add correct).
- `crates/sylva-cuda/benches/microbench.rs` — fairness-encoded launch-overhead + GB/s harness; `--test` smoke mode; correctness-first; CUDA-event timing.
- `scripts/cupy_baseline.py` — CuPy reference with the identical fairness protocol (Py3.12 venv).
- `VERSIONS.md` — finalized TOOL-04 artifact with all runtime fields + `proceed` verdict.

## Decisions Made

- **`cuda-12080` is required in both link modes.** cudarc's `build.rs` panics without a `cuda-1xxxx` version-selector feature even under `dynamic-loading` (the version selector is orthogonal to the link mode). The `wheel` feature is therefore `["cudarc/cuda-12080", "cudarc/dynamic-loading"]`, not `dynamic-loading` alone.
- **Feature toggle over manifest swap.** The two link modes are mutually exclusive (build.rs panics with >1 mode), so a Cargo `[features]` toggle (`default=cuda-static` for `cargo build/test/bench`; `wheel` for the maturin build) exercises both without editing the manifest between builds. maturin forwards `--no-default-features --features wheel` straight to cargo.
- **`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` not needed.** Pre-armed defensively (Pitfall 1) but the abi3 build succeeded against Py3.14 with no env var dependency and no interpreter fallback — maturin auto-generated the Windows import library (`Found pyo3 bindings with abi3-py3.10 support`).
- **`harness=false` microbench.** A plain `main()` with a `--test` smoke path keeps the `cargo bench -- --test` verify gate fast (correctness only) while the default run does full warmup+median timing — no criterion dependency added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `cuda-12080` to the `wheel` feature**
- **Found during:** Task 1 (first `cargo build --no-default-features --features wheel`)
- **Issue:** The plan/context specified the wheel features as `["driver","nvrtc","dynamic-loading"]`. That set **panics**: cudarc 0.19.8's `build.rs` requires a CUDA *version* feature (`cuda-1xxxx`) in addition to the link mode — `dynamic-loading` selects the link mode but not the binding ABI version. (Mirrors the Plan 01 launch-proof deviation, which is why `cuda-static` already carries `cuda-12080`.)
- **Fix:** `wheel = ["cudarc/cuda-12080", "cudarc/dynamic-loading"]`. The version selector is shared by both modes; only the link mode differs.
- **Files modified:** crates/sylva-cuda/Cargo.toml
- **Verification:** `cargo build --no-default-features --features wheel` exits 0; `maturin build` produced the cp310-abi3 wheel.
- **Commit:** ac62c4d (carried with the bench registration; the same Cargo.toml).

**Total deviations:** 1 auto-fixed (a blocking feature fix consistent with the Plan 01 link-mode deviation). No scope creep — no estimator/Philox/multi-feature logic; the PyO3 entrypoint exposes only the throwaway spike kernels; no `nvcc`/`cc`/`build.rs`.

## Known Stubs

None. The PyO3 wrappers call the real NVRTC launch core (no mock/placeholder), the smoke test asserts exact correctness against the installed wheel, and the microbench asserts `max_abs_err == 0` before reporting any timing.

## Authentication Gates

None — local toolchain spike, no network/auth surface.

## Issues Encountered

- Session-process PATH was stale (predates the Rust install) — `cargo`/`maturin`/`git commit` hooks invoked with `$HOME/.cargo/bin` prepended per command, per the environment note. `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` was exported in the wheel-build shell defensively (Pitfall 1), though it proved unnecessary.

## Next Phase Readiness

- **Phase 1 / Gate 1 complete.** All four TOOL requirements (01–04) and SC-6 are green; `VERSIONS.md` records `proceed`. The walking-skeleton chain is proven end-to-end on native Windows/MSVC.
- **Phase 2 ready:** the persisted Cargo workspace + `sylva-cuda` crate (D-04) is shaped for the `trait Backend` / `CpuBackend` / `ForestIR` drop-in. The abi3 wheel + maturin path is proven for the Phase-9 packaging polish. Only the throwaway spike kernel/launch logic is to be deleted/replaced; the structure, pins, and CI stay.

## Self-Check: PASSED

- Created files verified on disk: `scripts/import_smoke.py`, `scripts/cupy_baseline.py`, `crates/sylva-cuda/benches/microbench.rs`, `VERSIONS.md`, `01-03-SUMMARY.md` — all FOUND.
- Task commits verified in git log: `c99158e` (Task 1), `17c74de` (Task 2), `ac62c4d` (Task 3) — all FOUND.
- Wheel verified on disk: `target/wheels/sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl` — FOUND.

---
*Phase: 01-toolchain-spike-gate-1*
*Completed: 2026-06-20*
