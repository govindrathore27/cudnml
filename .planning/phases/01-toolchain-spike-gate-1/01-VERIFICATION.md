---
phase: 01-toolchain-spike-gate-1
verified: 2026-06-20T18:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 1: Toolchain Spike (Gate 1) Verification Report

**Phase Goal:** Prove the entire kernel-authoring and packaging path works natively on Windows before any algorithm is built — resolving the single biggest technical risk in PROJECT.md (cudarc + NVRTC + PyO3/maturin on native Windows/MSVC, no WSL).

**Verified:** 2026-06-20

**Status:** PASS

**Re-verification:** No — initial verification.

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | A throwaway hand-written CUDA C kernel compiles via cudarc 0.19.8 + NVRTC and launches on the local NVIDIA GPU, natively on Windows/MSVC with no WSL | VERIFIED | `nvrtc_launch_vector_add` and `histogram_privatized_matches_cpu` pass in `cargo test -p sylva-cuda --test toolchain_smoke -- --include-ignored` (3/3 passed, independently re-run and confirmed). `kernels.rs` contains the CUDA-C source strings; `nvrtc_launch.rs` uses `compile_ptx_with_opts(src, CompileOptions { arch: Some("sm_89"), options: vec!["-lineinfo"..], .. })`; `CudaContext::new(0)` + `load_module` + `launch_builder` are the live cudarc path. No WSL references in any source file. |
| SC-2 | `compute-sanitizer` runs against the spike kernel and reports clean, proving the toolchain is debuggable | VERIFIED | `sanitizer_histogram.rs` is a standalone test binary whose sole job is a single histogram launch; the SUMMARY records verbatim output from all four tools: memcheck `ERROR SUMMARY: 0 errors`, racecheck `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`, synccheck `ERROR SUMMARY: 0 errors`, initcheck `ERROR SUMMARY: 0 errors`. The kernel was compiled with `-lineinfo` (confirmed in `nvrtc_launch.rs:59`). VERSIONS.md records these outcomes. The orchestrator independently confirmed via re-run. |
| SC-3 | A minimal PyO3 + maturin `abi3` wheel builds and imports in a clean Python environment on Windows | VERIFIED | `target/wheels/sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl` exists on disk. `scripts/import_smoke.py` does `import sylva_cuda`, asserts `__file__` is in `site-packages`, calls `run_vector_add([1,2,3,4],[10,20,30,40])` and asserts `[11,22,33,44]`, exits 0. VERSIONS.md Wheel field: "Fresh `py -3.14 -m venv .venv-smoke`... import sylva_cuda resolved to `…\.venv-smoke\Lib\site-packages\…`, exit 0." `lib.rs` registers `#[pyfunction]` `py_run_vector_add` that calls through to `run_vector_add` and maps `CudaError` to `PyErr`. |
| SC-4 | Pinned, verified versions recorded (cudarc feature flags, rust-numpy/PyO3, CUDA toolkit) with documented kill-criteria result: proceed / WSL-fallback / stop | VERIFIED | `VERSIONS.md` contains all required pin fields: Rust 1.96.0 stable, MSVC cl.exe 14.44.35207, CUDA 12.8 / nvcc V12.8.93, maturin 1.14.1, Python 3.14.3, RTX 4060 Ti / sm_89 / driver 595.79, cudarc 0.19.8 with both feature sets documented, PyO3 0.29.0, rust-numpy 0.29.0 (noted not yet used in Phase 1). `Kill-criteria result: proceed` is present on line 143 with a full decision-tree walk. `Cargo.lock` committed at repo root. |
| SC-5 (Kill criterion) | If no kernel path yields a debuggable, packageable result — stop. Actual test: the proceed verdict is justified by all prior SC evidence | VERIFIED | All four legs of the kill-decision tree are green (SC-1..SC-3 above, SC-6 below). The VERSIONS.md decision tree is explicit: "not WSL-fallback (no native wheel/link wall) and not stop (every path sanitizes and launches natively)." The `proceed` verdict is supported by independent test re-runs. |
| SC-6 | A kernel-launch / vector-op microbenchmark confirms cudarc+NVRTC is not pathologically slow vs CuPy/raw-CUDA baseline — with no algorithm speed claim | VERIFIED | `benches/microbench.rs` implements the fairness protocol: 50 warmup launches discarded, median of 10 timed runs, CUDA-event device timing, correctness asserted first (`max_abs_err == 0` confirmed via re-run of `cargo bench -p sylva-cuda --bench microbench -- --test`). `scripts/cupy_baseline.py` implements the identical protocol in a Py3.12 venv with `cupy-cuda12x 14.1.1`. Recorded result: Sylva 4.85 µs/launch vs CuPy 7.98 µs/launch (0.61× ratio, inside the ≤2-3× bar); throughput 185 vs 237 GB/s. Both labeled "feasibility sanity check, no algorithm speed claim" in `microbench.rs` header, VERSIONS.md, and `cupy_baseline.py`. |

**Score:** 6/6 truths verified. 0 present-behavior-unverified.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | Workspace root with `[workspace]` + `crates/sylva-cuda` member | VERIFIED | Contains `[workspace]`, `resolver = "2"`, `members = ["crates/sylva-cuda"]`, `[workspace.package]` with `rust-version = "1.83"`, `license = "Apache-2.0"` |
| `crates/sylva-cuda/Cargo.toml` | cudarc 0.19.8 with `default-features = false`; pyo3 0.29.0 abi3-py310; no cublas/curand | VERIFIED | `cudarc = { version = "0.19.8", default-features = false, features = ["std", "driver", "nvrtc"] }`; `[features]` toggle: `cuda-static = ["cudarc/cuda-12080", "cudarc/dynamic-linking"]` and `wheel = ["cudarc/cuda-12080", "cudarc/dynamic-loading"]`. No `cublas`, `curand`, `cublaslt`, or `runtime` features present. `pyo3 = { version = "0.29.0", features = ["extension-module", "abi3-py310"] }`. |
| `pyproject.toml` | maturin build-backend + `[tool.maturin]` abi3 config | VERIFIED | `build-backend = "maturin"`, `requires = ["maturin>=1.14,<2.0"]`, `[tool.maturin]` with `manifest-path = "crates/sylva-cuda/Cargo.toml"` and `module-name = "sylva_cuda"` |
| `rust-toolchain.toml` | `channel = "stable"`, no nightly | VERIFIED | `channel = "stable"`, `targets = ["x86_64-pc-windows-msvc"]`, `profile = "default"`. The word "nightly" appears only in comments documenting what was rejected. |
| `VERSIONS.md` | All pin fields + `Kill-criteria result:` line + microbench label | VERIFIED | Contains `Kill-criteria result: proceed` (line 143), "feasibility sanity check, no algorithm speed claim" (lines 83, 86, 111), both cudarc feature sets, PyO3 0.29.0, rust-numpy 0.29.0, maturin 1.14.1, CUDA 12.8, driver 595.79, RTX 4060 Ti / sm_89, Python 3.14.3, compute-sanitizer verbatim output table, wheel filename and build command, PYO3_USE_ABI3_FORWARD_COMPATIBILITY outcome |
| `crates/sylva-cuda/src/kernels.rs` | CUDA-C source strings: vector_add + histogram_privatized (256-bin shared-mem, integer atomicAdd) | VERIFIED | `VECTOR_ADD_SRC` with `if (i < n)` bounds guard; `HISTOGRAM_PRIVATIZED_SRC` with `__shared__ unsigned int sh[256]`, cooperative zero-init, `__syncthreads()`, `atomicAdd(..., 1u)` (unsigned integer — no float atomic), two-level reduction to global; `BIN_COUNT = 256` named constant |
| `crates/sylva-cuda/src/nvrtc_launch.rs` | `compile_ptx_with_opts` + `sm_89` + `-lineinfo`; typed `CudaError`; no `.unwrap()` on device calls | VERIFIED | `compile_for_sm89` uses `CompileOptions { arch: Some("sm_89"), options: vec!["-lineinfo"..], .. }`. `CudaError` (thiserror) wraps `CompileError`, `DriverError`, `InvalidInput`. grep of `.unwrap()`/`.expect(` in source files returns zero matches in `src/`. Every cudarc call propagates via `?`. `unsafe` confined to single `launch` call with `// SAFETY:` comment. |
| `crates/sylva-cuda/src/lib.rs` | `#[pymodule]` + `#[pyfunction]` entrypoints calling `run_vector_add` / `run_histogram`; error mapped to `PyErr` | VERIFIED | `py_run_vector_add` and `py_run_histogram` registered in `#[pymodule] sylva_cuda`; `cuda_error_to_pyerr` maps `CudaError::InvalidInput` to `PyValueError`, `CudaError::Compile`/`Driver` to `PyRuntimeError`. `pub use nvrtc_launch::{run_histogram, run_vector_add, CudaError}` re-exports for integration tests. No `.unwrap()` across FFI. |
| `crates/sylva-cuda/tests/toolchain_smoke.rs` | `smoke_crate_links` + `nvrtc_launch_vector_add` + `histogram_privatized_matches_cpu` | VERIFIED | All three tests present and non-ignored; re-run confirms `3 passed; 0 failed; 0 ignored`. `nvrtc_launch_vector_add` asserts exact equality (`out[i] == a[i] + b[i]`) over N=10_000_000 with named const `VECTOR_ADD_N`. `histogram_privatized_matches_cpu` asserts exact integer equality vs CPU reference over 256 bins. |
| `crates/sylva-cuda/tests/sanitizer_histogram.rs` | Standalone compute-sanitizer target | VERIFIED | `sanitizer_histogram_single_launch` launches `run_histogram` once on a fixed 100k-element input and asserts CPU reference equality. Contains docs on how to invoke compute-sanitizer by full path. `-lineinfo` is wired in `compile_for_sm89`. |
| `crates/sylva-cuda/benches/microbench.rs` | Warmup + CUDA-event timing + median + correctness-first; `harness = false` | VERIFIED | 50-launch warmup (`WARMUP_LAUNCHES`), median of 10 (`TIMED_RUNS`), CUDA-event timing (`record_event` + `elapsed_ms`), `assert_vector_add_correct()` called before any timing. `--test` smoke mode exercises launch path without the full timing loop. Cargo `[[bench]] harness = false`. GB/s uses `ARRAYS_TOUCHED * BYTES_PER_F32 * n = 3 * 4 * N`. |
| `scripts/import_smoke.py` | `import sylva_cuda`; asserts `__file__` in site-packages; calls `run_vector_add`; exits 0 on correct result | VERIFIED | All four checks present: `import sylva_cuda`, `"site-packages" not in module_path` guard, `run_vector_add([1,2,3,4],[10,20,30,40])` call, exact result comparison, `print("OK: run_vector_add correct")`, `sys.exit(0)`. |
| `target/wheels/sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl` | Built wheel artifact on disk | VERIFIED | File confirmed present by `ls target/wheels/`. Filename matches the cp310-abi3 / win_amd64 requirement. |
| `Cargo.lock` | Full transitive pin at repo root | VERIFIED | `Cargo.lock` exists at repo root (confirmed by `ls`); committed in git (ec5f2c2). |
| `.github/workflows/ci.yml` | Honest CI — hosted lint-only + self-hosted GPU job; no false-green on hosted runners | VERIFIED | `lint` job on `windows-latest` runs only `cargo fmt --check` (no CUDA needed). `build-test` job pinned to `[self-hosted, windows, cuda, gpu]` runner with explicit comment: "never falsely green." GPU/wheel steps are not silently passing on hosted runners. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/sylva-cuda/src/nvrtc_launch.rs` | `crates/sylva-cuda/src/kernels.rs` | `compile_for_sm89` calls `compile_ptx_with_opts(HISTOGRAM_PRIVATIZED_SRC / VECTOR_ADD_SRC, ...)` with `arch: Some("sm_89")` | WIRED | `use crate::kernels::{BIN_COUNT, HISTOGRAM_PRIVATIZED_SRC, VECTOR_ADD_SRC}` at line 20; both constants used in `run_vector_add` and `run_histogram` |
| `crates/sylva-cuda/tests/toolchain_smoke.rs` | `crates/sylva-cuda/src/nvrtc_launch.rs` | `nvrtc_launch_vector_add` calls `sylva_cuda::run_vector_add` (re-exported from `nvrtc_launch`) and asserts out == a+b | WIRED | `sylva_cuda::run_vector_add(&a, &b).expect(...)` at line 41; `sylva_cuda::run_histogram(&bins).expect(...)` at line 84 |
| `crates/sylva-cuda/src/lib.rs` | `crates/sylva-cuda/src/nvrtc_launch.rs` | `#[pyfunction]` `py_run_vector_add` calls `run_vector_add(&a, &b).map_err(cuda_error_to_pyerr)` | WIRED | Line 77 in `lib.rs`; `pub use nvrtc_launch::{run_histogram, run_vector_add, CudaError}` re-export at line 26 |
| `scripts/import_smoke.py` | `crates/sylva-cuda/src/lib.rs` (via installed wheel) | `import sylva_cuda; sylva_cuda.run_vector_add(a, b)` calls through the PyO3 layer into the Rust NVRTC core | WIRED | VERSIONS.md confirms the clean-venv run: `run_vector_add([1,2,3,4],[10,20,30,40]) = [11,22,33,44]`, exit 0; module path confirmed inside `.venv-smoke\Lib\site-packages` |
| `Cargo.toml` (workspace) | `crates/sylva-cuda/Cargo.toml` | `members = ["crates/sylva-cuda"]` | WIRED | Line 8 in root `Cargo.toml` |
| `pyproject.toml` | `crates/sylva-cuda` | `[tool.maturin] manifest-path = "crates/sylva-cuda/Cargo.toml"` | WIRED | Line 18 in `pyproject.toml` |

---

### Data-Flow Trace (Level 4)

Not applicable. This phase produces a computational library (GPU kernels + PyO3 bindings), not a rendering/dashboard component. Data flow is kernel-argument-to-result, verified by exact-equality assertions in tests.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TOOL-01: NVRTC launches vector_add, returns exact a+b on 1e7 elements | `cargo test -p sylva-cuda --test toolchain_smoke -- --include-ignored` | 3 passed; 0 failed | PASS |
| Histogram matches CPU reference exactly | (same test run as above) | 3 passed; 0 failed | PASS |
| Microbench correctness-first check (max_abs_err == 0) | `cargo bench -p sylva-cuda --bench microbench -- --test` | "correctness: max_abs_err(out, a+b) = 0 [asserted before timing]" | PASS |
| abi3 wheel exists with correct filename | `ls target/wheels/` | `sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl` | PASS |
| TOOL-02: compute-sanitizer all-clean | Reported by executor + recorded in VERSIONS.md; independently confirmed by orchestrator re-run | 0 errors / 0 hazards across all 4 tools | PASS (orchestrator corroborated) |
| TOOL-03: clean-venv import + entrypoint call | import_smoke.py in fresh .venv-smoke (Python 3.14.3) | exit 0, "OK: run_vector_add correct", `__file__` in site-packages | PASS (orchestrator corroborated) |

---

### Prohibition Verification

| Prohibition | Check | Status |
|-------------|-------|--------|
| No `cublas` / `curand` / `cublaslt` / `runtime` cudarc features | `grep cublas\|curand\|cublaslt\|runtime crates/sylva-cuda/Cargo.toml` — only appears in comments documenting what was excluded | CLEAN |
| `default-features = false` on cudarc | Line 73 of `crates/sylva-cuda/Cargo.toml`: `cudarc = { version = "0.19.8", default-features = false, ... }` | CLEAN |
| No `nvcc` / `cc` CUDA compilation step | No `build.rs` file anywhere in the repo; `find . -name build.rs` returns empty. Kernels compiled via `compile_ptx_with_opts` at runtime. References to `nvcc`/`cc`/`build.rs` in comments only describe what was avoided. | CLEAN |
| No float `atomicAdd` in histogram kernel | `grep "float.*atomicAdd\|atomicAdd.*float" crates/` returns no matches. The histogram uses `atomicAdd(&sh[bins[i]], 1u)` (unsigned int). | CLEAN |
| No Extra Trees / Random Forest / SHAP / Philox logic | `grep -rn "ExtraTree\|RandomForest\|SHAP\|Philox\|BinnedMatrix\|ForestIR" crates/` returns only comments in `Cargo.toml` and `lib.rs` documenting their absence | CLEAN |
| No `.unwrap()` / `.expect()` on device calls (no silent fallback) | `grep -rn "\.unwrap()\|\.expect(" src/` in the crate source returns zero hits. The single `.unwrap()` in `benches/microbench.rs:111` is on `partial_cmp` for sorting host `f64` values in the median helper — not a device call. | CLEAN |
| `rust-toolchain.toml` pins `stable`, not nightly | `channel = "stable"` confirmed; "nightly" appears only in explanatory comments | CLEAN |
| Microbench does NOT claim algorithm speed | The "no algorithm speed claim" label appears in the microbench header docstring, inline print statements, `cupy_baseline.py`, and VERSIONS.md three times | CLEAN |
| No WSL involved | No WSL references in any source file; wheel built with `maturin build --release --no-default-features --features wheel` on native Windows/MSVC | CLEAN |

---

### Requirements Coverage

| Requirement | Plans | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| TOOL-01 | 01-01, 01-02 | NVRTC kernel launches native Windows, bit-exact result | SATISFIED | `nvrtc_launch_vector_add` passes (3/3 test run confirmed); exact a+b on N=10M; `sm_89` arch; no WSL |
| TOOL-02 | 01-02 | compute-sanitizer clean on representative kernel | SATISFIED | All four tools: 0 errors / 0 hazards; `-lineinfo` wired; `sanitizer_histogram.rs` is the dedicated target |
| TOOL-03 | 01-01, 01-03 | abi3 wheel imports in clean venv + calls Rust core | SATISFIED | `sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl` built with `dynamic-loading`; `import_smoke.py` confirmed exit 0 in fresh Python 3.14.3 venv |
| TOOL-04 | 01-01, 01-03 | Pinned VERSIONS.md + kill-decision verdict | SATISFIED | VERSIONS.md finalized with all runtime fields; `Kill-criteria result: proceed` with decision-tree walk and one-line rationale; `Cargo.lock` committed |
| SC-6 | 01-03 | Fairness-encoded microbench within ~2-3x baseline, no algorithm speed claim | SATISFIED | 0.61x per-launch ratio (well inside 2-3x bar); correctness asserted first; warmup+median protocol; baseline via CuPy 14.1.1 in Py3.12 venv with identical protocol |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/sylva-cuda/benches/microbench.rs` | 111 | `.unwrap()` on `partial_cmp` for sorting host f64 | Info | Not a device call; `partial_cmp` on finite f64 never returns `None`; acceptable in bench harness code |

No TBD/FIXME/XXX markers in source files. The word "TBD" appears only in the introductory sentence of VERSIONS.md (line 6) referencing a documentation writing convention for how the template was structured — all actual TBD slots have since been filled. No unresolved debt markers in any committed source file.

---

### Human Verification Required

None. All required behaviors are verified programmatically:
- GPU kernel launch and correctness: deterministic tests with exact equality assertions
- compute-sanitizer cleanliness: binary exit codes, reported by executor and corroborated by orchestrator
- abi3 wheel import: `import sylva_cuda` in a real clean venv with `__file__` path check and call assertion
- Microbench correctness gate: `cargo bench -- --test` exits 0 with "max_abs_err = 0"
- Prohibition checks: grep-verifiable

The one item that would normally require human eyes — visual confirmation of the compute-sanitizer output — is covered by the orchestrator's independent re-run and the verbatim output recorded in VERSIONS.md.

---

## Gate 1 Assessment

The phase goal was to resolve the single biggest technical risk before any algorithm is built: does cudarc + NVRTC + PyO3/maturin work natively on Windows/MSVC? The evidence is unambiguous.

**What was proven:**

1. A hand-written CUDA C kernel (`vector_add`, 1e7 f32 elements) compiles via NVRTC at runtime (no `nvcc`/`cc`/`build.rs`) for `sm_89`, launches on the RTX 4060 Ti, and returns bit-exact results. Independently re-run and confirmed 3/3 tests passing.

2. A representative privatized shared-memory histogram (the structure of the real Phase-4 hot path) is compute-sanitizer clean across all four tools. The kernel uses integer `atomicAdd` (not float), is organized with cooperative shared-mem zero-init and proper `__syncthreads()` placement, and was compiled with `-lineinfo` for source attribution.

3. A PyO3 0.29.0 + maturin 1.14.1 `abi3` wheel builds with cudarc `dynamic-loading` (the shipping config), installs into a fresh Python 3.14.3 venv, and calling `run_vector_add` through the FFI produces the correct result. Every `CudaError` maps to a typed `PyErr` — no `.unwrap()` anywhere on the device path.

4. VERSIONS.md is a complete, durable artifact recording every toolchain pin and the `proceed` verdict.

5. The microbench shows cudarc+NVRTC at 0.61x CuPy's per-launch overhead (lower, not higher) with 185 vs 237 GB/s throughput — well inside the ≤2-3x feasibility bar.

**The `proceed` verdict is justified.** Not `WSL-fallback` (no native build or link step failed), not `stop` (every path sanitizes and launches natively on Windows/MSVC). The decision is supported by concrete binary evidence at every link of the walking-skeleton chain.

**Scope hygiene is clean.** No Extra Trees / Random Forest / SHAP / Philox / BinnedMatrix / ForestIR logic is present. The spike is scope-bounded as designed.

**Risks and notes for Phase 2 (not blockers):**

- The `cuda-static` feature (`dynamic-linking`) used by `cargo test` requires the CUDA 12.8 toolkit to be present at build time. The CI `build-test` job is pinned to a self-hosted runner for this reason. Future phases should evaluate whether a `dynamic-loading` default would improve portability (it shifts the requirement from build-time to runtime, which is appropriate for a library).
- `rust-numpy 0.29.0` is pinned in VERSIONS.md but not yet used; it enters in Phase 2. The pin is correctly noted as resolved from the earlier "≈0.25.x" guess.
- The single `.unwrap()` in the median helper in `microbench.rs` is on a comparator for finite f64 values. Not a concern in practice, but could be replaced with `f64::total_cmp` for strictness in Phase 2 onward.

---

## Overall Verdict

**PASS — Phase 1 / Gate 1 goal achieved.**

All 6 roadmap success criteria are VERIFIED with direct code and binary evidence. All prohibitions are confirmed clean. The `proceed` kill-decision is supported by independent test re-runs on the live codebase.

---

_Verified: 2026-06-20_
_Verifier: Claude (gsd-verifier)_
