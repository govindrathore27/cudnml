# Walking Skeleton — Sylva (GPU-Native Forest Ensembles)

**Phase:** 1 (Toolchain Spike / Gate 1)
**Generated:** 2026-06-20

> This is a Rust-core + CUDA + Python-API **library**, not a web app. The web-flavored
> skeleton-template sections (routing / DB read-write / UI interaction / dev deploy) are
> mapped onto the equivalent links of the real library stack. The "feature" proven here is
> the toolchain itself: that a hand-written CUDA-C kernel can be compiled, launched,
> debugged, and shipped as an importable Python wheel — natively on Windows/MSVC, no WSL.

## Capability Proven End-to-End

A developer can run `pip install <sylva_cuda wheel>` in a clean Python 3.14 environment and then
`import sylva_cuda; sylva_cuda.run_vector_add(a, b)` — which compiles a hand-written CUDA-C kernel
via cudarc 0.19.8 + NVRTC (arch `sm_89`), launches it on the local RTX 4060 Ti, and returns the
correct result — proving the entire Rust → CUDA(NVRTC) → Python(abi3) path works natively on
Windows/MSVC with no WSL.

## The Five-Link Skeleton Chain

The thinnest end-to-end slice that touches every layer of the real architecture:

| # | Link | Maps to (skeleton-template term) | Proven by |
|---|------|----------------------------------|-----------|
| 1 | Cargo workspace + `sylva-cuda` crate builds with the pinned cudarc + PyO3 set | Project scaffold (framework/build/lint/test) | Plan 01 Task 2 (`cargo build`) |
| 2 | A CUDA-C kernel compiles via cudarc+NVRTC (`sm_89`) and launches on the RTX 4060 Ti, returning a correct result | "DB read/write" → device H2D → kernel → D2H roundtrip | Plan 02 Task 1 (`nvrtc_launch_vector_add`) |
| 3 | `compute-sanitizer` is clean against a representative privatized-histogram kernel | Build/test verification (the toolchain is debuggable) | Plan 02 Task 3 (racecheck/memcheck = 0 errors) |
| 4 | A PyO3 0.29 + maturin `abi3` wheel builds with cudarc `dynamic-loading` | "Dev deploy" → `maturin build --release` produces an importable artifact | Plan 03 Task 1 (`cp310-abi3-win_amd64` wheel) |
| 5 | The wheel imports in a clean venv and calls into the Rust core | "UI interaction" → `import sylva_cuda; sylva_cuda.run_vector_add()` | Plan 03 Task 2 (clean-venv import smoke) |

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Kernel-authoring path | cudarc 0.19.8 + hand-written CUDA C via **NVRTC** (runtime compile) | Native Windows/MSVC; no `nvcc`-at-build / `cc`-MSVC gap; rides mature CUDA tooling (compute-sanitizer, Nsight). Rust→PTX (`cust`) and `nvcc`-AOT ruled out (CLAUDE.md "What NOT to Use"). |
| NVRTC arch target | `sm_89` / `compute_89` | RTX 4060 Ti (Ada, compute capability 8.9) — D-03. |
| cudarc features | `default-features = false`; `driver` + `nvrtc` + (`cuda-12080` for the launch proof \| `dynamic-loading` for the wheel) | D-02 proves BOTH linking modes; D-03 drops cublas/curand (not needed in Phase 1). |
| Python binding | PyO3 0.29.0 (`extension-module`, `abi3-py310`) | One `cp310-abi3` wheel covers CPython ≥3.10 incl. 3.14; MSRV 1.83 floor. |
| Build/packaging | maturin 1.14.x; `dynamic-loading` shipping wheel | CUDA driver resolved at runtime; one wheel for any compatible CUDA. |
| rust-numpy ↔ PyO3 lock | rust-numpy **0.29.0** ↔ PyO3 0.29.0 | Resolved in 01-RESEARCH.md (supersedes the stale "≈0.25.x" CLAUDE.md figure). Only used if a numpy array is marshalled. |
| Toolchain | Rust **stable** ≥1.83 (`x86_64-pc-windows-msvc`); CUDA 12.8 (nvcc 12.8.93); driver 595.79; MSVC v143 for linking only | Stable, never nightly; MSVC links the extension, NVRTC compiles kernels. |
| Directory layout | Cargo workspace; `crates/sylva-cuda/`; `pyproject.toml` at root; `crates/sylva-cuda/src/{lib.rs,nvrtc_launch.rs,kernels.rs}` + `benches/` + `tests/` | D-04: persist the real skeleton, shaped so Phase 2's `trait Backend` + `CpuBackend` + `ForestIR` drop in cleanly. Many small files (CLAUDE.md). |
| Microbench baseline | CuPy (`cupy-cuda12x`) in a **separate Python 3.12 venv**; raw-CUDA-C in-process as fallback | CuPy ships NO cp314 Windows wheel (verified) — the project's 3.14 venv can't host it; same GPU keeps the comparison fair. D-06. |
| Determinism modeling | Histogram uses **integer** atomicAdd (associative) | Even in the throwaway spike, model the deterministic path (PITFALLS Pitfall 5 — no float atomics). |
| Error contract | cudarc Results propagated; PyO3 maps to PyErr | No silent CPU/GPU fallback, even in the spike. |

## Stack Touched in Phase 1

- [x] Project scaffold (Cargo workspace, `rust-toolchain.toml`, `.gitignore`, CI, `pyproject.toml`) — Plan 01
- [x] "Routing" = Rust↔Python FFI boundary (`#[pymodule] sylva_cuda` + `#[pyfunction]` entrypoint) — Plan 01 seam, Plan 03 fill
- [x] "DB read/write" = device H2D → kernel launch → D2H roundtrip (`vector_add` + `histogram_privatized`) — Plan 02
- [x] "UI interaction" = `import sylva_cuda; sylva_cuda.run_vector_add()` in a clean venv — Plan 03
- [x] "Dev deploy" = `maturin build --release` producing an importable `cp310-abi3-win_amd64` wheel — Plan 03
- [x] Verification = `cargo test` + `compute-sanitizer` (racecheck/memcheck/synccheck/initcheck) + clean-venv import smoke — Plans 02/03

## Out of Scope (Deferred to Later Slices)

> Explicit minimalism — this list prevents later phases from re-litigating Phase 1's scope.

- Any Extra Trees / Random Forest / SHAP / estimator / sklearn-parity logic (Phases 2, 4, 5, 8).
- `trait Backend`, `CpuBackend`, SoA `ForestIR`, the parity contract (Phase 2) — the skeleton is *shaped* for them but does not implement them.
- Philox-4×32-10 RNG (Phase 2/6); `cublas`/`curand` (not needed; Philox hand-rolled later).
- The real multi-feature SoA `BinnedMatrix`, sibling-histogram subtraction, breadth-first NodeScheduler, arena memory (Phases 3–5). The spike histogram is a single-feature 256-bin representative slice only.
- DLPack / `__cuda_array_interface__` zero-copy interop (v2 / later).
- CubeCL portability backend (Milestone 2).
- Any **algorithm speed claim** — the Phase 1 microbench is a feasibility sanity check only.
- The throwaway kernel/spike logic (`kernels.rs`, `nvrtc_launch.rs`, `microbench.rs`, spike entrypoints) is **deleted after the gate decision** per D-04; the workspace/maturin/CI/`VERSIONS.md` harness is **kept**.

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton without altering its core
architectural decisions (kernel-authoring path, abi3 packaging, directory layout):

- **Phase 2:** Device-neutral `trait Backend` + SoA `ForestIR` + pure-Rust `CpuBackend` (the correctness oracle) + Philox RNG + parity contract. Drops into `crates/` alongside `sylva-cuda`.
- **Phase 3:** Feature quantizer → SoA `BinnedMatrix`, bit-identical CPU↔GPU bins.
- **Phase 4:** Single GPU ExtraTree (privatized shared-mem histograms — the spike histogram grows up; breadth-first), bit-exact vs the CPU oracle, compute-sanitizer clean.
- **Phase 5:** Full forest + RandomForest + the four sklearn estimators (first real speed claim).
- **Phase 6:** `deterministic=True` + `fallback="error"` honest dispatch.
- **Phase 7:** The pre-registered (n×d) crossover benchmark (Gate 3).
- **Phases 8–9:** Exact tree SHAP (Gate 2) and Treelite export / packaging — both IR-only consumers.

## Artifacts This Phase Produces

> Every symbol / file / command this phase newly creates. The source-grounding pass reads
> this to exclude these from drift verification (they are net-new, not pre-existing references).

### Crates & modules
- **Cargo workspace** at repo root (`Cargo.toml` `[workspace]`, `Cargo.lock`, `rust-toolchain.toml`, `.gitignore`).
- **`sylva-cuda`** — the spike crate (`crates/sylva-cuda/Cargo.toml`), cdylib+lib. Becomes `CudaBackend` in Phase 4+.
- **Python extension module name:** `sylva_cuda` (the `#[pymodule]`, shipped as `sylva_cuda*.pyd` inside the wheel).

### Rust source files (THROWAWAY spike logic — deleted after the gate per D-04)
- `crates/sylva-cuda/src/lib.rs` — `#[pymodule] sylva_cuda`; `version()` probe; `#[pyfunction] run_vector_add` (and optional `run_histogram`).
- `crates/sylva-cuda/src/kernels.rs` — CUDA-C source strings: `vector_add`, `histogram_privatized`.
- `crates/sylva-cuda/src/nvrtc_launch.rs` — host functions `run_vector_add`, `run_histogram` (NVRTC compile→load→launch).
- `crates/sylva-cuda/benches/microbench.rs` — launch-overhead + GB/s harness.

### Test files
- `crates/sylva-cuda/tests/toolchain_smoke.rs` — `smoke_crate_links`, `nvrtc_launch_vector_add`, `histogram_privatized_matches_cpu`.
- `crates/sylva-cuda/tests/sanitizer_histogram.rs` — the isolated compute-sanitizer target binary.

### Python / build scripts
- `scripts/import_smoke.py` — clean-venv `import sylva_cuda` + entrypoint-call smoke test.
- `scripts/cupy_baseline.py` — CuPy (Py3.12 venv) microbench baseline (or raw-CUDA-C fallback).
- `pyproject.toml` — maturin build-backend + `[tool.maturin]` abi3 config.

### Durable artifacts (KEPT after the gate)
- `VERSIONS.md` — the TOOL-04 pin + kill-decision deliverable.
- `SKELETON.md` — this file (architectural backbone for later phases).
- `.github/workflows/ci.yml` — Windows CI (build + test).
- The Cargo workspace / `pyproject.toml` / CI harness/structure (only the spike kernel logic is deleted).

### Build / verification commands introduced
- `cargo build -p sylva-cuda` / `cargo test -p sylva-cuda` / `cargo bench -p sylva-cuda --bench microbench`
- `maturin build --release` (produces `cp310-abi3-win_amd64` wheel)
- `compute-sanitizer --tool {memcheck,racecheck,synccheck,initcheck} <target>`

### Wheel output
- `target/wheels/sylva_cuda-*-cp310-abi3-win_amd64.whl`
