# Stack Research

**Domain:** GPU-native tree-ensemble library (Rust performance core + CUDA kernels + Python sklearn-parity API)
**Researched:** 2026-06-19
**Confidence:** HIGH (binding/build/interop layers), MEDIUM (kernel-authoring path — alpha tooling on the leading edge)

---

## TL;DR — The Single Biggest Risk, Resolved

> **Author CUDA kernels as hand-written CUDA C/C++, compiled at RUNTIME with NVRTC, and drive them from Rust with the `cudarc` crate (driver API + NVRTC + cuBLAS/cuRAND).**
> **Do NOT use the Rust-CUDA project (`cust` + `rustc_codegen_nvvm`) as the primary path** — its published `cust` crate is frozen at 0.3.2 (Feb 2022) and modern usage requires a git dependency, which blocks publishing your own crate to crates.io and ships a fragile, unpaid-maintainer, LLVM-7-pinned toolchain on Windows.

**Fallback / future option:** **CubeCL** (`tracel-ai`, v0.10.0) — write kernels as borrow-checked Rust `#[cube]` functions that JIT to CUDA. Strategically attractive (one source → CUDA/HIP/Metal/WGPU, aligns with the "vendor-neutral backends deferred" note in PROJECT.md), but **alpha with breaking changes between minor versions**, so it is the *Plan B / Milestone-2 portability bet*, not the MVP foundation.

Rationale in full under [The Kernel-Authoring Decision](#the-kernel-authoring-decision-core-risk).

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Rust** (stable) | 1.83+ | Performance core, orchestration, CPU reference | Project constraint; MSRV floor is set by PyO3 0.29 (1.83). Use **stable**, not nightly — a hard reason to avoid `rustc_codegen_nvvm` (nightly-pinned). |
| **cudarc** | 0.19.8 (2026-06-19) | CUDA driver API + NVRTC runtime kernel compilation + cuBLAS/cuRAND bindings | Actively maintained (released *today*), 300k+ downloads/version, native `x86_64-pc-windows-msvc` builds, supports CUDA 11.4–13.0. Safe/result/sys layering. This is the de-facto modern Rust↔CUDA binding in 2026. |
| **Hand-written CUDA C/C++ kernels** | CUDA Toolkit 12.6+ / 13.0 | The histogram / split-score / scatter-partition hot path | Tree training is bandwidth- and atomic-contention-bound (per PROJECT.md): you need explicit shared-memory privatized histograms, warp intrinsics, and `atomicAdd`. CUDA C gives full, documented control over exactly these primitives. Compiled via NVRTC at runtime → **no `nvcc`-at-build-time, no MSVC-vs-cc headache.** |
| **PyO3** | 0.29.0 (2026-06-11) | Rust ↔ Python FFI for the sklearn-parity API | Standard, mature. MSRV 1.83. `abi3` (stable ABI) → one wheel covers Python 3.x. CPython 3.7–3.14 + free-threaded 3.13t/3.14t. |
| **maturin** | 1.14.1 | Build + package the Rust extension as a `pip`-installable wheel | The canonical PyO3 build backend. Native Windows wheel builds; `abi3` support; `maturin-action` for CI. |
| **rust-numpy** (`numpy` crate) | track PyO3 0.29 (≈0.25.x line) | Zero-copy host (CPU) numpy array interop | The PyO3-ecosystem bridge to `ndarray`; the correct host-side I/O path. Must match the PyO3 version exactly. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **ndarray** | 0.16.x | CPU reference backend data structures | The CPU correctness oracle / small-data path (Active requirement). Pairs with rust-numpy for zero-copy host transfer. |
| **rayon** | 1.x | Data-parallel CPU reference backend | Parallelize the CPU oracle so it is usable as a real small-`n` path, not just a test fixture. |
| **cuRAND (via cudarc) OR hand-rolled Philox-4×32-10** | cudarc 0.19.8 / Random123 algorithm | Counter-based RNG for split thresholds & feature subsampling | **Prescription: implement Philox-4×32-10 yourself in the Rust core AND inline in the CUDA kernel from the same key/counter scheme.** It is stateless, ~20 lines, vectorizes perfectly, gives bit-identical CPU↔GPU streams — essential for `deterministic=True` and the CPU/GPU differential tests. cuRAND's host-API Philox is fine but harder to bit-match across CPU and GPU. |
| **serde / serde_json** | 1.x | Treelite-compatible JSON export; model (de)serialization | Emit Treelite 4.x `import_from_json()`-compatible JSON (task_param / model_param / node schema) → FIL / Triton serving. |
| **half** | 2.x | (Optional) only if you later add fp16 I/O | Not MVP (dense float32 only) — listed so it is not reached for prematurely. |
| **thiserror / anyhow** | 1.x / 1.x | Error types in core / boundary | `thiserror` for library error enums (mapped to Python exceptions via PyO3); `anyhow` only at the bin/test edge. |
| **proptest** | 1.x | Property-based invariants (Active requirement) | Differential + invariant testing vs scikit-learn. |
| **approx** | 0.5.x | Float comparison in tests | Tolerance-based assertions for the CPU/GPU/sklearn oracle. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **CUDA Toolkit 12.6+ (or 13.0)** | NVRTC, headers, driver | Install full toolkit on Windows. cudarc supports 11.4–13.0; pick **12.x** for MVP (broadest driver/library compatibility) unless your GPU needs 13.0. Driver ≥580 required for CUDA 13. |
| **Visual Studio 2022 Build Tools (MSVC v143)** | Native Rust + PyO3 linking on Windows | Needed for `x86_64-pc-windows-msvc`. `cl.exe` must be on PATH. cudarc + NVRTC means you do **not** need `nvcc` to link your kernels, sidestepping the cc-crate MSVC gap. |
| **maturin + uv (or pip)** | `maturin develop` / `maturin build --release` | `maturin develop` for the inner loop; produces a wheel testable from Python immediately. |
| **cargo-nextest + cargo-llvm-cov** | Test runner + coverage | For the 80%-coverage target on the Rust core. |
| **compute-sanitizer** (CUDA Toolkit) | CUDA correctness tooling (Active requirement) | `memcheck`, `racecheck`, `initcheck`, `synccheck` — your "CUDA correctness tooling" requirement. Runs on Windows. |
| **Nsight Compute / Nsight Systems** | Kernel profiling for the crossover benchmark | Validates the bandwidth/atomics characterization and the pre-registered crossover surface. |

## Installation

```bash
# --- Rust core: Cargo.toml dependencies ---
# cudarc        = { version = "0.19", features = ["cuda-12060", "driver", "nvrtc", "cublas", "curand"] }
# pyo3          = { version = "0.29", features = ["extension-module", "abi3-py310"] }
# numpy         = "0.25"          # rust-numpy, must track pyo3 0.29
# ndarray       = "0.16"
# rayon         = "1"
# serde         = { version = "1", features = ["derive"] }
# serde_json    = "1"
# thiserror     = "1"
# [dev-dependencies] proptest = "1", approx = "0.5"
# [build-dependencies]  (none required for NVRTC path — kernels are .cu strings compiled at runtime)

# --- Python build toolchain ---
pip install "maturin>=1.14"          # or: uv tool install maturin
maturin develop --release            # dev inner loop
maturin build --release              # produce abi3 Windows wheel
```

> **Note on the cudarc feature flag:** pick the `cuda-12xxx` feature that matches your installed toolkit (e.g. `cuda-12060`), or use the `dynamic-loading` feature to defer the CUDA version to runtime — the latter is friendlier for distributing one wheel across users with different driver/toolkit versions. Verify the exact feature name against the cudarc 0.19.8 docs before committing.

---

## The Kernel-Authoring Decision (Core Risk)

Concrete comparison of the four candidate paths against *this* workload (irregular histogram build + split scoring + scatter partition; bandwidth/atomic-bound; needs determinism; must ship on Windows via pip/crates.io).

| Path | Maturity / Maintenance (2026) | Windows | How kernels are authored | Fit for irregular histogram/scatter | Verdict |
|------|------------------------------|---------|--------------------------|-------------------------------------|---------|
| **cudarc + hand-written CUDA C (NVRTC)** ✅ | **Active.** 0.19.8 released 2026-06-19; high download volume; single well-known maintainer + contributors. | **Native MSVC** ✅. No `nvcc` at build time. | CUDA C `.cu` source as Rust string/`include_str!`, compiled by NVRTC at runtime to PTX; loaded via driver API. Full access to shared mem, warp intrinsics, `atomicAdd`. | **Best.** You write exactly the privatized-histogram + scan + scatter kernels the workload demands, in the language they're documented in. | **RECOMMENDED (MVP core)** |
| **CubeCL** (`tracel-ai`) ⚠️ | **Alpha**, v0.10.0 (May 2026). Production-used by Burn, actively developed, but "expect breaking changes between minor versions." Standalone (no Burn dep). | Yes (CUDA + Vulkan + Metal). | `#[cube]` Rust functions, JIT to CUDA/HIP/Metal/WGSL. Supports `Atomic<u32>`/`atomic_add`, shared memory, `plane` (warp) intrinsics incl. exclusive scan, manual indexing. | **Good and improving** — has the primitives (atomics, plane scan) tree kernels need. Risk is API churn + thinner docs for advanced intrinsics. | **FALLBACK / portability bet (M2)** |
| **Rust-CUDA** (`rustc_codegen_nvvm` + `cust`) ❌ | Rebooted 2025 but **`cust` on crates.io frozen at 0.3.2 / Feb 2022**; modern use needs **git dependency**. Nightly-pinned codegen; LLVM 7.x; unpaid maintainers; README warns of bugs/safety issues. | Possible (prebuilt LLVM download + PATH hacks) but fragile. | Rust → NVVM IR → PTX via a custom rustc backend. | Workable in principle, but you're betting the core on an unstable, nightly + LLVM-7 + git-dependency toolchain. | **AVOID as primary** |
| **CUDA C via `nvcc` + `cc`/`bindgen` FFI** ❌ on Windows | `cc`/`bindgen_cuda` are mature, but **`cc-rs` compiles CUDA C++ only for GNU/Clang toolchains, not MSVC.** | **Broken on native MSVC**; effectively forces WSL2 or clang. | `.cu` compiled by `nvcc` at build time into a static lib, linked via `build.rs` + `bindgen`. | Same kernels as the recommended path, but AOT-compiled. | **AVOID on Windows** (NVRTC path gives the same kernels without the MSVC build break) |
| **wgpu / compute shaders** ❌ | Mature, but wrong tool. | Yes. | WGSL compute shaders. | **Poor.** WGSL lacks the mature 64-bit atomics / warp-shuffle / shared-memory ergonomics and CUDA-grade tooling (compute-sanitizer, Nsight) this atomic-heavy workload needs; also no CUDA-library/FIL ecosystem alignment. | **AVOID** (use CubeCL if you want portability) |
| **candle / Burn CUDA backend** ❌ | Mature host-side tensor libs. | Yes. | They are *tensor frameworks*, not custom-kernel authoring tools. Custom kernels still drop down to CubeCL (Burn) or raw CUDA (candle). | **N/A** — they don't remove the need to author the irregular kernels; they'd add a heavy dependency for tensor ops you don't have (no GEMM in the hot path). | **AVOID** (no GEMM hot path → no benefit) |

### Why NVRTC over AOT `nvcc`, concretely
1. **Eliminates the Windows MSVC build break** (`cc-rs` can't compile CUDA C for MSVC). NVRTC ships in the CUDA toolkit and compiles `.cu` strings to PTX at runtime regardless of host compiler.
2. **No `nvcc` invocation in `build.rs`** → simpler, more portable wheel builds via maturin.
3. **Runtime specialization**: you can template kernels on bin count / feature tiling / dtype and compile the exact variant — useful for the crossover-surface autotuning.
4. **Cost**: first-call JIT latency. Mitigate by caching compiled PTX/cubin (cudarc supports loading precompiled PTX) and compiling at model-construction time.

### Why CubeCL is the deliberate fallback, not the MVP
It directly serves the PROJECT.md note that "vendor-neutral backends [are] deferred" — one `#[cube]` source could later target AMD/Metal. But betting the *first* crossover-validation milestone on alpha tooling with inter-minor breaking changes adds schedule risk to the exact thing (the benchmark crossover) that gates the whole project. Recommendation: **build the kernels in CUDA C now; keep the kernel layer behind a Rust trait so a CubeCL backend can be added in Milestone 2 without disturbing the API.**

---

## Python Binding, Interop & Determinism Prescriptions

| Concern | Prescription | Confidence |
|---------|-------------|------------|
| **Binding layer** | PyO3 0.29 with `extension-module` + `abi3-py310` (or `abi3-py39`) features → one stable-ABI wheel per platform. | HIGH |
| **Build/packaging** | maturin 1.14.1, `pyproject.toml` with `[tool.maturin]`. Ship Windows `x86_64` wheel; the CUDA toolkit is a runtime prerequisite on the user's machine (document it; prefer cudarc `dynamic-loading`). | HIGH |
| **CPU reference** | Pure-Rust `ndarray` + `rayon` backend in the same crate, selected by `device="cpu"`. Serves as correctness oracle AND honest small-`n` path. Bit-share the Philox stream with the GPU path so differential tests are exact, not approximate. | HIGH |
| **Host (CPU) zero-copy** | `rust-numpy` (`numpy` crate, version-locked to PyO3 0.29) — borrow numpy arrays as `ndarray` views without copying. | HIGH |
| **Device (GPU) zero-copy** | Expose results via **`__cuda_array_interface__`** (dict protocol) **and** a **DLPack** capsule so CuPy / PyTorch / Numba / cuDF can consume Sylva GPU buffers without a host round-trip. Accept inputs the same way. | HIGH |
| **RNG** | **Counter-based Philox-4×32-10** reimplemented in the Rust core and inlined in the CUDA kernel from the same `(key=seed, counter=(tree, node, feature, draw))` scheme. Stateless → reproducible regardless of thread scheduling → satisfies `deterministic=True` bit-reproducibility. | HIGH |
| **Treelite export** | `serde_json`-emit a Treelite 4.x `import_from_json()`-compatible JSON (correct `task_param` / `model_param` / per-node schema) → FIL / Triton FIL backend serving. Validate against a Treelite round-trip in CI. | MEDIUM (schema details need verification against Treelite 4.x docs during the export phase) |
| **No silent fallback** | Device dispatch + `execution_report_` is application logic in the Rust core, not a stack choice — but note cudarc surfaces clean `Result`s for every CUDA call, which is exactly what the `fallback="error"` contract needs. | HIGH |

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| cudarc + CUDA C (NVRTC) | CubeCL | When you commit to multi-vendor (AMD/Metal/WGPU) portability and can absorb alpha API churn — target this for Milestone 2 behind a kernel trait. |
| cudarc + CUDA C (NVRTC) | Rust-CUDA (`cust`+nvvm) | Only if a future world ships `cust` properly on crates.io on stable Rust AND you want all-Rust kernels but reject CubeCL. Not foreseeable for MVP. |
| NVRTC (runtime compile) | `nvcc` AOT + `bindgen_cuda` | On **Linux/WSL2 only**, if you want AOT-compiled cubins and accept the clang/GNU toolchain. Pointless on native Windows. |
| Hand-rolled Philox | cuRAND host API (via cudarc) | If you only need GPU-side randomness and don't require bit-identical CPU↔GPU streams. The differential-test requirement argues against it. |
| PyO3 + maturin | cffi / ctypes over a C ABI | Never for this project — PyO3 is the user's stated stack and gives typed exceptions + abi3. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Rust-CUDA `cust` 0.3.2 (crates.io)** | Frozen Feb 2022; modern features only via **git dependency**, which **prevents publishing your crate to crates.io** and pins you to nightly Rust + LLVM 7.x. | cudarc 0.19.8 (driver API + NVRTC). |
| **`cc` crate to compile CUDA C on Windows** | `cc-rs` supports CUDA only on **GNU/Clang**, not MSVC — breaks the native Windows build. | NVRTC runtime compilation via cudarc (no host C compiler needed for kernels). |
| **`nvcc` AOT in `build.rs` on native Windows** | Forces a non-MSVC toolchain or WSL2; complicates maturin wheel builds. | NVRTC; keep `nvcc` for optional Linux/WSL profiling builds only. |
| **wgpu / WGSL compute shaders** | Immature 64-bit atomics / warp ops; no compute-sanitizer/Nsight; no FIL/Treelite alignment; this workload is atomic-heavy. | CUDA C (now) or CubeCL (portable later). |
| **candle / Burn as the engine** | Tensor frameworks with no benefit for a GEMM-free, irregular-kernel workload; heavy deps. | Author kernels directly; borrow only CubeCL (standalone) if you want its kernel DSL. |
| **cuda-oxide (NVlabs)** | Experimental Rust→PTX compiler; not production. | cudarc + CUDA C. |
| **Nightly Rust** | Needed only by `rustc_codegen_nvvm`; otherwise avoid for a shippable library. | Stable Rust 1.83+. |
| **cuDNN dependency** | No deep-learning ops in the hot path; adds a huge, version-fragile dependency. | cuBLAS/cuRAND via cudarc only if a specific kernel needs them (likely not for forests). |

## Stack Patterns by Variant

**If targeting native Windows 11 (the stated dev/bench host):**
- Use cudarc + NVRTC + MSVC + maturin. **No WSL required for the MVP build.**
- Install: CUDA Toolkit 12.x, VS 2022 Build Tools (MSVC v143), Rust stable, Python 3.10+, maturin.
- Avoid any `nvcc`-at-build-time path (the `cc`/MSVC incompatibility lives here).

**If you later add AOT cubins or hit an NVRTC limitation:**
- Switch *that build profile* to WSL2 (Ubuntu) where `nvcc` + `cc`/`bindgen_cuda` work cleanly with the GNU/Clang toolchain. Keep WSL2 as a *secondary profiling/AOT lane*, not the primary dev loop.

**If/when pursuing vendor-neutral backends (deferred per PROJECT.md):**
- Introduce a CubeCL backend behind the same kernel trait. Re-validate the crossover surface per backend.

## Windows-vs-WSL Build Implications (explicit)

- **Recommended MVP = native Windows.** The NVRTC path means the only Windows-specific requirement is MSVC for linking the Rust/PyO3 extension — which maturin + PyO3 handle as a first-class target. CUDA C kernels are compiled by NVRTC at runtime, so the broken `cc`+MSVC CUDA path is never exercised.
- **WSL2 is the fallback only if** you choose AOT `nvcc` compilation, need a Linux-only CUDA library, or want to mirror a Linux CI/serving target. WSL2 CUDA works (GPU passthrough via the Windows driver) but adds filesystem/perf-measurement friction for the benchmark crossover — keep benchmarks on the native host where the GPU is measured directly.
- **Determinism note:** the Philox + NVRTC approach is host-OS-independent, so deterministic-mode results should match across native Windows and WSL2 — useful for cross-checking.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| pyo3 0.29 | numpy (rust) ≈0.25 | **Hard lock** — rust-numpy must match the PyO3 minor it was built against. Verify the exact compatible `numpy` crate version when pinning. |
| pyo3 0.29 | Rust ≥1.83 | MSRV floor for the whole project. |
| pyo3 0.29 | maturin ≥1.x | abi3 + free-threaded supported. |
| cudarc 0.19.8 | CUDA 11.4–13.0 | Choose `cuda-12xxx` feature matching installed toolkit, or `dynamic-loading` to defer to runtime. |
| CUDA 13.0 | NVIDIA driver ≥580 | If you stay on CUDA 12.x, older drivers are fine — prefer 12.x for MVP breadth. |
| CubeCL 0.10 (fallback) | — | Alpha: expect breaking changes across minor versions; pin exactly and budget for upgrades. |

## Unverified / Flagged

- **rust-numpy exact version for PyO3 0.29** — stated as ≈0.25.x from ecosystem versioning convention; **verify the precise compatible release** against the rust-numpy changelog before pinning.
- **cudarc 0.19.8 exact feature-flag names** (e.g. `cuda-12060`, `nvrtc`, `dynamic-loading`) — confirm against the 0.19.8 docs.rs feature list; the *capabilities* (driver/NVRTC/cuBLAS/cuRAND, Windows MSVC, CUDA 11.4–13.0) are HIGH-confidence verified.
- **Treelite 4.x JSON schema field names** (`task_param`, `model_param`, node layout) — confirmed that a custom JSON import path exists (MEDIUM); the exact schema must be pinned against Treelite 4.x docs during the export phase.
- **CubeCL advanced intrinsic coverage** for every kernel you'll need — atomics, shared memory, and plane scan are confirmed present; thoroughly spike before adopting as a backend.
- **maturin 1.14.1 release date** not surfaced (version confirmed latest on PyPI).

## Sources

- crates.io API — cudarc 0.19.8 (2026-06-19), cust 0.3.2 (2022-02-16, stale), pyo3 0.29.0 (2026-06-11) — version verification (HIGH)
- pypi.org/pypi/maturin/json — maturin 1.14.1 latest (HIGH)
- github.com/Rust-GPU/Rust-CUDA + issue #280 + rust-gpu.github.io blog/guide — reboot status, CUDA 12/13, LLVM 7.x, Windows PATH, `cust` crates.io publishing gap (MEDIUM)
- docs.rs/crate/cudarc — driver/NVRTC/cuBLAS/cuRAND layering, CUDA 11.4–13.0, windows-msvc builds (HIGH)
- github.com/tracel-ai/cubecl + thomasantony.com CubeCL writeup + HN — v0.10.0 alpha, `#[cube]`, `Atomic<u32>`/atomic_add, plane intrinsics/exclusive sum, standalone (MEDIUM)
- PyO3 releases + nandann.com PyO3 0.28/0.29 guide — abi3 subclassing, CPython 3.7–3.14, free-threaded, MSRV 1.83 (HIGH)
- developer.nvidia.com (ML framework interop) + docs.cupy.dev interoperability — DLPack + `__cuda_array_interface__` as standard zero-copy protocols (HIGH)
- treelite.readthedocs.io 4.7 import tutorial + NVIDIA Triton FIL backend docs — JSON import + checkpoint serialization for FIL (MEDIUM)
- thesalmons.org Random123 paper + cuRAND docs + OpenRAND — Philox-4×32-10 as the standard reproducible counter-based parallel RNG (HIGH)
- docs.rs/cc + github.com/narsil/bindgen_cuda + cicoria.com CUDA-on-WSL2 — `cc-rs` CUDA = GNU/Clang only (MSVC gap), WSL2 implications (HIGH)

---
*Stack research for: GPU-native tree-ensemble library (Rust + CUDA + Python)*
*Researched: 2026-06-19*
