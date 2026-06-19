# Phase 1: Toolchain Spike (Gate 1) - Research

**Researched:** 2026-06-20
**Domain:** Rust↔CUDA↔Python toolchain bring-up on native Windows/MSVC (cudarc 0.19.8 + NVRTC + PyO3 0.29 + maturin abi3); kernel-launch microbenchmark
**Confidence:** HIGH (version pins + cudarc API + feature flags verified against crates.io/docs.rs/PyPI; CuPy cp314 gap verified; compute-sanitizer behavior CITED)

## Summary

Phase 1 is a **throwaway feasibility spike** that resolves PROJECT.md's single biggest technical risk: proving the entire kernel-authoring + packaging path (Rust → CUDA via cudarc+NVRTC → Python wheel via PyO3+maturin) works natively on Windows/MSVC with no WSL, and is debuggable (`compute-sanitizer`) and packageable (`abi3` wheel imports in a clean env). No Extra Trees / Random Forest / SHAP algorithm logic is built. The deliverable is a *decision* (proceed / WSL-fallback / stop) plus a durable `VERSIONS.md` artifact recording every pin (TOOL-04).

The two hardest version pins flagged as UNVERIFIED in CLAUDE.md/SUMMARY.md are now **resolved with HIGH confidence**: (1) **rust-numpy is `0.29.0`** (released 2026-06-13), version-locked to PyO3 0.29.0 — confirmed directly from rust-numpy's own `Cargo.toml` (`pyo3 = { version = "0.29.0" }`). The "≈0.25.x" figure in CLAUDE.md was stale and is superseded. (2) **CuPy has NO Python 3.14 (cp314) Windows wheel** — `cupy-cuda12x` 14.1.1 ships cp310–cp313 win_amd64 only. This confirms D-06's flag and forces the microbench-baseline fallback. All cudarc 0.19.8 feature-flag names (`cuda-12080`, `driver`, `nvrtc`, `dynamic-loading`, `static-linking`) and the NVRTC compile→load→launch API surface (`compile_ptx_with_opts` + `CompileOptions { arch: Some("sm_89") }` → `CudaContext::load_module` → `module.load_function` → `stream.launch_builder(&f).launch(cfg)`) are verified against docs.rs.

**Primary recommendation:** Pin the dependency set exactly as in the Standard Stack table below. Bring the toolchain up in this order — install Rust stable ≥1.83 / verify MSVC `cl.exe` / verify CUDA 12.8 `nvrtc` / install maturin — then prove the static-link launch path first (fastest signal), then the privatized-histogram kernel under `compute-sanitizer racecheck`, then the `dynamic-loading` `abi3` wheel in a clean venv, then the microbench. For the CuPy cp314 gap, use **a separate Python 3.12 venv just for the CuPy reference** (Option b) as the cleanest baseline; keep a raw-CUDA-C launch-overhead baseline (Option a) as the in-process fallback.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CUDA-C kernel source (vector-add, histogram) | L1 CUDA Kernels (`.cu` strings) | — | Authored as CUDA C, compiled by NVRTC at runtime; no `nvcc`-at-build |
| NVRTC compile + module load + launch | L2 Backend / cudarc host code | L1 | cudarc driver+nvrtc API owns PTX compile, module load, launch config |
| Device buffer alloc / H2D / D2H | L0 GPU Memory / cudarc | L2 | `stream.alloc`, `memcpy_stod`/`memcpy_dtov` — spike-scale only |
| Rust↔Python FFI (`#[pyfunction]`/`#[pymodule]`) | L4 PyO3 binding seam | — | Thin: expose one "run the spike kernel" entrypoint, no algorithm logic |
| Wheel build + packaging (abi3, dynamic-loading) | Build toolchain (maturin) | L4 | maturin produces the `cp310-abi3-win_amd64` wheel |
| Microbench timing harness | Validation / host orchestration | L2 | Warmup + CUDA-event timing for launch overhead + GB/s |
| Version pinning artifact (VERSIONS.md) | Documentation / TOOL-04 | — | Durable record + kill-decision; the actual phase deliverable |

**Note on scope:** This is a *spike*. The "tiers" above are the same conceptual layers the real product uses (per ARCHITECTURE.md L0–L5), but the spike implements only the thinnest vertical slice through them — deliberately, so the toolchain risk is proven once against the real layering shape (D-04: persist the skeleton).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rust (stable) | ≥1.83 (current stable ~1.95/1.96) | Core + host orchestration; build the extension | `[VERIFIED: crates.io]` PyO3 0.29 MSRV floor is 1.83; current stable far exceeds it. Use **stable**, never nightly (avoids rust-cuda nightly-pin). |
| cudarc | 0.19.8 | CUDA driver API + NVRTC runtime kernel compile + module load + launch | `[VERIFIED: crates.io]` latest 0.19.8 (2026-06-19); 173k weekly downloads; supports CUDA 11.04–13.03; native windows-msvc |
| PyO3 | 0.29.0 | Rust↔Python FFI for the spike entrypoint | `[VERIFIED: crates.io]` latest 0.29.0 (2026-06-11); 3.37M weekly downloads; `abi3-py310` → one stable-ABI wheel |
| rust-numpy (`numpy` crate) | **0.29.0** | (Spike: optional) zero-copy host numpy↔ndarray | `[VERIFIED: crates.io + rust-numpy Cargo.toml]` 0.29.0 (2026-06-13) pins `pyo3 = "0.29.0"` exactly. **Resolves the CLAUDE.md "≈0.25.x UNVERIFIED" flag.** |
| maturin | 1.14.1 | Build + package the Rust extension as a pip wheel | `[VERIFIED: PyPI]` latest 1.14.1; canonical PyO3 build backend; native Windows abi3 wheels |
| CUDA Toolkit (NVRTC) | 12.8 (installed; nvcc 12.8.93) | NVRTC headers/lib for runtime kernel compile + `compute-sanitizer` | `[VERIFIED: local env]` cudarc feature `cuda-12080` matches it exactly |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ndarray | 0.16.x | Host array type behind rust-numpy | Only if the spike marshals a numpy array; otherwise omit `[ASSUMED]` (pin during use) |
| CuPy (`cupy-cuda12x`) | 14.1.1 | Microbench reference baseline | `[VERIFIED: PyPI]` **cp310–cp313 only — NO cp314 wheel.** Install in a separate Py3.12 venv (see Microbench Baseline) |
| VS 2022 Build Tools (MSVC v143) | — | Linker for the Rust/PyO3 extension on `x86_64-pc-windows-msvc` | `cl.exe` must be on PATH. **Not** used to compile kernels (NVRTC does that). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cudarc + CUDA C (NVRTC) | Rust-CUDA (`cust`+`rustc_codegen_nvvm`) | `[CITED: .claude/CLAUDE.md]` Rejected: `cust` frozen on crates.io (0.3.2/2022), nightly-pinned, LLVM 7.x, fragile Windows. AVOID as primary. |
| NVRTC (runtime compile) | `nvcc` AOT + `cc`/`bindgen_cuda` in build.rs | `[CITED: .claude/CLAUDE.md]` `cc-rs` compiles CUDA only for GNU/Clang, **breaks on native MSVC**. WSL-only. NVRTC sidesteps it entirely. |
| static `cuda-12080` | `dynamic-loading` only | D-02 says prove BOTH: static for the launch proof, dynamic for the shipping wheel. Not either/or. |
| CuPy baseline (Py3.12 venv) | raw CUDA-C baseline kernel | Both are valid (D-06). Raw-CUDA-C is in-process (no version hassle) but more code; CuPy venv is the standard, easy reference. Recommend CuPy venv primary, raw-CUDA-C fallback. |

**Installation:**

```toml
# Cargo.toml — sylva-cuda spike crate (dependencies)
# default-features = false is REQUIRED to drop cublas/cublaslt/curand/runtime (D-03: not needed in Phase 1)
[dependencies]
cudarc = { version = "0.19.8", default-features = false, features = ["driver", "nvrtc", "cuda-12080"] }
pyo3   = { version = "0.29.0", features = ["extension-module", "abi3-py310"] }
# numpy  = "0.29.0"   # only if the spike marshals a numpy array (optional for the launch proof)

# For the SHIPPING wheel build (D-02), swap the cudarc features to dynamic-loading:
# cudarc = { version = "0.19.8", default-features = false, features = ["driver", "nvrtc", "dynamic-loading"] }
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.14,<2.0"]
build-backend = "maturin"

[tool.maturin]
# abi3 wheel: one cp310-abi3-win_amd64 wheel works on any CPython >= 3.10 (incl. 3.14)
# (PyO3 0.27+/maturin 1.9.4+ sets PYO3_BUILD_EXTENSION_MODULE automatically)
```

```bash
# Build commands
maturin develop                  # inner loop: build + install into current venv
maturin build --release          # produce the distributable cp310-abi3-win_amd64 wheel
```

**Version verification:** All versions above were confirmed during research (2026-06-20) against crates.io, docs.rs, and PyPI. The planner should still add a first-task re-verify (`cargo add` resolves exact patch; `pip index versions maturin`) since these registries move.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| cudarc | crates.io | since 2022-09 | 173k/wk | github.com/.../cudarc (Coreylowman) | OK | Approved |
| pyo3 | crates.io | since 2017-07 | 3.38M/wk | github.com/pyo3/pyo3 | OK | Approved |
| numpy (rust-numpy) | crates.io | since 2017-05 | 565k/wk | github.com/PyO3/rust-numpy | OK | Approved |
| ndarray | crates.io | since 2015-12 | 1.53M/wk | github.com/rust-ndarray/ndarray | OK | Approved |
| rayon | crates.io | since 2015 | (high) | github.com/rayon-rs/rayon | OK | Approved |
| maturin | PyPI | est. since 2018 | n/a (API gap) | github.com/pyo3/maturin | SUS → cleared | Approved — see note |
| cupy-cuda12x | PyPI | est. since 2018 | n/a (API gap) | cupy.dev | SUS → cleared | Approved — see note |

**Note on the two PyPI `SUS` verdicts:** Both flags are `too-new` + `unknown-downloads`. These are **false positives** from the legitimacy seam reading only the *latest release date* (maturin 1.14.1 published 2026-06-19; cupy-cuda12x 14.1.1 published 2026-06-01) and a registry download-count API gap — not genuine newness. maturin is the canonical PyO3 build backend (github.com/pyo3/maturin, the same org as PyO3) and cupy-cuda12x is the official CuPy CUDA-12 wheel (cupy.dev). Both were independently verified against their official sources in this session. No `checkpoint:human-verify` needed; treat as Approved.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** maturin, cupy-cuda12x — both cleared as false positives above.

## Architecture Patterns

### System Architecture Diagram

```
                       PHASE 1 SPIKE — vertical slice through the real layering
                       ────────────────────────────────────────────────────────
  [clean Python venv]                                     [Microbench harness (Python or Rust)]
        │  import sylva_spike                                       │  loop N launches, CUDA-event timing
        ▼                                                           ▼
  ┌─────────────────────┐   PyO3 #[pymodule]   ┌──────────────────────────────────────┐
  │ L4  abi3 .pyd        │◄────────────────────│  one #[pyfunction] entrypoint:        │
  │ (cp310-abi3-win_amd64)│                     │  run_vector_add() / run_histogram()  │
  └─────────┬───────────┘                       └───────────────┬──────────────────────┘
            │ (dynamic-loading wheel resolves CUDA driver at runtime)                │
            ▼                                                                        ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ L2/L1  cudarc host code                                                            │
  │   CUDA_C_SRC: &str  ──[nvrtc::compile_ptx_with_opts(arch="sm_89")]──► Ptx          │
  │   CudaContext::new(0) ──load_module(ptx)──► CudaModule                             │
  │   module.load_function("vector_add" / "histogram_privatized") ──► CudaFunction     │
  │   stream.alloc + memcpy_stod ──► CudaSlice<f32>                                    │
  │   unsafe { stream.launch_builder(&f).arg(&a).arg(&out).launch(cfg) }               │
  │   memcpy_dtov ──► host Vec<f32>  (verify correctness)                              │
  └──────────────────────────────────────────────────────────────────────────────────┘
            ▼                                              ▲
  ┌─────────────────────┐                    [compute-sanitizer --tool racecheck/memcheck/
  │ L0  RTX 4060 Ti      │                     synccheck/initcheck wraps the host process]
  │  sm_89, 8 GiB        │
  └─────────────────────┘
```

Two compile-mode proofs, per D-02:
- **Launch proof** — Cargo features `driver,nvrtc,cuda-12080` (static link against installed CUDA 12.8). Fastest "is the toolchain alive" signal.
- **Wheel proof** — Cargo features `driver,nvrtc,dynamic-loading` (CUDA driver resolved at runtime). This is the shipping config; validates Pitfall-10 packaging.

### Recommended Project Structure

Per D-04 (persist the skeleton) and ARCHITECTURE.md's target layout — lay down the real workspace now, delete only the throwaway kernel/spike logic after the gate:

```
sylva/                          # repo root (cargo workspace)
├── Cargo.toml                  # [workspace] members; pinned versions
├── pyproject.toml              # [tool.maturin], abi3
├── VERSIONS.md                 # TOOL-04 durable artifact (the deliverable)
├── crates/
│   └── sylva-cuda/             # spike lives here; becomes CudaBackend in Phase 4+
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs          # PyO3 #[pymodule] + spike entrypoints (THROWAWAY logic)
│       │   ├── nvrtc_launch.rs # compile→load→launch host code (THROWAWAY)
│       │   └── kernels.rs      # CUDA_C_SRC strings: vector_add, histogram_privatized (THROWAWAY)
│       └── benches/
│           └── microbench.rs   # launch-overhead + GB/s harness (THROWAWAY)
└── python/
    └── sylva_spike/            # thin; or rely on maturin's module output
```

> **Planner note (D-04 is Claude-discretion):** if the persisted layout proves spike-specific, the planner may instead use a flat `spike/` scratch dir. The skeleton is the *default* because it matches the user's "structure early, many small files" convention and avoids re-scaffolding in Phase 2.

### Pattern 1: NVRTC compile → module load → launch (cudarc 0.19.x)

**What:** Author the kernel as a CUDA-C string, compile to PTX at runtime with `compile_ptx_with_opts` setting `arch = Some("sm_89")`, load the module, get the function, launch with a `LaunchConfig`.
**When to use:** Every kernel in the spike (and the whole project's GPU path).
**Example:**
```rust
// Source: docs.rs/cudarc/0.19.8 (nvrtc::safe, driver) [VERIFIED: docs.rs]
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const SRC: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"#;

let opts = CompileOptions { arch: Some("sm_89"), ..Default::default() };
let ptx  = compile_ptx_with_opts(SRC, opts)?;          // CUDA-C string -> PTX (NVRTC)

let ctx    = CudaContext::new(0)?;                     // device 0 = RTX 4060 Ti
let module = ctx.load_module(ptx)?;                    // Arc<CudaModule>
let f      = module.load_function("vector_add")?;      // CudaFunction
let stream = ctx.default_stream();

let a = stream.memcpy_stod(&host_a)?;                  // H2D
let b = stream.memcpy_stod(&host_b)?;
let mut out = stream.alloc_zeros::<f32>(n)?;
let cfg = LaunchConfig::for_num_elems(n as u32);
let mut builder = stream.launch_builder(&f);
builder.arg(&a).arg(&b).arg(&mut out).arg(&n);
unsafe { builder.launch(cfg)?; }                       // launching is unsafe
let result = stream.memcpy_dtov(&out)?;                // D2H, verify
```
> **API confidence:** the call *sequence* (`compile_ptx_with_opts` → `CudaContext::new` → `load_module` → `load_function` → `launch_builder(...).launch`) is `[VERIFIED: docs.rs]`. The exact builder arg-chaining syntax (`.arg()` ordering, whether `&n` vs `&(n as i32)`) is `[ASSUMED]` from the 0.19.x docs idiom — the planner should mark "confirm exact `launch_builder` signature against `cargo doc` at execution."

### Pattern 2: Representative privatized-histogram kernel (mirrors Phase-4)

**What:** A throwaway shared-memory privatized histogram using `atomicAdd`, shaped to mirror the real Phase-4 hot path so `racecheck` validates something meaningful (D-01 + CONTEXT "lean representative").
**Layout (from ARCHITECTURE.md — privatized shared-mem + two-level reduction):**
- **Bins:** 256 bins (uint8 bin domain; PITFALLS.md Pitfall 4 recommends 128–256 to fit shared-mem budget). For the spike, **one feature, 256 bins** is sufficient to exercise the primitive.
- **Shared memory:** one private `__shared__ unsigned int hist[256]` per block (1 KiB) — each block accumulates its own histogram via `atomicAdd` into shared memory, then one reduced `atomicAdd` merge to global. This is the privatization that cuts global-atomic contention (Pitfall 4) and the exact thing `racecheck` must clear.
- **Block/grid:** block = 256 threads; grid = `ceil(n / 256)`; each thread reads one element, computes its bin, `atomicAdd(&hist[bin], 1)` in shared mem; after `__syncthreads()`, threads cooperatively flush `hist[]` to global with `atomicAdd`.
- **Why this shape:** it is the *minimum* faithful slice of `histogram.cu`'s "privatized shared-mem + two-level reduction" (ARCHITECTURE.md L1). It deliberately does **not** add sibling-subtraction, multi-feature, or the SoA BinnedMatrix — those are Phase-4, out of spike scope.

**Example:**
```cuda
// Source: ARCHITECTURE.md Pattern (privatized shared-mem histogram) [CITED: .planning/research/ARCHITECTURE.md]
extern "C" __global__ void histogram_privatized(const unsigned char* bins, unsigned int* global_hist, int n) {
    __shared__ unsigned int sh[256];
    int t = threadIdx.x;
    for (int b = t; b < 256; b += blockDim.x) sh[b] = 0;
    __syncthreads();                                   // initcheck: sh must be zeroed before use
    int i = blockIdx.x * blockDim.x + t;
    if (i < n) atomicAdd(&sh[bins[i]], 1u);            // shared-mem atomic (racecheck target)
    __syncthreads();                                   // synccheck: required before flush
    for (int b = t; b < 256; b += blockDim.x) atomicAdd(&global_hist[b], sh[b]);
}
```

### Anti-Patterns to Avoid

- **Compiling kernels with `nvcc` in build.rs on native Windows:** `[CITED: .claude/CLAUDE.md]` `cc-rs` MSVC gap breaks this. Use NVRTC runtime compile only.
- **Float `atomicAdd` in the histogram:** `[CITED: PITFALLS.md Pitfall 5]` non-associative → nondeterministic. The spike histogram uses *integer* counts (associative, safe) — keep it integer even in the spike so it models the deterministic path.
- **Reporting "data already on GPU" microbench numbers as a speed claim:** `[CITED: PITFALLS.md Pitfalls 1/13]` this phase makes **no algorithm speed claim**. The microbench is a feasibility sanity check only.
- **Enabling default cudarc features:** pulls in cublas/curand/runtime not needed in Phase 1 (D-03). Always `default-features = false`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CUDA-C → PTX compilation | A custom `nvcc` shell-out + temp-file dance | `cudarc::nvrtc::compile_ptx_with_opts` | In-process NVRTC, no build-time toolchain, MSVC-safe |
| Module load + kernel launch | Raw `cuModuleLoadData`/`cuLaunchKernel` FFI | `cudarc` driver API (`load_module`/`launch_builder`) | Safe Rust wrappers, Result errors (needed for no-silent-fallback) |
| Python wheel packaging | Hand-rolled setuptools + cffi | maturin + PyO3 `abi3` | Canonical, native Windows abi3, one wheel for all CPython ≥3.10 |
| Kernel correctness checking | Print-debugging GPU values | `compute-sanitizer` (racecheck/memcheck/synccheck/initcheck) | Mature CUDA tooling; the whole reason cudarc+CUDA-C was chosen over Rust→PTX |
| Microbench timing | `std::time::Instant` around launch | CUDA events (or CuPy's timer) + warmup | CPU-side timing misses async launch/exec overlap; events measure device time |

**Key insight:** The entire kernel-authoring decision (cudarc+NVRTC over Rust-CUDA/nvcc-AOT) exists precisely so the project rides the *mature* CUDA toolchain (NVRTC, compute-sanitizer, Nsight) instead of hand-rolling around an immature one. Honor that — every "don't hand-roll" here is a direct consequence.

## Common Pitfalls

### Pitfall 1: Python 3.14 abi3 forward-compatibility build failure
**What goes wrong:** Building a PyO3 extension against a Python 3.14 interpreter can fail with "the configured Python interpreter version (3.14) is newer than PyO3's maximum supported version" even for abi3 targets, if PyO3's known-max is below the interpreter.
**Why it happens:** `[CITED: github.com/PyO3/pyo3/issues/5505]` PyO3 gates on interpreter version; for abi3 builds where the interpreter (3.14.3) is newer than PyO3's tested ceiling, it errors unless told to proceed.
**How to avoid:** Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` in the build environment when building the abi3 wheel on Python 3.14. PyO3 0.29 lists `abi3-py314` and broad 3.14 support, so it may build cleanly — but **pre-arm the env var** so the wheel build doesn't fail mid-spike. Verify at execution.
**Warning signs:** maturin build aborts with an interpreter-version error referencing PyO3's max version.

### Pitfall 2: CuPy has no cp314 Windows wheel (microbench baseline blocked)
**What goes wrong:** `pip install cupy-cuda12x` in the project's Python 3.14.3 venv fails — no cp314 win_amd64 wheel exists (14.1.1 ships cp310–cp313 only).
**Why it happens:** `[VERIFIED: PyPI cupy-cuda12x/json]` CuPy hasn't published 3.14 Windows wheels yet.
**How to avoid:** Use **a dedicated Python 3.12 venv just for the CuPy reference** (Option b — cleanest). The CuPy baseline runs in its own process/venv against the same GPU/driver; the cudarc path runs in the 3.14 venv. Both hit the same RTX 4060 Ti, so the comparison is fair. **Fallback (Option a):** a raw-CUDA-C launch-overhead baseline compiled via the same NVRTC path, measured in-process — avoids the venv split but is more code and a weaker "independent reference."
**Warning signs:** `No matching distribution found for cupy-cuda12x` on the 3.14 venv.

### Pitfall 3: `default-features = true` pulls in cublas/curand
**What goes wrong:** cudarc's 8 default features include `cublas`, `cublaslt`, `curand`, `runtime` — none needed in Phase 1, and they add link/load surface (and may demand libraries not desired in the dynamic-loading wheel).
**How to avoid:** `[VERIFIED: docs.rs/crate/cudarc/0.19.8/features]` always `default-features = false`, then opt into exactly `["driver", "nvrtc", <link-mode>]`.

### Pitfall 4: compute-sanitizer "no source correlation"
**What goes wrong:** sanitizer runs but can't attribute errors to kernel source lines.
**Why it happens:** `[CITED: PITFALLS.md Pitfall 11]` typically a symptom of the Rust→PTX path (the rejected one). With NVRTC-compiled CUDA-C it should symbolicate; to improve line info, add `--generate-line-info` (`-lineinfo`) to the NVRTC `options` in `CompileOptions`.
**How to avoid:** compile the spike kernels with line-info in `CompileOptions.options`; run sanitizer against the wheel-importing Python process or a Rust test binary.

### Pitfall 5: MSVC `cl.exe` not on PATH
**What goes wrong:** Rust/PyO3 link step fails on `x86_64-pc-windows-msvc` because the linker can't find MSVC.
**Why it happens:** VS Build Tools installed but the shell isn't a "Developer" shell / PATH not set.
**How to avoid:** `[CITED: .claude/CLAUDE.md]` confirm `cl.exe` resolves (run from a VS Developer prompt or after `vcvars64.bat`). **This is the ONLY thing MSVC is needed for** — NVRTC compiles kernels with no host C compiler, so the `cc`/MSVC CUDA gap is never exercised.

## Code Examples

### Microbench timing (launch overhead + throughput) — fairness-encoded
```rust
// Fairness rules baked in (PITFALLS.md 1/2/13): warmup, repeated runs, same GPU/driver,
// device-event timing, NO algorithm speed claim. [CITED: ROADMAP Comparative Baseline Study]
// 1. Warmup: launch the kernel ~50 times discarded (JIT/first-launch costs excluded).
// 2. Launch-overhead metric: time a tight loop of N empty/near-empty launches, divide by N (µs/launch).
// 3. Throughput metric: vector_add on a FIXED 1e7-element f32 vector; bytes = 3*4*1e7 (2 reads+1 write);
//    GB/s = bytes / median_time. Report MEDIAN of >=10 timed runs, not mean.
// 4. Correctness: assert max_abs_err(out, a+b) == 0 (exact for add) before reporting any timing.
// 5. Same device/driver/CUDA for both paths; record them in VERSIONS.md alongside numbers.
// Pass bar: cudarc+NVRTC per-launch overhead <= ~2-3x the CuPy/raw baseline AND vector-op correct.
```

### compute-sanitizer invocation (Windows)
```bash
# Source: docs.nvidia.com/compute-sanitizer [CITED]
# Run against the test/bench binary or the python process that imports the wheel + launches the kernel.
compute-sanitizer --tool memcheck   --launch-timeout 0  <target.exe / python spike.py>
compute-sanitizer --tool racecheck                       <target>   # shared-mem hazards (histogram)
compute-sanitizer --tool synccheck                       <target>   # __syncthreads misuse
compute-sanitizer --tool initcheck                       <target>   # uninitialized device global reads
# "Clean" = trailing line "========= ERROR SUMMARY: 0 errors" for each tool.
# Add NVRTC option -lineinfo so errors carry source line attribution.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| rust-numpy ≈0.25.x (CLAUDE.md guess) | rust-numpy **0.29.0** (locks pyo3 0.29.0) | 2026-06-13 release | Pin resolved; no guesswork |
| `nvcc` AOT in build.rs | NVRTC runtime compile via cudarc | mature in 0.19.x | MSVC-native, no build-time CUDA toolchain |
| Rust→PTX (rust-cuda/cust) | cudarc + hand-written CUDA C | project decision | Rides mature CUDA tooling; stable Rust |
| Per-Python-version wheels | single `cp310-abi3` wheel | PyO3 abi3 | One wheel covers CPython ≥3.10 incl. 3.14 |

**Deprecated/outdated:**
- `cust` 0.3.2 on crates.io (frozen 2022) — do not use `[CITED: .claude/CLAUDE.md]`.
- The "≈0.25.x" rust-numpy figure in CLAUDE.md — superseded by verified 0.29.0.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Exact `launch_builder(...).arg().launch(cfg)` arg syntax / `LaunchConfig` shape in 0.19.8 | Pattern 1 | LOW — call sequence verified; only chaining syntax may differ. Confirm via `cargo doc` at execution. |
| A2 | PyO3 0.29 builds the abi3 wheel cleanly on Py3.14, with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` as the safety net | Pitfall 1 | MEDIUM — if the env var is also insufficient, build the wheel against a Py3.10–3.13 interpreter (abi3 wheel still runs on 3.14). |
| A3 | A separate Py3.12 venv is the cleanest CuPy baseline path | Microbench/Pitfall 2 | LOW — raw-CUDA-C fallback (Option a) is documented; either satisfies the microbench. |
| A4 | `-lineinfo` via `CompileOptions.options` gives sanitizer source attribution | Pitfall 4 | LOW — standard NVRTC/sanitizer behavior; verify output at execution. |
| A5 | Current Rust stable is ~1.95/1.96 (sources disagreed 1.95 vs 1.96) | Standard Stack | NONE for the gate — any stable ≥1.83 satisfies PyO3 0.29 MSRV. Verify exact at install. |
| A6 | ndarray 0.16.x is the version to pair with rust-numpy 0.29 | Supporting | LOW — only matters if the spike marshals numpy; pin against rust-numpy 0.29's re-export at use. |

**All version pins in the Standard Stack core table are VERIFIED, not assumed.** The assumptions above are execution-time confirmations, not unresolved decisions.

## Open Questions

1. **Does the abi3 wheel build cleanly on Python 3.14.3, or does it need `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`?**
   - What we know: PyO3 0.29 advertises `abi3-py314` and 3.14 support; issue #5505 shows older PyO3 needs the env var.
   - What's unclear: whether 0.29 specifically needs it against a 3.14.3 interpreter.
   - Recommendation: set the env var defensively in the build task; if still failing, build against a 3.10–3.13 interpreter (the abi3 wheel runs on 3.14 regardless). Plan a fallback task.

2. **Exact `launch_builder` argument-passing API in cudarc 0.19.8.**
   - What we know: the method exists; the launch flow is verified.
   - What's unclear: precise `.arg()` chaining + whether scalar `n` is passed by ref/typed.
   - Recommendation: first kernel task runs `cargo doc --open` on cudarc and pins the exact call site.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU (sm_89) | All GPU proof | ✓ | RTX 4060 Ti, cc 8.9, 8 GiB | — |
| CUDA Toolkit + NVRTC | NVRTC compile, sanitizer | ✓ | 12.8 (nvcc 12.8.93) | — |
| NVIDIA driver | dynamic-loading launch | ✓ | 595.79 | — |
| Python | wheel import target | ✓ | 3.14.3 | Py3.12 venv for CuPy |
| Rust stable | build everything | ✗ NOT INSTALLED | need ≥1.83 | — (blocking; first install task) |
| MSVC v143 (`cl.exe`) | link the extension | ⚠ verify | VS2022 Build Tools | — (blocking if absent) |
| maturin | wheel build | ✗ install | 1.14.1 | pip install maturin |
| CuPy (`cupy-cuda12x`) | microbench baseline | ⚠ no cp314 wheel | 14.1.1 (cp310–313) | Py3.12 venv OR raw-CUDA-C baseline |

**Missing dependencies with no fallback (blocking — must be first tasks):**
- **Rust stable ≥1.83** — not on PATH (CONTEXT specifics). Install via rustup; verify `rustc --version` and target `x86_64-pc-windows-msvc`.
- **MSVC v143 / `cl.exe`** — verify on PATH; install VS 2022 Build Tools if absent.

**Missing dependencies with fallback:**
- **CuPy cp314** — use a separate Python 3.12 venv for the CuPy reference (recommended), or a raw-CUDA-C baseline kernel.

## Validation Architecture

> Nyquist validation is enabled (`workflow.nyquist_validation: true`). This is a toolchain spike — validation is *toolchain-claim* validation (does the path build/launch/sanitize/package), not algorithm correctness.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `cargo test` / `cargo nextest` (Rust); `pytest` smoke test (wheel import); `compute-sanitizer` (GPU correctness) |
| Config file | none yet — Wave 0 lays `Cargo.toml [workspace]`, `pyproject.toml`, an `import` smoke test |
| Quick run command | `cargo test -p sylva-cuda` (launch proof + correctness assert) |
| Full suite command | `cargo test` + `compute-sanitizer --tool racecheck/memcheck <bin>` + `python -c "import sylva_spike"` in clean venv |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TOOL-01 | CUDA-C kernel compiles via NVRTC + launches on GPU (Windows/MSVC, no WSL) | integration | `cargo test -p sylva-cuda nvrtc_launch_vector_add` (asserts `out == a+b`) | ❌ Wave 0 |
| TOOL-02 | `compute-sanitizer` reports clean on the histogram kernel | gpu-correctness | `compute-sanitizer --tool racecheck <test-bin>` → "ERROR SUMMARY: 0 errors" | ❌ Wave 0 |
| TOOL-03 | abi3 wheel builds + imports in a clean Python env | packaging-smoke | `maturin build --release` then fresh venv `pip install <wheel> && python -c "import sylva_spike"` | ❌ Wave 0 |
| TOOL-04 | Pinned versions recorded + kill-decision documented | artifact | manual check: `VERSIONS.md` exists with all pins + proceed/WSL/stop verdict | ❌ Wave 0 |
| (SC-6) | Microbench: launch overhead ≤~2–3× baseline AND vector op correct | microbench | `cargo bench --bench microbench` + compare vs CuPy/raw baseline | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p sylva-cuda` (fast launch + correctness signal).
- **Per wave merge:** full `cargo test` + `compute-sanitizer` race/mem on the histogram + clean-venv wheel import.
- **Phase gate:** all four TOOL reqs green + microbench within pass bar + `VERSIONS.md` written with the kill-decision, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `Cargo.toml` workspace + `crates/sylva-cuda/Cargo.toml` with pinned deps — covers TOOL-01/03
- [ ] `pyproject.toml` with `[tool.maturin]` abi3 — covers TOOL-03
- [ ] `tests/` Rust integration test asserting `vector_add` correctness — covers TOOL-01
- [ ] `benches/microbench.rs` harness with warmup + event timing — covers SC-6
- [ ] clean-venv import smoke test script — covers TOOL-03
- [ ] `VERSIONS.md` template — covers TOOL-04
- [ ] Tool installs: rustup (Rust ≥1.83), `pip install maturin`, verify `cl.exe` + `nvrtc` — prerequisite tasks

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a local numerical-toolchain spike with no network surface, no auth, no user data, no persistence. Most ASVS categories are N/A. The relevant surface is supply-chain + memory-safety at the FFI/CUDA boundary.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | local library |
| V5 Input Validation | partial | Validate array length/dtype before any device launch (out-of-bounds GPU reads = silent corruption — PITFALLS Security table) |
| V6 Cryptography | no | none in spike (Philox is a later phase) |
| V14 Configuration / Supply Chain | yes | Pin exact versions (Cargo.lock + VERSIONS.md); legitimacy-audited deps; no unverified packages |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds GPU read from unvalidated array shape | Tampering / DoS | Validate `n`/shape/contiguity at the Rust↔Python boundary before launch; `if (i < n)` guard in kernel |
| Unsafe Rust at the CUDA FFI boundary | Tampering | Minimize `unsafe` to the launch call; wrap in checked cudarc Result handling; run `compute-sanitizer memcheck` |
| Supply-chain (slopsquat / compromised crate) | Spoofing | All deps legitimacy-audited (see audit table); pin exact versions; commit Cargo.lock |
| License contamination (GPL/Snap ML source copied into spike) | — | `[CITED: .claude/CLAUDE.md]` Apache-2.0 only; the spike kernels are trivial original code — never copy restrictively-licensed source |

## Project Constraints (from CLAUDE.md)

- **Apache-2.0 only** — never copy GPL/Snap ML source; reuse algorithms from papers, document provenance. (Spike kernels are trivial original code.)
- **Tech stack is fixed:** Rust core + Python API via PyO3 + maturin; NVIDIA CUDA only; native Windows (WSL only as documented fallback).
- **Kernel-authoring decision is locked:** cudarc 0.19.8 + NVRTC + hand-written CUDA C. Rust-CUDA/`cust`, `nvcc`-AOT-on-MSVC, wgpu, candle/Burn-as-engine, cuDNN, nightly Rust are all in "What NOT to Use."
- **`default-features = false` on cudarc** — no `cublas`/`curand` in Phase 1 (D-03).
- **No silent CPU/GPU fallback** — even in the spike, surface clean `Result`s; cudarc's per-call Results are the mechanism.
- **Coding style:** many small files (200–400 lines typical, 800 max); comprehensive error handling; no hardcoded values; Rust ownership idioms.
- **GSD workflow enforcement:** file edits go through a GSD command (this phase is planned via `/gsd-plan-phase`).

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (Spike kernel choice):** Implement BOTH (a) a trivial elementwise op (vector add / SAXPY on a fixed ~1e7-element float32 array) as the microbench-comparison kernel, AND (b) a separate small shared-memory privatized histogram kernel using `atomicAdd` to give `compute-sanitizer racecheck` something meaningful.
- **D-02 (Linking modes):** Prove BOTH — static `cuda-12080` for the launch proof, AND `dynamic-loading` for the `abi3` wheel (the actual shipping config).
- **D-03 (Environment, verified 2026-06-20):** RTX 4060 Ti / cc 8.9 → `sm_89`; CUDA 12.8 → `cuda-12080`; driver 595.79; Python 3.14.3 → `abi3-py310`; cudarc features `driver` + `nvrtc` (+ `dynamic-loading` for wheel, + static `cuda-12080` for launch proof); NO `cublas`/`curand`.

### Claude's Discretion
- **D-04 (Scaffolding):** Default = persist the project skeleton (real Cargo workspace + `sylva` crate + `pyproject.toml`/maturin + CI + pinned `Cargo.toml`); delete only throwaway kernel/spike logic after the gate. Planner may revert to a pure `spike/` scratch dir if the layout proves spike-specific.
- **D-05 (Kill-criteria & timebox):** Default = ~2 working-day timebox. Native NVRTC+MSVC fails only on wheel/link → try WSL fallback for that build profile; kernels won't compile/launch/sanitize anywhere within the box → full stop + reconsider stack (the KILL CRITERION). Planner may tighten/loosen.
- **D-06 (Microbench baseline):** Default = CuPy (`cupy-cuda12x`) reference for per-launch overhead (µs) + throughput (GB/s), pass bar ≤~2–3× baseline with correct vector op. ⚠ Researcher confirmed CuPy has **no cp314 wheel** → use a separate Python 3.11/3.12 venv for the CuPy reference (recommended), or a raw-CUDA-C baseline kernel. Feasibility sanity check, NOT a speed gate, NO algorithm speed claim.
- Histogram representativeness: lean **representative** — mirror the real Phase-4 privatized-per-block layout (256 bins, shared-mem private histogram, two-level atomic reduction) per ARCHITECTURE.md.

### Deferred Ideas (OUT OF SCOPE)
- None pulled into the spike. Explicitly later-phase: CubeCL portability backend, Philox RNG, DLPack/`__cuda_array_interface__` zero-copy interop, all estimator/algorithm work, `cublas`/`curand`, sibling-subtraction, multi-feature SoA BinnedMatrix.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TOOL-01 | Throwaway CUDA-C kernel builds via cudarc 0.19.8 + NVRTC and launches on GPU, native Windows/MSVC (no WSL) | Pattern 1 (verified compile→load→launch API); feature flags `driver,nvrtc,cuda-12080` verified; `arch="sm_89"` verified; MSVC-only-for-linking confirmed |
| TOOL-02 | `compute-sanitizer` runs against the spike kernel and reports clean | Pattern 2 (representative histogram); compute-sanitizer Windows support + tool list CITED; `-lineinfo` for source attribution; "ERROR SUMMARY: 0 errors" = clean |
| TOOL-03 | Minimal PyO3 + maturin `abi3` wheel builds + imports in clean Python env on Windows | Standard Stack (PyO3 0.29 + maturin 1.14.1 + `abi3-py310` + `dynamic-loading`); Pitfall 1 (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY` for Py3.14); clean-venv import smoke test |
| TOOL-04 | Pinned versions recorded (cudarc flags, rust-numpy↔PyO3, CUDA toolkit) + kill-decision (proceed/WSL/stop) | All pins resolved & verified (rust-numpy 0.29.0↔pyo3 0.29.0 is the key one); D-05 decision tree; `VERSIONS.md` artifact spec |

## WSL-Fallback Decision Tree (D-05 / KILL CRITERION)

```
Start: native Windows NVRTC + MSVC path, ~2 working-day timebox
  │
  ├─ Kernel compiles via NVRTC + launches on GPU (TOOL-01)?
  │     NO  ──► retry sm_89 arch / driver / CUDA path within box.
  │            Still NO anywhere (incl. WSL) ──► ❌ FULL STOP (KILL): reconsider stack.
  │     YES ─┐
  │          ▼
  ├─ compute-sanitizer clean on histogram (TOOL-02)?
  │     NO  ──► fix kernel (race/init/sync) — this is a kernel bug, not a toolchain kill.
  │            Toolchain itself can't sanitize anywhere ──► ❌ FULL STOP (KILL).
  │     YES ─┐
  │          ▼
  ├─ abi3 wheel builds + imports in clean venv (TOOL-03)?
  │     NO, fails ONLY on wheel/link step ──► 🟡 try WSL fallback for the BUILD PROFILE
  │            (keep kernels/launch native; mirror Linux build).  Document as WSL-fallback result.
  │     NO, fails everywhere incl. WSL ──► ❌ FULL STOP (KILL).
  │     YES ─┐
  │          ▼
  └─ Microbench within ~2–3× baseline + correct (SC-6)?
        NO (pathologically slow) ──► investigate; not an automatic kill, but record honestly.
        YES ──► ✅ PROCEED. Write VERSIONS.md verdict = "proceed".
```

**`VERSIONS.md` (TOOL-04 deliverable) must record:**
- Rust toolchain version + target triple (`x86_64-pc-windows-msvc`)
- cudarc `0.19.8` + exact feature set used (launch-proof: `driver,nvrtc,cuda-12080`; wheel: `driver,nvrtc,dynamic-loading`)
- PyO3 `0.29.0` (`extension-module`, `abi3-py310`) + rust-numpy `0.29.0` (if used)
- maturin `1.14.1`
- CUDA toolkit `12.8` (nvcc 12.8.93), driver `595.79`, GPU `RTX 4060 Ti / sm_89`
- Python `3.14.3`; whether `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` was needed
- CuPy baseline env (Py3.12 venv + `cupy-cuda12x 14.1.1`) OR raw-CUDA-C baseline note
- Microbench numbers (µs/launch, GB/s, both paths) — **labeled "feasibility sanity check, no algorithm speed claim"**
- **The kill-decision: proceed / WSL-fallback / stop**, with one-line rationale
- Committed `Cargo.lock` reference

## Sources

### Primary (HIGH confidence)
- crates.io API — cudarc 0.19.8 (2026-06-19, 173k/wk), pyo3 0.29.0 (2026-06-11), numpy/rust-numpy 0.29.0 (2026-06-13), ndarray 0.16, rayon — version + legitimacy verification
- github.com/PyO3/rust-numpy `Cargo.toml` — `pyo3 = { version = "0.29.0" }` → **rust-numpy 0.29.0 ↔ PyO3 0.29.0 lock confirmed**
- docs.rs/crate/cudarc/0.19.8/features — feature flags `cuda-12080`, `driver`, `nvrtc`, `dynamic-loading`, `static-linking`, default set
- docs.rs/cudarc/0.19.8 (nvrtc::safe, driver) — `compile_ptx_with_opts`, `CompileOptions.arch`, `CudaContext::new`/`load_module`/`load_function`, `launch_builder`/`LaunchConfig`
- pypi.org/pypi/cupy-cuda12x/json — 14.1.1 ships cp310–cp313 win_amd64, **no cp314** (microbench baseline gap confirmed)
- pypi.org/pypi/maturin/json — 1.14.1 latest

### Secondary (MEDIUM confidence)
- github.com/PyO3/pyo3 issue #5505 — Python 3.14 abi3 build requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` on older PyO3
- pyo3.rs/v0.29.0 — abi3-py310/py314, Python 3.14 support
- docs.nvidia.com/compute-sanitizer + NERSC/CSC docs — memcheck/racecheck/synccheck/initcheck, Windows support, "ERROR SUMMARY: 0 errors"
- .claude/CLAUDE.md Technology Stack — kernel-authoring decision matrix, What NOT to Use, MSVC-for-linking-only, version compatibility table
- .planning/research/ARCHITECTURE.md — privatized shared-mem histogram layout (Pattern 2 representativeness)
- .planning/research/PITFALLS.md — Pitfalls 1/2/4/5/9/10/11/13 (microbench fairness, packaging, sanitizer, atomics)

### Tertiary (LOW confidence / verify at execution)
- Exact `launch_builder` arg-chaining syntax in 0.19.8 (call sequence verified; chaining idiom assumed)
- Current Rust stable exact value (1.95 vs 1.96 across sources; non-blocking, any ≥1.83 works)

## Metadata

**Confidence breakdown:**
- Standard stack / version pins: **HIGH** — every core pin verified against crates.io/docs.rs/PyPI this session; the two CLAUDE.md-flagged unknowns (rust-numpy version, CuPy cp314) both resolved.
- cudarc NVRTC API: **HIGH** for the call sequence + feature flags + `arch`; MEDIUM for exact builder syntax (verify via `cargo doc`).
- Architecture / histogram representativeness: **HIGH** — mirrors ARCHITECTURE.md's already-pinned Phase-4 layout.
- Packaging (abi3 on Py3.14): **MEDIUM** — env-var workaround documented; fallback (build on 3.10–3.13 interpreter) available.
- Pitfalls / fairness rules: **HIGH** — sourced from PITFALLS.md + ROADMAP Comparative Baseline Study.

**Research date:** 2026-06-20
**Valid until:** 2026-07-20 (30 days; fast-moving crates — re-verify cudarc/maturin patch versions at execution, but the API surface and major pins are stable)
