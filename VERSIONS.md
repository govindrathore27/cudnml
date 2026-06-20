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
| cudarc | **0.19.8** | launch-proof (`cuda-static`, the Cargo default): `["std", "driver", "nvrtc", "cuda-12080", "dynamic-linking"]` | See **Deviation** below — `dynamic-linking` is required; the plan's 3-feature list panics. Selected by the `default = ["cuda-static"]` feature. |
| cudarc | **0.19.8** | shipping wheel (Plan 03 / D-02, the `wheel` feature): `["std", "driver", "nvrtc", "cuda-12080", "dynamic-loading"]` | CUDA resolved at **runtime** via the driver — one wheel for any compatible CUDA. Built via `maturin build --release --no-default-features --features wheel`. `cuda-12080` (the version selector) is required in BOTH modes; only the link MODE (`dynamic-linking` vs `dynamic-loading`) differs. Both modes are **provably exercised** (Plan 02 launch proof = static; Plan 03 wheel = dynamic-loading). |
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

### Link-mode toggle (Plan 03) — both D-02 modes provably exercised

The two link modes are **mutually exclusive** (cudarc's `build.rs` panics unless
exactly one of `{dynamic-loading, dynamic-linking, static-linking,
fallback-dynamic-loading}` is active), so they cannot coexist in one dependency
line. Plan 03 implements a **Cargo feature toggle** in
`crates/sylva-cuda/Cargo.toml`:

- `default = ["cuda-static"]` → `cudarc/cuda-12080 + cudarc/dynamic-linking` —
  the Plan 01/02 launch proof. `cargo build` / `cargo test` / `cargo bench` use
  this and stay green (3/3 tests pass).
- `wheel` → `cudarc/cuda-12080 + cudarc/dynamic-loading` — the shipping abi3
  wheel. Built via `maturin build --release --no-default-features --features
  wheel`; maturin passes the flags straight through to cargo (no cargo-args form
  needed). Note `cuda-12080` is the CUDA *version* selector and is required in
  **both** modes — cudarc panics without a `cuda-1xxxx` feature even under
  `dynamic-loading`; only the link MODE differs.

Both modes were built and exercised this session: static (`cargo test` launch
proof, still green) and dynamic-loading (the wheel, imported + launched in a
clean venv — see "Wheel / abi3 runtime fields").

## Microbench numbers — *feasibility sanity check, no algorithm speed claim*

Phase 1 is a toolchain feasibility spike. Any kernel-launch / vector-op timing
recorded here is a **feasibility sanity check, no algorithm speed claim**. No
end-to-end speed comparison is made until Phase 5; the authoritative (n×d)
crossover is Phase 7 (per ROADMAP comparative-study fairness note).

**Fairness protocol (both paths identical):** WARMUP 50 discarded launches; MEDIAN
of 10 timed runs (not mean); CUDA-event device timing; FIXED 1e7-element f32
vector; bytes = 3·4·N (two reads + one write); GB/s = bytes / median_time;
per-launch overhead = a 1000-launch tight loop / 1000; **correctness asserted
(`max_abs_err(out, a+b) == 0`) BEFORE any timing**. Same GPU / driver / CUDA 12.8
for both. Harness: `crates/sylva-cuda/benches/microbench.rs` (Rust, cudarc+NVRTC)
and `scripts/cupy_baseline.py` (CuPy, Python 3.12 venv).

| Metric | Sylva (cudarc + NVRTC) | CuPy 14.1.1 baseline |
|--------|------------------------|----------------------|
| Per-launch overhead (µs/launch) | **4.85** | **7.98** |
| Throughput (GB/s, vector_add 1e7 f32) | **185.42** | **237.48** |
| Vector-op correctness | exact (max_abs_err = 0) | exact (max_abs_err = 0) |

**Per-launch overhead ratio (Sylva / CuPy) = 4.85 / 7.98 ≈ 0.61×** — Sylva's
launch overhead is **lower** than CuPy's, far inside the ≤ ~2–3× pass bar.
Throughput is ~78 % of CuPy's (185 vs 237 GB/s on an RTX 4060 Ti with ~288 GB/s
theoretical peak), i.e. the same order of magnitude — not pathologically slow.

> **Comparison verdict:** PASS (SC-6). The cudarc+NVRTC path launches and moves
> data at the same order of magnitude as CuPy, with lower per-launch overhead and
> an exact vector op. **Feasibility sanity check, no algorithm speed claim** — no
> end-to-end / estimator speed conclusion is drawn from these numbers.

## Wheel / abi3 runtime fields

| Field | Value |
|-------|-------|
| Wheel filename | **`sylva_cuda-0.1.0-cp310-abi3-win_amd64.whl`** (`target/wheels/`) |
| Wheel build command | `maturin build --release --no-default-features --features wheel` |
| cudarc features in wheel | `["std", "driver", "nvrtc", "cuda-12080", "dynamic-loading"]` (runtime CUDA resolution) |
| `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` needed? | **No** — pre-armed defensively (Pitfall 1) but the maturin abi3 build did **not** require it. maturin auto-generated the Windows import library without pinning a specific interpreter (`🔗 Found pyo3 bindings with abi3-py3.10 support`); no interpreter fallback to Py3.10–3.13 was needed. The env var was set in the build env but the build would have succeeded without it. |
| Wheel import in clean venv (TOOL-03) | **PASS.** Fresh `py -3.14 -m venv .venv-smoke` (Python **3.14.3**), `pip install` the wheel ONLY (no repo source on PYTHONPATH), ran `scripts/import_smoke.py`: `import sylva_cuda` resolved to `…\.venv-smoke\Lib\site-packages\sylva_cuda\__init__.py` (the installed wheel, not the source tree), `run_vector_add([1,2,3,4],[10,20,30,40]) = [11,22,33,44]`, printed `OK: run_vector_add correct`, exit 0. The cp310-abi3 wheel runs on Python 3.14.3 as designed; dynamic-loading resolved the CUDA driver at runtime. The 5-link skeleton chain (Rust → CUDA/NVRTC → Python/abi3) is **closed**. |
| CuPy baseline env (Python 3.12 venv) | **`.venv-cupy`**, Python **3.12.8**, `cupy-cuda12x` **14.1.1** (cp312 wheel; numpy 2.4.6, cuda-pathfinder 1.5.5). CuPy has no cp314 wheel (D-06 / Pitfall 2), so a separate Py3.12 venv hosts the baseline against the same RTX 4060 Ti. Raw-CUDA-C fallback was **not** needed — CuPy installed and ran cleanly. |

## compute-sanitizer outcomes (TOOL-02, recorded from Plan 01-02)

All four `compute-sanitizer` tools reported clean against the isolated histogram
launch (NVRTC-compiled with `-lineinfo` for source attribution). Verbatim trailing
summary line from each tool (exit 0; full evidence in `01-02-SUMMARY.md`):

| Tool | Verbatim summary line |
|------|-----------------------|
| memcheck  | `========= ERROR SUMMARY: 0 errors` |
| racecheck | `========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| synccheck | `========= ERROR SUMMARY: 0 errors` |
| initcheck | `========= ERROR SUMMARY: 0 errors` |

Sanitizer: `…\CUDA\v12.8\compute-sanitizer\compute-sanitizer.exe` (not on PATH;
invoked by full path). No hazard → no kernel fix-and-rerun iteration was needed.

## Kill-criteria result

**Kill-criteria result: `proceed`.**

**Rationale (one line):** All four toolchain requirements pass natively on
Windows/MSVC with no WSL — TOOL-01 (NVRTC `vector_add` launches on the RTX 4060 Ti,
`max_abs_err = 0` over 1e7 elems), TOOL-02 (`compute-sanitizer` clean across all
four tools), TOOL-03 (the `dynamic-loading` cp310-abi3 wheel imports and calls into
the Rust core from a clean Python 3.14.3 venv) — and the SC-6 microbench is well
inside the pass bar (per-launch overhead 0.61× CuPy, throughput 185 vs 237 GB/s,
vector op exact), so the cudarc + NVRTC + PyO3/maturin premise is proven and the
project continues to Phase 2.

**Decision-tree walk (D-05 WSL-Fallback Decision Tree):**
1. Kernel compiles via NVRTC + launches on GPU (TOOL-01)? **YES** (Plan 01-02).
2. compute-sanitizer clean on histogram (TOOL-02)? **YES** (all 4 tools, 0 errors).
3. abi3 wheel builds + imports in clean venv (TOOL-03)? **YES** (native Windows;
   no wheel/link wall → no WSL fallback needed).
4. Microbench within ~2–3× baseline + correct (SC-6)? **YES** (0.61× per-launch
   overhead, exact vector op).
→ **✅ PROCEED.** Not `WSL-fallback` (no native wheel/link step failed) and not
`stop` (every path sanitizes and launches natively).

**Supply-chain integrity:** `Cargo.lock` is committed at the repo root and pins
the full transitive Rust dependency graph (T-01-01 / V14). The CuPy baseline pin
(`cupy-cuda12x 14.1.1`, cleared as a false-positive SUS in 01-RESEARCH.md) is
recorded above; it runs only in the throwaway, isolated `.venv-cupy`.

Status: the entire walking-skeleton chain (Cargo workspace builds+links →
NVRTC compile+launch → sanitizer-clean histogram → abi3 wheel builds → wheel
imports + calls into the Rust core in a clean venv) is **complete end-to-end** on
native Windows/MSVC. Phase 1 / Gate 1 is decided: **proceed**.
