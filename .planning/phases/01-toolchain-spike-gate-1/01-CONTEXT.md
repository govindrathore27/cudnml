# Phase 1: Toolchain Spike (Gate 1) - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the **entire kernel-authoring + packaging path works natively on Windows** before any algorithm is built — resolving PROJECT.md's single biggest technical risk. Concretely (TOOL-01..04):

1. A hand-written CUDA C kernel compiles via **cudarc 0.19.8 + NVRTC** and launches on the local NVIDIA GPU, natively on **Windows/MSVC, no WSL**.
2. `compute-sanitizer` runs against the spike kernel(s) and reports clean (toolchain is debuggable).
3. A minimal **PyO3 0.29 + maturin `abi3`** wheel builds and imports in a clean Python environment on Windows.
4. Pinned, verified versions are recorded (cudarc feature flags, rust-numpy↔PyO3, CUDA toolkit) with a documented **kill-criteria result: proceed / WSL-fallback / stop**.

This is a **throwaway feasibility spike + microbench**, NOT algorithm work. No Extra Trees / Random Forest / SHAP logic is built here. The comparative study is a kernel-launch/vector-op **MICROBENCHMARK only — explicitly no algorithm speed claim**.

</domain>

<decisions>
## Implementation Decisions

### Spike kernel choice (USER-DECIDED)
- **D-01:** **Layered approach** — implement BOTH (a) a trivial elementwise op (vector add / SAXPY on a fixed ~1e7-element float32 array) as the microbench baseline-comparison kernel, AND (b) a separate small **shared-memory privatized histogram kernel using `atomicAdd`** to genuinely exercise shared memory + atomics so that `compute-sanitizer racecheck` has something meaningful to validate. Rationale: the toolchain risk and the real Phase-4 hot-path primitive risk are the same risk — prove both once, while keeping the "is the toolchain alive" signal cleanly separated from the "are the hard primitives debuggable" signal.

### Version pinning / cudarc linking (USER-DECIDED + environment-detected)
- **D-02:** **Prove both linking modes.** Run the fast kernel-launch proof with the **static `cuda-12080`** feature (link against the installed CUDA 12.8 toolkit), AND build the `abi3` wheel with the **`dynamic-loading`** feature so the actual shipping configuration (CUDA resolved at runtime via the driver, one wheel for any compatible CUDA) is validated in Phase 1 rather than deferred. PROJECT.md's recommended shipping mode is dynamic-loading; this decision proves it now instead of trusting it.
- **D-03 (environment-pinned, verified 2026-06-20):**
  - GPU: **NVIDIA GeForce RTX 4060 Ti**, **compute capability 8.9** (Ada), 8 GiB VRAM → NVRTC arch target **`sm_89` / `compute_89`**.
  - CUDA toolkit: **12.8** installed (nvcc 12.8.93) → cudarc feature **`cuda-12080`**.
  - Driver: **595.79** (supports CUDA 13 too; no reason to leave the installed 12.8 for the MVP — matches PROJECT.md "12.x for MVP breadth").
  - Python: **3.14.3** → **`abi3-py310`** floor covers it.
  - cudarc features needed in Phase 1: **`driver` + `nvrtc`** (+ `dynamic-loading` for the wheel; + static `cuda-12080` for the launch proof). **No `cublas` / `curand`** — not needed until later (Philox is hand-rolled; no GEMM in the hot path).

### Scaffolding fate (CLAUDE DISCRETION — planner may override)
- **D-04:** **Persist the project skeleton.** Lay down the real Cargo workspace + `sylva` crate + `pyproject.toml`/maturin layout + CI config + pinned `Cargo.toml` now; delete only the throwaway *kernel/spike logic* after the gate decision, keeping the harness/structure for Phase 2. Chosen as the default because it fits the user's coding-style convention ("many small files, structure early") and avoids re-scaffolding. User skipped this question — planner is free to revert to a pure `spike/` scratch dir if the layout proves spike-specific.

### Kill-criteria & timebox (CLAUDE DISCRETION — planner may override)
- **D-05:** **~2 working-day timebox.** Decision logic:
  - Native NVRTC+MSVC path fails **only on the wheel/link step** → try the documented **WSL fallback** for that build profile.
  - Kernels won't compile / launch / sanitize cleanly **anywhere** within the box → **full stop + reconsider the stack** (this is the TOOL-04 / Success-Criterion-5 KILL CRITERION).
  - Aligns with the user's session-planning guideline ("timebox any debugging detour; do not sink the session into one fix"). User skipped this question — planner may tighten/loosen the box.

### Microbench baseline (CLAUDE DISCRETION — researcher must verify feasibility)
- **D-06:** Prefer **CuPy** (`cupy-cuda12x`) as the microbench reference (easy, pip-installable, standard "is the toolchain alive" baseline) for per-launch overhead (µs) and elementwise throughput (GB/s), pass bar ≤ ~2–3× the baseline per-launch overhead with a correct vector-op result. **⚠ FLAG for researcher:** CuPy may not yet ship wheels for **Python 3.14** — verify wheel availability; if unavailable, fall back to (a) a **raw CUDA-C** baseline kernel, or (b) a separate Python 3.11/3.12 venv just for the CuPy reference. This is a feasibility sanity check, **not a speed gate**, and carries **no algorithm speed claim**.

### Claude's Discretion
- Histogram-kernel representativeness (match the real Phase-4 bin layout vs a generic shared-mem atomicAdd kernel) — user skipped; **lean representative** (match the real privatized-per-block layout where Phase-4 design is already settled in PROJECT.md/ARCHITECTURE.md) since it maximizes de-risking, but the planner/researcher may choose a generic kernel if the Phase-4 layout isn't pinned enough yet.
- D-04, D-05, D-06 defaults above are all Claude's discretion (user delegated).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Stack & kernel-authoring decision (the core risk this phase resolves)
- `.claude/CLAUDE.md` — full Technology Stack section: cudarc 0.19.8 + NVRTC + hand-written CUDA C decision, the kernel-authoring decision matrix, "What NOT to Use," Windows-vs-WSL build implications, version compatibility table.
- `.planning/research/STACK.md` — recommended stack and alternatives considered.
- `.planning/research/ARCHITECTURE.md` — technical architecture incl. the real Phase-4 privatized-histogram design the spike histogram kernel should mirror.
- `.planning/research/PITFALLS.md` — pitfalls incl. the binding comparative-study fairness rules (Pitfalls 1, 2, 13) governing the microbench.
- `.planning/research/SUMMARY.md` — research synthesis; fairness/honesty calibration.
- `.planning/research/FEATURES.md` — feature/requirement landscape.

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 1: Toolchain Spike (Gate 1)" — goal, 6 success criteria, Comparative Baseline Study (microbench spec, pass bar), KILL CRITERION; plus the binding comparative-study fairness note in the Overview.
- `.planning/REQUIREMENTS.md` — TOOL-01..04 (the 4 requirements this phase must satisfy).
- `.planning/PROJECT.md` — project constraints (Windows 11, NVIDIA CUDA only, Apache-2.0, no silent fallback).
- `.planning/STATE.md` — current blockers incl. the comparative-study fairness protocol.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **None — greenfield.** The repository currently contains only `.planning/` docs and config; no Rust crate, no Python package, no kernels exist yet. Phase 1 lays the first code.

### Established Patterns
- No code patterns yet. Conventions to honor from the user's global rules: **many small files (200–400 lines typical, 800 max)**, comprehensive error handling, no hardcoded values, immutability where idiomatic (Rust pointer/ownership idioms take precedence per language note).

### Integration Points
- The persisted skeleton (D-04) becomes the integration point for Phase 2's `trait Backend` + `CpuBackend` + `ForestIR`. Keep the Cargo workspace and maturin/pyproject layout shaped so those drop in cleanly.

</code_context>

<specifics>
## Specific Ideas

- **Setup prerequisite (not a decision):** ⚠ `rustc` is **not on PATH** as of 2026-06-20 — Rust **stable ≥ 1.83** (PyO3 0.29 MSRV floor) must be installed and on PATH before any build work. Capture this as the first task in the plan.
- Toolchain present and verified locally: CUDA 12.8 (nvcc 12.8.93), driver 595.79, Python 3.14.3, RTX 4060 Ti (sm_89). Missing: Rust, maturin, VS 2022 Build Tools / MSVC v143 (verify `cl.exe` on PATH), and CuPy (if chosen as baseline — see D-06 flag).
- The spike must record pinned versions in a durable artifact (e.g. a `VERSIONS.md` / lockfiles) per TOOL-04, including cudarc feature flags, the rust-numpy↔PyO3 0.29 compatible version (PROJECT.md flags ≈0.25.x as **unverified** — researcher must confirm the exact compatible `numpy` crate release), and the CUDA toolkit version.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within Phase 1 scope. The CubeCL portability backend, Philox RNG, DLPack/`__cuda_array_interface__` zero-copy interop, and all estimator/algorithm work are explicitly later-phase concerns and were not pulled into the spike.

</deferred>

---

*Phase: 1-Toolchain Spike (Gate 1)*
*Context gathered: 2026-06-20*
