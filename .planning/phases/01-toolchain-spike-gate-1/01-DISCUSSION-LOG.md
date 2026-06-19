# Phase 1: Toolchain Spike (Gate 1) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 1-Toolchain Spike (Gate 1)
**Areas discussed:** Spike kernel choice, Version pinning, Scaffolding fate, Kill-criteria & baseline

---

## Spike kernel choice

| Option | Description | Selected |
|--------|-------------|----------|
| Representative histogram | Small shared-mem privatized histogram + atomicAdd; proves toolchain AND de-risks Phase-4 primitive | |
| Trivial elementwise | Vector add on 1e7 float32 — roadmap's literal microbench; fastest green signal | |
| Both (layered) | Elementwise for the microbench baseline + a separate histogram kernel for racecheck | ✓ |

**User's choice:** Both (layered)
**Notes:** Cleanly separates the "is the toolchain alive" signal (elementwise) from the "are the hard primitives debuggable" signal (histogram/atomicAdd → racecheck). The toolchain risk and the real hot-path primitive risk are proven together.

### Follow-up: histogram representativeness (skipped)

| Option | Description | Selected |
|--------|-------------|----------|
| Match real layout | Privatized per-block histograms keyed like the real Phase-4 bin layout | |
| Generic atomicAdd | Any small shared-mem histogram over a few bins | |
| You decide | Planner/researcher picks based on Phase-4 detail settled | (deferred) |

**User's choice:** Skipped → Claude's discretion (lean "match real layout").

---

## Version pinning

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic-loading | cudarc `dynamic-loading`; CUDA resolved at runtime; one wheel for any compatible CUDA | |
| Static cuda-12080 | Pin/link the installed 12.8 toolkit; simpler but bakes CUDA version into the wheel | |
| Prove both | Static `cuda-12080` for the launch proof + dynamic-loading wheel for the shipping config | ✓ |

**User's choice:** Prove both
**Notes:** Validates the actual shipping configuration (dynamic-loading) in Phase 1 instead of deferring/trusting it. Environment-detected pins locked alongside: RTX 4060 Ti / sm_89, CUDA 12.8 → `cuda-12080`, driver 595.79, Python 3.14.3 → `abi3-py310`, features `driver`+`nvrtc` (no cuBLAS/cuRAND).

---

## Scaffolding fate

| Option | Description | Selected |
|--------|-------------|----------|
| Persist skeleton | Lay down real Cargo workspace + sylva crate + pyproject/maturin now; delete only spike kernels | ✓ (discretion) |
| Pure scratch | Everything in throwaway `spike/`, deleted wholesale after the gate | |
| You decide | Planner chooses based on reusability | |

**User's choice:** Skipped → Claude's discretion. Defaulted to "persist skeleton" (fits "many small files / structure early" convention; avoids re-scaffolding in Phase 2). Planner may revert to pure scratch.

---

## Kill-criteria & baseline

| Option | Description | Selected |
|--------|-------------|----------|
| 2-day box, WSL only if MSVC blocks | ~2 working days; WSL only if native wheel/link step blocks; stop if no clean sanitize anywhere | ✓ (discretion) |
| 1-day box, aggressive | ~1 day; any native blocker → WSL → stop; higher false-negative risk | |
| Open-ended until proven | No hard timebox; grind until proceed/WSL/stop is unambiguous | |

**User's choice:** Skipped → Claude's discretion. Defaulted to the 2-day timebox (aligns with the user's "timebox debugging detours" guideline).
**Notes:** Microbench baseline defaulted to CuPy with a researcher-flag that CuPy may lack Python 3.14 wheels (fallback: raw CUDA-C baseline or a 3.11/3.12 venv). Feasibility sanity check only — no algorithm speed claim.

---

## Claude's Discretion

- Histogram-kernel representativeness (lean "match real Phase-4 layout").
- Scaffolding fate → "persist skeleton."
- Kill-criteria/timebox → "~2-day box, WSL only if MSVC/link blocks, else stop."
- Microbench baseline → CuPy with a Python-3.14 wheel-availability flag for the researcher.

## Deferred Ideas

None — discussion stayed within Phase 1 scope.
