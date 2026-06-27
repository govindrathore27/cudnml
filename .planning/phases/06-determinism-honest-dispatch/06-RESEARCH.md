# Phase 6: Determinism & Honest Dispatch - Research

**Researched:** 2026-06-27
**Domain:** Hardening byte-reproducible (`deterministic=True`) GPU forest training onto the now-correct Phase 4/5 kernels (eliminating every remaining source of GPU non-determinism), measuring deterministic-mode overhead (Sylva-vs-Sylva), enforcing `device="auto|cuda|cpu"` dispatch with `fallback="error"` (no silent CPU fallback; unsupported configs RAISE), and building a full `execution_report_` (selected backend + reason, every input conversion, bytes transferred, fallback status). Implemented in Rust core + CUDA-C via NVRTC + a PyO3/maturin estimator surface, on native Windows/MSVC (sm_89 / RTX 4060 Ti).
**Confidence:** HIGH for the dispatch contract, the `execution_report_` shape (extends the shipped `QuantizeReport`), the byte-compare gate idiom (already proven in `tests/determinism.rs`), and the determinism-mechanism inventory (grounded in shipped CPU oracle + the Phase 4/5 locked integer-accumulation decisions). MEDIUM for *which* GPU non-determinism sources actually survive Phases 4/5 (depends on whether 4/5 already adopted the fully-deterministic baseline as their locked decisions imply — flagged Open Question 1), and for the deterministic-vs-fast split (whether a *non-deterministic faster* path even exists to measure overhead against — Open Question 2).

> **Phase-ordering note (important for the planner):** Phases 3, 4, and 5 are **planned but NOT yet executed** (STATE.md: `current_phase: 03`, completed_phases: 2). This research is therefore written against the *locked design decisions* in the Phase 4/5 RESEARCH/PLAN docs (integer/fixed-point histograms, host-side scoring, `-fmad=false`, fixed tie-break, fixed scan order), not against shipped GPU code. Several findings hinge on whether 4/5 land exactly as designed; those are flagged. The integer/deterministic-accumulation architecture decision is explicitly "designed in from Phase 4, hardened in Phase 6" (ROADMAP/STATE), so Phase 6's job is largely **enforcement, surfacing (report/flag/test), and measurement** — not inventing determinism from scratch.

## User Constraints (from CONTEXT.md)

> No `06-CONTEXT.md` exists yet (this phase has not been through `/gsd-discuss-phase`). The constraints below are extracted from the binding upstream sources (PROJECT.md, ROADMAP.md, STATE.md, CLAUDE.md) and must be treated by the planner as locked until a CONTEXT.md supersedes them.

### Locked Decisions (from PROJECT.md / ROADMAP / STATE / CLAUDE.md)
- **No silent CPU fallback.** `device="auto|cuda|cpu"`, `fallback="error"`; an unmet `device="cuda"` request RAISES a typed error. This is a *core project differentiator* vs cuml.accel / H2O4GPU silent fallback (PROJECT.md Key Decisions; DET-03).
- **`execution_report_` is mandatory** and explains every decision + input conversion (PROJECT.md Active requirements; DET-04).
- **Deterministic mode must be bit-reproducible** — byte-identical models across two same-seed runs, verified by *exact binary comparison*, NOT `allclose` (PROJECT.md Constraints "Correctness"; DET-01).
- **Integer/deterministic accumulation** is a near-rewrite-if-deferred architecture decision, designed in from Phase 4, **hardened here** (STATE.md Blockers/Concerns).
- **Comparative Baseline Study is Sylva-vs-Sylva** for the overhead number + a *qualitative-only* external determinism-gap claim (cuML RF / LightGBM-GPU are NOT byte-reproducible). **No cross-library speed claim** (ROADMAP Phase 6 study; binding fairness note).
- **cudarc 0.19.8 + hand-written CUDA-C via NVRTC, native Windows/MSVC, Apache-2.0, stable Rust 1.83+** (CLAUDE.md). `-fmad=false`, never `--use_fast_math`.

### Claude's Discretion
- Internal module layout of the dispatch + report code (subject to "many small files" CLAUDE.md rule).
- Whether the `execution_report_` is a Rust struct serialized to JSON across the seam (recommended — extends `QuantizeReport`) vs a Python-side dict.
- Whether a *separate* non-deterministic fast path is built at all, or determinism is the only mode and "overhead" is measured against a documented hypothetical/micro-variant (Open Question 2).

### Deferred Ideas (OUT OF SCOPE)
- Zero-copy GPU input via `__cuda_array_interface__` / DLPack (API-V2-02) — not this phase.
- Multi-GPU determinism (SCALE-V2-01).
- The authoritative end-to-end (n×d) crossover (Phase 7, BENCH-01..03) — Phase 6 only produces a Sylva-vs-Sylva overhead data point that *feeds* Phase 7.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **DET-01** | `deterministic=True` yields byte-identical models across two same-seed runs (exact binary comparison, NOT `allclose`), via integer/fixed-point histogram accumulation + canonical reduction order + fixed tie-breaking | The byte-compare gate already exists for CPU (`tests/determinism.rs`, serde_json string equality). Phase 4/5 lock the integer histogram + host-side scoring + fixed tie-break + fixed scan order. Phase 6 hardens + extends the gate to the GPU path under `deterministic=True`, and audits the *remaining* non-determinism sources (multi-stream ordering, arena reuse, any residual float reduction). See **Remaining GPU Non-Determinism Inventory** + Pitfalls 1–4. |
| **DET-02** | The documented performance cost of deterministic mode is measured and reported | Sylva-vs-Sylva overhead measurement (the internal apples-to-apples baseline). Expected ~95–98% throughput retention (small overhead — reported, not gated). See **Validation Architecture** + Open Question 2 (does a non-deterministic faster path exist to measure against?). |
| **DET-03** | `device="auto"|"cuda"|"cpu"` dispatch with `fallback="error"` — no silent CPU fallback; unsupported configs raise | The typed-error spine is shipped (`CudaError`→`SylvaError`→`PyErr`, `cuda_error_to_pyerr`, `sylva_error_to_pyerr`). Phase 5 builds the *minimal* dispatch in the estimator layer; Phase 6 makes it the full contract: `auto` selection logic + every unsupported path a typed RAISE. See **Dispatch State Machine** + Pattern 2/3. |
| **DET-04** | `execution_report_` reports selected backend + reason, every input conversion (dtype/layout/H2D), bytes transferred, and fallback status | Extends the shipped `QuantizeReport` (which already carries `input_dtype`, `input_c_contiguous`, `binned_bytes`, `edges_bytes`, `h2d_executed`, `h2d_note` and is explicitly marked "Phase 6 will extend into a full `execution_report_`"). See **Pattern 1** + the `ExecutionReport` schema. |
</phase_requirements>

## Summary

Phase 6 is **not a new-algorithm phase** — it is a *hardening, enforcement, surfacing, and measurement* phase layered on the correct-but-not-yet-contract-complete forest from Phases 4/5. Three of its four requirements are mostly already de-risked by upstream locked decisions: the integer/fixed-point accumulation that makes determinism *possible* is designed into Phase 4 (ET counts) and Phase 5 (RF scan, sibling subtraction as integer-only, `sample_weight` as fixed-point), and the typed no-silent-fallback error spine (`CudaError`/`SylvaError`/`cuda_error_to_pyerr`) is shipped from Phase 1. The byte-identical gate idiom — serialize two same-seed `ForestIR`s and assert exact string equality — is already running for the CPU backend in `crates/sylva-core/tests/determinism.rs`. So Phase 6's real work is: (1) **prove** that gate holds for the GPU path under `deterministic=True` by **auditing and eliminating the GPU-specific non-determinism sources that a histogram tree builder can still leak even with integer histograms** (multi-stream completion ordering, arena buffer-reuse ordering, any float reduction or `atomicAdd` that slipped onto the hot path, and non-associative scoring); (2) **build the full `execution_report_`** by extending the shipped `QuantizeReport` into an `ExecutionReport` that records the selected backend + reason, every dtype/layout coercion, the H2D byte count, and the fallback status; (3) **enforce the dispatch contract** so `device="cuda"` on a machine with no usable GPU RAISES a typed error rather than silently using the CPU; and (4) **measure** the Sylva-vs-Sylva deterministic overhead and qualitatively confirm the external determinism gap.

The single most important conceptual point for the planner: **a `deterministic=True` flag is only meaningful if there is something it changes.** If Phases 4/5 already made the *only* training path fully deterministic (which their locked decisions — integer histograms, host-side scoring, fixed scan/tie-break — strongly imply), then `deterministic=True` is the *default and only* behavior and the "performance cost" (DET-02) is ~0% because there is no faster non-deterministic path to trade against. The realistic interpretation, consistent with the ~95–98% throughput-retention target in the ROADMAP, is that Phase 6 introduces a **deterministic mode that constrains otherwise-faster relaxations** — specifically (a) single-stream / serialized multi-stream tree scheduling instead of free multi-stream overlap, and (b) a deterministic arena allocation/reuse order — and the *non-deterministic* mode is what enables free stream overlap. The overhead is the cost of giving up that overlap. **This deterministic-vs-fast axis must be settled before planning** (Open Question 2): it determines whether DET-02 measures a real toggle or documents "deterministic is the only mode; overhead = 0, here is why."

**Primary recommendation:** Plan Phase 6 as three slices. **Slice 1 — `execution_report_` (DET-04):** add `crates/sylva-core/src/report.rs` defining an `ExecutionReport` struct (superset of `QuantizeReport`: `selected_backend`, `selection_reason`, `requested_device`, `fallback_policy`, `fallback_status`, `conversions: Vec<InputConversion>`, `bytes_h2d`, `bytes_d2h`, `deterministic: bool`), thread it through `CudaBackend::fit`/`CpuBackend::fit` (returning it alongside or attached to the IR), and expose it on every estimator as `execution_report_`. **Slice 2 — honest dispatch (DET-03):** add `crates/sylva-core/src/dispatch.rs` (device probe + `auto` selection + `fallback="error"` enforcement, returning a typed `SylvaError::DeviceUnavailable`/`UnsupportedConfig` — a new variant) wired into the Python `_dispatch.py` so an unmet `device="cuda"` RAISES. **Slice 3 — determinism harden + measure (DET-01, DET-02):** add a `deterministic: bool` to `TrainConfig`, route the GPU scheduler to its deterministic constraints (serialized streams + deterministic arena order), add the GPU same-seed byte-identical gate (mirror `tests/determinism.rs` for `CudaBackend`), run the four-tool `compute-sanitizer` (esp. `racecheck`) as the non-determinism canary, and build the Sylva-vs-Sylva overhead bench + the qualitative cuML/LightGBM-GPU determinism-gap note. The GPU two-run byte-compare under `deterministic=True` is the **DET-01 gate**; the typed-raise-on-unmet-cuda tests are the **DET-03 gate**; the `execution_report_` field-assertion tests are the **DET-04 gate**; the overhead number is **reported, not gated** (DET-02).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `deterministic=True` byte-reproducibility enforcement | L3 host orchestration (Rust `sylva-cuda` scheduler) + L1 kernels (already integer) | test tier (byte-compare gate) | The non-determinism that survives integer histograms is *ordering* (streams, arena reuse) — a host-scheduler concern — plus any residual float reduction (L1). The flag toggles host scheduling constraints. |
| Remaining-non-determinism audit (streams, arena, residual floats, scan order) | L3 host + L1 CUDA-C | `compute-sanitizer racecheck` (canary) | This is the core Phase-6 investigation; integer counts are order-free but *which buffer / which stream finishes first* is not, unless serialized. |
| `execution_report_` (DET-04) | L3 device-neutral (`sylva-core::report`, extends `QuantizeReport`) | exposed at L4 Python as an attribute | Single shared report struct written by both backends; serde to JSON across the seam (the shipped `QuantizeReport` pattern). |
| Honest dispatch `device=`/`fallback="error"` (DET-03) | L3 typed errors (`sylva-core::dispatch` + `SylvaError`) | L4 Python `_dispatch.py` maps to typed `PyErr` | The decision (which backend, why) is application logic; the no-silent-fallback contract is a typed RAISE, never a degrade. Spine already shipped (Phase 1 error mapping). |
| Device availability probe (is a usable CUDA GPU present?) | L3 host (`sylva-cuda`, cudarc `CudaContext::new(0)` as a `Result`) | feeds `auto` selection + the report's `selection_reason` | cudarc surfaces a clean `Result` for context init — exactly what `fallback="error"` needs; no panic. |
| Deterministic-overhead measurement (DET-02, Sylva-vs-Sylva) | bench/test tier (Python harness) | reads L4 estimators | Sylva deterministic vs Sylva non-deterministic on identical inputs/seed; feeds Phase 7. |
| Qualitative external determinism-gap confirmation | study tier (Python, cuML/LightGBM-GPU) | documented, not gated | Establishes the differentiator without a speed claim; cuML on Windows gated behind human-verify (carried from Phase 5 OQ7). |

## Standard Stack

> Phase 6 introduces **no new external dependencies.** Everything is pinned and proven in Phases 1–5. This phase adds *source files* (`report.rs`, `dispatch.rs`, a `deterministic` config field, the Python `_dispatch.py` enforcement, byte-compare GPU tests, an overhead bench) — not packages.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cudarc | 0.19.8 | `CudaContext::new(0)` as the device-availability probe (`Result`, no panic); stream control for deterministic serialization; the arena from Phase 5 | `[VERIFIED: crates/sylva-cuda/Cargo.toml + VERSIONS.md]` Every cudarc call is already `?`-propagated to `CudaError` — the exact substrate `fallback="error"` needs. |
| sylva-core | (workspace) | `TrainConfig` (gains `deterministic`), `SylvaError` (gains dispatch variants), `QuantizeReport` → `ExecutionReport`, `ForestIR`, byte-compare idiom | `[VERIFIED: crates/sylva-core/src]` The report, config, and error types all live here and are explicitly marked for Phase-6 extension. |
| serde / serde_json | 1.x | `ExecutionReport` round-trip across the PyO3 seam; the byte-identical gate (serialize two IRs, compare strings) | `[VERIFIED: report.rs + ir.rs derive Serialize; tests/determinism.rs uses serde_json string equality]` The shipped determinism-test mechanism. |
| PyO3 | 0.29.0 | Expose `execution_report_` attribute + the `device`/`fallback`/`deterministic` params; map dispatch errors to typed `PyErr` | `[VERIFIED: Cargo.toml abi3-py310; lib.rs cuda_error_to_pyerr, pyseam.rs sylva_error_to_pyerr]` The no-silent-fallback FFI mapping is already the established pattern. |
| thiserror | 1.x | New `SylvaError` dispatch variants (`DeviceUnavailable`, `UnsupportedConfig`) | `[VERIFIED: error.rs SylvaError, nvrtc_launch.rs CudaError]` Extend the existing enums; no new crate. |
| compute-sanitizer | CUDA 12.8 | `racecheck` as the non-determinism canary on the deterministic GPU path | `[VERIFIED: VERSIONS.md; Phase-1 four-tool clean]` A data race is a non-determinism source; racecheck is the proof. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | 1.9.0 | `check_estimator` still green after `device`/`fallback`/`deterministic` params + `execution_report_` are added; estimator-clone semantics for the new params | `[VERIFIED: .venv-parity scikit_learn-1.9.0]` New constructor params must be stored verbatim (no `__init__` logic) and survive clone. |
| pytest | (in `.venv-parity`) | DET-01..04 Python gates + the overhead bench harness | The Python test/bench runner; mirrors `python/tests/` and `python/benchmarks/`. |
| cuML (RAPIDS) / LightGBM-GPU | pin in study manifest | The *qualitative* external determinism-gap references only (run two same-seed fits, show byte difference) | `[ASSUMED]` Windows availability problematic (cuML Linux-first); gate behind `checkpoint:human-verify` — carried from Phase 5 OQ7. Document "not available on host" honestly if it won't run. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extend `QuantizeReport` → `ExecutionReport` (Rust struct, serde) | A Python-side dict assembled in `_dispatch.py` | The Rust struct keeps the report device-neutral, written identically by both backends, and reuses the shipped serde round-trip + tests. A Python dict scatters the truth across layers and can't be unit-tested in Rust. **Recommend the Rust struct.** |
| Deterministic = serialized streams + deterministic arena (a real toggle) | Determinism is the only mode (no fast path) | If 4/5 already made the only path deterministic, the "toggle" is vacuous and DET-02 overhead = 0 (document why). If a free-multi-stream fast path exists, the toggle is real and overhead is the cost of serialization. **Open Question 2 — must be settled.** |
| `CudaContext::new(0)` as availability probe | A separate `nvidia-smi`/driver query | cudarc's context init already returns a typed `Result` that captures "no GPU / no driver / wrong CUDA" — reuse it; don't shell out. |
| New `SylvaError::DeviceUnavailable` variant | Reuse `InvalidConfig` | A distinct variant lets the Python layer map "you asked for cuda but there is none" to a precise exception type/message — better for the no-silent-fallback UX and for tests. **Recommend a new variant.** |

**Installation:** No new packages. Phase 6 adds Rust source under `crates/sylva-core/src/` and `crates/sylva-cuda/src/cuda_backend/`, Python under `python/sylva/_dispatch.py`, tests, and a bench. Study-only cuML/LightGBM-GPU baselines are external references (gate install behind human-verify).

**Version verification:** All Sylva pins are already committed in `Cargo.lock` + `VERSIONS.md` (cudarc 0.19.8, PyO3 0.29.0, numpy 0.29, CUDA 12.8, sm_89, driver 595.79, Rust 1.96.0). sklearn 1.9.0 confirmed present in `.venv-parity`. No registry re-check needed for runtime deps.

## Package Legitimacy Audit

> Phase 6 adds **no new packages.** The only externally-invoked tools are the *qualitative determinism-gap* baselines (cuML, LightGBM-GPU), which are study references, not Sylva runtime or build dependencies. Sylva's stack (cudarc, pyo3, numpy, ndarray, rayon, serde, thiserror) was vetted and pinned in Phases 1–2 and is in the committed `Cargo.lock`.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| cudarc | crates.io | active (0.19.8) | 300k+/version | github.com/coreylowman/cudarc | OK (pinned Phase 1) | Approved (in use) |
| pyo3 / numpy / ndarray / rayon / serde / thiserror | crates.io | mature | very high | (canonical repos) | OK (pinned Phase 1–2) | Approved (in use) |
| scikit-learn | PyPI | mature (1.9.0) | very high | github.com/scikit-learn/scikit-learn | OK | Approved (present) |
| cuml-cu12 / lightgbm | PyPI/RAPIDS | mature | high | github.com/rapidsai/cuml, github.com/microsoft/LightGBM | OK (study ref only) | Approved as qualitative ref; Windows install gated behind human-verify |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*`cuml-cu12` / `lightgbm`(-GPU) exact package names + Windows install paths are `[ASSUMED]`; the planner must gate the study-baseline install behind a `checkpoint:human-verify` (cuML is Linux-first; LightGBM GPU build on Windows needs a specific wheel/CMake path). The Phase-6 gates do NOT depend on these — the external gap is qualitative and may be reported as "not available on host."*

## Architecture Patterns

### System Architecture Diagram

```
  PYTHON USER LAYER (sylva/  — abi3 wheel)
  ────────────────────────────────────────────────────────────────
  est = RandomForestClassifier(device="auto"|"cuda"|"cpu",
                               fallback="error", deterministic=True, random_state=42)
  est.fit(X, y)        ──►  est.execution_report_   (DET-04, read after fit)
        │
        ▼
  _dispatch.py : resolve_backend(requested_device, fallback)        (DET-03)
        │   ┌─────────────────────────────────────────────────────────────┐
        │   │ probe = cuda_available()?   (cudarc CudaContext::new(0) -> Result)│
        │   │ requested="cuda" & !probe & fallback="error" ──► RAISE typed err │
        │   │ requested="cuda" & !probe & fallback="warn"  ──► (NOT in MVP)    │
        │   │ requested="auto"  ──► cuda if probe else cpu (record reason)     │
        │   │ requested="cpu"   ──► cpu (record reason)                        │
        │   └─────────────────────────────────────────────────────────────┘
        │            │ selected_backend + selection_reason
        ├────────────┴──────────────┐
        ▼                           ▼
  RUST CudaBackend (sylva-cuda)   RUST CpuBackend (sylva-core)
        │                           │
        │  build ExecutionReport (DET-04) as it runs:
        │   • requested_device, selected_backend, selection_reason
        │   • conversions: [{from_dtype:float64,to:float32}, {layout:F→C}, ...]
        │   • bytes_h2d (binned + raw X + y), bytes_d2h (node arrays)
        │   • fallback_policy, fallback_status="none"
        │   • deterministic: true
        │
        ▼   ── DETERMINISTIC GPU FIT (deterministic=True constrains scheduling) ──
        │   integer histograms (Phase 4/5)  → order-free counts
        │   host-side scoring (Phase 4 lock) → exact criterion.rs op order, no device float reduce
        │   fixed (feature, threshold_bits) tie-break (Phase 4 lock)
        │   fixed scan order for RF argmax (Phase 5 lock)
        │   ★ DETERMINISM-CRITICAL (Phase 6 enforcement):
        │     • serialize tree streams (or fixed-order merge) — no free multi-stream overlap
        │     • deterministic arena buffer assignment/reuse order
        │     • no global float atomicAdd anywhere on the hot path
        ▼
  ForestIR  ──serde──►  byte-compare two same-seed runs == identical   (DET-01 gate)
        │
        └──► overhead bench: deterministic vs non-deterministic fit time (DET-02, Sylva-vs-Sylva)
             qualitative: cuML/LightGBM-GPU two same-seed fits differ (gap, no speed claim)
```

### Recommended Project Structure
```
crates/sylva-core/src/
├── report.rs           # NEW: ExecutionReport (superset of QuantizeReport) + InputConversion
│                       #   (move/re-export QuantizeReport here or compose it; serde; tests)
├── dispatch.rs         # NEW: device-neutral dispatch types — RequestedDevice, FallbackPolicy,
│                       #   SelectedBackend, SelectionReason; the resolve() decision fn (pure,
│                       #   takes a `cuda_available: bool` probe result → typed Result)
├── config.rs           # EXTEND: TrainConfig gains `deterministic: bool` (+ validate())
├── error.rs            # EXTEND: SylvaError gains DeviceUnavailable / UnsupportedConfig variants
└── quantize/report.rs  # EXTEND or re-home: QuantizeReport becomes a sub-record of ExecutionReport

crates/sylva-cuda/src/cuda_backend/
├── mod.rs              # EXTEND: CudaBackend::fit returns (ForestIR, ExecutionReport); honors cfg.deterministic
├── availability.rs     # NEW: cuda_available() -> Result<bool> via CudaContext::new(0) (no panic)
├── scheduler.rs        # EXTEND: deterministic=true ⇒ serialized/fixed-order stream schedule
├── arena.rs            # EXTEND (Phase 5 file): deterministic buffer assignment order under deterministic=true
└── report_build.rs     # NEW: helpers to populate ExecutionReport (bytes_h2d/d2h tallies, conversions)

python/sylva/
├── _dispatch.py        # EXTEND (Phase 5 file): full device="auto|cuda|cpu" + fallback="error" RAISE;
│                       #   no silent fallback; record selection_reason into the report
├── _base.py            # EXTEND: store device/fallback/deterministic params verbatim (no __init__ logic);
│                       #   expose self.execution_report_ after fit
└── ensemble.py         # EXTEND: the four classes inherit the new params

python/tests/
├── test_dispatch.py            # DET-03: device="cuda" with no GPU RAISES; auto picks correctly; cpu forced
├── test_execution_report.py    # DET-04: report has backend+reason, conversions, bytes_h2d, fallback status
└── gpu_determinism/
    └── test_deterministic_byte_identical.py  # DET-01: two same-seed deterministic GPU fits → byte-identical

crates/sylva-cuda/tests/
└── deterministic_cpu_gpu.rs    # DET-01 (Rust): two CudaBackend fits (deterministic=true) byte-identical;
                                #   + (optional) GPU==CPU under deterministic=true

python/benchmarks/
└── determinism_overhead.py     # DET-02: Sylva deterministic vs non-deterministic fit-time overhead (%)
                                #   + qualitative cuML/LightGBM-GPU same-seed byte-diff note
```

### Pattern 1: `ExecutionReport` extends the shipped `QuantizeReport` (DET-04)
**What:** Define one device-neutral report struct that both backends populate and that becomes `execution_report_`. It is a *superset* of the shipped `QuantizeReport` (which already has `input_dtype`, `input_c_contiguous`, `binned_bytes`, `edges_bytes`, `h2d_executed`, `h2d_note` and is explicitly annotated "Phase 6 will extend this"). Keep the quantize record as a nested field rather than duplicating.
**When to use:** Always — it is the DET-04 deliverable. Written incrementally during `fit`, returned alongside the IR, exposed as the estimator attribute.
**Example:**
```rust
// Source: extends crates/sylva-core/src/quantize/report.rs (the shipped QuantizeReport).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectedBackend { Cpu, Cuda }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputConversion {
    pub what: String,        // "dtype" | "layout" | "contiguity"
    pub from: String,        // "float64" | "F-order" | "non-contiguous"
    pub to: String,          // "float32" | "C-order" | "contiguous-copy"
    pub bytes_copied: usize, // 0 if zero-copy
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub requested_device: String,        // "auto" | "cuda" | "cpu"
    pub selected_backend: SelectedBackend,
    pub selection_reason: String,        // "auto: CUDA device 0 available" | "explicit cpu" | ...
    pub fallback_policy: String,         // "error" (MVP) | "warn" (not in MVP)
    pub fallback_status: String,         // "none" | "n/a" ; NEVER a silent CPU substitution
    pub deterministic: bool,
    pub conversions: Vec<InputConversion>,
    pub bytes_h2d: usize,                // total host→device (binned + raw X + y)
    pub bytes_d2h: usize,                // total device→host (node arrays)
    pub quantize: QuantizeReport,        // the shipped per-quantize record, nested
}
```
**Anti-pattern:** Assembling the report as a Python dict — it can't be unit-tested in Rust, drifts from the backend truth, and duplicates the shipped serde round-trip.

### Pattern 2: Honest dispatch as a pure decision function over a probe (DET-03)
**What:** Make backend selection a *pure* function of `(requested_device, fallback_policy, cuda_available)` returning a typed `Result<SelectedBackend, SylvaError>`. The only impure part is the probe (`cuda_available()` via `CudaContext::new(0)`), isolated so the decision is unit-testable without a GPU. **Every unmet `device="cuda"` is a RAISE, never a degrade.**
**When to use:** At the top of every `fit` (and the estimator records the result into the report).
**Example:**
```rust
// Source: original; uses the shipped typed-error spine (SylvaError) + cudarc Result.
pub fn resolve_backend(
    requested: RequestedDevice,     // Auto | Cuda | Cpu
    fallback: FallbackPolicy,       // Error  (MVP only supports Error)
    cuda_available: bool,           // from availability::cuda_available()
) -> Result<(SelectedBackend, String /*reason*/), SylvaError> {
    match requested {
        RequestedDevice::Cpu  => Ok((SelectedBackend::Cpu, "explicit device=cpu".into())),
        RequestedDevice::Cuda => {
            if cuda_available {
                Ok((SelectedBackend::Cuda, "explicit device=cuda; CUDA device 0 available".into()))
            } else {
                // NO SILENT FALLBACK — the differentiator. RAISE.
                Err(SylvaError::DeviceUnavailable(
                    "device=\"cuda\" requested with fallback=\"error\" but no usable CUDA device \
                     was found (CudaContext::new(0) failed). Refusing to silently use CPU.".into()))
            }
        }
        RequestedDevice::Auto => Ok(if cuda_available {
            (SelectedBackend::Cuda, "auto: CUDA device 0 available".into())
        } else {
            (SelectedBackend::Cpu,  "auto: no CUDA device; using CPU".into())
        }),
    }
}
```
**Anti-pattern:** Catching a CUDA init error inside `fit` and continuing on CPU. That is exactly the silent fallback the project exists to avoid.

### Pattern 3: Device availability probe via cudarc `Result` (no panic, no shell-out)
**What:** `cuda_available()` attempts `CudaContext::new(0)` and maps the outcome to `bool` (or to a richer reason). cudarc returns a `DriverError` if there is no driver/device/incompatible CUDA — exactly the clean `Result` the contract needs.
**When to use:** For `auto` selection and to enforce `fallback="error"` on an explicit `cuda` request.
**Example:**
```rust
// Source: original; cudarc CudaContext::new returns Result (Phase-1 proven init pattern).
pub fn cuda_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()   // no panic; no nvidia-smi shell-out
}
// (Optionally return Result<(), CudaError> to surface WHY for selection_reason.)
```

### Pattern 4: Deterministic GPU scheduling — serialize ordering, keep counts integer (DET-01)
**What:** With `deterministic=True`, constrain every *ordering* the GPU could vary: (a) build/merge trees in a fixed order rather than via free multi-stream overlap whose completion order is nondeterministic; (b) assign/reuse arena buffers in a deterministic order; (c) ensure no float `atomicAdd` and no warp-shuffle tree-sum on the parity path (counts stay integer = associative; scoring is host-side in fixed `criterion.rs` order, per the Phase-4 lock). The *non-deterministic* (faster) mode relaxes (a)/(b) for stream overlap.
**When to use:** The deterministic training path. The byte-compare gate is the proof.
**Anti-pattern:** Believing "integer histograms ⇒ fully deterministic." Integer *counts* are order-free, but *which stream/buffer finishes first*, and any leftover float reduction, are not. The Phase-6 audit must close those.

### Anti-Patterns to Avoid
- **Silent CPU fallback on an unmet `device="cuda"`** — the entire project differentiator; always RAISE a typed error (DET-03).
- **A float `atomicAdd` or warp-shuffle reduction surviving onto the deterministic hot path** — re-introduces run-to-run non-determinism even with integer histograms (DET-01). `racecheck` + the byte-compare gate are the canaries.
- **Free multi-stream overlap under `deterministic=True`** — completion ordering is nondeterministic; serialize or use a fixed-order merge.
- **`execution_report_` assembled in Python** — drifts from backend truth; build the Rust struct and serialize it.
- **Measuring DET-02 as a cross-library speed claim** — it is Sylva-vs-Sylva only; the external comparison is *qualitative* (does cuML/LightGBM-GPU reproduce byte-identically? — no).
- **Logic in estimator `__init__` for the new `device`/`fallback`/`deterministic` params** — `check_estimator` fails; store verbatim, validate/resolve in `fit`.
- **A vacuous `deterministic` flag** — if there is no non-deterministic path, don't pretend there is overhead; document "deterministic is the only mode; overhead = 0 because X" (Open Question 2).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Execution record | A fresh report type from scratch | Extend the shipped `QuantizeReport` → `ExecutionReport` | It already has dtype/contiguity/bytes/h2d fields and is annotated "Phase 6 extends this"; reuse its serde + tests. |
| Device availability check | An `nvidia-smi` shell-out / FFI to the driver | `cudarc::driver::CudaContext::new(0).is_ok()` | cudarc already returns a clean typed `Result`; the Phase-1 init pattern. No subprocess. |
| Typed error mapping across FFI | A new exception path | `cuda_error_to_pyerr` / `sylva_error_to_pyerr` (shipped) | The no-silent-fallback FFI contract is already implemented; add variants, reuse the mapper. |
| Byte-identical gate | A custom binary diff / `allclose` tolerance | serde_json string equality of two `ForestIR`s (shipped `tests/determinism.rs` idiom) | DET-01 is *exact* binary comparison; the CPU gate already does this — mirror it for GPU. |
| Same-seed determinism proof | A new RNG-replay scheme | The shipped Philox `(seed,tree,node,feature,draw)` keying + `tests/determinism.rs` (incl. the 1-thread-pool order-independence test) | Already proves order-independence for CPU; the GPU path must reproduce the same property. |
| `get_params`/`set_params`/clone of new params | Custom param plumbing | sklearn `BaseEstimator` (store `device`/`fallback`/`deterministic` as same-named attrs) | Free, correct, `check_estimator`-clean. |
| Integer histogram / fixed-point weights | A determinism rework | The Phase 4/5 locked integer count + fixed-point `sample_weight` accumulation | Determinism is *designed in* upstream; Phase 6 enforces + surfaces, not rebuilds. |

**Key insight:** Phase 6 hand-rolls **only** the dispatch decision logic, the `ExecutionReport` superset, the `deterministic` scheduling constraints in the GPU scheduler/arena, and the measurement/test harness. The determinism *substrate* (integer accumulation, fixed tie-break/scan, host scoring) and the typed-error *spine* both already exist — Phase 6's value is making them a **contract the user can see, trust, and verify**.

## Runtime State Inventory

> Phase 6 is a config/contract/measurement phase, not a rename/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string. **None found in any category — verified by scope (this phase adds a `deterministic` config field, a dispatch decision, a report struct, and tests/benches; it renames nothing and migrates no datastore).** This section is otherwise N/A for Phase 6.

## Common Pitfalls

### Pitfall 1: "Integer histograms make it deterministic" — the ordering gap
**What goes wrong:** The team assumes that because Phase 4/5 use integer count histograms (order-free), the GPU forest is automatically byte-reproducible. Two same-seed runs still differ because **multi-stream tree scheduling completes in a nondeterministic order**, or the **arena reuses buffers in a run-dependent order**, subtly changing which node-array slot a tree's nodes land in (and thus the serialized IR), or because a **residual float reduction** (e.g. an on-device score that slipped in instead of host-side scoring) sums in warp order.
**Why it happens:** Determinism of *values* (integer counts) ≠ determinism of *layout/ordering*. The serialized `ForestIR` is sensitive to node ordering and tree-slot assignment.
**How to avoid:** Under `deterministic=True`: assign tree→node-array offsets in fixed tree index order (mirror `assemble_forest` global-offset logic); serialize streams or use a fixed-order merge; assert no on-device float reduction is on the path (host-side scoring is the Phase-4 lock); run `racecheck`. Add the two-run byte-compare gate as the regression canary.
**Warning signs:** Two same-seed GPU fits agree on accuracy but differ byte-for-byte; difference disappears when forced to one stream.

### Pitfall 2: Silent fallback creeps in via error handling
**What goes wrong:** A `match`/`try` around CUDA init or a kernel launch "helpfully" falls back to CPU on error, violating DET-03 — the exact behavior cuml.accel/H2O4GPU are criticized for.
**Why it happens:** Defensive coding instinct; the CPU backend is right there and "works."
**How to avoid:** The dispatch decision is made *once*, *before* fit, by `resolve_backend` (Pattern 2). Inside `fit`, a CUDA error is a typed RAISE, never a CPU retry. `fallback="error"` is the only MVP policy; a `warn`/`silent` policy is explicitly out of scope. Add a test that monkeypatches `cuda_available()→False` and asserts `device="cuda"` RAISES.
**Warning signs:** A test that requests `cuda` on a GPU-less CI runner *passes* by quietly training on CPU.

### Pitfall 3: The `deterministic` flag has nothing to toggle (DET-02 measures noise)
**What goes wrong:** If Phases 4/5 already made the only path fully deterministic, `deterministic=True` changes nothing, and the "overhead" measured for DET-02 is just timing jitter — reported as a meaningless ~0% ± noise, or worse, presented as if a real tradeoff was measured.
**Why it happens:** The locked integer-accumulation + host-scoring + fixed-order decisions may *already* be fully deterministic, leaving no faster relaxed mode.
**How to avoid:** Settle Open Question 2 before planning. Either (a) define a genuine non-deterministic fast mode (free multi-stream overlap) so DET-02 measures the serialization cost (~95–98% retention target), or (b) document that determinism is the only mode, DET-02 overhead = 0, and explain *why* (the design is deterministic by construction) — which is a legitimate, honest answer to "measure and report the cost."
**Warning signs:** The overhead bench reports numbers indistinguishable from run-to-run jitter; there is no code branch on `cfg.deterministic`.

### Pitfall 4: RF scan/argmax or sibling-subtraction order leaks non-determinism
**What goes wrong:** The RF best-split prefix-scan (Phase 5) or sibling-subtraction (Phase 5) uses a reduction whose order varies, or the argmax tie-break isn't a strict total order — so two same-seed RF runs pick different splits on ties.
**Why it happens:** Hillis-Steele/Blelloch scans and argmax reductions are easy to write order-dependently; subtraction must be integer-only.
**How to avoid:** Fixed scan order (Phase-5 lock), integer-only sibling subtraction (Phase-5 lock), and the exact `(feature, threshold_bits)` total-order tie-break (Phase-4 lock). Phase 6 *verifies* these via the byte-compare gate on RF specifically (the path most prone to ordering leaks). Test RF and ET separately — RF-only non-determinism points straight at scan/argmax.
**Warning signs:** ET is byte-reproducible but RF is not; divergence concentrated on datasets with split ties.

### Pitfall 5: `execution_report_` omits a conversion that actually happened
**What goes wrong:** The user passes float64 / F-contiguous / non-contiguous input; the backend silently coerces it; the report claims zero conversions — defeating the "explain every input conversion" contract (DET-04) and hiding a perf cost.
**Why it happens:** The coercion happens deep in the numpy→ndarray boundary and isn't threaded back to the report.
**How to avoid:** Record an `InputConversion` at every coercion point (dtype cast, layout copy, contiguity copy), with `bytes_copied`. Add a test that feeds float64 F-order input and asserts the report lists both the dtype cast and the layout copy with nonzero bytes. The shipped `QuantizeReport.input_c_contiguous` is the seed of this.
**Warning signs:** Report shows `conversions: []` for an input that wasn't float32 C-contiguous.

### Pitfall 6: New estimator params break `check_estimator`
**What goes wrong:** Adding `device`/`fallback`/`deterministic` with validation in `__init__`, or not storing them as same-named attributes, fails `check_no_attributes_set_in_init` / `check_parameters_default_constructible` / clone.
**How to avoid:** Store verbatim in `__init__`; validate in `fit`; ensure clone round-trips them. Keep `device="cpu"` as the `check_estimator` test instance so GPU-less CI can run the API gate (the Phase-5 idiom).
**Warning signs:** `check_estimator` red after adding the new params.

### Pitfall 7: Over-claiming the determinism differentiator
**What goes wrong:** The study implies "we are faster AND deterministic, cuML is neither" — a cross-library speed claim Phase 6 is explicitly forbidden from making.
**How to avoid:** DET-02 is Sylva-vs-Sylva overhead only. The external comparison is *qualitative*: run cuML RF / LightGBM-GPU twice same-seed, show the models differ byte-for-byte (or document "could not run on host"). No speed number vs cuML/LightGBM. Phase 7 owns speed.
**Warning signs:** A determinism section with a cuML wall-clock comparison in it.

## Code Examples

### DET-01 GPU two-run byte-identical gate (mirrors the shipped CPU gate)
```rust
// Source: extends crates/sylva-core/tests/determinism.rs (serde_json string-equality idiom)
// to the CudaBackend under deterministic=true.
#[test]
fn gpu_deterministic_two_runs_byte_identical() {
    let (x, y) = fixed_seed_dataset();
    let cfg = TrainConfig { deterministic: true, seed: 42, algo: Algo::RandomForest,
                            bootstrap: true, n_estimators: 16, /* .. */ };
    let backend = CudaBackend::new().expect("cuda");
    let (ir1, _r1) = backend.fit_with_report(x.view(), y.view(), &cfg).unwrap();
    let (ir2, _r2) = backend.fit_with_report(x.view(), y.view(), &cfg).unwrap();
    assert_eq!(serde_json::to_string(&ir1).unwrap(),
               serde_json::to_string(&ir2).unwrap(),
               "deterministic=true: two same-seed GPU fits must be byte-identical (NOT allclose)");
}
```

### DET-03 no-silent-fallback (Python, monkeypatched probe)
```python
# Source: original; exercises resolve_backend's RAISE-on-unmet-cuda contract without a GPU.
import pytest
from sylva.ensemble import RandomForestClassifier
from sylva import _dispatch

def test_cuda_requested_but_unavailable_raises(monkeypatch):
    monkeypatch.setattr(_dispatch, "cuda_available", lambda: False)
    est = RandomForestClassifier(device="cuda", fallback="error")
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        est.fit(X, y)                       # must RAISE, never silently use CPU

def test_auto_uses_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(_dispatch, "cuda_available", lambda: False)
    est = RandomForestClassifier(device="auto").fit(X, y)
    assert est.execution_report_["selected_backend"] == "Cpu"
    assert "no CUDA device" in est.execution_report_["selection_reason"]
```

### DET-04 execution_report_ assertions (Python)
```python
# Source: original; asserts the report explains backend, reason, conversions, bytes, fallback.
def test_report_records_dtype_and_layout_conversion():
    import numpy as np
    Xf64 = np.asfortranarray(X.astype(np.float64))   # force dtype + layout coercion
    est = RandomForestClassifier(device="cpu", deterministic=True).fit(Xf64, y)
    rep = est.execution_report_
    assert rep["selected_backend"] == "Cpu"
    assert rep["fallback_status"] == "none"
    assert rep["deterministic"] is True
    kinds = {c["what"] for c in rep["conversions"]}
    assert "dtype" in kinds and "layout" in kinds
    assert any(c["bytes_copied"] > 0 for c in rep["conversions"])
    assert rep["bytes_h2d"] >= 0   # 0 for CPU backend; > 0 for cuda
```

### DET-02 Sylva-vs-Sylva overhead (Python bench)
```python
# Source: original; the internal apples-to-apples overhead. NO cross-library speed claim.
def measure_determinism_overhead(X, y, repeats=7):
    det   = _timed_fits(RandomForestClassifier(device="cuda", deterministic=True),  X, y, repeats)
    nondet= _timed_fits(RandomForestClassifier(device="cuda", deterministic=False), X, y, repeats)
    overhead_pct = 100.0 * (median(det) - median(nondet)) / median(nondet)
    # Report retention = nondet/det; target ~95–98% (small overhead). REPORTED, not gated.
    return {"overhead_pct": overhead_pct, "retention_pct": 100.0*median(nondet)/median(det)}
# If deterministic is the ONLY mode (OQ2 resolution b): skip this and document overhead=0 + why.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Silent CPU fallback (cuml.accel, H2O4GPU) | Explicit `device=`/`fallback="error"` + `execution_report_` | this phase (DET-03/04) | The user-facing differentiator: every decision is visible and refusable. |
| "GPU RF is non-reproducible, live with it" (cuML/LightGBM-GPU) | Byte-reproducible `deterministic=True` via integer accumulation + fixed ordering | this phase (DET-01), substrate from Phase 4/5 | The determinism wedge; the qualitative gap vs cuML/LightGBM-GPU is the evidence. |
| Float atomic histograms / free stream overlap | Integer histograms + serialized/fixed-order deterministic scheduling | Phase 4/5 lock, hardened here | Bit-reproducibility; the small measured overhead (DET-02) is the cost of serialization. |
| `QuantizeReport` (quantize step only) | `ExecutionReport` (whole-fit decision + conversions + transfers) | this phase (DET-04) | One auditable record per fit. |

**Deprecated/outdated:**
- Do not reach for `--use_fast_math` / default FMA on the deterministic path (carried from Phase 3/4: `-fmad=false`).
- Do not catch-and-fallback inside `fit` (the silent-fallback anti-pattern).
- cudarc safe `alloc_zeros` is synchronous — fine for the deterministic arena; don't assume `cudaMallocAsync` ordering semantics from it (Phase-5 note).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Phases 4/5 land their locked decisions (integer histograms, host-side scoring, `-fmad=false`, fixed tie-break, fixed scan order) so Phase 6 mostly *enforces + surfaces + measures* determinism rather than inventing it. | Summary / Pitfall 1 | If 4/5 left a float reduction or free-stream overlap as the *only* path, Phase 6 must do real kernel/scheduler determinism work (larger scope). The remaining-non-determinism audit (a planned Phase-6 task) is exactly what surfaces this — front-load it. |
| A2 | A meaningful non-deterministic *faster* mode exists (free multi-stream overlap + relaxed arena order) so DET-02 measures a real ~2–5% overhead. | Summary / Pitfall 3 / OQ2 | If determinism is the only mode, DET-02 overhead = 0 and must be *documented as such* (still satisfies "measure and report the cost"), not faked. Single biggest scoping fork — OQ2. |
| A3 | The serialized `ForestIR` is sensitive to tree/node *ordering*, so deterministic tree→node-array offset assignment (fixed tree-index order) is required, not just integer counts. | Pitfall 1 | If the IR were order-insensitive (it is not — `assemble_forest` adjusts child ids/leaf offsets by global offset), ordering wouldn't matter. It does; the byte-compare gate catches any regression. |
| A4 | `CudaContext::new(0)` returning `Ok` is a sufficient "usable CUDA device" probe for `auto`/`fallback="error"`. | Pattern 3 | A context that inits but later fails on NVRTC/launch would pass the probe yet fail in fit — but that failure is *still a typed RAISE* (no silent fallback), so the contract holds; only the `selection_reason` granularity suffers. Acceptable. |
| A5 | Extending `QuantizeReport` into `ExecutionReport` (nesting the quantize record) is the right shape, vs a flat replacement. | Pattern 1 | If a flatter schema is preferred for the Python attribute, it's a cosmetic refactor; the field set is the load-bearing part. Low risk. |
| A6 | cuML/LightGBM-GPU are NOT byte-reproducible under same-seed runs (the qualitative gap). `[ASSUMED]` from ecosystem knowledge (PROJECT.md ecosystem note). | Study / Pitfall 7 | If one *is* byte-reproducible, the differentiator weakens — but DET-01/03/04 still stand on their own (honest dispatch + report are independent wedges). Verify empirically in the study; if it won't run on Windows, report "not verified on host" honestly. |
| A7 | `device`/`fallback`/`deterministic` can be added to the four estimators without breaking `check_estimator` (stored verbatim, validated in `fit`). | Pitfall 6 | If a check objects to an unknown param, document via `expected_failed_checks` with a reason (the Phase-5 mechanism). Low risk — they are plain constructor params. |

**If this table is empty:** it is not — A1 (do 4/5 land deterministically?) and A2 (is there a non-deterministic fast path to measure against?) are the two the orchestrator/discuss-phase must resolve before locking the plan; they set Phase 6's scope.

## Open Questions (RESOLVED)

> Left open with recommendations per instructions; the orchestrator resolved them before planning — each is locked in the plans:
> - **OQ1 RESOLVED:** front-load a non-determinism AUDIT task (Plan 06-03 Task 1) that enumerates surviving GPU non-determinism in the shipped 4/5 train path and pins each fix (re-validated against shipped code at execution).
> - **OQ2 RESOLVED:** add a `deterministic=false` fast mode (scheduler branches on `cfg.deterministic`) so DET-02 overhead is a real Sylva-vs-Sylva number; if no real free-overlap path exists, document overhead≈0 with the reason (Plan 06-03 Tasks 2–3).
> - **OQ3 RESOLVED:** `deterministic` default = `True` (safe mode) (Plan 06-02 `_base.py`).
> - **OQ4 RESOLVED:** `fallback="error"` only in MVP; any other value → typed UnsupportedConfig raise; `device="cuda"` unmet → `SylvaError::DeviceUnavailable` (Plan 06-02).
> - **OQ5 RESOLVED:** ExecutionReport reports zero `bytes_h2d` for the CPU path (Plan 06-01 test `cpu_report_has_zero_h2d`).
> - **OQ6 RESOLVED:** external determinism-gap study (cuML/LightGBM-GPU not byte-reproducible) behind a `checkpoint:human-verify`; gates do not depend on it (Plan 06-03 Task 4).

> Left open with recommendations per instructions; the orchestrator resolves them before planning.

1. **How much GPU non-determinism actually survives Phases 4/5?**
   - What we know: 4/5 lock integer histograms, host-side scoring, `-fmad=false`, fixed tie-break, fixed scan order. Integer *counts* are order-free.
   - What's unclear: whether *ordering* sources remain — free multi-stream tree overlap (completion order), arena buffer-reuse order, tree→node-array offset assignment order, and whether any float reduction slipped onto the hot path. These are not closed by integer histograms alone.
   - Recommendation: **Front-load a "remaining non-determinism audit" as Phase 6 Task 1** (read the shipped 4/5 GPU code once it exists; enumerate every stream, every arena reuse, every reduction). Drive it with the two-run byte-compare gate + `racecheck`. Treat the audit's findings as the real scope of the DET-01 work.

2. **Is there a non-deterministic *faster* mode, or is determinism the only mode?** (Sets DET-02's meaning.)
   - What we know: ROADMAP targets ~95–98% throughput retention, implying a measurable (small) overhead — i.e. a faster non-deterministic mode exists to retain *against*.
   - What's unclear: whether the 4/5 design even *has* a faster relaxed path, or whether it is deterministic by construction (overhead = 0).
   - Recommendation: **Define a genuine non-deterministic mode = free multi-stream tree overlap + relaxed arena order**, gated by `cfg.deterministic == false`, so DET-02 measures the serialization cost honestly. If the team decides determinism is the only mode, then DET-02 is satisfied by *documenting* overhead = 0 with the reason (deterministic by construction) — both are honest; pick one explicitly before planning.

3. **Where does `deterministic` live in the config + API, and what is its default?**
   - What we know: `TrainConfig` has no `deterministic` field yet (`config.rs`). sklearn determinism is normally implied by `random_state`.
   - Recommendation: Add `deterministic: bool` to `TrainConfig` and a `deterministic=` estimator param. **Default `deterministic=True`** (the project's correctness stance is "deterministic mode must be bit-reproducible"; make the safe mode the default, opt *out* for speed). Confirm the default with the user — it affects every benchmark's reported numbers.

4. **`fallback` policy surface — `error` only, or also `warn`/`silent`?**
   - What we know: PROJECT.md/DET-03 specify `fallback="error"` and "no silent fallback."
   - Recommendation: **MVP supports `fallback="error"` only**; accept the param for forward-compat but RAISE `UnsupportedConfig` on any other value (never implement `silent`). A `warn` mode (use CPU but loudly report it) is a reasonable v2 idea — defer. Confirm whether `warn` is wanted at all.

5. **Does the `execution_report_` need to capture GPU memory/transfer for the CPU backend too?**
   - What we know: CPU has no H2D; the shipped `QuantizeReport` already sets `h2d_executed=false` with an N/A note.
   - Recommendation: The CPU report sets `bytes_h2d=0`, `bytes_d2h=0`, `selected_backend=Cpu`, and still records dtype/layout conversions (those happen on CPU too). Keep one struct for both backends; zero is a valid, honest value.

6. **Should the qualitative external determinism-gap study run cuML, LightGBM-GPU, or both — and how on Windows?**
   - What we know: cuML is Linux-first; LightGBM GPU needs a specific Windows build; the benchmark host is Windows (carried Phase-5 OQ7).
   - Recommendation: Attempt **LightGBM-GPU on Windows** (more likely to install) and **cuML under WSL2** if available; gate both behind a `checkpoint:human-verify`. If neither runs, **document the gap from published evidence + "not reproduced on host"** rather than faking a result. The Phase-6 gates do not depend on this.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU (sm_89) | DET-01 GPU byte-compare gate, DET-02 overhead bench, dispatch `cuda` path | ✓ | RTX 4060 Ti, driver 595.79 | none (CUDA-only MVP) — but DET-03 *tests* run GPU-less via monkeypatched probe |
| CUDA Toolkit + NVRTC | GPU fit + `compute-sanitizer racecheck` (non-determinism canary) | ✓ | 12.8 | none |
| compute-sanitizer | racecheck/memcheck on the deterministic path | ✓ | CUDA 12.8 | none — required for the determinism canary |
| cudarc 0.19.8 | `CudaContext::new(0)` probe + stream control | ✓ | pinned | none |
| Rust stable + MSVC v143 | Build sylva-core/sylva-cuda + abi3 wheel | ✓ | rustc 1.96.0, cl.exe 14.44 | none |
| scikit-learn | `check_estimator` still-green after new params | ✓ | 1.9.0 (.venv-parity) | none — the API gate needs it |
| cuML (RAPIDS) | Qualitative determinism-gap ref ONLY | ✗/uncertain on native Windows | pin in study | WSL2, or documented "not available on host" (OQ6) |
| LightGBM (GPU) | Qualitative determinism-gap ref ONLY | uncertain (needs GPU build) | pin in study | documented "not available on host" (OQ6) |

**Missing dependencies with no fallback:** none that block the DET-01..04 gates (those need only the proven CUDA toolchain + sklearn, all present; DET-03 even runs GPU-less via a monkeypatched probe).
**Missing dependencies with fallback:** cuML / LightGBM-GPU for the *qualitative* gap (fallback: WSL2 or honest "unavailable" — the Phase-6 gates do not depend on them).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `#[test]`/`tests/` (cargo-nextest) for the report struct, dispatch decision fn, and GPU byte-compare; Python pytest for dispatch RAISE, `execution_report_` assertions, `check_estimator`; Python bench for overhead |
| Config file | none for Rust (cargo built-in); Python mirrors `python/tests/` + `python/benchmarks/` |
| Quick run command | `cargo test -p sylva-core dispatch report` + `cargo test -p sylva-cuda --test deterministic_cpu_gpu` |
| Full suite command | `cargo test -p sylva-core` + `cargo test -p sylva-cuda` + `pytest python/tests/` + `python python/benchmarks/determinism_overhead.py` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DET-01 | Two same-seed `deterministic=True` GPU fits → byte-identical serialized `ForestIR` (clf+reg, ET+RF), exact comparison NOT allclose | integration (byte-compare) | `cargo test -p sylva-cuda --test deterministic_cpu_gpu` | ❌ Wave 0 |
| DET-01 | No data race on the deterministic hot path (non-determinism canary) | sanitizer | `compute-sanitizer --tool racecheck <forest-fit exe>` | ❌ Wave 0 |
| DET-01 | (Python) two same-seed deterministic fits via the estimator → byte-identical model bytes | Python | `pytest python/tests/gpu_determinism/test_deterministic_byte_identical.py` | ❌ Wave 0 |
| DET-02 | Sylva deterministic vs non-deterministic fit-time overhead (%) measured + reported (Sylva-vs-Sylva; ~95–98% retention target) | bench (report) | `python python/benchmarks/determinism_overhead.py` | ❌ Wave 0 |
| DET-02 | (qualitative) cuML/LightGBM-GPU two same-seed fits differ byte-for-byte (gap documented, no speed claim) | study (report) | same bench, qualitative section (gated human-verify) | ❌ Wave 0 |
| DET-03 | `device="cuda"` + no usable GPU + `fallback="error"` → typed RAISE (no silent CPU) | unit (Rust) + Python | `cargo test -p sylva-core resolve_backend` + `pytest test_dispatch.py::test_cuda_requested_but_unavailable_raises` | ❌ Wave 0 |
| DET-03 | `device="auto"` selects cuda-if-present-else-cpu; `device="cpu"` forces cpu; reason recorded | unit (Rust) + Python | `cargo test -p sylva-core resolve_backend` + `pytest test_dispatch.py` | ❌ Wave 0 |
| DET-03 | `check_estimator` still green with new `device`/`fallback`/`deterministic` params | Python (parametrized) | `pytest python/tests/test_check_estimator.py` | ✅ (Phase 5) extend |
| DET-04 | `execution_report_` has selected_backend + reason, fallback_status, deterministic flag | Python + Rust | `pytest test_execution_report.py` + `cargo test -p sylva-core report` | ❌ Wave 0 |
| DET-04 | Report lists every input conversion (dtype/layout/contiguity) with bytes_copied; bytes_h2d/d2h tallied | Python | `pytest test_execution_report.py::test_report_records_dtype_and_layout_conversion` | ❌ Wave 0 |
| DET-04 | `ExecutionReport` serde round-trip is identity (extends the shipped QuantizeReport test) | unit (Rust) | `cargo test -p sylva-core report::serde_round_trip` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the touched gate — `cargo test -p sylva-core dispatch report` (decision + report) or `cargo test -p sylva-cuda --test deterministic_cpu_gpu` (DET-01); for Python tasks, `pytest python/tests/test_dispatch.py -x` or `test_execution_report.py -x`.
- **Per wave merge:** full `cargo test -p sylva-core` + `cargo test -p sylva-cuda` + `pytest python/tests/` (incl. `check_estimator` regression) + `racecheck` on a deterministic forest fit.
- **Phase gate:** DET-01 two-run byte-identical (clf+reg, ET+RF) green under `deterministic=True` AND deterministic path `racecheck`-clean AND DET-03 RAISE-on-unmet-cuda + auto/cpu selection tests green AND `check_estimator` still green with the new params AND DET-04 report-field assertions + serde round-trip green AND DET-02 overhead measured/reported (Sylva-vs-Sylva) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/sylva-core/src/report.rs` + tests — `ExecutionReport`/`InputConversion` (extends `QuantizeReport`); serde round-trip test (DET-04)
- [ ] `crates/sylva-core/src/dispatch.rs` + tests — `resolve_backend` pure decision fn (DET-03)
- [ ] `crates/sylva-core/src/error.rs` — `DeviceUnavailable`/`UnsupportedConfig` variants (DET-03)
- [ ] `crates/sylva-core/src/config.rs` — `deterministic: bool` field + validate (DET-01)
- [ ] `crates/sylva-cuda/src/cuda_backend/availability.rs` — `cuda_available()` probe (DET-03)
- [ ] `crates/sylva-cuda/src/cuda_backend/{scheduler.rs, arena.rs}` — deterministic ordering under `deterministic=true` (DET-01)
- [ ] `crates/sylva-cuda/tests/deterministic_cpu_gpu.rs` — GPU two-run byte-identical gate (DET-01)
- [ ] `python/sylva/_dispatch.py` (extend Phase-5) — full `device`/`fallback="error"` RAISE + reason recording (DET-03)
- [ ] `python/sylva/_base.py` (extend) — store new params verbatim; expose `execution_report_` (DET-03/04)
- [ ] `python/tests/{test_dispatch.py, test_execution_report.py}` + `gpu_determinism/` — DET-03/04/01 Python gates
- [ ] `python/benchmarks/determinism_overhead.py` — DET-02 Sylva-vs-Sylva overhead + qualitative external gap
- [ ] A "remaining non-determinism audit" task (read shipped 4/5 GPU code; enumerate streams/arena/reductions) — OQ1, front-loaded

*(No existing dispatch/report/determinism-mode infrastructure on the GPU path. The shipped `QuantizeReport`, `tests/determinism.rs` (CPU byte-compare), and the `cuda_error_to_pyerr`/`sylva_error_to_pyerr` mappers are the templates. The Phase-5 estimator classes + `check_estimator` harness are extended, not created.)*

## Security Domain

> `security_enforcement` is enabled (absent/true in config = enabled, ASVS level 1). Local GPU compute library + a Python FFI surface; relevant controls are input validation at the dispatch boundary, memory safety of any deterministic-scheduling changes, and not leaking environment detail through the report.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (local library). |
| V3 Session Management | no | N/A. |
| V4 Access Control | no | N/A. |
| V5 Input Validation | yes | Validate `device` ∈ {auto,cuda,cpu}, `fallback` ∈ {error} (MVP), `deterministic` ∈ {bool} at the estimator boundary → typed `ValueError`/`UnsupportedConfig` *before* any device work. An unmet `device="cuda"` is a typed RAISE, not an OOB or a silent degrade. |
| V6 Cryptography | no | Philox is non-cryptographic; never used for secrets. The report contains no secrets. |

### Known Threat Patterns for Rust core + CUDA-C via NVRTC + PyO3 boundary
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Silent CUDA error swallow → wrong backend / silent CPU fallback | Tampering | Dispatch decided once before fit; CUDA errors `?`-propagated to typed RAISE; `fallback="error"` is the only MVP policy; test asserts RAISE on unmet cuda. |
| Use-after-free / race from deterministic-scheduling changes to the arena/streams | Tampering | Confine stream/arena changes; `compute-sanitizer` memcheck + racecheck on a deterministic forest fit; racecheck doubles as the non-determinism canary. |
| Information disclosure via `execution_report_` (leaking absolute paths, driver internals, env) | Info-disclosure | The report records *categories* (dtype, layout, byte counts, backend, reason) — never absolute filesystem paths, env vars, or driver build strings. Review the `selection_reason` strings to ensure they describe the decision, not the host. |
| Untrusted `device`/`fallback`/`deterministic` values | Tampering | Whitelist-validate at the boundary → typed error; never `eval`/dynamic-dispatch on the raw string. |
| License contamination (copying cuML/LightGBM determinism or dispatch code while studying the gap) | (Legal/IP) | Apache-2.0 discipline: the determinism mechanism is reimplemented from the integer-accumulation algorithm; the external study only *observes* cuML/LightGBM behavior (run + compare bytes), never copies their source. |

## Project Constraints (from CLAUDE.md)

- **No silent fallback** — `device="cuda"` unmet RAISES a typed error; every cudarc/dispatch call is a `Result`/typed `PyErr`; no `.unwrap()`/`.expect()` across FFI; the dispatch decision is explicit, never a catch-and-degrade. (The project's headline differentiator.)
- **cudarc 0.19.8 + hand-written CUDA-C via NVRTC** — the only sanctioned kernel path; deterministic-mode changes are to *scheduling/arena ordering*, not a new kernel toolchain. No CubeCL/Rust-CUDA/nvcc/wgpu.
- **Native Windows / MSVC, no WSL** for the build (WSL allowed only as the cuML qualitative-gap lane — OQ6).
- **`-fmad=false`, never `--use_fast_math`** on any compiled path (carried from Phase 3/4; determinism depends on it).
- **Integer/deterministic accumulation** — designed in from Phase 4, **hardened here**; sibling subtraction + `sample_weight` stay integer/fixed-point.
- **Apache-2.0 reimplementation discipline** — never copy cuML/LightGBM/XGBoost/sklearn/Snap-ML source, including while studying the determinism gap.
- **Stable Rust 1.83+** (on 1.96.0), **`-D warnings`** clippy bar, non-deprecated cudarc APIs.
- **PyO3 0.29 abi3-py310 + maturin**; new estimator params stored verbatim (no `__init__` logic; `check_estimator`-clean).
- **Many small files** (200–400 lines, 800 max); organize by domain (`report.rs`, `dispatch.rs`, `availability.rs`).
- **Comparative-study fairness binding** — Phase 6's study is **Sylva-vs-Sylva** overhead + a *qualitative* external gap; **NO cross-library speed claim** (Phase 7 owns speed).

## Sources

### Primary (HIGH confidence)
- `crates/sylva-core/src/quantize/report.rs` — the shipped `QuantizeReport` (the explicit Phase-6 `execution_report_` extension point: `input_dtype`, `input_c_contiguous`, `binned_bytes`, `edges_bytes`, `h2d_executed`, `h2d_note`).
- `crates/sylva-core/tests/determinism.rs` — the shipped byte-identical gate idiom (serde_json string equality of two same-seed `ForestIR`s; the 1-thread-pool order-independence test) — the DET-01 template.
- `crates/sylva-core/src/{error.rs,config.rs,pyseam.rs}` — `SylvaError` (to extend), `TrainConfig` (no `deterministic` yet), the test-only PyO3 seam + `sylva_error_to_pyerr` no-silent-fallback mapping.
- `crates/sylva-cuda/src/{lib.rs,nvrtc_launch.rs}` — `CudaError`, `cuda_error_to_pyerr`, the cudarc `Result`-everywhere / no-`.unwrap()` device pattern, `CudaContext::new` init (the availability-probe substrate).
- `.planning/{ROADMAP.md,REQUIREMENTS.md,STATE.md,PROJECT.md}` — Phase-6 goal + 6 success criteria, DET-01..04, the no-silent-fallback differentiator, the integer-accumulation "hardened in Phase 6" decision, the Sylva-vs-Sylva study + binding fairness note.
- `.planning/phases/04-single-gpu-extratree/04-RESEARCH.md` + `.../05-...RESEARCH.md` — the locked determinism substrate (integer histograms, host-side scoring, `-fmad=false`, fixed tie-break/scan order, integer-only sibling subtraction, fixed-point `sample_weight`) Phase 6 enforces.
- `.claude/CLAUDE.md` — kernel-authoring constraints, Apache-2.0 discipline, no-silent-fallback, `-fmad=false`, Windows/MSVC.

### Secondary (MEDIUM confidence)
- cudarc 0.19.8 docs — `CudaContext::new` returns `Result` (clean availability probe); safe alloc is synchronous (fine for deterministic arena). https://docs.rs/cudarc/latest/cudarc/driver/
- scikit-learn 1.9 estimator-checks — new constructor params must be stored verbatim / clone-able; `expected_failed_checks` for any documented exception. https://scikit-learn.org/stable/developers/develop.html
- NVIDIA determinism guidance — float `atomicAdd` and warp-shuffle reductions are run-to-run nondeterministic; integer atomics are associative/order-free (the basis for integer-histogram determinism). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### Tertiary (LOW confidence)
- General knowledge that cuML RF / LightGBM-GPU are not byte-reproducible under same-seed runs (PROJECT.md ecosystem note + common GPU-tree-builder behavior) — `[ASSUMED]`; to be *empirically* confirmed in the qualitative study, or reported as "not reproduced on host."
- The ~95–98% throughput-retention target (ROADMAP) as the realistic determinism-overhead band — `[ASSUMED]` until measured (DET-02); the actual number is whatever the Sylva-vs-Sylva bench reports.

## Metadata

**Confidence breakdown:**
- Dispatch contract (DET-03): HIGH — the typed-error spine + cudarc `Result` probe are shipped; the decision fn is a pure function over a probe.
- `execution_report_` (DET-04): HIGH — extends the shipped `QuantizeReport` with a serde-tested superset.
- Determinism mechanism (DET-01): MEDIUM — the substrate (integer accumulation, fixed ordering) is locked upstream, but *which* GPU non-determinism sources survive 4/5 depends on those phases landing as designed (OQ1); the audit is front-loaded for exactly this reason.
- Overhead measurement (DET-02): MEDIUM — depends on whether a non-deterministic fast mode exists to measure against (OQ2); the bench is straightforward either way (measure, or document overhead = 0 with reason).

**Research date:** 2026-06-27
**Valid until:** ~2026-07-27 (stable — toolchain + sklearn pinned; the volatile elements are the two scoping forks OQ1/OQ2, which are design decisions resolved by the orchestrator, not moving external facts). Note: this research is written against Phase 4/5 *plans* (not yet executed); re-validate the remaining-non-determinism inventory against the shipped 4/5 GPU code when Phase 6 begins.
