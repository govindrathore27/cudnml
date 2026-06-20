---
phase: 02-cpu-oracle-contracts-forest-ir
plan: 01
subsystem: infra
tags: [rust, sylva-core, trait-backend, forest-ir, philox, rng, serde, ndarray]

requires:
  - phase: 01-toolchain-spike-gate-1
    provides: Cargo workspace + thiserror error-enum pattern the new crate extends
provides:
  - New device-neutral crate crates/sylva-core (NO cudarc, NO pyo3 — ENG-01)
  - Backend{fit,predict} + HistogramBackend{quantize,build_histograms,eval_splits,partition} traits (CUDA-free)
  - SoA ForestIR (f32, consumer-complete incl. SHAP-cover + Treelite fields) with serde round-trip + structural validation
  - Philox-4x32-10 RNG keyed (seed,tree,node,feature,draw) bit-matching verified Random123 KAT vectors
  - TrainConfig / Criterion / MaxFeatures / Algo / Task / SylvaError
affects: [02-02 ExtraTrees, 02-03 RandomForest, 02-04 invariants, 02-05 parity, phase-4 CudaBackend, phase-8 SHAP, phase-6 export]

tech-stack:
  added: [ndarray 0.16.1, rayon 1.12.0, serde 1.0.228, serde_json 1.0.150, thiserror 1.0.69, proptest 1.11.0, approx 0.5.1]
  patterns: [SoA ForestIR, two-trait device-neutral seam (assoc type Bins), counter-based RNG keyed by coordinate, thiserror typed errors no .unwrap]

key-files:
  created:
    - crates/sylva-core/Cargo.toml
    - crates/sylva-core/src/lib.rs
    - crates/sylva-core/src/error.rs
    - crates/sylva-core/src/config.rs
    - crates/sylva-core/src/backend.rs
    - crates/sylva-core/src/ir.rs
    - crates/sylva-core/src/rng/mod.rs
    - crates/sylva-core/src/rng/kat.rs
  modified:
    - Cargo.toml

key-decisions:
  - "Adopted RESEARCH Option A: Backend{fit,predict} + separate HistogramBackend{quantize,build_histograms,eval_splits,partition} with assoc `type Bins` — all 5 ENG-01 op-names in the contract surface, no CUDA types crossing"
  - "Philox key schedule: round 0 uses the original key; rounds 1..9 bump-then-round (Random123 philox4x32_R) — reproduces the published KATs"
  - "KAT vectors confirmed authoritative: a from-spec Philox reproduces all 3 canonical Random123 philox4x32x10 vectors (mutual validation)"

patterns-established:
  - "ForestIR is the single SoA shared representation; forward-design fields (node_sample_count/node_weighted_count/impurity) carried now for SHAP cover + Treelite export (D-03)"
  - "f32 end-to-end in the IR (D-05) — keeps the Phase-4 GPU==CPU bit-exact gate reachable"
  - "Counter-based RNG keyed by (seed,tree,node,feature,draw) → tree-parallelism is order-independent; Phase-4 CUDA copy bit-verifiable vs frozen KATs"

requirements-completed: [ENG-01, ENG-02, ENG-06]

duration: ~20min
completed: 2026-06-20
status: complete
---

# Phase 2 / Plan 02-01: sylva-core contracts + ForestIR + Philox Summary

**A device-neutral, CUDA-free `sylva-core` crate: the `Backend`/`HistogramBackend` trait seam (ENG-01), the consumer-complete SoA `ForestIR` (ENG-02), and a Philox-4×32-10 RNG that bit-matches the verified canonical Random123 KAT vectors (ENG-06).**

## Performance
- **Duration:** ~20 min (completed inline after two subagent dispatches died on the session limit)
- **Completed:** 2026-06-20
- **Tasks:** 4 (Tasks 1–3 auto + Task 4 KAT checkpoint)
- **Files created:** 8 (+1 modified)

## Accomplishments
- New `crates/sylva-core` workspace member that builds + `cargo clippy -D warnings` clean; **`cargo tree -p sylva-core` shows 0 cudarc and 0 pyo3 entries** (ENG-01 enforced structurally).
- `trait Backend {fit,predict}` + `trait HistogramBackend {quantize,build_histograms,eval_splits,partition}` with associated `type Bins` — all five ENG-01 device-op names exist CUDA-free; documented why no device type crosses.
- SoA `ForestIR` (f32, D-05) carrying every train/predict field PLUS the D-03 forward-design fields (`node_sample_count`, `node_weighted_count` = SHAP cover / Treelite `data_count`/`sum_hess`; `impurity` = export gain). serde round-trip + `validate_structure` invariants pass.
- Philox-4×32-10 keyed `(seed,tree,node,feature,draw)` + `u32_to_unit_f32` + `pack_counter` + `philox_uniform`. **9/9 tests pass**, incl. all three KAT vectors.
- `TrainConfig::validate` rejects bad hyperparameters with a typed `SylvaError` (no panic).

## Task Commits
1. **Tasks 1–4 (crate + contracts + IR + Philox/KAT)** — `d8f8b31` (feat)
2. **SUMMARY + tracking** — this commit (docs)

_The four plan tasks (crate/traits, ForestIR, Philox, KAT verify) were committed as one cohesive crate commit because the contract files are mutually referential (`backend.rs` imports `ForestIR`; `lib.rs` declares all modules) — an intermediate per-task commit would not compile. Each commit builds; clippy-clean; 9 tests pass._

## Task 4 — Philox KAT verification (the human-verify checkpoint)
Research flagged the three literal KAT output triples as `[ASSUMED]` (canonical `kat_vectors.txt` was unreachable). Resolution: the orchestrator supplied the three canonical Random123 `philox4x32×10` vectors (all-zero, all-ones, π-mixed), and a **from-spec** Philox implementation (verified constants M0=0xD2511F53, M1=0xCD9E8D57, W0=0x9E3779B9, W1=0xBB67AE85; 10 rounds; round-0-unbumped schedule) **reproduces all three exactly**. A wrong implementation would not match published vectors, and wrong vectors would not match a correct implementation — the mutual match confirms both. The vectors are frozen in `rng/kat.rs` with a provenance comment and are the Phase-4 bit-match oracle. **Checkpoint satisfied.**

## Files Created/Modified
- `Cargo.toml` — added `crates/sylva-core` workspace member
- `crates/sylva-core/Cargo.toml` — ndarray/rayon/serde/serde_json/thiserror; dev proptest/approx; NO cudarc/pyo3
- `src/error.rs` — `SylvaError` thiserror enum
- `src/config.rs` — `TrainConfig` + `Criterion`/`MaxFeatures`/`Algo`/`Task` + boundary validation
- `src/backend.rs` — `Backend` + `HistogramBackend` traits + `Predictions` + device-neutral placeholder handles
- `src/ir.rs` — SoA `ForestIR` + `validate_structure` + serde round-trip tests
- `src/rng/mod.rs` — Philox-4×32-10 + conversions + KAT/range/injectivity/determinism tests
- `src/rng/kat.rs` — the three frozen, verified KAT vectors
- `src/lib.rs` — module wiring + public re-exports

## Decisions Made
- Two-trait split (Option A) over a single 5-method trait — keeps the CPU oracle's recursive `fit`/`predict` clean while still naming the GPU histogram ops in the contract; the associated `type Bins` is the device-neutrality seam.
- Single cohesive crate commit (see Task Commits note) for buildability.

## Deviations from Plan
- **Committed Tasks 1–4 together** rather than four atomic commits — the contract files are mutually referential and an intermediate commit would not compile. No scope change; all acceptance criteria met.
- **Executed inline** (not via gsd-executor) — two subagent dispatches returned 0 tokens on the session limit; the workflow's documented inline fallback after a terminal dispatch failure.

## Issues Encountered
- One clippy `manual_range_contains` lint in a test → rewrote as `(0.0..1.0).contains(&x)`. Now clean.
- Stale session PATH — `cargo` invoked with `$HOME/.cargo/bin` prepended per command.

## Next Phase Readiness
- **Wave 2 (02-02) ready:** `Backend` trait + `ForestIR` + `TrainConfig` + `philox_uniform` are the imports the ExtraTrees `CpuBackend` builds on. f32 IR + per-tree RNG keying are in place for the deterministic recursive builder.

---
*Phase: 02-cpu-oracle-contracts-forest-ir*
*Completed: 2026-06-20*
