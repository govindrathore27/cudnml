# Phase 2 Discussion Log

**Date:** 2026-06-20
**Mode:** discuss (default, interactive)
**Phase:** 2 — CPU Oracle, Contracts & Forest IR

Human-reference audit trail of the discussion. Not consumed by downstream agents — see `02-CONTEXT.md` for the decisions that flow into research/planning.

## Gray areas presented (user selected all four)

1. NaN / missing-value routing
2. CpuBackend scope this phase
3. ForestIR design horizon
4. sklearn parity bar strictness

(Plus a fifth surfaced mid-discussion: numeric precision.)

## Decisions

### NaN / missing-value routing (ENG-05)
- **Options:** Simple deterministic default-direction / Learned (XGBoost-style) default-direction / sklearn-strict reject.
- **Selected:** **Simple deterministic default-direction** (higher-sample-count child → `default-child`; tie→left). Learned direction deferred.

### CpuBackend scope (ENG-03, MVP)
- **Options:** ET-first MVP slice / Both ET + RF this phase.
- **Selected:** **Both ET + RF this phase** (full SC-2).

### ForestIR design horizon (ENG-02)
- **Options:** Design for known consumers now / Minimal-for-training, evolve later.
- **Selected:** **Design for known consumers** (carry tree-SHAP cover/sample counts + Treelite-export-compatible fields).

### sklearn parity bar (ENG-04 / EST-07 / SC-6)
- **Options:** Strict distributional (accuracy CI + KS on split stats) / Pragmatic (accuracy/proba CI only).
- **Selected:** **Strict distributional**.

### Numeric precision (surfaced follow-up)
- **Options:** f32 end-to-end / f64 CPU oracle + f32 GPU.
- **Selected:** **f32 end-to-end** (enables Phase-4 CPU==GPU bit-exact; sklearn parity stays distributional).

## Handed to research / planner (not user decisions)
- RF split-finding must mirror sklearn's candidate-threshold best-split for parity.
- Exact parity thresholds: CI width, KS p-value, tree count.
- Exact `ForestIR` field set for SHAP/export forward-design.
- Philox-4×32-10 test vectors documented for Phase-4 bit-match.

## Deferred ideas
- Learned (impurity-optimal) default-direction — later enhancement.
- `sample_weight` + full estimator API (EST-02/05) — Phase 5.
- Quantizer (QUANT-*) — Phase 3; auto CPU dispatch / `execution_report_` (DET-*) — Phase 6.
- GPU work (GPU-*) — Phase 4; SHAP — Phase 8; Treelite export — Phase 6.

## Scope creep
- None — discussion stayed within Phase 2 scope.
