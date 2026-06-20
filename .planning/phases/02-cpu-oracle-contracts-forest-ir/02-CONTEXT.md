# Phase 2: CPU Oracle, Contracts & Forest IR - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up the **device-neutral contracts** and a **trusted pure-Rust CPU backend** that trains and predicts ExtraTrees + RandomForest correctly — the bit-level **correctness oracle** that makes every later GPU result verifiable and enables GPU-less cloud CI. Concretely (ENG-01..06, EST-07):

1. A device-neutral `trait Backend` (quantize / build_histograms / eval_splits / partition / predict) with **no CUDA types crossing the trait boundary** (ENG-01).
2. A SoA `ForestIR` (feature_id / threshold / left / right / default-child / leaf-value arrays) as the **single shared representation** — written by training, read read-only by inference, SHAP, and export (ENG-02).
3. A pure-Rust `CpuBackend` (`ndarray` + `rayon`) that trains + predicts **ET and RF** correctly, serving as the differential-test oracle and the `device="cpu"` / small-data path (ENG-03).
4. The documented **parity contract** (ENG-04) + a stateless **Philox-4×32-10** RNG in Rust keyed by `(seed, tree, node, feature, draw)` (ENG-06).
5. A defined, consistently-implemented **NaN / missing-value routing policy** (ENG-05).
6. **Differential tests vs scikit-learn** + property-based invariants pass (EST-07); the Comparative Baseline Study confirms accuracy/distribution **PARITY** with sklearn ET/RF (like-for-like).

**MVP mode.** This is the contracts + CPU-oracle layer. **No GPU/CUDA code, no quantizer (Phase 3), no estimators-API surface (Phase 5), no SHAP/export.** The throwaway Phase-1 spike kernels (`sylva-cuda` `run_vector_add`/`run_histogram`) are NOT consumed here — Phase 2's logic lives in a new device-neutral crate.

</domain>

<decisions>
## Implementation Decisions

### NaN / missing-value routing (USER-DECIDED)
- **D-01:** **Simple deterministic default-direction.** At each split, missing/NaN rows route to a deterministic default child — the **higher-sample-count child** — recorded in the `ForestIR` `default-child` array; **tie → left child** (deterministic, Claude's discretion). Cheap, deterministic, and trivially **CPU/GPU bit-matchable** (ENG-05 requires consistency across CPU and GPU). scikit-learn's trees have no missing-value story, so the differential tests run on clean (non-NaN) data and no parity is broken; the default-direction path is exercised by dedicated NaN fixtures (SC-4). Learned/impurity-optimal default-direction (XGBoost-style) is explicitly **deferred** as a later enhancement, not built now.

### CpuBackend scope this phase (USER-DECIDED)
- **D-02:** **Both ExtraTrees AND RandomForest this phase** (classifier + regressor for each). RF brings bootstrap resampling + best-split (impurity-search over candidate thresholds) vs ET's random thresholds; both get full differential + property-based parity coverage. Honors the project's "single tree before forest" within each (build/verify the single-tree path first, then the forest), but does **not** slice RF out to a later phase — SC-2 is met in full this phase.

### ForestIR design horizon (USER-DECIDED)
- **D-03:** **Design the SoA `ForestIR` now for all known downstream consumers**, not minimal-for-training. The arrays carry, in addition to the training/predict essentials (feature_id / threshold / left / right / default-child / leaf-value), the fields that **tree-SHAP (Phase 8)** needs (per-node **sample/cover counts**) and **Treelite-export (Phase 6)** compatibility in mind. Rationale: STATE flags the CPU↔GPU parity contract + IR as a **near-rewrite risk if under-designed**; the IR is the single shared representation every later phase reads, so a small upfront design cost avoids a cross-backend rewrite. (Exact field set → research, see flagged items.)

### sklearn parity bar (USER-DECIDED)
- **D-04:** **Strict distributional parity.** The differential-test gate (EST-07 / SC-6) requires BOTH (a) accuracy / predicted-probability agreement within a **tight CI**, AND (b) a **KS test on aggregate split statistics** (feature-selection frequency, threshold distribution) across many trees. ExtraTrees splits are random, so this is **distributional** equivalence, never per-tree structural match (consistent with ENG-04: Sylva's own bit-identical CPU↔GPU RNG + *distributional* equivalence to sklearn, NOT bit-replay of sklearn's serial PRNG). The oracle's whole value is trustworthiness, so the bar is strict. (Exact CI width / KS p-value / tree count → research.)

### Numeric precision (USER-DECIDED)
- **D-05:** **float32 end-to-end.** `CpuBackend` and `ForestIR` compute and store in **f32**, matching the future GPU path so the **Phase-4 `GPU == CPU oracle` bit-exact gate** is achievable. Parity to scikit-learn (which computes thresholds in f64) stays **distributional** — the last-bit f32/f64 differences are absorbed by the D-04 CI/KS bar. Consistent with PROJECT.md "dense float32 only" for the MVP.

### Claude's Discretion (delegated / flagged for research)
- **RNG / determinism:** implement **Philox-4×32-10** in Rust now with **documented test vectors** that Phase-4's CUDA copy must bit-match. Per-tree RNG keyed by `(seed, tree, node, feature, draw)` makes `rayon` tree-parallelism order-independent → deterministic regardless of thread scheduling (no reliance on reduction order).
- **`device="cpu"` dispatch:** this phase exposes the `CpuBackend` as the **explicit `device="cpu"`** path + the differential oracle only. Auto small-data dispatch, the `fallback="error"` contract, and `execution_report_` are **Phase 6 (DET)** — not built here; no silent fallback.
- **RF split-finding** must mirror scikit-learn's candidate-threshold best-split search closely enough to satisfy the D-04 distributional bar — researcher to confirm the exact algorithm alignment (and whether RF here is exact/sort-based, since there is no quantizer until Phase 3).
- **Exact parity thresholds** (CI width, KS p-value, number of trees) — researcher picks defensible statistical values.
- **Exact `ForestIR` field set** for SHAP/export forward-design — researcher reads ARCHITECTURE.md + Treelite 4.x JSON schema + tree-SHAP requirements (per-node cover/sample counts) to pin the array set.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 2: CPU Oracle, Contracts & Forest IR" — goal, 7 success criteria, the Comparative Baseline Study (sklearn ET/RF parity, dataset shapes, pass bar), Mode: mvp.
- `.planning/REQUIREMENTS.md` — ENG-01, ENG-02, ENG-03, ENG-04, ENG-05, ENG-06, EST-07 (the 7 requirements this phase satisfies).
- `.planning/PROJECT.md` — constraints: sklearn semantics + determinism, no silent fallback, dense float32 MVP, Apache-2.0 (reimplement from algorithm; never copy GPL/sklearn source).
- `.planning/STATE.md` — Blockers/Concerns: the four architecture decisions flagged as near-rewrite-if-deferred (breadth-first level-at-a-time build → Ph4, privatized histograms → Ph4, integer/deterministic accumulation, **CPU↔GPU parity contract → Ph2**); binding comparative-study fairness protocol.

### Architecture & technical research (the contracts this phase locks)
- `.planning/research/ARCHITECTURE.md` — `trait Backend` shape, SoA `ForestIR` design, the parity contract, and the Phase-4 privatized-histogram layout the IR/backend must stay compatible with.
- `.planning/research/PITFALLS.md` — comparative-study fairness (Pitfalls 1, 2, 13); determinism / accumulation-order pitfalls; integer-vs-float accumulation.
- `.planning/research/STACK.md` — `ndarray` 0.16 + `rayon` 1.x for the CPU backend; `proptest` / `approx` for invariant + tolerance tests.
- `.planning/research/SUMMARY.md` — research synthesis; honesty/fairness calibration.
- `.claude/CLAUDE.md` — Technology Stack (ndarray/rayon roles, Philox prescription, thiserror error enums), coding style (many small files, no hardcoded values, f32 MVP).

### Phase-1 foundation this builds on
- `.planning/phases/01-toolchain-spike-gate-1/01-01-SUMMARY.md` + `VERSIONS.md` — the persisted Cargo workspace + maturin/pyproject skeleton + pinned toolchain the new device-neutral crate slots into; the `thiserror`/no-`.unwrap()` error pattern established in `sylva-cuda`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Cargo workspace skeleton** (`Cargo.toml` `[workspace]`, `[workspace.package]` license/MSRV, `rust-toolchain.toml`) — Phase 2 adds a new **device-neutral crate** (e.g. `crates/sylva-core`) as a second workspace member holding `trait Backend`, `ForestIR`, `CpuBackend`, and Philox. **No CUDA deps** in this crate (ENG-01).
- **maturin/pyproject + `#[pymodule]` seam** in `crates/sylva-cuda/src/lib.rs` — the PyO3 export surface exists; Phase-2 estimator API is Phase 5, so the seam stays as-is for now.
- **Error pattern:** `sylva-cuda`'s `thiserror` `CudaError` + Results-not-`.unwrap()` convention is the template for a core error enum.

### Established Patterns
- f32 numerics (D-05); many small files (200–400 lines); no hardcoded values; deterministic-by-construction via Philox keying; honest/non-silent error surfacing.

### Integration Points
- `trait Backend` is implemented by `CpuBackend` now and `CudaBackend` (Phase 4) later — keep CUDA types out of the trait (ENG-01).
- `ForestIR` is written by training and read read-only by inference (this phase), SHAP (Ph8), and Treelite export (Ph6) — design-for-known-consumers (D-03).
- `ndarray` 0.16 + `rayon` 1.x become first real CPU-backend deps (Phase 1 deliberately omitted them per D-03 of Phase 1).

</code_context>

<specifics>
## Specific Ideas

- **f32 end-to-end** (D-05) is the load-bearing precision decision — it makes the Phase-4 CPU==GPU bit-exact gate reachable.
- **Philox-4×32-10** must ship with **documented test vectors** in this phase so Phase-4's CUDA reimplementation can be proven bit-identical against them.
- Comparative Baseline Study dataset (from ROADMAP): `make_classification` (~20k×50) and/or a Covertype subset, fixed seed, identical hyperparameters across Sylva and sklearn; ET-vs-ET and RF-vs-RF only (never crossed).
- NaN test fixtures are required in the suite (SC-4) to exercise the D-01 default-direction path.

</specifics>

<deferred>
## Deferred Ideas

- **Learned (impurity-optimal) default-direction** for missing values — XGBoost-style; D-01 ships the simple deterministic version now, learned direction is a later enhancement.
- **`sample_weight`** end-to-end (EST-05) and the full estimator API surface (EST-02, `fit`/`predict_proba`/`check_estimator` parity) — **Phase 5**.
- **Quantizer / binning** (QUANT-01/02) — **Phase 3**; RF best-split in Phase 2 operates without a quantizer.
- **Auto small-data CPU dispatch, `fallback="error"`, `execution_report_`** (DET-*) — **Phase 6**.
- **All GPU/CUDA work** (GPU-*), including the privatized-histogram kernel the IR stays compatible with — **Phase 4**.
- **SHAP** (Phase 8) and **Treelite export** (Phase 6) — the IR is *designed for* them now (D-03) but neither is implemented this phase.

</deferred>

---

*Phase: 2-CPU Oracle, Contracts & Forest IR*
*Context gathered: 2026-06-20*
