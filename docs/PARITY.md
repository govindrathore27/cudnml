# Sylva Parity Contract

**Requirement:** ENG-04
**Phase:** 2 — CPU Oracle, Contracts & Forest IR
**Status:** Locked (Phase 2 deliverable)

This document defines what "parity" means for Sylva and what it explicitly does not mean.
No vague "compatible" claims are made here. Each point distinguishes exact bit-level
guarantees from distributional statistical guarantees.

---

## Point 1 — Sylva-internal bit-exact reproducibility (EXACT)

For any fixed `(seed, TrainConfig, X, y)` tuple:

- Two calls to `CpuBackend::fit` produce a **byte-identical** serialized `ForestIR`.
  Byte-identical means `serde_json::to_string(&ir_a) == serde_json::to_string(&ir_b)` —
  not approximate, not structurally similar: every byte of JSON is the same.
- This holds regardless of whether rayon builds trees in parallel or in serial order,
  because all random draws use a **stateless counter-based Philox-4x32-10 RNG** keyed by
  `(seed, tree_id, node_id, feature_id, draw_counter)`. There is no shared mutable RNG
  state between trees.

**This is the bar the Phase-4 GPU path is held to.** When `CudaBackend::fit` is
implemented, it must produce a byte-identical serialized `ForestIR` for any seed that
`CpuBackend` was given — not merely similar accuracy, but bit-exact reconstruction.

**Tested by:** `tests/determinism.rs` — 12 tests asserting exact string equality for all
four estimators (ExtraTrees/RandomForest x clf/reg).

---

## Point 2 — CPU-to-GPU RNG identity (EXACT, enabled Phase 4)

Sylva's randomness is defined by documented constants, counter layout, and conversion:

**Algorithm:** Philox-4x32-10 (Salmon et al., Random123; reimplemented from the
DEShawResearch reference header, Apache-2.0).

**Constants (frozen — Phase 4 reproduces these bit-for-bit in CUDA):**

| Constant | Value | Role |
|----------|-------|------|
| `PHILOX_M0` | `0xD2511F53` | multiplier for counter word 0 |
| `PHILOX_M1` | `0xCD9E8D57` | multiplier for counter word 2 |
| `PHILOX_W0` | `0x9E3779B9` | Weyl increment for key word 0 |
| `PHILOX_W1` | `0xBB67AE85` | Weyl increment for key word 1 |
| Rounds | 10 | round count |

**Counter layout (frozen — the parity-contract coordinate scheme):**

```
key     = [ seed as u32,  (seed >> 32) as u32 ]
counter = [ tree_id,  node_id,  feature_id,  draw_counter ]
```

Each `(key, counter)` pair is a unique coordinate; there is no shared state between
trees, nodes, or features. `draw_counter` is incremented per draw within a
`(tree, node, feature)` context.

**uint32 to f32 conversion (frozen — must be identical in CUDA):**

```
u32_to_unit_f32(x) = (x >> 8) as f32 * (1.0 / 16_777_216.0)
```

This uses the top 24 bits (matching f32's 24-bit mantissa for a clean dyadic uniform).
Output range: `[0, 1)`, never reaching 1.0.

**Known-answer test (KAT) vectors** are frozen in `crates/sylva-core/src/rng/kat.rs`.
Phase 4's CUDA implementation must reproduce these exact outputs for the three KAT
inputs before any GPU training result is accepted as valid.

**Note:** This guarantee is a Phase 4 deliverable. Phase 2 ships the Rust reference
implementation and KAT vectors. Phase 4 reproduces the same computation in CUDA and
verifies bit-for-bit against the KAT vectors before the `GPU == CPU` gate is declared.

---

## Point 3 — sklearn equivalence (DISTRIBUTIONAL, NOT bit-level)

Sylva does **not** reproduce scikit-learn's per-tree structure for a given seed,
and making it do so is not a goal. The reason is fundamental:

scikit-learn's tree splitters use a stateful serial 32-bit xorshift PRNG (`our_rand_r`
in `sklearn/utils/_random.pyx`) advanced sequentially as features and thresholds are
drawn within a node. A parallel counter-based RNG cannot replay this serial draw order.
Attempting to do so would require serializing the entire tree build, defeating the
purpose of counter-based RNG and Sylva's determinism guarantee.

**What "parity" means for sklearn equivalence:**

Sylva is statistically equivalent to scikit-learn, demonstrated by:

1. **Accuracy / R-squared agreement within a tight confidence interval** across many
   seeds: `|acc_sylva - acc_sklearn|` is within the seed-to-seed variance of sklearn's
   own runs (i.e., indistinguishable from sklearn's natural run-to-run variation).

2. **KS test on aggregate split statistics** (feature-selection frequency and threshold
   distribution per feature, pooled across many trees and seeds): the two-sample
   Kolmogorov-Smirnov test fails to reject the null hypothesis that Sylva and sklearn
   draw from the same distribution (p-value > 0.05).

These checks are like-for-like only:

- ExtraTrees Sylva vs. sklearn `ExtraTreesClassifier` / `ExtraTreesRegressor`
- RandomForest Sylva vs. sklearn `RandomForestClassifier` / `RandomForestRegressor`

Comparing ET to RF, or Sylva to a quantized GPU implementation with different
hyperparameters, is not a valid parity check.

**Requirement reference:** ENG-04 (documented parity contract), D-04 (strict
distributional parity gate), EST-07 / SC-6 (CI + KS parity gate).

**Tested by:** Phase 5 Python harness (`python/tests/parity/test_distributional_parity.py`)
using `scipy.stats.ks_2samp` against sklearn on `make_classification(20_000, 50)` and a
Covertype subset.

---

## Point 4 — f32 precision (D-05)

All Sylva compute and storage is **f32** (single-precision float), matching the future
GPU path so the Phase-4 `GPU == CPU` bit-exact gate is achievable.

scikit-learn uses f64 for thresholds and impurity computation. Last-bit f32/f64
differences exist and are expected. These differences are absorbed by the distributional
tolerance band (Point 3) and do not constitute a parity failure.

No speed claims are made about f32 vs f64 computation in this document. Performance
numbers are reported separately in the Comparative Baseline Study (SC-7), not asserted
here.

---

## What Sylva explicitly does NOT guarantee

- Per-tree structural identity with scikit-learn for any given seed. This is infeasible
  (Point 3) and is not a design goal.
- Bit-level reproduction of scikit-learn's `our_rand_r` serial PRNG sequence.
- Accuracy that is numerically identical to sklearn's output (f32 vs f64 differences
  exist; they are within the distributional tolerance).
- Any speed claim over other implementations. Training time is reported informally; no
  formal guarantee is made in this document.

---

## Deferred items

- **Phase 4:** CUDA KAT verification (`CudaBackend` reproduces the three KAT vectors in
  `rng/kat.rs` bit-for-bit before the `GPU == CPU` bit-exact gate is opened).
- **Phase 5:** Full distributional parity harness (CI + KS on large datasets, like-for-like
  ET/RF vs sklearn). Thresholds calibrated empirically from sklearn-vs-sklearn null spread.
- **Phase 6 / DET:** `fallback="error"` device dispatch policy (no silent CPU fallback when
  GPU is unavailable). The dispatch contract is orthogonal to the parity contract defined here.

---

*ENG-04 parity contract locked Phase 2, 2026-06-20.*
*CPU oracle confirmed bit-exact and distributional parity defined.*
*Next review: Phase 4 (CUDA KAT gate) and Phase 5 (distributional gate).*
