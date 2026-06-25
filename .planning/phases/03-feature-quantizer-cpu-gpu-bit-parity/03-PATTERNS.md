# Phase 3: Feature Quantizer (CPU/GPU Bit-Parity) - Pattern Map

**Mapped:** 2026-06-26
**Files analyzed:** 10 (8 new Rust/fixture + 2 new Python harness)
**Analogs found:** 10 / 10 (every new file has a strong in-repo analog)

> **Read-only note:** all paths below are real and verified against the tree.
> The research-cited layout (`crates/sylva-core/src/quantize/{mod,edges,binned_matrix,assign,report,parity}.rs`)
> matches the actual crate structure (sibling of `cpu/`, `rng/` module dirs). Python harness goes
> under the existing `python/tests/parity/` sibling dir as `python/tests/quantize_parity/`.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/sylva-core/src/quantize/mod.rs` | module/public surface | transform | `crates/sylva-core/src/rng/mod.rs` + `lib.rs` re-export block | role-match |
| `crates/sylva-core/src/quantize/edges.rs` | core algorithm (quantile edge build) | transform/batch | `crates/sylva-core/src/cpu/split_et.rs` (`feature_edges`-style numeric, `FEATURE_THRESHOLD`, sequential determinism) | role-match |
| `crates/sylva-core/src/quantize/binned_matrix.rs` | model/data structure (SoA) | transform | `crates/sylva-core/src/ir.rs` (`ForestIR` SoA + serde + `validate_structure`) | exact |
| `crates/sylva-core/src/quantize/assign.rs` | core algorithm (searchsorted + NaN routing) | transform | `crates/sylva-core/src/cpu/predict.rs` (`traverse_tree` NaN-first) | exact |
| `crates/sylva-core/src/quantize/report.rs` | model/record struct | event-driven (record) | `crates/sylva-core/src/parity.rs` (`SplitStats` serde record) | role-match |
| `crates/sylva-core/src/quantize/parity.rs` | model/fixture + (de)serialize | file-I/O (JSON) | `crates/sylva-core/src/parity.rs` (`SplitStats` serde + `parity` module role) | exact |
| Boundary validation (in `mod.rs`/`edges.rs`) | validation | request-response | `crates/sylva-core/src/config.rs` `TrainConfig::validate` + `predict.rs` boundary checks | exact |
| Rust unit/property tests (`#[cfg(test)]` + `tests/`) | test | — | `crates/sylva-core/tests/invariants.rs` (proptest) + in-file `#[cfg(test)] mod tests` | exact |
| `python/tests/quantize_parity/test_numpy_oracle.py` + `test_sklearn_distributional.py` + `test_throughput.py` | test (Python baseline) | request-response | `python/tests/parity/test_distributional_parity.py` + `datasets.py` + `conftest.py` | exact |
| Golden-vector JSON fixture file | config/fixture | file-I/O | `crates/sylva-core/src/parity.rs` serde JSON round-trip pattern | role-match |

## Pattern Assignments

### `crates/sylva-core/src/quantize/binned_matrix.rs` (model, SoA struct)

**Analog:** `crates/sylva-core/src/ir.rs`

**SoA struct + derive + doc convention** (ir.rs:20-65). Mirror exactly: `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]`, per-array `Vec<...>` fields with `///` doc on each, scalar shape fields (`n_rows`, `n_features`, `n_bins`) at the bottom, doc-link cross-references (`[`LEAF_FEATURE`]`-style). Use the same `pub const` sentinel style (ir.rs:16-18) for `MISSING_BIN` / dtype thresholds.

```rust
// ir.rs:22-23 — the exact derive line the new BinnedMatrix and BinEdges must copy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForestIR {
```

**`validate_structure`-style invariant check returning typed error, never panic** (ir.rs:88-146). The `BinnedMatrix`/`BinEdges` must expose an equivalent `validate()` that checks array-length agreement (`offsets.len() == n_features + 1`, monotone offsets — directly analogous to the `tree_offsets` checks at ir.rs:108-124) and returns `SylvaError::InvalidInput`/`InvalidIr`.

```rust
// ir.rs:118-124 — copy this monotone-offsets check shape for BinEdges.offsets (CSR)
for w in self.tree_offsets.windows(2) {
    if w[1] < w[0] {
        return Err(SylvaError::InvalidIr("tree_offsets must be non-decreasing".into()));
    }
}
```

**Accessor helper** (ir.rs:82-84 `tree_node_range`) → the research's `BinEdges::feature(f) -> &[f32]` CSR slice accessor follows this `#[inline]`/`Range`-returning idiom.

**Serde round-trip + validate unit tests** (ir.rs:149-219): copy the `serde_round_trip`, `validate_accepts_well_formed`, `validate_rejects_*` test trio verbatim in structure for `BinnedMatrix`/`BinEdges`.

---

### `crates/sylva-core/src/quantize/assign.rs` (core algorithm, searchsorted + NaN routing)

**Analog:** `crates/sylva-core/src/cpu/predict.rs`

**NaN-FIRST routing — the load-bearing convention** (predict.rs:83-105, esp. 92-99). The new `assign_bin` MUST check `v.is_nan()` before any `partition_point` compare, exactly as `traverse_tree` checks `is_nan()` before `v <= threshold`. This is RESEARCH Pitfall 4. Cite this file in the doc comment as the convention source.

```rust
// predict.rs:91-104 — the NaN-before-compare idiom assign_bin must mirror
let v = x[[row, feat]];
node = if v.is_nan() {
    ir.default_child[node] as usize          // → assign.rs: return missing_bin_idx
} else if v <= ir.threshold[node] {          // → assign.rs: partition_point(|&e| e <= v)
    ir.left_child[node] as usize
} else {
    ir.right_child[node] as usize
};
```

**Module-doc D-0x decision callouts** (predict.rs:1-11) — open the file with a `//!` doc block that names the decisions it implements (D-06 side='right', D-09 missing bin) and cross-references RESEARCH Pitfalls, exactly as predict.rs cites "D-01, RESEARCH Pitfall 2".

**`#[inline]` hot helper** (predict.rs:83-84) — `assign_bin` is the per-cell hot path; mark `#[inline]` like `traverse_tree`.

**Boundary feature-count validation** (predict.rs:28-37) — copy the `n_features != ir.n_features` → `SylvaError::InvalidInput` guard shape for the quantizer's X-vs-edges feature check.

---

### `crates/sylva-core/src/quantize/edges.rs` (core algorithm, exact-quantile build)

**Analog:** `crates/sylva-core/src/cpu/split_et.rs`

**Apache-2.0 reimplementation header + algorithm-in-doc** (split_et.rs:1-16). Copy this exact module-doc shape: state "Reimplemented from numpy `quantile` method='linear' algorithm description (Apache-2.0; NOT copied from numpy/sklearn source)", then a numbered algorithm sketch. This satisfies the Security-Domain license-contamination control.

**Named numeric constant, not a magic number** (split_et.rs:24-26). Bin count / dtype thresholds live in a `QuantizeConfig` field (D-07 default 255) plus `pub const` named constants in the split_et.rs `FEATURE_THRESHOLD` style — never inline literals.

```rust
// split_et.rs:24-26 — the named-constant + sklearn-cite convention edges.rs follows
/// Feature range threshold below which a feature is considered constant ...
/// Matches sklearn's `FEATURE_THRESHOLD` ≈ 1e-7.
pub const FEATURE_THRESHOLD: f32 = 1e-7;
```

**Determinism: sequential per-column, parallel-across-columns only.** split_et.rs accumulates in fixed order per node; edges.rs must keep each column's sort/quantile sequential and only `rayon`-parallelize across features (RESEARCH Standard Stack note + Anti-Pattern). The f64 index-math / f32 result cast point (RESEARCH A1) is the one calibration the planner must spike before locking golden vectors.

---

### `crates/sylva-core/src/quantize/report.rs` (record struct, minimal execution_report_)

**Analog:** `crates/sylva-core/src/parity.rs` (`SplitStats`)

**Serde record struct read by Python** (parity.rs:35-59). `QuantizeReport` copies the `SplitStats` pattern: `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]`, plain scalar fields (input dtype string, contiguity flag, computed payload bytes), `///` doc per field, and a module doc (parity.rs:1-22) explaining it is consumed downstream (here: Phase 6 extends it; H2D explicitly marked N/A per D-11).

---

### `crates/sylva-core/src/quantize/parity.rs` + golden-vector JSON fixture (fixture + serde)

**Analog:** `crates/sylva-core/src/parity.rs`

**This is the closest structural twin in the repo** — same filename, same "parity-support utilities consumed by an out-of-Rust harness via serde_json" role (parity.rs:1-22 module doc). The golden-vector fixture struct (`{ seed, n_bins, X, expected_bins (column-major u8), edges }` per RESEARCH Pattern 3) copies the `SplitStats` serde-record shape and the serde round-trip test (parity.rs:294-304).

```rust
// parity.rs:294-304 — the serde round-trip test the golden-fixture loader must replicate
let json = serde_json::to_string(&stats).expect("serialize SplitStats");
let back: SplitStats = serde_json::from_str(&json).expect("deserialize SplitStats");
assert_eq!(stats, back, "... serde round-trip must be identical");
```

**Contract metadata in the fixture:** per RESEARCH Pitfall 2, embed in the fixture/its doc the Phase-4 GPU requirement: "literal `v <= edge` compare against uploaded host edges; no FMA; compile assignment path WITHOUT `--use_fast_math`."

---

### `crates/sylva-core/src/quantize/mod.rs` + `lib.rs` re-export (module surface)

**Analog:** `crates/sylva-core/src/lib.rs` (re-export block) + `rng/mod.rs` (sibling module-dir precedent)

**Add `pub mod quantize;` to lib.rs:9-15 and re-export the public surface** in the lib.rs:22-27 block (`pub use quantize::{Quantizer, QuantizeConfig, BinnedMatrix, BinEdges, BinDtype, QuantizeReport};`). RESEARCH Open-Question 2 resolves: build `Quantizer` **standalone** (do NOT force `CpuBackend` to implement `HistogramBackend`); leave the `backend.rs:52 BinEdges` placeholder for Phase-4 trait wiring.

```rust
// lib.rs:22-27 — extend this exact re-export block
pub use parity::{split_statistics, SplitObservation, SplitStats};
// + pub use quantize::{Quantizer, QuantizeConfig, BinnedMatrix, BinEdges, ...};
```

---

### Python harness: `python/tests/quantize_parity/` (numpy edge-exact + sklearn distributional + throughput)

**Analog:** `python/tests/parity/` (`test_distributional_parity.py`, `datasets.py`, `conftest.py`)

**Reuse the whole harness skeleton — this is an exact match.**

- **Fairness-protocol module docstring** (test_distributional_parity.py:1-25): open each new test file with the `[FP-1..6]`-style protocol block; for quantize, the gate framing is "numpy = edge-exact 100% GATE; sklearn `_BinMapper` = distributional informational only" (RESEARCH Tolerances + Pitfall 3).
- **Dataset loader module** (datasets.py:1-124): copy `load_make_classification` (make_classification, `random_state=DATASET_SEED`, `.astype(np.float32)`) — RESEARCH pins 100k×100 f32, bins 128/256. Reuse the `Dataset` NamedTuple + fixed-seed constants pattern (datasets.py:26-83).
- **Version manifest + autouse print** (conftest.py:42-71, 143-146): copy `VERSION_MANIFEST` (numpy/sklearn/python/sylva_commit) and the autouse session print — satisfies the FP-5 pinned-versions requirement for D-12.
- **Cold/warm separated, informational throughput** (test_distributional_parity.py:262-319): copy the `time.perf_counter()` cold-then-warm two-fit structure for the rows/s microbench; **report, never gate** (D-12, foundational-phase fairness — matches the SC-7 "INFORMATIONAL ONLY - no speed claim" framing).
- **Sylva access:** the harness imports `import sylva_core_pyseam as sylva` and calls seam fns (test_distributional_parity.py:41, 123). The new quantize harness needs a `pyseam` quantize entry point (a `quantize_cpu` seam fn returning the binned matrix + a JSON report) — mirror the existing `sylva.fit_cpu` / `sylva.split_statistics` seam pattern in `crates/sylva-core/src/pyseam.rs`.

```python
# test_distributional_parity.py:262-270 — the cold/warm two-fit microbench shape to copy
t0 = time.perf_counter()
_ = sylva.fit_cpu(...)          # → quantize_cpu(...) cold
sylva_cold = time.perf_counter() - t0
t0 = time.perf_counter()
ir_handle = sylva.fit_cpu(...)  # → quantize_cpu(...) warm
sylva_warm = time.perf_counter() - t0
```

**numpy edge-exact oracle:** implement `numpy_oracle_bins` exactly as RESEARCH Code-Examples (np.quantile method='linear', `np.searchsorted(side='right')`, NaN→missing_idx) and assert 100% bit-equality vs the Rust `BinnedMatrix` — this is the SC-4 GATE.

---

### Rust tests (in-file `#[cfg(test)]` + `tests/`)

**Analog:** `crates/sylva-core/tests/invariants.rs` (proptest) + in-file `mod tests` across ir.rs/parity.rs

**Property invariants via proptest** (invariants.rs:1-19 doc + bounded-shape budget). Copy the bounded-shape, ≤32-cases proptest discipline for the quantize properties RESEARCH names: bins ∈ [0, n_bins); monotone (x1 ≤ x2 ⇒ bin1 ≤ bin2); NaN → missing bin; edge round-trip. Use the deterministic `Array2::from_shape_fn` dataset helpers (invariants.rs:26-51) — no external RNG.

**Degenerate-column tests** (RESEARCH Pitfall 5): add explicit unit tests for constant column → 0 edges → bin 0, all-NaN column → all missing bin, `<n_data_bins` distinct → fewer effective bins. Typed `SylvaError` only for truly-invalid input (empty matrix, n_bins out of range) — following the `config.rs:79-109` `validate` reject pattern.

## Shared Patterns

### Typed-error boundary validation (no panic, no `.unwrap()`)
**Source:** `crates/sylva-core/src/error.rs:8-21` + `crates/sylva-core/src/config.rs:79-109` (`TrainConfig::validate`)
**Apply to:** `quantize/mod.rs` entry point, `edges.rs`, `binned_matrix.rs::validate`
Reuse the existing `SylvaError` enum (`InvalidInput` for bad X shape/dtype, `InvalidConfig` for n_bins out of `[2,256]`/`[2,65536]`). Do NOT add new error variants unless a case doesn't fit; the three existing variants cover this phase.
```rust
// config.rs:80-83 — the reject-with-typed-error shape every quantize validation copies
if self.n_estimators == 0 {
    return Err(SylvaError::InvalidConfig("n_estimators must be >= 1".into()));
}
```

### Module doc that names decisions + cites RESEARCH/source
**Source:** `crates/sylva-core/src/cpu/predict.rs:1-11`, `cpu/split_et.rs:1-16`, `ir.rs:1-6`
**Apply to:** every new `quantize/*.rs` file
Open with `//!` summarizing the file, the D-0x decisions it implements, and (for assign/edges) the RESEARCH Pitfall it guards. This is a hard, consistently-followed repo convention.

### Serde-record consumed by the Python harness
**Source:** `crates/sylva-core/src/parity.rs` (`SplitStats`) + the `pyseam` seam usage in `test_distributional_parity.py:41,123`
**Apply to:** `quantize/report.rs` (`QuantizeReport`), `quantize/parity.rs` (golden fixture), and the new `quantize_cpu` pyseam fn
`#[derive(...Serialize, Deserialize)]` + a serde round-trip test; the Python side does `json.loads(sylva.<fn>(...))`.

### Determinism: fixed-order accumulation, parallel only across independent units
**Source:** `crates/sylva-core/src/cpu/split_et.rs` (fixed-order per-node) + RESEARCH Standard-Stack rayon note
**Apply to:** `edges.rs` (sequential per-column sort/quantile; `rayon` across features only) — never a non-deterministic within-column reduction.

### License-contamination guard (reimplement from algorithm, never copy)
**Source:** `crates/sylva-core/src/cpu/split_et.rs:1-5` Apache-2.0 reimplementation header
**Apply to:** `edges.rs`, `assign.rs` — cite the numpy/sklearn *formula*, never the source.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| — | — | — | None. Every new file has a strong in-repo analog; the repo's `parity.rs` / `ir.rs` / `predict.rs` / `split_et.rs` and the `python/tests/parity/` harness cover all roles this phase introduces. |

> Note: the `HistogramBackend::quantize` seam (`backend.rs:75-97`) exists but is a Phase-4 placeholder
> (`struct BinEdges;` at backend.rs:52). Per RESEARCH Open-Question 2 the new `Quantizer` is standalone
> and does NOT bind this trait now — so it is an integration *reference*, not an analog to copy.

## Metadata

**Analog search scope:** `crates/sylva-core/src/` (ir, backend, error, config, parity, lib, cpu/{predict,split_et}, rng/mod), `crates/sylva-core/tests/` (invariants), `python/tests/parity/` (test_distributional_parity, datasets, conftest)
**Files scanned:** 12 read in full/targeted; crate + python-harness layout enumerated
**Pattern extraction date:** 2026-06-26
