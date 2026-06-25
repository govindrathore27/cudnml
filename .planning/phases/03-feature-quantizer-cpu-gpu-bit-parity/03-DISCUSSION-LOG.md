# Phase 3: Feature Quantizer (CPU/GPU Bit-Parity) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-26
**Phase:** 3-Feature Quantizer (CPU/GPU Bit-Parity)
**Areas discussed:** Edge construction, CPU↔GPU bit-parity mechanism, Bin count / dtype / missing-value bin, execution_report_ hook scope
**Mode:** `--all` (all gray areas auto-selected, discussed interactively)

---

## Edge construction

| Option | Description | Selected |
|--------|-------------|----------|
| Exact full-data quantiles | Edges from full column via exact percentile/sort; no RNG; trivially CPU↔GPU parity-able; sklearn baseline becomes distributional | ✓ (Claude's discretion) |
| Subsample w/ Sylva Philox | Subsample large columns via reproducible Philox; parity preserved; more complex | |
| Match sklearn _BinMapper | Replicate sklearn subsample=200k + its RNG; exact sklearn parity but bit-matching sklearn PRNG (the Phase-2 trap) | |

**User's choice:** "you decide" → Claude locked **Exact full-data quantiles** (D-01).

| Option (dup edges) | Description | Selected |
|--------|-------------|----------|
| Collapse to unique edges | sklearn _BinMapper-style np.unique dedupe; low-cardinality → fewer effective bins | ✓ |
| Keep fixed n_bins | Always emit n_bins even if empty/degenerate | |

**User's choice:** Collapse to unique edges (D-02).

| Option (edge value) | Description | Selected |
|--------|-------------|----------|
| Midpoints (sklearn style) | Edge = midpoint between consecutive quantile values | |
| Quantile values directly | Use quantile sample values themselves as edges | ✓ |

**User's choice:** Quantile values directly (D-03).
**Notes:** Choosing quantile-values + dedupe shifts the correctness anchor toward `numpy.quantile`; the sklearn `_BinMapper` (midpoint) comparison becomes distributional-only.

---

## CPU↔GPU bit-parity mechanism

| Option (GPU scope) | Description | Selected |
|--------|-------------|----------|
| Build GPU kernel this phase | Write NVRTC quantize kernel + live CI parity now (matches SC-2 literally) | |
| CPU + parity scaffold only | CPU quantizer + parity harness now; GPU kernel lands Phase 4 | ✓ |

**User's choice:** CPU + parity scaffold only (D-05).
**Notes:** Flagged — narrows ROADMAP SC-2 for this phase to "CPU quantizer + enforced parity contract"; live CPU↔GPU CI proof + QUANT-02 live half move to Phase 4.

| Option (parity design) | Description | Selected |
|--------|-------------|----------|
| Edges on CPU, GPU assigns only | Edges computed once on host, uploaded; GPU runs only searchsorted compare → bit-identical by construction | ✓ |
| Edges computed on both | Full independent GPU quantile/sort; higher parity risk | |

**User's choice:** Edges on CPU, GPU assigns only (D-04).

| Option (boundary) | Description | Selected |
|--------|-------------|----------|
| Right (value < edge) | bin = count of edges ≤ value (digitize right=False / searchsorted 'right') | ✓ (Claude's discretion) |
| Left (value ≤ edge) | bin = count of edges < value (searchsorted 'left') | |
| You decide | Align with numpy.quantile baseline + Phase-4 split semantics | |

**User's choice:** "You decide" → Claude locked **side='right'** (D-06).

---

## Bin count / dtype / missing-value bin

| Option (bin count) | Description | Selected |
|--------|-------------|----------|
| 255 data + 1 missing = 256 | Matches sklearn HGBT; fits uint8; internal config, public param Phase 5 | ✓ (Claude's discretion) |
| 128 default | Smaller histograms; more shared-mem headroom | |
| You decide | Pick default + config-knob-vs-constant | |

**User's choice:** "you decide" → Claude locked **255 + 1 = 256, internal config field** (D-07).

| Option (dtype) | Description | Selected |
|--------|-------------|----------|
| Auto by bin count | ≤256 → uint8, >256 → uint16; both paths tested | ✓ |
| uint8 only (MVP) | Cap ≤256, add uint16 later | |

**User's choice:** Auto by bin count (D-08).

| Option (missing bin) | Description | Selected |
|--------|-------------|----------|
| Reserved missing bin (sklearn) | NaN → dedicated highest-index bin; split routing via D-01; out-of-range clamps | ✓ |
| Map missing to bin 0 | NaN folds into lowest bin | |

**User's choice:** Reserved missing bin (D-09).

---

## execution_report_ hook scope

> Asked twice via AskUserQuestion; user did not answer either time. Per the workflow's
> answer-validation rule (and consistent with the "you decide" pattern in earlier areas),
> Claude proceeded with the recommended options.

| Option (report scope) | Description | Selected |
|--------|-------------|----------|
| Minimal quantize record | dtype + contiguity + byte size into a small struct Phase 6 extends | ✓ (Claude's discretion) |
| Full execution_report_ now | Complete report surface this phase | |
| You decide | Right granularity for SC-3 | |

**Selected:** Minimal quantize record (D-10).

| Option (bytes field) | Description | Selected |
|--------|-------------|----------|
| Record computed payload size | Byte size of BinnedMatrix + edges; mark H2D N/A this phase | ✓ (Claude's discretion) |
| Defer bytes to Phase 4 | Leave bytes-transferred unimplemented | |

**Selected:** Record computed payload size (D-11).

| Option (microbench) | Description | Selected |
|--------|-------------|----------|
| CPU vs baseline now | Sylva CPU rows/s vs numpy.quantile + sklearn _BinMapper; GPU in Phase 4 | ✓ (Claude's discretion) |
| Skip until GPU exists | Defer whole microbench to Phase 4 | |

**Selected:** CPU vs baseline now (D-12).

---

## Claude's Discretion

- Edge algorithm (D-01: exact full-data quantiles), boundary rule (D-06: side='right'), bin count + config-field default (D-07), and the entire execution_report_ area (D-10/D-11/D-12) were delegated to Claude — either via explicit "you decide" or via non-answer after retry.

## Deferred Ideas

- Live GPU quantize kernel + live CPU↔GPU CI parity run + GPU throughput microbench → Phase 4.
- Subsampled quantile edges (with Philox) → profiling-driven later optimization.
- Public sklearn `max_bins` param → Phase 5.
- Full execution_report_ / dispatch / fallback → Phase 6.
- Learned optimal missing-value direction → still deferred from Phase 2.
</content>
