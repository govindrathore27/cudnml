# Phase 7: Crossover Benchmark (Gate 3) - Research

**Researched:** 2026-06-27
**Domain:** Pre-registered end-to-end (n×d) benchmark crossover surface — the authoritative study that defines project success and enforces the KILL CRITERION
**Confidence:** HIGH for harness architecture, fairness protocol, kill-criterion mechanics, and pre-registration structure (all grounded in binding upstream documents). MEDIUM for cuML/oneDAL Windows availability (environment-dependent; confirmed problematic but not run here). LOW for exact crossover coordinates (empirical unknown — that is what the phase measures).

> **Phase-ordering note:** Phases 3–6 are planned but NOT yet executed at research time (STATE.md: `current_phase: 03`, `completed_phases: 2`). Phase 7 depends on Phase 6 being complete. This research is written against the locked design decisions from Phases 1–6 planning documents; harness integration points reference structures and files as designed, not yet shipped. The Phase-5 harness (`python/benchmarks/comparative_study.py`, `study_manifest.py`) is the designed precursor this phase generalizes.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **BENCH-01** | A scripted, reproducible benchmark measures end-to-end training time from numpy (including H2D transfer + quantization), cold and warm, with pinned hardware/driver/CUDA/package versions | Sections: Standard Stack (harness), Architecture Patterns (timing contract), Pitfalls (cold vs warm, end-to-end region), Validation Architecture (reproducibility gate). The Phase-5 `comparative_study.py` + `study_manifest.py` are the designed precursor to extend. |
| **BENCH-02** | Baselines include scikit-learn ET/RF (`n_jobs=-1`), oneDAL/sklearnex, and cuML RF (labeled as a different algorithm); accuracy parity reported alongside speed | Sections: Standard Stack (baseline libraries), Pitfalls (Pitfall 13 fairness, Pitfall 2 baseline strength), Open Questions (cuML/oneDAL Windows availability). oneDAL/sklearnex and cuML require a human-verify checkpoint. |
| **BENCH-03** | A published (n×d) crossover surface identifies where GPU beats the strongest CPU baseline and where it does not; pre-registered kill criteria evaluated | Sections: Architecture Patterns (grid spec, heatmap), Kill Criterion (BENCH-03 embeds the KILL CRITERION), Validation Architecture (crossover surface artifact gate). The kill criterion is an explicit decision checkpoint, not an implicit assertion. |

</phase_requirements>

## User Constraints (from upstream binding documents)

> No `07-CONTEXT.md` exists (this phase has not gone through `/gsd-discuss-phase`). The constraints below are extracted from PROJECT.md, ROADMAP.md, REQUIREMENTS.md, STATE.md, and the binding fairness note at the top of ROADMAP.md. Treat them as locked until a CONTEXT.md supersedes them.

### Locked Decisions

**From PROJECT.md / ROADMAP.md:**
- **Success is defined by the crossover, not an arbitrary speedup number.** The pre-registered (n×d) crossover surface (where GPU ET beats the strongest CPU baseline end-to-end from numpy) IS the definition of project success (PROJECT.md Constraints).
- **KILL CRITERION is binding.** If NO region of the (n×d) surface shows end-to-end GPU Extra Trees beating the strongest CPU baseline → the core premise is false → PIVOT to the SHAP/determinism layer (which can ride on cuML) rather than continuing broad build-out. This is not a soft warning; it is the pre-registered decision criterion (ROADMAP Phase 7 Success Criterion 4).
- **Fairness rules are binding on every comparative study (non-negotiable).** From the ROADMAP comparative-study fairness note and PITFALLS.md Pitfalls 1, 2, 13:
  - Compare **equivalent algorithms only**: ET vs ET, RF vs RF — never ExtraTrees vs RandomForest as if identical.
  - Time **end-to-end from numpy** including dtype coercion + H2D + quantization — never "data already on GPU" in reported numbers.
  - Separate **cold** (first call) vs **warm** (subsequent calls) runs.
  - Use the **strongest** CPU baseline: sklearn `n_jobs=-1` AND oneDAL/sklearnex (not single-threaded).
  - cuML RF is the GPU baseline, **labeled as a different algorithm** (RF, not ET).
  - **Pin all hardware/driver/CUDA/package versions** (extend the Phase-5 MANIFEST).
  - **Repeated runs** (≥5 per cell); report median + spread.
  - **Accuracy parity reported alongside every speed cell** — a fast but less accurate result is not a win.
  - **Report failures and OOM honestly** rather than hiding them.
- **Pre-registration is mandatory.** The grid, datasets, hyperparameters, and pass/kill criteria must be **frozen before running** — no p-hacking, no moving goalposts after seeing results (ROADMAP Phase 7 / PITFALLS 13).
- **Per-phase studies (Phases 1–6) feed this authoritative crossover.** Any conflicts between a per-phase data point and the Phase-7 surface must be explained (ROADMAP Phase 7 Success Criterion 5).
- **cudarc 0.19.8 + hand-written CUDA C via NVRTC, native Windows/MSVC, Apache-2.0, stable Rust 1.83+** (CLAUDE.md). The harness must produce results on the Windows dev box (RTX 4060 Ti, CUDA 12.8, driver 595.79).
- **The crossover study times the `deterministic=False` (fast) mode** for the primary speed cells, with `deterministic=True` overhead reported separately (per Phase-6 DET-02 design; the crossover claim is about the fastest honest path, not the deterministic one — the determinism differentiator is separately documented).

### Claude's Discretion
- The exact (n, d) grid coordinates beyond the minimum anchors (Covertype full, Higgs/large synthetic), subject to staying within GPU memory.
- Visualization format for the heatmap (matplotlib/seaborn vs a plain markdown table is acceptable; the surface artifact must exist in a published file).
- Whether the MANIFEST extends `python/benchmarks/study_manifest.py` in-place or creates `python/benchmarks/crossover_manifest.py` (recommend a new file to avoid mutating the Phase-5 harness).
- The tolerance for accuracy-parity assertions (must be documented + justified; mirrors Phase-5 convention).

### Deferred Ideas (OUT OF SCOPE for Phase 7)
- Exact tree SHAP benchmarks (Phase 8).
- Treelite export / FIL inference throughput (Phase 9).
- Multi-GPU scaling (v2+).
- Sparse/CSR input benchmarks (v2+, DATA-V2-01).
- XGBoost rf-mode comparison beyond what was done in Phase 5 (it was an informational number there; this phase is ET/RF vs sklearn+oneDAL+cuML only).
- The SHAP feasibility study (Phase 8).

---

## Summary

Phase 7 is the project's **defining gate**. It exists to run the pre-registered (n×d) crossover benchmark that either confirms the GPU ET training advantage or triggers the kill criterion. Every earlier phase exists to enable this study; every later phase is contingent on its result.

The harness this phase builds is a **generalization of the Phase-5 `comparative_study.py`**: whereas Phase 5 ran one large dataset end-to-end and gated on accuracy parity (with speed reported informatively), Phase 7 runs a **2D grid of (n, d) cells** across synthetic and real datasets, each cell timed end-to-end from numpy (cold + warm, repeated), with accuracy beside every speed cell, and produces a published **crossover surface** — a map of where Sylva GPU ET wins, where it loses, and the boundary.

The critical structural addition over Phase 5 is the **pre-registration document**: before any cell is measured, the grid, datasets, hyperparameters, baselines, pass bar, and the exact kill criterion wording must be committed to the repo. This makes the benchmark protocol tamper-evident (you cannot choose the grid after seeing results). The pre-registration document (`07-PRE-REGISTRATION.md`) is a Task 0 blocking gate; nothing else runs until it is committed.

The **kill criterion** is encoded as an explicit checkpoint in the plan, evaluated after all cells are measured but before any downstream work. If the surface shows no win region, the checkpoint fires and the project pivots to the SHAP/determinism layer (which can ride on cuML). If the surface shows a win region, the crossover is published and Phase 8 is unblocked.

**Primary recommendation:** Plan Phase 7 as four slices: (1) pre-registration commit (blocking gate, before any measurement); (2) harness + manifest build (extends Phase-5 scaffolding, adds the (n×d) grid loop, OOM-safe, markdown/JSON output); (3) measurement run on the dev box (dev-box-only; not CI); (4) kill-criterion checkpoint + crossover publication (surface artifact + a reconciliation note against per-phase studies).

The Phase 6 studies (DET-02 Sylva-vs-Sylva overhead) provide a data point that informs the crossover: the crossover times `deterministic=False` (the fast path) for the primary speed cells, and `deterministic=True` results are reported separately in the manifest with the measured overhead percentage.

**Crossover timing mode:** Time `deterministic=False` (fast mode) as the primary speed cells. Report `deterministic=True` alongside as a separate column with the Phase-6 overhead fraction applied (or by re-running). This is the most favorable honest number for Sylva and the correct comparison for the speed claim. The determinism differentiator is a separate documented property, not the primary speed metric.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pre-registration document | Planning artifact tier (`.planning/phases/07-*/07-PRE-REGISTRATION.md`) | Git history (commit timestamp is the tamper-evidence) | Must be committed before measurement runs; tamper-evidence comes from the git commit predating results. |
| (n×d) grid loop + timing harness | Python benchmarks tier (`python/benchmarks/crossover_study.py`) | Extends `comparative_study.py` from Phase 5 | Runs on the dev box; not a CI artifact. Must import Sylva via the maturin-built wheel. |
| Version / hardware manifest | Python benchmarks tier (`python/benchmarks/crossover_manifest.py`) | Extends `study_manifest.py` from Phase 5 | Extends Phase-5 manifest with grid spec, oneDAL/sklearnex, cuML, hardware exact pins. |
| Crossover surface visualization | Python benchmarks tier (matplotlib/seaborn or markdown table) | Published as a file in `results/` or `docs/` | The published artifact that constitutes BENCH-03. |
| Kill-criterion checkpoint | Human decision tier (`checkpoint:human-verify`) | Documented in plan as a blocking gate | The decision (proceed vs pivot) is a human call; the harness only computes the surface. |
| Fairness-rule enforcement | Harness code (`crossover_study.py` prohibitions baked in) | Pre-registration document (protocol locked) | Encoding fairness rules as code prohibitions (no ET-vs-RF, no kernel-only timing, n_jobs=-1) is the defence against accidental p-hacking. |
| Accuracy parity gate (per cell) | Test tier (`test_crossover_accuracy_parity.py`) | CI-portable (device='cpu', small datasets) | Accuracy parity is the gate; speed is reported, not gated. |
| Baseline availability resolution | `checkpoint:human-verify` (for oneDAL, cuML) | Study manifest records result honestly | cuML on Windows is historically unavailable; oneDAL/sklearnex may need WSL2 for GPU path. |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Sylva** (`python/sylva/`) | Phase 6 HEAD | The thing being benchmarked | The library under test — imported via `maturin develop` or wheel install |
| **scikit-learn** | 1.9.0 (pinned in venv-parity) | Primary CPU baseline ET/RF + accuracy reference | `[VERIFIED: .venv-parity scikit_learn-1.9.0]` — like-for-like ET vs ET, RF vs RF, `n_jobs=-1` |
| **numpy** | pinned in venv-parity | Dataset generation, dtype coercion source (float64→float32), timing region start point | `[VERIFIED: conftest.py VERSION_MANIFEST]` |
| **scikit-learn-intelex / sklearnex** | latest stable (≈2024.x) | oneDAL-accelerated sklearn (strongest CPU baseline — replaces sklearn's BLAS with oneDAL on Intel CPUs) | `[ASSUMED]` — must be installed and verified on the measurement host; Windows install via `pip install scikit-learn-intelex` is documented. If `patch_sklearn()` is called, it accelerates sklearn's own estimators in-place. Gate behind `checkpoint:human-verify`. |
| **cuML** (`cuml-cu12`) | RAPIDS 24.x or 25.x | GPU RF baseline (labeled: different algorithm from ET) | `[ASSUMED]` — Linux-first; native Windows wheel unavailable; WSL2 path or "not available on host" per the Phase-5 OQ7 resolution. Gate behind `checkpoint:human-verify`. |
| **matplotlib / seaborn** | any recent | Crossover heatmap visualization | `[ASSUMED]` — standard Python visualization stack; install in benchmark env only. |
| **time / timeit** (stdlib) | stdlib | Wall-clock timing of `fit(X, y)` from numpy | Standard; `time.perf_counter()` is the correct monotonic timer for benchmarking on Windows. |
| **json / csv** (stdlib) | stdlib | Emit results as JSON + CSV for the surface artifact | Standard; no external serialization dependency. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **sklearn.datasets** | (via scikit-learn) | `make_classification` grid generation; `fetch_covtype` real anchor | Always — grid is synthetic + real anchors |
| **scipy.stats** | (via conftest) | Wilcoxon signed-rank test or similar for "is GPU win statistically significant" (informational only) | Optional annotation on each win-region cell; does not change the surface verdict |
| **pytest** | pinned | Accuracy-parity gate test (`test_crossover_accuracy_parity.py`) | CI-portable accuracy gate (device='cpu') |
| **pandas** | optional | Tabular surface output for the markdown table | Optional; numpy + json are sufficient; pandas adds nicely-formatted output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `time.perf_counter()` wall-clock | CUDA events (`cudaEventElapsedTime`) | CUDA events measure only device time; they exclude H2D transfer, dtype coercion, quantization — exactly what PITFALL 1 says must be included. Use wall-clock of the Python API call. |
| `make_classification` synthetic grid | Only real datasets | Synthetic gives full control over (n, d); real datasets anchor a few specific points. Both are needed: synthetic for the grid, real (Covertype, Higgs subset) for ecological validity. |
| Extending `comparative_study.py` in-place | New `crossover_study.py` | The Phase-5 harness was designed for one large dataset + a few baselines; the Phase-7 grid loop is structurally different (outer loop over grid cells, per-cell OOM handling, surface accumulation). A new file (`crossover_study.py`) avoids mutating a working Phase-5 artifact while reusing its helpers. |
| Heatmap in matplotlib | Markdown table only | Both should be produced: the table is machine-readable and fits in docs; the heatmap is the visual summary. The table is the canonical artifact; the heatmap is the visual supplement. |

**Installation (benchmark environment only — not Sylva runtime deps):**
```bash
# On the Windows measurement host (in benchmark venv, not the main Sylva venv)
pip install scikit-learn-intelex            # oneDAL/sklearnex — gate behind human-verify
# cuML: likely unavailable on Windows; see human-verify checkpoint (WSL2 or skip)
pip install matplotlib seaborn pandas       # visualization / tabular output
```

**Version verification:** All Sylva pins are in `Cargo.lock` + `VERSIONS.md` (cudarc 0.19.8, PyO3 0.29.0, CUDA 12.8, sm_89, driver 595.79, Rust 1.96.0). Baseline pins belong in `crossover_manifest.py` (MANIFEST dict), committed alongside the pre-registration document.

---

## Package Legitimacy Audit

> Phase 7 adds **no new Sylva runtime dependencies**. The only additions are benchmark-environment-only packages. They are study tools, not Sylva build or runtime deps.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| scikit-learn | PyPI | mature (>10 yr) | 50M+/mo | github.com/scikit-learn/scikit-learn | OK | Approved (already in use) |
| scikit-learn-intelex | PyPI | active (~3 yr) | high | github.com/intel/scikit-learn-intelex | `[ASSUMED]` | Benchmark env only — planner must add `checkpoint:human-verify` before install on the measurement host |
| cuml-cu12 | RAPIDS channel | active | high | github.com/rapidsai/cuml | `[ASSUMED]` | Benchmark env only; Linux-first; Windows availability gated behind `checkpoint:human-verify` (carried from Phase-5 OQ7) |
| matplotlib | PyPI | mature (>15 yr) | very high | github.com/matplotlib/matplotlib | OK | Benchmark env, visualization only |
| seaborn | PyPI | mature (>10 yr) | high | github.com/mwaskom/seaborn | OK | Optional heatmap; benchmark env only |
| pandas | PyPI | mature (>15 yr) | very high | github.com/pandas-dev/pandas | OK | Optional tabular output; benchmark env only |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

`scikit-learn-intelex` and `cuml-cu12` are `[ASSUMED]` — discovered via docs/training knowledge, not verified via Context7 or official docs in this session. Planner must gate both installs behind `checkpoint:human-verify`. Neither is a Sylva dependency; failure to install either is reported honestly as "baseline not available on measurement host" per the fairness protocol.

---

## Architecture Patterns

### System Architecture Diagram

```
  PRE-REGISTRATION GATE (committed before any measurement)
  ─────────────────────────────────────────────────────────
  07-PRE-REGISTRATION.md ──► git commit (timestamp = tamper evidence)
        │
        │  locks: grid spec, datasets, hyperparams, baselines, pass/kill bar
        ▼
  MEASUREMENT HARNESS (dev box only — not CI)
  ─────────────────────────────────────────────────────────
  crossover_study.py
        │
        ├─ for (n, d) in GRID:
        │     ├─ X, y = make_classification(n, d, seed=DATASET_SEED)  [float64]
        │     ├─ for impl in [sylva_et, sklearn_et, sklearnex_et?,
        │     │               sylva_rf, sklearn_rf, sklearnex_rf?,
        │     │               cuml_rf?]:  # cuML labeled "different algorithm"
        │     │     ├─ cold_times = []
        │     │     │     # TIMED REGION START (time.perf_counter)
        │     │     │     X32 = X.astype(np.float32)   # dtype coercion INSIDE timed region
        │     │     │     model.fit(X32, y)             # includes H2D + quantize + train
        │     │     │     # TIMED REGION END
        │     │     ├─ warm_times = [model.fit(X32, y) for _ in range(N_REPEATS-1)]
        │     │     ├─ acc = model.score(X32_test, y_test)
        │     │     └─ record {n, d, impl, cold_median, warm_median, cold_spread,
        │     │                warm_spread, accuracy, oom=False/True, error=None}
        │     │
        │     │  OOM guard: wrap each fit in try/except; record oom=True, continue
        │     └─ accumulate to surface_results[n][d]
        │
        ├─ real anchors: Covertype full (581k×54), Higgs subset (1M×28)
        │     └─ same per-impl loop; single (n,d) point per anchor
        │
        ├─ write results/crossover_results.json  (canonical)
        ├─ write results/crossover_results.csv   (tabular)
        └─ generate crossover_surface.png (heatmap: win/lose/tie per cell)
                  + crossover_table.md  (markdown table: all cells)

  KILL-CRITERION CHECKPOINT (human gate, after measurement)
  ─────────────────────────────────────────────────────────
  checkpoint:human-verify ──► evaluate: does any cell show GPU ET warm time < best CPU warm time?
        │
        ├─ YES (crossover found): publish surface → unblock Phase 8
        └─ NO  (no win region): KILL CRITERION fires → document → pivot to SHAP/determinism on cuML

  ACCURACY-PARITY GATE (CI-portable, device='cpu', small dataset)
  ─────────────────────────────────────────────────────────
  test_crossover_accuracy_parity.py
        └─ Sylva ET/RF vs sklearn ET/RF within tolerance (device='cpu')
           [existing gate from Phase 5 — verify still green after Phase 6]
```

### Recommended Project Structure
```
python/benchmarks/
├── __init__.py                          # (already exists from Phase 5)
├── comparative_study.py                 # Phase-5 harness — DO NOT MODIFY
├── study_manifest.py                    # Phase-5 manifest — DO NOT MODIFY
├── crossover_study.py                   # NEW: (n×d) grid loop harness
├── crossover_manifest.py                # NEW: extended manifest (grid+baselines+hardware)
├── grid_spec.py                         # NEW: GRID constant + dataset loaders for grid sizes
└── visualize_surface.py                 # NEW: heatmap + markdown table generator

.planning/phases/07-crossover-benchmark-gate-3/
├── 07-PRE-REGISTRATION.md              # NEW (TASK 0): frozen protocol before any measurement
└── 07-RESEARCH.md                       # this file

results/                                 # NEW directory for published surface artifacts
├── crossover_results.json              # canonical per-cell results
├── crossover_results.csv               # tabular
├── crossover_table.md                  # markdown crossover table
└── crossover_surface.png               # heatmap visualization

python/tests/
└── test_crossover_accuracy_parity.py   # NEW (or reuse Phase-5 gate): accuracy gate CI-portable
```

### Pre-Registration Document Structure (07-PRE-REGISTRATION.md)

The pre-registration document is the **anti-p-hacking contract**. It must be committed with a git timestamp before any measurement cell is run. The planner should produce it as a blocking task (Task 0). It must contain:

1. **Grid specification:** Exact list of (n, d) cells, e.g.:
   - n ∈ {10k, 50k, 100k, 250k, 500k, 1M} (rows)
   - d ∈ {20, 50, 100, 200} (features)
   - The full Cartesian product (or a documented subset with justification)
2. **Real anchors:** Covertype full (581k×54), Higgs subset (1M×28) — fixed shapes from sklearn/UCI.
3. **Fixed hyperparameters (identical across implementations):** n_estimators=200, max_depth=12, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, bootstrap=False (ET), bootstrap=True (RF), criterion="gini" / "squared_error", random_state=DATASET_SEED.
4. **Baseline set and labels:**
   - CPU primary: `sklearn.ensemble.ExtraTreesClassifier` / `RandomForestClassifier` with `n_jobs=-1`
   - CPU accelerated: `sklearnex.ensemble.ExtraTreesClassifier` / `RandomForestClassifier` (`patch_sklearn()` or direct) — **strongest CPU baseline**
   - GPU (different algorithm): `cuml.ensemble.RandomForestClassifier` (labeled "cuML RF — different algorithm from ET")
   - Sylva GPU ET: `sylva.ExtraTreesClassifier(device='cuda', deterministic=False)` — primary speed claim
   - Sylva GPU RF: `sylva.RandomForestClassifier(device='cuda', deterministic=False)` — secondary
   - Sylva CPU ET/RF (for transfer-cost analysis): `sylva.ExtraTreesClassifier(device='cpu')`
5. **Timing protocol:** `time.perf_counter()` wall-clock; timed region = `model.fit(X_float32, y)` from a host numpy float32 array, including dtype coercion from float64 if input was float64 (PITFALL 1); cold = first call; warm = N_REPEATS subsequent calls; report median + IQR.
6. **OOM policy:** Catch `MemoryError` / CUDA OOM; record `oom=True`; continue; do not hide.
7. **Pass bar:** A win region exists if GPU ET warm time < best CPU baseline warm time for at least one (n, d) cell with accuracy parity (< 1% accuracy difference from sklearn ET).
8. **KILL CRITERION (exact wording, pre-registered):** "If no cell in the grid shows Sylva GPU ExtraTreesClassifier warm `fit(X, y)` time — measured end-to-end from a host float64 numpy array, including dtype coercion + H2D + quantization — beating the BEST of {sklearn ET n_jobs=-1, sklearnex ET} warm times by ≥ 5% while maintaining accuracy parity (test-set accuracy within 1% of sklearn ET), then the core premise of the GPU advantage is not confirmed on this hardware. The project shall pivot to the SHAP/determinism layer (which can ride on cuML) or to upstreaming rather than continuing broad GPU training build-out."
9. **Hardware / version snapshot:** GPU model, driver, CUDA version, Sylva git SHA, all baseline versions — committed with the pre-registration.
10. **Reconciliation note:** "This study is the authoritative Phase-7 crossover. Any conflict between a per-phase study (Phases 1–6) speed number and the Phase-7 surface number for the same (n, d) point will be explained in the crossover report."

### Pattern 1: End-to-end timing region (BENCH-01 — the canonical timing contract)

**What:** Every timed `fit` call must include the full round-trip from a host numpy array: dtype coercion, H2D, quantization, and GPU training. Never start the timer after data is already on device.

**When to use:** Every cell in the grid, every baseline. No exceptions.

```python
# Source: extends comparative_study.py pattern (Phase-5 plan 05-06-PLAN.md task 1)
import time
import numpy as np

def timed_fit(model, X_f64: np.ndarray, y: np.ndarray,
              n_cold: int = 1, n_warm: int = 5) -> dict:
    """
    Time model.fit() end-to-end from host float64 numpy array.
    Dtype coercion (float64 -> float32) is INSIDE the timed region.
    cold = first n_cold calls (model uninitialized / no warm GPU caches)
    warm = subsequent n_warm calls (GPU memory resident, caches warm)
    Returns dict with cold_times, warm_times, median_cold, median_warm, iqr_warm.
    """
    cold_times = []
    for _ in range(n_cold):
        t0 = time.perf_counter()
        X32 = X_f64.astype(np.float32)  # coercion INSIDE timed region
        model.fit(X32, y)
        cold_times.append(time.perf_counter() - t0)

    warm_times = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        X32 = X_f64.astype(np.float32)  # coercion INSIDE timed region every call
        model.fit(X32, y)
        warm_times.append(time.perf_counter() - t0)

    import statistics
    return {
        "cold_times": cold_times,
        "warm_times": warm_times,
        "median_cold_s": statistics.median(cold_times),
        "median_warm_s": statistics.median(warm_times),
        "iqr_warm_s": (sorted(warm_times)[3*len(warm_times)//4]
                       - sorted(warm_times)[len(warm_times)//4])
                      if len(warm_times) > 2 else 0.0,
    }
```

**Anti-pattern:** Starting the timer after `X32 = X_f64.astype(np.float32)`. That omits the coercion cost — a form of benchmark dishonesty (PITFALL 1 and PITFALL 13 from PITFALLS.md).

### Pattern 2: Like-for-like enforcement (BENCH-02 — algorithmic parity contract)

**What:** Each Sylva implementation is compared only to its algorithmic counterpart. ET vs ET; RF vs RF. cuML RF is labeled as a different algorithm.

**When to use:** Every cell in the grid. The harness must structurally prevent ET-vs-RF comparisons.

```python
# Source: extends comparative_study.py prohibitions (Phase-5 plan 05-06-PLAN.md prohibitions)
from dataclasses import dataclass

@dataclass
class BenchmarkImpl:
    name: str          # e.g. "sklearn_et", "sylva_gpu_et", "cuml_rf_DIFFERENT_ALGO"
    algorithm: str     # "et" | "rf"   -- crossover surface groups by algorithm
    backend: str       # "cpu" | "gpu"
    is_reference: bool # True for sklearn (used for accuracy parity check)
    factory: object    # callable() -> fitted estimator

# Crossover surface is computed per-algorithm group:
# ET group: sylva_gpu_et vs [sklearn_et, sklearnex_et]
# RF group: sylva_gpu_rf vs [sklearn_rf, sklearnex_rf, cuml_rf (labeled "different algo")]
# NEVER: sylva_gpu_et vs cuml_rf  (ET vs RF -- forbidden by fairness protocol)
```

**Anti-pattern:** Populating a single results dict with all implementations without an algorithm tag, then comparing across algorithm families. The harness must enforce the algorithm grouping structurally, not by convention.

### Pattern 3: OOM-safe grid loop (prevents silent result gaps)

**What:** Every `fit` call is wrapped in a try/except that catches CUDA OOM and host OOM. The run continues; the cell records `oom=True`. Large (n, d) cells are expected to OOM for some implementations.

**When to use:** Every cell in the grid.

```python
# Source: original; extends the OOM-honest pattern from comparative_study.py
def safe_timed_fit(model_factory, X_f64, y, **kwargs) -> dict:
    try:
        model = model_factory()
        times = timed_fit(model, X_f64, y, **kwargs)
        return {"oom": False, "error": None, **times}
    except (MemoryError, Exception) as e:
        error_str = str(e)
        is_oom = "out of memory" in error_str.lower() or isinstance(e, MemoryError)
        return {"oom": is_oom, "error": error_str,
                "median_cold_s": None, "median_warm_s": None}
```

**Anti-pattern:** Letting an OOM crash terminate the grid loop, silently losing all remaining cells. Also anti-pattern: omitting the OOM result from the published surface, making the grid appear complete when it is not.

### Pattern 4: Kill-criterion evaluation (explicit human checkpoint)

**What:** After all cells are measured and the surface JSON is written, a `checkpoint:human-verify` task compares Sylva GPU ET warm times to the best CPU baseline warm time per cell. The checkpoint prompts the human to evaluate: any cell where GPU ET warm time < best CPU warm time with accuracy parity? If yes → crossover confirmed → proceed. If no → KILL CRITERION fires → document → pivot.

**When to use:** Once, after the grid run is complete and results are written to disk.

The checkpoint task must present:
1. The surface table (all cells: n, d, sylva_gpu_et_warm_s, best_cpu_warm_s, speedup_ratio, accuracy_delta, oom flags)
2. The pre-registered kill criterion wording (verbatim from `07-PRE-REGISTRATION.md`)
3. A clear binary question: "Does any cell satisfy the win condition (speedup ≥ 5% with accuracy delta < 1%)?"
4. Resume signals for both outcomes:
   - YES: "Proceed to Phase 8. Record the crossover coordinates and publish the surface."
   - NO: "KILL CRITERION fires. Pivot to SHAP/determinism layer atop cuML. Record the decision in STATE.md and ROADMAP.md."

### Pattern 5: oneDAL/sklearnex patching (strongest CPU baseline)

**What:** scikit-learn-intelex (`sklearnex`) patches sklearn's estimators in-place via `patch_sklearn()`. After patching, `sklearn.ensemble.ExtraTreesClassifier` runs on Intel oneDAL. This is the strongest available CPU baseline and must be included.

**When to use:** On the measurement host (after human-verify checkpoint confirms it is installed). If sklearnex is not available on the host, record "oneDAL baseline: not available on measurement host" — do not fake it.

```python
# Source: [ASSUMED] from scikit-learn-intelex documentation
# NOTE: patch_sklearn() must be called BEFORE importing sklearn estimators for the session.
# The benchmark harness should support both patched and unpatched modes.
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX_AVAILABLE = True
except ImportError:
    SKLEARNEX_AVAILABLE = False
    # Record in manifest: sklearnex_version: "not available on measurement host"

# Then: from sklearn.ensemble import ExtraTreesClassifier
# If SKLEARNEX_AVAILABLE, this ExtraTreesClassifier runs on oneDAL.
# Treat this as the "strongest CPU baseline" per PITFALL 2 and PITFALL 13.
```

**Anti-pattern:** Using sklearn without oneDAL as the "strongest CPU baseline" when oneDAL is available. Beating a weaker CPU baseline is benchmark dishonesty (PITFALL 13).

### Anti-Patterns to Avoid
- **Starting the timer after dtype coercion** — omits a real user cost; constitutes benchmark dishonesty (PITFALL 1). Coercion is inside the timed region.
- **Comparing Sylva ET to cuML RF or sklearn RF as if equivalent** — algorithmic dishonesty (PITFALL 13). ET vs ET; RF vs RF.
- **Reporting warm times only without cold** — cold times expose the H2D transfer amortization cost that matters for one-shot users (PITFALL 1, PITFALL 2).
- **Using sklearn `n_jobs=1` as the CPU baseline** — this is the single-threaded strawman baseline (PITFALL 13). Must use `n_jobs=-1`.
- **Hiding OOM cells** — OOM regions are part of the honest surface; their absence would imply Sylva handles arbitrarily large inputs.
- **Moving the goalposts after seeing results** — the grid and kill criterion must be committed before measurement (pre-registration is the structural enforcement).
- **Reporting GPU kernel time or "data already on GPU" time as the speed claim** — the reported number is `fit(X_float64)` wall-clock (PITFALL 1). Any "kernel-only" number may be reported as an internal diagnostic but never as the speed claim.
- **P-hacking the grid** — choosing grid points retroactively to include cells that show wins and exclude cells that show losses. The pre-registered grid is run in full; cells are not dropped from the surface.
- **Reporting speed without accuracy alongside** — every speed cell must have an accuracy number beside it (PITFALL 13 / ROADMAP binding fairness note). A fast model that is less accurate is not a win.
- **Faking or silently skipping the cuML or oneDAL baseline** — gate behind human-verify; record "not available on host" honestly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dataset generation for the grid | A custom generator | `sklearn.datasets.make_classification` + `fetch_covtype` | Already used in Phase-5 `datasets.py`; fixed-seed reproducible; no new code needed |
| Wall-clock timer on Windows | A CUDA event timer | `time.perf_counter()` for the Python API call | `perf_counter` on Windows is high-resolution and includes the full H2D cost; CUDA events measure only device time — exactly what PITFALL 1 says to avoid |
| OOM detection | Custom GPU memory query | `try/except MemoryError` + catch CUDA OOM exception string | Simple, portable, works for both CPU (MemoryError) and GPU (CUDA RuntimeError with "out of memory") |
| Version manifest extension | A new manifest from scratch | Extend `study_manifest.MANIFEST` shape from Phase-5 `study_manifest.py` | Established pattern; add cuML/oneDAL/grid columns to the same dict shape |
| Heatmap | A custom rendering library | `matplotlib.pyplot.imshow` or `seaborn.heatmap` | Standard; produces the surface.png artifact; the markdown table is the canonical artifact |
| Pre-registration tamper-evidence | A third-party timestamp service | Git commit timestamp of `07-PRE-REGISTRATION.md` | The git history is the tamper record; no external service needed |
| Kill-criterion decision logic | Automated code that pivots the project | A `checkpoint:human-verify` task | The kill criterion is a human strategic decision, not an automated code path; the harness only computes the surface and presents it |

**Key insight:** Phase 7 is almost entirely a measurement and documentation phase, not an algorithm phase. The GPU kernel, the Python API, the dispatch contract, and the accuracy parity are all proven upstream. This phase's entire value is: (a) the pre-registered protocol (tamper-evident), (b) the grid loop (systematic coverage), (c) the published surface (the evidence), and (d) the kill-criterion checkpoint (the decision). Hand-rolling any of these supporting utilities would be pure distraction.

---

## Common Pitfalls

### Pitfall 1: Timing the wrong region — "data already on GPU"
**What goes wrong:** The harness starts timing after the data is already a device-resident float32 array. The reported time omits dtype coercion (~n×d×4 bytes of CPU work) and H2D transfer (~n×d×4 bytes over PCIe). This produces a number that looks like a win but collapses when a user tries to replicate it from a standard numpy pipeline.
**Why it happens:** Separating "data prep" from "training" feels intuitive; it mirrors how microbenchmarks are written. But PROJECT.md is explicit: success is "measured end-to-end including transfers."
**How to avoid:** Start `time.perf_counter()` before `X.astype(np.float32)` and end it after `model.fit(X32, y)` returns. This is the timing region in Pattern 1 above. The pre-registration document must specify this exactly.
**Warning signs:** Reported GPU speedup drops by 2-10x when re-measured with coercion inside the timed region. `execution_report_.bytes_h2d` is large relative to training time on small n.

### Pitfall 2: Using sklearn without n_jobs=-1 (the weak-baseline trap)
**What goes wrong:** The CPU baseline is sklearn with default `n_jobs=1` or `n_jobs=None`. On an 8-16 core machine this is 8-16x slower than the actual strongest CPU path. Beating this baseline is benchmark dishonesty (PITFALL 13 from PITFALLS.md).
**Why it happens:** Default `n_jobs` is None (single-threaded) in sklearn. It's easy to forget.
**How to avoid:** The hyperparameter dict in `grid_spec.py` must always include `n_jobs=-1` for sklearn baselines. The pre-registration document must specify this. Add a test assertion: `assert clf_params.get("n_jobs") == -1` before any sklearn baseline run.
**Warning signs:** sklearn baseline times are surprisingly fast at small n or surprisingly slow; suspiciously large speedups at all grid points.

### Pitfall 3: P-hacking the grid (retroactive grid modification)
**What goes wrong:** After seeing preliminary results, the grid is modified to add more large-n cells (where GPU wins) and drop small-n cells (where CPU wins), producing a cherry-picked surface that misrepresents the honest crossover.
**Why it happens:** Confirmation bias; it's tempting to focus on the win region.
**How to avoid:** The pre-registration commit (Task 0) must happen before any measurement run. The pre-registration document specifies the exact grid and is immutable once committed. The surface is reported for all pre-registered cells, including cells where CPU wins and cells that OOM. Any deviation from the pre-registered grid must be documented with a justification.
**Warning signs:** The reported surface omits cells that are in the pre-registration; cells are added to the surface that are not in the pre-registration.

### Pitfall 4: Comparing ET to RF anywhere in the surface
**What goes wrong:** The surface includes a Sylva ET vs cuML RF cell (or a Sylva ET vs sklearn RF cell) as if they are equivalent algorithms. ET and RF are different algorithms: ET uses random split thresholds (no best-split search); RF uses an exact best-split search over quantized bins. They are expected to have different accuracy and different training cost.
**Why it happens:** Both are ensemble methods; the distinction is easy to elide in code. The ROADMAP fairness note is explicit: "compare equivalent algorithms only — ExtraTrees vs ExtraTrees, RF vs RF — never ExtraTrees vs RandomForest as if identical."
**How to avoid:** The `BenchmarkImpl.algorithm` tag (Pattern 2) enforces grouping. The pre-registration document specifies the exact comparison pairs. Add an assertion in the harness that the crossover surface only computes speedup ratios within the same algorithm group.
**Warning signs:** A cell labeled "ET speedup vs cuML RF" in the surface; any cross-algorithm comparison in the results JSON.

### Pitfall 5: Missing the Higgs or Covertype real anchor
**What goes wrong:** The surface is computed only on synthetic `make_classification` grids. The crossover coordinates on real data (with its feature distributions, class imbalance, and structure) may differ significantly from the synthetic grid. Claiming a win on synthetic data and calling it the authoritative crossover lacks ecological validity.
**Why it happens:** Synthetic data is controllable and reproducible; real data introduces download/caching complexity.
**How to avoid:** Include at least Covertype full (581k×54, available via `sklearn.datasets.fetch_covtype`) and a Higgs subset (1M×28, available via OpenML or direct download) as fixed real-data anchor points in the pre-registration. These are the same anchors used in the Phase-5 harness design.
**Warning signs:** Surface JSON contains only `make_classification_*` dataset names; no `covertype` or `higgs` entry.

### Pitfall 6: The kill criterion is soft (no hard pivot)
**What goes wrong:** The kill criterion checkpoint is structured as "note that GPU might not win" rather than as a binding decision point with a documented pivot plan. The project continues broad build-out after a failed crossover because the kill criterion language was vague.
**Why it happens:** It is uncomfortable to encode a project-ending decision. The temptation is to hedge ("marginal results may warrant further investigation").
**How to avoid:** The pre-registration document must contain the exact kill criterion wording (Pattern 4 above), including the specific win condition (e.g., ≥ 5% speedup on ≥ 1 cell with accuracy parity). The checkpoint task must present the binary question to the human and require a recorded decision: "PROCEED" or "PIVOT." The plan must include a documented pivot path (SHAP/determinism layer on cuML) so it is not a crisis if the kill criterion fires.
**Warning signs:** The checkpoint language says "review results" without a specific pass/fail threshold; the pivot path is undefined; the decision is not recorded in STATE.md.

### Pitfall 7: cuML RF is labeled as a GPU ET competitor
**What goes wrong:** The surface includes "Sylva ET vs cuML RF" as a primary comparison row. This is wrong on two levels: (a) ET and RF are different algorithms; (b) cuML RF is a GPU baseline, not a CPU one. The ROADMAP is explicit: cuML RF must be labeled as "a different algorithm" and used only in the RF-vs-RF comparison, not the ET-vs-ET comparison.
**Why it happens:** cuML has no GPU ET implementation, creating a temptation to use cuML RF as the "best available GPU reference for ET."
**How to avoid:** The surface has separate ET-group and RF-group sub-tables. In the ET group, the GPU alternative is "none (cuML has no GPU ET)" — that gap is explicitly noted. In the RF group, cuML RF appears as a GPU RF comparison with the "different algorithm" label.
**Warning signs:** cuML RF appears in the ET-group comparison; "GPU ET speedup vs cuML RF" appears in the surface.

### Pitfall 8: Warm times are over-reported (not truly warm)
**What goes wrong:** "Warm" times are measured after only one prior fit on a different dataset or a much smaller dataset. The GPU caches and allocations are not actually warm for the target (n, d). This produces optimistically fast warm times.
**Why it happens:** It's easy to conflate "the model has been fit once" with "the GPU caches are warm for this exact (n, d) grid point."
**How to avoid:** Warm runs for a given (n, d) cell must all use the SAME (n, d) data. The first call (cold) initializes GPU memory for this (n, d). Subsequent calls (warm) reuse the arena and pinned memory. A cell's warm time is the median of calls 2 through N_REPEATS, all on the same data.

---

## Kill Criterion — Full Specification

> This section is the canonical definition of the KILL CRITERION. The pre-registration document reproduces it verbatim (locked before measurement).

**Trigger condition:** After all pre-registered (n×d) cells have been measured and the results have been written to `results/crossover_results.json`, evaluate:

```
exists (n, d) in PRE_REGISTERED_GRID such that:
  sylva_gpu_et_warm_median(n, d) < best_cpu_et_warm_median(n, d) * (1 - WIN_THRESHOLD)
  AND accuracy_delta(n, d) < ACCURACY_THRESHOLD
  AND NOT sylva_gpu_et_oom(n, d)
  AND NOT best_cpu_et_oom(n, d)
```

Where:
- `WIN_THRESHOLD = 0.05` (5% faster than best CPU baseline warm time)
- `ACCURACY_THRESHOLD = 0.01` (test-set accuracy within 1% of sklearn ET)
- `best_cpu_et_warm_median(n, d) = min(sklearn_et_warm_median(n,d), sklearnex_et_warm_median(n,d))` — strongest CPU baseline

**If True (win region exists):** Crossover is confirmed. Record the coordinates and proceed to Phase 8.

**If False (no win region):** The core premise is NOT confirmed on this hardware. Fire the KILL CRITERION. Document in STATE.md and ROADMAP.md:
> "Phase 7 kill criterion fired: no cell in the pre-registered (n×d) grid showed GPU ET warm fit time ≥ 5% faster than the strongest CPU baseline with accuracy parity. Core premise (GPU ET beats optimized CPU on large dense workloads, end-to-end from numpy, on RTX 4060 Ti) is NOT confirmed. Pivoting to the SHAP/determinism layer (which can ride on cuML) rather than continuing broad GPU training build-out."

**Pivot path (pre-defined if kill fires):** Redirect effort to (a) `sylva-shap` exact tree SHAP attributions consuming ForestIR (Phase 8 is still valuable even without GPU training), (b) the determinism + honest dispatch differentiators (Phase 6 results remain valid), (c) potential upstream contribution to cuML of the SHAP + dispatch work. The ForestIR is stable and Phase 8 / Phase 9 remain executable on the CPU backend.

**Reconciliation with per-phase studies:** The Phase 5 study SC-6/7 produced a preliminary end-to-end comparison on Covertype + large synthetic. If the Phase-7 surface contradicts the Phase-5 data point (e.g., Phase-5 showed a win on Covertype but Phase-7 shows a loss), the crossover report must explain the discrepancy (hyperparameter differences, warm vs cold, sklearnex inclusion, etc.).

---

## Pre-Registration Grid Recommendation

> The exact grid is Claude's discretion per CONTEXT constraints. The planner should encode the following as the pre-registration Task 0 content, subject to discussion with the user.

**n (rows):** {10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000}
**d (features):** {20, 50, 100, 200}
**Total synthetic cells:** 24 (full Cartesian product)
**Real anchors:** Covertype full (~581k×54), Higgs subset (~1M×28) [2 additional points]
**Total cells:** ~26 per implementation per algorithm (ET and RF separate)
**Expected OOM cells:** Large (n=1M, d=200) cells likely OOM for some implementations; record honestly.
**n_estimators = 200, max_depth = 12 for all cells** (matching Phase-5 hyperparameter set from `CLF_HYPERPARAMS`).

**Rationale for grid bounds:**
- Lower bound (n=10k): GPU is expected to lose here; including it establishes the win boundary honestly.
- Upper bound (n=1M, d=200): likely OOM or very slow for some baselines; tests the large-data regime where GPU should win.
- d range: 20 (sparse feature space, fast splits) to 200 (wide matrix, more bins, more histogram work).
- The crossover is expected somewhere in the n=100k–500k range based on the Phase-5 preliminary result and the RTX 4060 Ti HBM bandwidth advantage.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Benchmarking GPU "kernel time" only | End-to-end wall-clock from numpy (`fit(X_f64)`) | PROJECT.md requirement (always) | Includes H2D + coercion; honest claim |
| Single-threaded CPU baseline | `n_jobs=-1` sklearn + oneDAL/sklearnex | PITFALLS.md research (2026-06-19) | Strongest honest CPU baseline |
| "GPU is faster" (universal claim) | Published (n×d) crossover surface with win AND loss regions | This phase | Honest, defensible, not misleading |
| Implicit kill criteria | Pre-registered, exact, committed before measurement | PROJECT.md / PITFALLS.md | Tamper-evident; not post-hoc |
| Benchmarking without accuracy | Accuracy parity reported alongside every speed cell | Binding fairness note (ROADMAP) | Speed without accuracy is not a win |
| cuML RF vs ET as if equivalent | cuML RF labeled "different algorithm", in RF-vs-RF group only | ROADMAP / PITFALLS 13 | Algorithmic honesty |

**Deprecated/outdated:**
- "Data already on GPU" benchmarking mode is not acceptable as the reported number. It may exist as an internal diagnostic but must never be in published results.
- Single-run timing (n_repeats=1) is not acceptable; statistical spread (IQR) must be reported.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The crossover times `deterministic=False` (fast mode) as the primary speed claim. This assumes Phase 6 delivers a real non-deterministic fast path (free multi-stream overlap) distinct from `deterministic=True`. | Timing mode (Summary) | If Phase 6 determined that determinism is the only mode (DET-02 overhead ≈ 0), then there is no distinction and the crossover simply times the only mode. Low risk — either way the harness times `model.fit()` from numpy. |
| A2 | `scikit-learn-intelex` (sklearnex) installs and runs correctly on the Windows measurement host (or under WSL2). | Standard Stack | If sklearnex is unavailable, the strongest CPU baseline falls back to sklearn `n_jobs=-1`. The win bar shifts down (GPU must only beat sklearn, not oneDAL). Must be resolved by human-verify checkpoint and recorded honestly. |
| A3 | cuML RF can be run on the measurement host (WSL2 path) or is recorded as "not available." | Standard Stack / BENCH-02 | If unavailable, the RF-vs-RF GPU comparison is missing. This does not invalidate the ET crossover. Record "cuML RF: not available on measurement host" in the manifest. |
| A4 | The Higgs dataset (~1M×28 float32) is obtainable via OpenML or direct download in the benchmark environment. | Grid spec | If unavailable, use a `make_classification` 1M×28 synthetic anchor instead. Pre-registration document must specify the fallback. |
| A5 | Accuracy parity between Sylva GPU ET and sklearn ET is within 1% on all cells (it was confirmed on Covertype in the Phase-5 design; Phase 7 adds more cells). | Kill criterion accuracy threshold | If accuracy diverges on larger datasets, the kill criterion may fire even on cells where the speed target is met. Accuracy divergence at scale would be a Phase-5/6 correctness bug, not a Phase-7 issue. |
| A6 | The pre-registration document is committed before any measurement cell is run. The git timestamp is the tamper evidence. | Pre-registration | If measurement runs happen before the commit, the pre-registration is meaningless. Must be enforced as Task 0 blocking all other harness tasks. |
| A7 | The crossover exists somewhere in the pre-registered grid (optimistic baseline assumption, not a requirement). | Kill criterion | If it does not, the kill criterion fires and the project pivots. This is explicitly the expected possible outcome — planning must include the documented pivot path. |

---

## Open Questions

1. **oneDAL/sklearnex on Windows — is it the CPU or GPU accelerated path?**
   - What we know: `scikit-learn-intelex` patches sklearn in-place via `patch_sklearn()`. On Intel CPUs, it routes to oneDAL which uses multi-threading and BLAS optimizations. On AMD CPUs the acceleration is partial. The measurement host (Windows, RTX 4060 Ti) may have an Intel or AMD CPU — unknown.
   - What's unclear: Whether sklearnex provides meaningful acceleration over plain sklearn `n_jobs=-1` on the specific host CPU, and whether `patch_sklearn()` correctly accelerates ExtraTreesClassifier or only RandomForest.
   - Recommendation: The human-verify checkpoint for sklearnex must verify not just `import sklearnex` but also that `ExtraTreesClassifier` is among the patched estimators (`sklearnex.get_patch_map()` or similar). If ET is not patched, use sklearn `n_jobs=-1` as the only ET CPU baseline. Record the result in the manifest.

2. **Higgs dataset availability and reproducibility**
   - What we know: The Higgs boson dataset (~11M rows × 28 features) is standard in ML benchmarking; subsets of 1M rows are commonly used. It is available via OpenML (dataset ID 23512) or direct download.
   - What's unclear: Whether OpenML download is reliable from the measurement host without a proxy; whether a consistent 1M-row reproducible subset can be obtained.
   - Recommendation: Pre-register a specific subset strategy: "rows 0..1_000_000 of the full Higgs dataset as downloaded from OpenML ID 23512, seed-shuffled with DATASET_SEED before splitting." Fall back to `make_classification(n=1_000_000, n_features=28, ...)` if OpenML is unavailable; document the substitution.

3. **What mode does Phase 6 land in? (does `deterministic=False` exist?)**
   - What we know: Phase 6 was designed to add a `deterministic: bool` toggle and measure the Sylva-vs-Sylva overhead. The Phase-6 research document marks "Open Question 2: does a non-deterministic faster path exist?" as resolved: "add a `deterministic=false` fast mode."
   - What's unclear: Phase 6 has not yet executed (STATE.md: phases 3–6 pending). The actual shipped behavior may differ.
   - Recommendation: Phase 7 harness should pass `deterministic=False` to Sylva; if this param is not supported in Phase-6's shipped estimator API, fall back to default params and document. The crossover times whatever the default (fastest honest) mode is.

4. **The exact win threshold in the kill criterion (5% vs a different value)**
   - What we know: The pre-registration requires a specific threshold. 5% is the proposed value.
   - What's unclear: Whether 5% is the right threshold given measurement noise (IQR of repeated runs). If the IQR exceeds 5%, a "win" and a "loss" are statistically indistinguishable.
   - Recommendation: Set `WIN_THRESHOLD = 0.05` in the pre-registration. Additionally require that the win cell shows a speedup that exceeds the observed IQR of the GPU warm times. If no cell exceeds both thresholds, the kill criterion fires. Include this IQR requirement in the pre-registration wording.

5. **Should the surface publish both warm and cold? Which wins the "crossover" label?**
   - What we know: BENCH-01 requires both cold and warm. The kill criterion is based on warm (amortized cost, the relevant metric for repeated training runs in a pipeline).
   - What's unclear: Whether a cold-only win (GPU wins on first call) but warm loss is a meaningful result.
   - Recommendation: The primary crossover surface is **warm times**. Cold times are reported in the full table but do not change the kill criterion outcome. A cold-only win is interesting but not sufficient (it means the GPU only wins when CUDA context initialization amortizes over a single fit — not the typical use case).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Sylva wheel | ✓ (per Phase-1/5 completion) | 3.10+ (pinned in venv-parity) | — |
| CUDA 12.8 + RTX 4060 Ti + driver 595.79 | GPU timing cells | ✓ (per VERSIONS.md) | CUDA 12.8, sm_89 | — (no GPU = no GPU cells) |
| scikit-learn 1.9.0 | CPU baseline | ✓ (confirmed in .venv-parity) | 1.9.0 | — (required) |
| scikit-learn-intelex / sklearnex | oneDAL CPU baseline | `[ASSUMED]` unknown | unknown | "oneDAL not available" — skip, record honestly |
| cuML (cuml-cu12) | GPU RF baseline | `[ASSUMED]` unavailable on Windows | — | "cuML not available on host" — skip, record (WSL2 path) |
| matplotlib + seaborn | Heatmap visualization | `[ASSUMED]` standard | any recent | Markdown table only (heatmap is optional) |
| Covertype dataset | Real anchor | ✓ (used in Phase-5 `datasets.py`) | via sklearn.datasets.fetch_covtype | — |
| Higgs dataset (OpenML) | Real anchor (large n) | `[ASSUMED]` | unknown | `make_classification(1M×28)` fallback |
| Sylva wheel (Phase 6 output) | The thing being benchmarked | Not yet (Phase 6 pending) | Phase 6 HEAD at execution time | — (required) |

**Missing dependencies with no fallback:**
- scikit-learn (required; already present)
- CUDA runtime + Sylva GPU build (required for GPU cells; Phase 6 must complete first)

**Missing dependencies with fallback:**
- sklearnex: if unavailable, sklearn `n_jobs=-1` is the only CPU baseline (document as "oneDAL not tested on host")
- cuML: if unavailable on Windows, WSL2 path or "not available" documented honestly
- Higgs dataset: fall back to `make_classification(1M×28)` synthetic anchor

---

## Validation Architecture

> `workflow.nyquist_validation` is `true` in `.planning/config.json` — this section is required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (Python) |
| Config file | none — see Wave 0 |
| Quick run command | `cd python && python -m pytest tests/test_crossover_accuracy_parity.py -q` |
| Full suite command | `cd python && python benchmarks/crossover_study.py` (dev box; not CI) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | Harness measures end-to-end from numpy (H2D + coercion + train), cold + warm | Integration (dev box) | `python python/benchmarks/crossover_study.py --dry-run` | ❌ Wave 0 |
| BENCH-01 | Manifest records all pinned versions + hardware + grid spec | Unit | `pytest tests/test_crossover_manifest.py -q` | ❌ Wave 0 |
| BENCH-01 | Timed region includes dtype coercion (float64→float32 inside timer) | Unit | assert in `crossover_study.py` (static assertion + test) | ❌ Wave 0 |
| BENCH-02 | Accuracy parity: Sylva ET within tolerance of sklearn ET (CI-portable, device='cpu') | Unit/Integration | `pytest tests/test_crossover_accuracy_parity.py -q` | ❌ Wave 0 |
| BENCH-02 | Fairness rules asserted: no ET-vs-RF comparison in results JSON | Unit | `pytest tests/test_crossover_fairness_rules.py -q` | ❌ Wave 0 |
| BENCH-02 | baseline_availability checkpoint resolves sklearnex + cuML presence | Manual gate | `checkpoint:human-verify` | ❌ Wave 0 |
| BENCH-03 | Pre-registration document committed before measurement run | Manual gate | `checkpoint:human-verify` (Task 0) | ❌ Wave 0 |
| BENCH-03 | Surface JSON written after grid run (all cells present including OOM) | Integration (dev box) | verify `results/crossover_results.json` exists and is complete | ❌ Wave 1 (run phase) |
| BENCH-03 | Kill-criterion decision recorded (PROCEED or PIVOT) in STATE.md | Manual gate | `checkpoint:human-verify` (post-measurement) | ❌ Wave 2 |
| BENCH-03 | Crossover surface published (markdown table + heatmap PNG) | Integration | `python python/benchmarks/visualize_surface.py` | ❌ Wave 2 |

### Fairness-Rule Assertions (baked into the harness — not separate tests)

The harness (`crossover_study.py`) must assert at runtime:
- [ ] All sklearn baselines have `n_jobs=-1` in params
- [ ] No cell's `algorithm_comparison` field crosses ET→RF or RF→ET
- [ ] Every speed cell has an accuracy field alongside it
- [ ] OOM cells are present in the JSON (not silently dropped)
- [ ] The timed region starts before `X.astype(np.float32)` and ends after `model.fit()` returns

### Sampling Rate
- **Per task commit:** `cd python && python -m pytest tests/test_crossover_manifest.py tests/test_crossover_accuracy_parity.py tests/test_crossover_fairness_rules.py -q`
- **Per wave merge:** full CI test suite + `python python/benchmarks/crossover_study.py --dry-run` (parses, imports, runs 1 cell on cpu)
- **Phase gate:** Full grid measurement on dev box + kill-criterion checkpoint + surface published

### Wave 0 Gaps
- [ ] `python/benchmarks/crossover_study.py` — main harness (BENCH-01/03)
- [ ] `python/benchmarks/crossover_manifest.py` — extended manifest (BENCH-01/02)
- [ ] `python/benchmarks/grid_spec.py` — GRID constant + dataset loaders for grid sizes
- [ ] `python/benchmarks/visualize_surface.py` — heatmap + markdown table generator (BENCH-03)
- [ ] `python/tests/test_crossover_accuracy_parity.py` — accuracy gate, CI-portable (BENCH-02)
- [ ] `python/tests/test_crossover_manifest.py` — manifest completeness check (BENCH-01)
- [ ] `python/tests/test_crossover_fairness_rules.py` — fairness-rule assertion tests (BENCH-02/13)
- [ ] `.planning/phases/07-crossover-benchmark-gate-3/07-PRE-REGISTRATION.md` — pre-registration document (BENCH-03, Task 0 blocking gate)
- [ ] `results/` directory — for surface artifacts (BENCH-03)
- [ ] Framework install: `pip install matplotlib seaborn` (in benchmark env)

*(Accuracy-parity unit tests may reuse `python/tests/parity/conftest.py` and `datasets.py` fixtures from Phase 5 — these exist.)*

---

## Security Domain

> `security_enforcement: true` in config.json; `security_asvs_level: 1`; security_block_on: "high".

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A — offline benchmark, no auth surface |
| V3 Session Management | no | N/A — no sessions |
| V4 Access Control | no | N/A — local dev tool |
| V5 Input Validation | partial | Validate n, d, hyperparameter ranges in `grid_spec.py` before constructing datasets (prevent accidental out-of-memory by out-of-range grid values) |
| V6 Cryptography | no | N/A |

### Known Threat Patterns for this Phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Benchmark dishonesty (unfair comparison) | Repudiation | Pre-registration commit before measurement; fairness-rule assertions in harness code |
| Result JSON tampering (modifying published surface after measurement) | Tampering | Git commit of results with timestamp; document that results are immutable once committed |
| pip install of benchmark baseline packages (sklearnex, cuML) | Tampering | Gate behind `checkpoint:human-verify`; these are study-env-only packages, not Sylva runtime deps; verify package source before install |
| Unsafe numpy/array indexing if grid dimensions exceed GPU memory silently | Elevation of Privilege | OOM guard in every `safe_timed_fit` call; catch and record, never crash silently |
| Apache-2.0 compliance | Information Disclosure | Never copy GPL / Snap ML source; all baseline packages must be Apache-2.0 or BSD-compatible; cuML is Apache-2.0 confirmed |

---

## Sources

### Primary (HIGH confidence)
- `.planning/ROADMAP.md` — Phase 7 success criteria (6 criteria), KILL CRITERION exact wording, Comparative Baseline Study spec, comparative-study fairness note (binding) — the authoritative source for this phase
- `.planning/REQUIREMENTS.md` — BENCH-01, BENCH-02, BENCH-03 exact text
- `.planning/research/PITFALLS.md` — Pitfalls 1, 2, 13 (binding fairness rules, timing region, baseline strength, algorithm parity)
- `.planning/research/SUMMARY.md` — ecosystem context, stack, competitive gaps
- `.planning/STATE.md` — fairness-blocker note; per-phase study design pattern; current state (phase 3 executing)
- `.planning/phases/05-full-forest-randomforest-sklearn-estimators/05-06-PLAN.md` — Phase-5 harness design (`comparative_study.py`, `study_manifest.py`); the exact precursor this phase generalizes
- `.planning/phases/06-determinism-honest-dispatch/06-RESEARCH.md` — Phase-6 `deterministic=False` fast mode design (DET-02); timing mode for Phase-7 crossover cells
- `python/tests/parity/conftest.py` — VERSION_MANIFEST pattern; dataset fixtures reused in Phase-7 accuracy gate
- `python/tests/parity/datasets.py` — dataset loaders, DATASET_SEED, hyperparameter sets

### Secondary (MEDIUM confidence)
- `[ASSUMED]` scikit-learn-intelex (sklearnex) install and behavior on Windows — Windows support documented in official Intel repos but not verified on the measurement host this session
- `[ASSUMED]` cuML on Windows — historically Linux-first; WSL2 path documented in Phase-5 OQ7 resolution
- `[ASSUMED]` Higgs dataset availability via OpenML on the measurement host

### Tertiary (LOW confidence)
- `[ASSUMED]` exact crossover coordinates — empirically unknown; that is what Phase 7 measures; no prior-session measurement

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Sylva + sklearn are proven in earlier phases; baselines are industry-standard
- Architecture (harness design): HIGH — extends the Phase-5 pattern which is already designed; pre-registration structure is standard benchmark practice
- Pitfalls (fairness rules): HIGH — all grounded in PITFALLS.md binding research + ROADMAP fairness note
- Kill criterion mechanics: HIGH — exact wording reproduced from ROADMAP; decision structure is unambiguous
- Baseline availability (sklearnex, cuML): LOW — environment-dependent; not verified on measurement host in this session
- Crossover outcome: LOW — empirically unknown (that is the purpose of Phase 7)

**Research date:** 2026-06-27
**Valid until:** Until Phase 6 ships (at which point the exact Sylva API + dispatch params must be verified against shipped code); indefinitely for the structural patterns (pre-registration, timing region, fairness rules, kill criterion).
