# Feature Research

**Domain:** GPU-native tree-ensemble ML library (Extra Trees + Random Forest) with sklearn-parity API and exact high-depth tree SHAP
**Researched:** 2026-06-19
**Confidence:** MEDIUM-HIGH (sklearn parity surface: HIGH; WOODELF-HD SHAP: MEDIUM, verified against real published sources; competitive surface: MEDIUM)

---

## Headline Verification Result (read first)

The single highest-risk claim in `PROJECT.md` — the "WOODELF-HD" exact high-depth tree SHAP engine — **is NOT fabricated. It maps to real, published, verifiable work.** Two corrections to the project blueprint are required, and one strategic implication must reach the roadmap:

1. **The method is real.** "Woodelf" is a unified SHAP algorithm published at **AAAI 2026** (Alexander Nadel, Ron Wettenstein), with the long-form paper *"From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm"* (arXiv [2511.09376](https://arxiv.org/abs/2511.09376)). **"WoodelfHD"** is the high-depth extension, *"WOODELF-HD: Efficient Background SHAP for High-Depth Decision Trees"* (arXiv [2604.10569](https://arxiv.org/html/2604.10569v1), 2026, Wettenstein / Nadel / Boker, Reichman University + Technion).
2. **Complexity correction.** The blueprint's `O(2^D · D)` is slightly off. The *published* WoodelfHD background-SHAP time complexity is **`O(mTL + nTLD + TL·2^D·D²)`** (space `O(2^D·D)`), improving on original Woodelf's `O(...+ TL·3^D·D)`. The headline win is the **3^D → 2^D** reduction via the Strassen-like scheme. Use `2^D·D²`, not `2^D·D`, in any requirement or perf model.
3. **Strategic implication (important).** WoodelfHD **already has a GPU implementation by its own authors** ("WoodelfHD GPU achieves 3 hrs at depth 21"). So the differentiator is **NOT "invent GPU exact high-depth SHAP"** — that exists. The wedge is **"ship exact high-depth SHAP integrated natively into a from-scratch GPU forest, on Apache-2.0, as a first-class part of the estimator (`.shap_values()` / matching FIL-served models)."** The roadmap must not budget research time for *inventing* the algorithm; it must budget for *porting / re-implementing a published algorithm* (check the upstream license before porting — see Anti-Features & Sources).

The "UFDP" and "Strassen-like" terms in the blueprint are also real: **UFDP = Unique-Feature Decision Pattern** (one bit per unique feature on a root→leaf path; gives the `O(D)` space term), and the **Strassen-like multiplication** exploits `M₁=0, M₄=M₂+M₃` block structure to compute two inner products instead of four, vectorized in `O(D)` operations. Both are described in arXiv 2604.10569.

---

## Feature Landscape

### Table Stakes (Users Expect These)

These are the minimum for the phrase "scikit-learn-compatible" to be honest. Missing any of the core set means users cannot drop the estimator into existing pipelines, and the project's entire "don't leave the sklearn idiom" value prop collapses. Confidence on this whole table is **HIGH** — the sklearn estimator contract is stable and well-documented ([Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html)).

#### Core estimator contract (the four classes)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `fit(X, y)` returning `self` | Universal sklearn entry point; required by Pipeline/GridSearchCV | HIGH | Must set fitted attrs with trailing underscore; must validate X/y |
| `predict(X)` | Baseline inference | MEDIUM | Classifier returns class labels (not indices); regressor returns float |
| `predict_proba(X)` (classifiers) | Needed for ROC/AUC, calibration, soft voting; cuML historically lagged here | MEDIUM | Mean of per-tree leaf class distributions; must sum to 1 per row |
| `predict_log_proba(X)` (classifiers) | sklearn parity; trivial once `predict_proba` exists | LOW | `log(predict_proba)` |
| `score(X, y)` | Used by `cross_val_score`, default CV | LOW | accuracy (clf) / R² (reg); inherit from mixin semantics |
| `apply(X)` | Returns leaf indices per tree; used by feature-engineering pipelines | MEDIUM | Optional-ish but expected from forest estimators |
| `decision_path(X)` | sklearn forest API completeness | HIGH | Sparse path matrix; defer-able but expected eventually |

#### Constructor parameters — drop-in parity set

These must exist with **matching names, defaults, and semantics** or `clone()`/`get_params()`-driven tooling (GridSearchCV, pipelines) silently misbehaves. Semantics matter more than presence: a param that exists but means something different is worse than absent.

| Param | Why Expected | Complexity | Notes |
|-------|--------------|------------|-------|
| `n_estimators` (default 100) | Defines the ensemble | LOW | Embarrassingly parallel over trees on GPU |
| `criterion` (`gini`/`entropy`/`log_loss` clf; `squared_error`/`absolute_error`/`friedman_mse`/`poisson` reg) | Core split objective | MEDIUM–HIGH | **Scope decision: MVP = `gini`+`entropy` (clf), `squared_error` (reg).** `absolute_error`/`poisson` are expensive on GPU; defer with a clear `NotImplementedError`, not silent substitution |
| `max_depth` (None) | Primary capacity control | MEDIUM | Unbounded depth interacts with SHAP cost (`2^D`); see dependencies |
| `min_samples_split` (2) | Stopping rule | LOW | int or float fraction |
| `min_samples_leaf` (1) | Stopping rule; affects leaf stats | MEDIUM | int or float; constrains split acceptance |
| `min_weight_fraction_leaf` (0.0) | Weighted stopping | MEDIUM | Couples to `sample_weight` (see below) |
| `max_features` (`sqrt` clf / `1.0` reg) | The defining RF/ET randomization knob | MEDIUM | Accept `sqrt`,`log2`,None,int,float; per-node feature subsampling |
| `max_leaf_nodes` (None) | Best-first growth cap | HIGH | Best-first growth is a different scheduler than depth-first; **candidate defer** |
| `min_impurity_decrease` (0.0) | Split acceptance threshold | LOW | Cheap to honor in split scoring |
| `bootstrap` (True RF / **False ET**) | Distinguishes RF from ET sampling | MEDIUM | ET default False is a key semantic difference |
| `max_samples` (None) | Subsample size when bootstrapping | LOW | int/float; only valid when `bootstrap=True` |
| `oob_score` (False) | Out-of-bag generalization estimate | HIGH | cuML notably lacks this ([cuML #3361](https://github.com/rapidsai/cuml/issues/3361)) — parity opportunity, but costs a second pass; **defer to v1.x** |
| `n_jobs` (None) | CPU parity knob | LOW | Maps to host orchestration; mostly a no-op on the GPU path but must be accepted |
| `random_state` (None) | Reproducibility seed | HIGH | **This is where the determinism differentiator lives** — see Differentiators |
| `verbose` (0) | Logging | LOW | |
| `warm_start` (False) | Incremental tree addition | HIGH | **Anti-feature for v1** — see below; param must still be accepted and rejected honestly if True |
| `class_weight` (None, clf) | Imbalanced data | MEDIUM | `balanced`/`balanced_subsample`/dict; couples to sample weighting |
| `ccp_alpha` (0.0) | Cost-complexity pruning | HIGH | Minimal cost-complexity pruning is nontrivial on GPU; **defer**, accept+reject `>0` |

#### Fitted attributes (trailing underscore)

| Attribute | Why Expected | Complexity | Notes |
|-----------|--------------|------------|-------|
| `classes_` (clf) | Label decoding; required by many tools | LOW | Sorted unique labels |
| `n_classes_` (clf) | Parity | LOW | |
| `n_features_in_` | **Required by modern check_estimator**; input validation | LOW | Set in `fit` |
| `feature_names_in_` | Set when X is a DataFrame | LOW | Parity with sklearn ≥1.0 |
| `n_outputs_` | Multi-output bookkeeping (v1 single-output) | LOW | =1 for v1 |
| `estimators_` | List of fitted trees; introspection, plotting, SHAP | MEDIUM | Need a tree object with sklearn-ish `tree_` structure for ecosystem tools |
| `feature_importances_` | Most-used forest attribute; cuML returns NaN after `as_sklearn()` conversion | MEDIUM | Gini/MDI importance; **parity opportunity** — cuML's gap ([search result](https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.ensemble.randomforestclassifier/)) is a credibility win |
| `oob_score_` / `oob_decision_function_` | Only if `oob_score=True` | HIGH | Defer with the `oob_score` param |
| `estimators_samples_` | Bootstrap index introspection | LOW | |

#### Estimator protocol / "what sklearn-compatible actually requires"

| Requirement | Why Expected | Complexity | Notes |
|-------------|--------------|------------|-------|
| `get_params(deep=True)` / `set_params(**params)` | **Necessary** for GridSearchCV, Pipeline; `set_params` must return `self` | LOW | Free if you inherit `BaseEstimator` and store all `__init__` args unmodified on `self` (no validation/mutation in `__init__`) |
| Cloneable via `sklearn.base.clone` | GridSearchCV deep-copies estimators before each fit | LOW | Works automatically iff `__init__` only stores params verbatim (the "no logic in `__init__`" rule) |
| `__init__` stores args without transformation | Hard requirement for clone/get_params to round-trip | LOW | Common failure mode; enforce in tests |
| Mixin tags (`ClassifierMixin`/`RegressorMixin`, `__sklearn_tags__`) | Tells sklearn the estimator type; affects scoring & checks | LOW | Use `_estimator_type` / tags API |
| Passing `sklearn.utils.estimator_checks.check_estimator` | The objective definition of "compatible" | HIGH | Run it in CI as a gate. Expect to either pass or **explicitly document each skipped check with reason**. Some checks (sparse input, sample_weight invariance) will fail by design given v1 scope — use `_xfail_checks` / documented expected-failures rather than faking support |
| `__repr__` from BaseEstimator | Notebook/printing parity | LOW | Free via BaseEstimator |
| Picklability / serialization | Save/load trained models | MEDIUM | Must serialize GPU-trained tree structure to host; couples to Treelite export |

> **`sample_weight` decision (called out by the brief):** `fit(X, y, sample_weight=...)` is **table stakes for genuine parity** and is exercised by `check_estimator`. However, weighted histogram accumulation adds real GPU complexity (weighted bin sums, weighted impurity, weighted leaf values). **Recommendation: MVP = accept `sample_weight=None` cleanly and raise a clear error for non-None; promote to v1.x.** This is defensible *only if* the determinism/SHAP differentiators land — without sample_weight the estimator will fail some `check_estimator` invariance checks, so document it as a known, scoped gap rather than pretending. Flag for requirements: this is a parity-vs-effort tradeoff the product owner must ratify.

---

### Differentiators (Competitive Advantage)

These align directly with `PROJECT.md` Core Value. The project does **not** win on raw speed alone (cuML RF is mature); it wins on the combination below. Confidence MEDIUM (competitive landscape verified June 2026; exact cuML gaps confirmed via issue tracker).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **GPU-native Extra Trees (clf+reg)** | The actual ecosystem gap: cuML focuses on RF; ExtraTrees on GPU is underserved. Random thresholds delete RF's atomic-heavy best-split search → most GPU-amenable ensemble | HIGH | The core wedge. Shared histogram engine with RF; ET is the "easy seam" (no per-feature argmax over thresholds) |
| **Exact high-depth tree SHAP (`.shap_values()`, WoodelfHD)** | Verifiable, exact (not sampled), scales past depth ~12 where classic TreeSHAP/GPUTreeSHAP degrade; first-class in the estimator | VERY HIGH | **Verified real method** (AAAI 2026 + arXiv 2604.10569). Complexity `O(mTL+nTLD+TL·2^D·D²)`. Background (interventional) SHAP primary; path-dependent also supported. **Risk: must re-implement a published algo; check upstream code license before porting (see Anti-Features).** |
| **Deterministic GPU training contract (`deterministic=True`)** | Bit-reproducible models — atomics-based GPU histograms are normally nondeterministic; almost nobody offers a *guaranteed* reproducible GPU forest | HIGH | Requires deterministic reduction order (no race-y atomic adds), fixed RNG streams per (tree,node,feature). Documented perf cost. This is a genuine, defensible differentiator |
| **Non-silent device dispatch + `execution_report_`** | `device="auto"|"cuda"|"cpu"`, `fallback="error"`; never silently runs on CPU or silently up/down-casts. Audit-friendly vs cuml.accel/H2O4GPU silent fallback | MEDIUM | `execution_report_` records every device decision + dtype conversion. Cheap to build, high trust value, strong story for regulated users |
| **`feature_importances_` that actually works** | cuML returns NaN after sklearn conversion; this returns real MDI importances | MEDIUM | Low cost, direct parity win |
| **Variance-based redundant-tree pruning** | Drop trees that don't reduce ensemble variance → smaller/faster models at equal accuracy | MEDIUM | **Method provenance UNVERIFIED in literature** — "blueprint fix #5". Treat as an *engineering heuristic to be validated empirically*, not a cited technique. Flag: requires its own small validation study before it's advertised |
| **Treelite-compatible export → FIL serving** | Train on GPU here, serve anywhere via the standard inference path (FIL/Treelite) | MEDIUM | Treelite model spec is stable and Apache-2.0-friendly. Export = serialize tree structure to Treelite's format. Unlocks the broader RAPIDS inference ecosystem |
| **CPU reference backend (correctness oracle)** | Differential-testable against sklearn; small-data path; the thing the non-silent dispatch falls *toward* (explicitly) | MEDIUM | Doubles as the oracle for the determinism/SHAP tests |
| **Zero-copy GPU input (CUDA Array Interface / DLPack)** | Accept CuPy/cuDF/torch GPU arrays without host round-trip | MEDIUM | Honors the "measure transfers end-to-end" constraint; strong for users already on-GPU |

> **Differentiator priority for the wedge:** GPU-native ExtraTrees + Deterministic mode + non-silent dispatch are the *defensible core* (hard for cuML to instantly match, low fabrication risk). Exact high-depth SHAP is the *headline feature* but carries port-not-invent + licensing diligence. Variance pruning is the *weakest* differentiator (unverified provenance) — keep it optional and validate before marketing.

---

### Anti-Features (Commonly Requested, Often Problematic)

All ratified in `PROJECT.md` Out of Scope; restated with feature-level reasons and the honest-rejection requirement (each accepted-but-unsupported param must raise a clear error, never silently no-op).

| Feature | Why Requested | Why Problematic (v1) | Alternative |
|---------|---------------|----------------------|-------------|
| Gradient boosting (GBDT) | "Trees on GPU = XGBoost" expectation | Sequential, saturated market (XGBoost 3.3/LightGBM/CatBoost own it); no defensible wedge | Point users to XGBoost; stay bagging-only |
| Sparse / CSR input | Real-world tabular is often sparse | Changes missing-vs-zero semantics, histogram construction, row partitioning — near-rewrite | Dense float32 only; post-MVP RFC |
| Native categorical features | CatBoost-style convenience | Target-statistics encoding out of MVP reach; semantics-heavy | User one-hot/ordinal-encodes upstream |
| Multi-GPU / multi-node | "Scale to huge data" | Massive complexity before single-GPU value is proven | Single GPU; revisit after crossover validated |
| `warm_start=True` | Incremental tree addition | Mutable-fit state fights the immutable/deterministic model contract; extra bookkeeping | Refit; accept param, **error if True** |
| Monotonic constraints | Regulated/feature-direction needs | Constrained split search complicates the GPU histogram-eval hot path | Defer; not in sklearn ET/RF anyway |
| `sample_weight` | Imbalanced data, parity, `check_estimator` | Weighted histograms add real GPU complexity | **MVP: error on non-None; promote v1.x** (flagged tradeoff above) |
| `ccp_alpha` pruning | sklearn parity | Minimal cost-complexity pruning is a serial tree-walk; awkward on GPU | Accept, error if `>0`; v1.x |
| `max_leaf_nodes` best-first growth | sklearn parity | Different scheduler than depth-first growth | Depth-first only in v1; accept None |
| SpMM / Tensor-Core histograms | "Use the GPU's matmul units" | **Debunked premise** — tree training is bandwidth/atomic-bound, no GEMM in hot path; Tensor Cores idle | Optimize for memory bandwidth + atomic contention instead |
| INT8/INT4 quantized inference, selected matrix inversion, BLAS autotuning (blueprint fixes #1,#3,#4,#6) | Listed in the incoherent "7 fixes" blueprint | Belong to other subfields (DL inference, sparse linalg, spatial stats) — not tree training | Out of scope permanently |

---

## Feature Dependencies

```
[Histogram / split-evaluation engine]  (prefix-sum scan kernels, blueprint fix #7)
    ├──requires──> [Feature quantization / binning]
    └──enables───> [GPU Random Forest]
                       └──simplifies-into──> [GPU Extra Trees]  (random thresholds = no best-split argmax)

[GPU Extra Trees / RF training]
    ├──requires──> [random_state / RNG stream design]
    │                   └──enables──> [deterministic=True mode]  (deterministic reduction order)
    ├──produces──> [estimators_ / tree_ structures]
    │                   ├──requires-for──> [feature_importances_]
    │                   ├──requires-for──> [Treelite/FIL export]
    │                   └──requires-for──> [exact high-depth SHAP (WoodelfHD)]
    └──requires──> [CPU reference backend]  (differential-test oracle)

[Non-silent device dispatch + execution_report_]
    ├──wraps──> [all fit/predict entry points]
    └──requires──> [CPU reference backend]  (the explicit fallback target)

[sklearn estimator contract: get_params/set_params/clone/check_estimator]
    └──requires──> [__init__ stores params verbatim]  (no logic in __init__)

[exact high-depth SHAP]
    ├──cost-scales-with──> [max_depth]  (2^D · D² term)
    └──conflicts-with──> [unbounded max_depth + very deep trees]  (SHAP becomes intractable past ~D≈21)

[variance-based tree pruning] ──enhances──> [GPU Extra Trees / RF]  (optional, post-train, UNVERIFIED method)

[zero-copy GPU input] ──enhances──> [GPU training]  (avoids host round-trip; honors end-to-end transfer measurement)
```

### Dependency Notes

- **Histogram engine → ExtraTrees → RandomForest:** Build the shared histogram/split-scoring engine first. **ExtraTrees is genuinely simpler** (random split thresholds → no per-feature argmax over candidate thresholds, far fewer atomics), so the recommended build order is *engine → ExtraTrees → add RF best-split search*. This inverts the naive "RF first" instinct and matches `PROJECT.md`'s "Extra Trees is the seam" thesis.
- **`random_state` → `deterministic=True`:** Determinism is not a bolt-on. RNG stream layout (per tree/node/feature) and deterministic reduction order must be designed into the histogram kernels from the start; retrofitting determinism onto race-y atomic adds is a rewrite.
- **`estimators_`/tree structures → importances, export, SHAP:** All three downstream features consume the trained tree representation. Design the in-memory tree format once, with Treelite-compatibility and SHAP-traversal needs in mind, so it isn't re-shaped three times.
- **SHAP cost ⟂ depth:** The `2^D·D²` term means SHAP is the feature that makes `max_depth` a *first-class cost knob*. Document that exact high-depth SHAP is practical to ~D≈21 (the paper's GPU figure) and budgets blow up beyond that.
- **`check_estimator` ⟂ scoped gaps:** Because `sample_weight`, sparse, and `ccp_alpha` are deferred, `check_estimator` will surface expected failures. Wire it into CI with explicit documented expected-failures rather than removing the gate.
- **CPU backend underpins two differentiators:** It's both the differential-test oracle (vs sklearn) and the explicit target of non-silent dispatch. It is not optional polish; it is load-bearing.

---

## MVP Definition

### Launch With (v1)

- [ ] **Histogram/split engine + prefix-sum scan kernels** — the bandwidth-bound core that everything sits on
- [ ] **`ExtraTreesClassifier` + `ExtraTreesRegressor`** (dense float32, single GPU) — the wedge
- [ ] **`RandomForestClassifier` + `RandomForestRegressor`** — shared engine, adds best-split search
- [ ] **Drop-in sklearn contract**: `fit/predict/predict_proba/predict_log_proba/score`, `get_params/set_params`, clone-able, core constructor params (`n_estimators`, `max_depth`, `max_features`, `min_samples_*`, `bootstrap`, `max_samples`, `criterion`={gini,entropy,squared_error}, `random_state`, `n_jobs`, `class_weight`), core fitted attrs (`classes_`, `n_classes_`, `n_features_in_`, `feature_names_in_`, `n_outputs_`, `estimators_`, `feature_importances_`) — *essential or the value prop is void*
- [ ] **`deterministic=True` mode** — primary differentiator, must be designed in from day one
- [ ] **Non-silent device dispatch + `execution_report_`** (`device`, `fallback="error"`) — cheap, high-trust differentiator
- [ ] **CPU reference backend** — correctness oracle + dispatch target
- [ ] **`check_estimator` CI gate** with documented expected-failures for scoped gaps
- [ ] **Differential tests vs sklearn + property-based invariants** — proves "matches semantics, not just accuracy"

### Add After Validation (v1.x)

- [ ] **Exact high-depth tree SHAP (WoodelfHD)** — *headline* feature, but gated on (a) successful port/re-implementation of the published algo and (b) **upstream license diligence** before any code reuse. Could be promoted into v1 if licensing/port risk clears early — flag for requirements owner. Trigger: core forest validated + crossover benchmark passes.
- [ ] **Treelite/FIL export** — trigger: tree format stabilized; unlocks serving story
- [ ] **`sample_weight` support** — trigger: weighted-histogram kernel proven; closes a `check_estimator` gap
- [ ] **`oob_score` / `oob_score_`** — trigger: second-pass cost acceptable; direct cuML-parity gap
- [ ] **Zero-copy GPU input (CUDA Array Interface/DLPack)** — trigger: on-GPU users request it
- [ ] **Variance-based redundant-tree pruning** — trigger: **its own empirical validation study passes** (unverified method; do not ship as a headline claim until measured)

### Future Consideration (v2+)

- [ ] `ccp_alpha`, `max_leaf_nodes` (best-first growth), `absolute_error`/`poisson` criteria — full sklearn-param closure
- [ ] Sparse/CSR input — post-MVP RFC (near-rewrite)
- [ ] Native categorical, multi-output-heavy, multi-GPU — deferred per PROJECT.md
- [ ] `warm_start` — only if a concrete incremental use case emerges

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Histogram/split engine + scan kernels | HIGH | HIGH | P1 |
| ExtraTrees clf+reg (GPU) | HIGH | HIGH | P1 |
| RandomForest clf+reg (GPU) | HIGH | HIGH | P1 |
| sklearn drop-in contract (params/attrs/methods/clone) | HIGH | MEDIUM | P1 |
| `deterministic=True` mode | HIGH | HIGH | P1 |
| Non-silent dispatch + `execution_report_` | MEDIUM | LOW | P1 |
| CPU reference backend | MEDIUM | MEDIUM | P1 |
| `feature_importances_` (working) | MEDIUM | LOW | P1 |
| `check_estimator` gate + differential tests | HIGH | MEDIUM | P1 |
| Exact high-depth SHAP (WoodelfHD) | HIGH | VERY HIGH | P2 (port + license risk) |
| Treelite/FIL export | MEDIUM | MEDIUM | P2 |
| `sample_weight` | MEDIUM | MEDIUM | P2 |
| `oob_score` | LOW | HIGH | P2 |
| Zero-copy GPU input | MEDIUM | MEDIUM | P2 |
| Variance-based tree pruning | LOW–MEDIUM | MEDIUM | P3 (unverified) |
| ccp_alpha / max_leaf_nodes / extra criteria | LOW | HIGH | P3 |

---

## Competitor Feature Analysis

| Feature | scikit-learn (CPU) | cuML / RAPIDS (GPU) | Our Approach |
|---------|--------------------|--------------------|--------------|
| Extra Trees on GPU | N/A (CPU only) | **Underserved** (RF-focused) | **Native first-class — the wedge** |
| Random Forest on GPU | N/A | Mature (cuML 26.6 + cuml.accel) | Shared engine; parity semantics; not trying to out-speed cuML RF alone |
| `predict_proba` | Yes | Yes (historically lagged, [#836](https://github.com/rapidsai/cuml/issues/836)) | Yes, day one |
| `feature_importances_` | Yes | **NaN after sklearn conversion** | **Yes, working — parity win** |
| `oob_score` | Yes | **Missing** ([#3361](https://github.com/rapidsai/cuml/issues/3361)) | v1.x — parity win |
| Deterministic GPU training | Deterministic (CPU) | Not guaranteed (atomic nondeterminism) | **Guaranteed bit-reproducible mode — differentiator** |
| Silent CPU fallback | N/A | cuml.accel can fall back silently | **Explicit, non-silent, `execution_report_` — differentiator** |
| Exact tree SHAP | TreeSHAP `O(TLD²)` (shap lib, CPU) | GPUTreeSHAP (up to 19× SHAP / 340× interactions, V100) | **WoodelfHD-style exact high-depth, integrated in-estimator** |
| Exact SHAP past depth ~12–21 | Degrades | GPUTreeSHAP still `O(TLD²)`-style | **WoodelfHD `2^D·D²` background SHAP scales deeper** |
| Inference/serving | sklearn predict | FIL + Treelite (mature) | Export to Treelite → reuse FIL ecosystem |

### Exact-SHAP reality check (grounded alternatives, for the roadmap)

If WoodelfHD porting/licensing proves infeasible, the **verified fallbacks** are:
1. **Lundberg TreeSHAP** — exact, `O(TLD²)` (T trees, L leaves, D depth), CPU; the `shap` library standard. Real, citable ([Lundberg et al., *Explainable AI for Trees*](https://arxiv.org/pdf/1905.04610)).
2. **GPUTreeSHAP** (Mitchell, Frank, Holmes; NVIDIA + Waikato; arXiv [2010.13972](https://arxiv.org/abs/2010.13972), PeerJ CS) — exact GPU reformulation of TreeSHAP via bin-packing variable-size subproblems into SIMT tasks; up to **19× (SHAP) / 340× (interactions)** vs multicore CPU on a V100. Apache-2.0, [rapidsai/gputreeshap](https://github.com/rapidsai/gputreeshap). **This is the safe, proven GPU exact-SHAP baseline** the roadmap can fall back to without literature risk.
3. **Linear TreeSHAP** (Yu, Xu, Bifet, Read; NeurIPS 2022) — exact, lower-order alternative formulation.

So the SHAP feature is *grounded either way*: the ambitious path (WoodelfHD, scales to deep trees) is real and verified; the conservative path (GPUTreeSHAP) is real, GPU-ready, and Apache-2.0.

---

## Sources

**sklearn parity surface (confidence HIGH):**
- [Developing scikit-learn estimators — official dev guide](https://scikit-learn.org/stable/developers/develop.html) (get_params/set_params, clone via `__sklearn_clone__`, BaseEstimator, no-logic-in-`__init__`)
- [sklearn.utils.estimator_checks.check_estimator](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html)
- ExtraTrees/RandomForest constructor signatures (sklearn 1.x stable API; author knowledge, HIGH confidence)

**WOODELF / WOODELF-HD exact SHAP (confidence MEDIUM — verified real, cross-checked):**
- [WOODELF-HD: Efficient Background SHAP for High-Depth Decision Trees, arXiv 2604.10569 (2026)](https://arxiv.org/html/2604.10569v1) — `O(mTL+nTLD+TL·2^D·D²)`, Strassen-like scheme (`M₁=0, M₄=M₂+M₃`), UFDP = Unique-Feature Decision Pattern, CPU+GPU, depth-21 in ~3 hrs on GPU
- [From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm, arXiv 2511.09376](https://arxiv.org/abs/2511.09376) — base Woodelf, AAAI 2026 (Nadel & Wettenstein), linear-time background SHAP, 162s CPU / 16s GPU on 3M×127

**Grounded SHAP alternatives (confidence MEDIUM–HIGH):**
- [GPUTreeShap, arXiv 2010.13972 / PeerJ CS](https://arxiv.org/abs/2010.13972) and [rapidsai/gputreeshap (Apache-2.0)](https://github.com/rapidsai/gputreeshap)
- [Lundberg et al., Explainable AI for Trees (TreeSHAP O(TLD²)), arXiv 1905.04610](https://arxiv.org/pdf/1905.04610)
- [Linear TreeSHAP, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/a5a3b1ef79520b7cd122d888673a3ebc-Paper-Conference.pdf)

**Competitive surface (confidence MEDIUM):**
- [cuML RandomForestClassifier docs (26.06)](https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.ensemble.randomforestclassifier/)
- [cuML #3361 — OOB & feature importance gap](https://github.com/rapidsai/cuml/issues/3361), [#836 — predict_proba history](https://github.com/rapidsai/cuml/issues/836)

---

## Flags for Requirements Definition (downstream consumer)

1. **`sample_weight`** — recommend MVP defer (error on non-None) vs full parity. Tradeoff against `check_estimator` invariance failures. **Owner must ratify.**
2. **Exact high-depth SHAP** — method is REAL but requires *re-implementing/porting a published algorithm* + **upstream license diligence** before code reuse. Not "invent"; budget as "port + validate." Conservative fallback = GPUTreeSHAP (Apache-2.0, proven).
3. **Variance-based tree pruning** — **method provenance UNVERIFIED** in literature ("blueprint fix #5"). Treat as engineering heuristic; require its own validation study before advertising. Keep P3/optional.
4. **Complexity number correction** — use `O(2^D·D²)` (not `2^D·D`) for SHAP perf modeling; SHAP practical to ~D≈21.
5. **Build order** — engine → ExtraTrees → RandomForest (ET is the simpler seam), not the naive RF-first order.
6. **Determinism is not a bolt-on** — RNG stream layout + deterministic reduction order must be in the kernel design from phase one.

---
*Feature research for: GPU-native tree-ensemble library with sklearn parity + exact high-depth tree SHAP*
*Researched: 2026-06-19*
