# Phase 9: Treelite Export & Packaging - Research

**Researched:** 2026-06-27
**Domain:** Treelite 4.x ModelBuilder export, TL2cgen compiled CPU inference, abi3 Windows wheel packaging, FIL/nvForest GPU inference availability
**Confidence:** MEDIUM (Treelite 4.x API verified against docs; FIL Windows gap confirmed; maturin wheel approach proven from Phase 1; `import_from_json()` removal confirmed indirectly — schema is ASSUMED)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXP-01 | `sylva-export` serializes `ForestIR` to a Treelite 4.x `import_from_json()`-compatible JSON | **CRITICAL FINDING:** `import_from_json()` was a Treelite 3.x API — it does not appear in Treelite 4.x docs or `frontend.py`. The 4.x path is the Python `ModelBuilder` API. The requirement text references 3.x; the implementation must use 4.x `ModelBuilder`. See §Critical Finding below and §Architecture Patterns Pattern 1. |
| EXP-02 | Exported model round-trips through Treelite/FIL and produces matching predictions (CI test) | FIL is Linux/WSL2 only. Windows CI path uses TL2cgen (compiled CPU). See §Environment Availability and §Pitfall 3. |
| EXP-03 | `abi3` Windows wheel validated in a fresh environment with CUDA driver dynamic-loading; documented install path | Directly follows the Phase 1 proven pattern; generalizes to the full `sylva` package. See §Pattern 3. |
</phase_requirements>

---

## Summary

Phase 9 makes Sylva models **portable** (Treelite export → TL2cgen/FIL inference) and the library **installable** (abi3 Windows wheel). It is an IR-only consumer: it reads `ForestIR` produced by Phase 5 (now stable) and adds no training logic.

**Critical finding that rewrites the phase framing:** CLAUDE.md references `Treelite 4.x import_from_json()`-compatible JSON, and REQUIREMENTS.md EXP-01 inherits that language. However, research confirms that `import_from_json()` is a **Treelite 3.x API**. In Treelite 4.x (current stable: 4.7.0), the programmatic model-creation path is the **Python `ModelBuilder` API** (`treelite.model_builder.ModelBuilder`). The function `import_from_json()` does not appear in the Treelite 4.x Python `frontend.py` or any 4.x documentation page. The practical implementation of EXP-01 therefore means: write a Python `sylva-export` module (or function) that constructs a `treelite.Model` from `ForestIR` using `ModelBuilder`, not by serializing to a bespoke JSON string.

**Second critical finding — FIL on Windows:** FIL and the replacement `nvForest` (RAPIDS) do not run natively on Windows — they require Linux or WSL2. The EXP-02 round-trip CI test on Windows must use **TL2cgen** (compiled CPU path, toolchain `'msvc'`) as the testable inference target. FIL/nvForest can be documented as the GPU inference target for Linux/cloud serving, tested in a separate Linux CI step behind a `checkpoint:human-verify`.

**Third finding — packaging:** Phase 1 proven the abi3 + dynamic-loading wheel for the spike crate. Phase 9 generalizes this to the full `sylva` package (sylva-core + sylva-cuda + Python estimators). The wheel validation pattern is identical; CUDA toolkit remains a runtime prerequisite (not bundled).

**Primary recommendation:** Implement `sylva-export` as a Python module (installable with the wheel) that uses `treelite.ModelBuilder` to build a `treelite.Model` from `ForestIR`. The round-trip CI test calls `treelite.gtil.predict()` (pure Python inference, works on Windows) or TL2cgen for the compiled path. Write an INSTALL.md documenting the runtime prerequisites.

## Critical Finding: `import_from_json()` is a 3.x API

The requirement text `"Treelite 4.x import_from_json()-compatible JSON"` reflects the MEDIUM-confidence item flagged in CLAUDE.md ("schema details need verification against Treelite 4.x docs during the export phase"). That verification has now been performed:

- Treelite 4.0 removed the JSON-string import path (`import_from_json()` does not appear in the 4.x API docs or `treelite/python/treelite/frontend.py` on the mainline branch). [CITED: treelite.readthedocs.io/en/latest/treelite-api.html]
- The 4.x programmatic model creation path is `treelite.model_builder.ModelBuilder` with `start_tree()`, `start_node()`, `numerical_test()`, `leaf()`, `commit()`. [CITED: treelite.readthedocs.io/en/4.1.2/tutorials/builder.html]
- The 3.x JSON format (with `task_param`, `model_param`, node `split_feature_id`, etc.) is the input format for `import_from_json()` which no longer exists in 4.x.

**Plan implication:** EXP-01 should be re-read as "serialize ForestIR to a Treelite 4.x `Model` object via `ModelBuilder`." There is no JSON intermediary needed. The planner should flag this to the user for confirmation as an open question.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ForestIR → Treelite Model conversion | Python `sylva-export` module | Rust pyseam (reads IR) | ForestIR is already JSON-serializable via serde; the Treelite `ModelBuilder` is a Python API — mapping lives in Python. |
| Round-trip CI test (predict parity) | Python test (`pytest`) | TL2cgen on Windows | Testing the export against Treelite's own inference (gtil or TL2cgen) is a Python-layer concern. |
| abi3 wheel build | Rust + maturin (same as Phase 1) | pyproject.toml update | Build infrastructure already proven; Phase 9 updates the package scope. |
| FIL/nvForest GPU inference | Linux/WSL2 only | Not CI-testable on Windows | RAPIDS does not support native Windows; this is a deployment target, not a local CI target. |
| INSTALL.md / docs | Documentation | — | Runtime prerequisites (CUDA toolkit, NVIDIA driver) must be documented clearly. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| treelite | 4.7.0 | Tree model exchange format; `ModelBuilder` to build `Model` from ForestIR; `gtil.predict()` for pure-Python inference check | `[VERIFIED: pip index versions treelite]` latest stable; DMLC/RAPIDS maintained; the universal serving format cited in CLAUDE.md |
| tl2cgen | 1.0.0 | Compile treelite `Model` to C code + shared lib (.dll on Windows); `Predictor` for CPU inference round-trip | `[VERIFIED: pip index versions tl2cgen]` split from Treelite 4.0; official Windows MSVC toolchain support |
| maturin | 1.14.1 | Build abi3 wheel for the full `sylva` package (already installed) | `[VERIFIED: maturin --version]` the proven Phase 1 wheel tool; same version |
| serde + serde_json | 1.x (workspace) | ForestIR serialization already in place; may also serialize the `sylva` model checkpoint | `[VERIFIED: crates/sylva-core/Cargo.toml]` already used in the crate |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| treelite.gtil | bundled with treelite 4.7 | Pure-Python inference on a treelite `Model` (no compilation needed); fastest way to test round-trip parity | Use for the primary CI round-trip test on Windows (no MSVC compilation step) |
| numpy | ≥1.25 (runtime dep) | Feed `np.float32` arrays to `treelite.gtil.predict()` and `tl2cgen.Predictor.predict()` | Always required for array I/O |
| pytest | 7+ | Host the round-trip parity test in CI | Standard test framework |
| nvforest | (Linux/WSL2 only) | GPU inference via RAPIDS `nvForest`; loads treelite `Model` objects; deprecates FIL | Use only in Linux/WSL2 CI or serving documentation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| treelite 4.x `ModelBuilder` | treelite 3.x JSON `import_from_json()` | 3.x is outdated; 4.x is the maintained API; 3.x JSON format is frozen and unsupported going forward |
| TL2cgen for CI round-trip | treelite.gtil only | `gtil` is simpler (no compilation), suitable for CI; TL2cgen exercises the compiled inference path and is worth a separate throughput benchmark plan |
| nvForest (FIL replacement) | Old cuml.fil | `cuml.fil` is deprecated; `nvForest` is the forward path — but both Linux-only |

**Installation (development environment):**
```bash
pip install treelite==4.7.0 tl2cgen==1.0.0
```

**Version verification:**
```bash
pip index versions treelite    # 4.7.0
pip index versions tl2cgen     # 1.0.0
```

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| treelite | PyPI | ~6 yrs (0.90 → 4.7.0) | Unknown via seam | github.com/dmlc/treelite (Apache 2.0, 800+ stars) | SUS (seam: no repo URL in PyPI metadata) | **Cleared — official DMLC/RAPIDS project; well-known in ML serving ecosystem; Apache 2.0; 4.7.0 released 2026-03-06** |
| tl2cgen | PyPI | ~3 yrs | Unknown via seam | github.com/dmlc/tl2cgen | SUS (seam: no repo URL) | **Cleared — official DMLC split from Treelite 4.0; part of the same RAPIDS/DMLC ecosystem; 1.0.0 stable** |
| maturin | PyPI | 5+ yrs | Unknown via seam | github.com/pyo3/maturin (seam-returned) | SUS (seam: too-new for 1.14.1) | **Cleared — official PyO3 project; already proven in Phase 1; 1.14.1 installed** |

**Packages removed due to SLOP verdict:** none

**Packages flagged as suspicious SUS:** All three cleared as well-known projects. The seam's SUS signals are false positives (PyPI does not always expose repository URLs in metadata; the packages are established DMLC/RAPIDS and PyO3 projects).

*Note: Download counts unavailable through the seam for PyPI packages without repository URL in metadata. These packages are confirmed via official documentation, GitHub repositories, and direct pip index verification.*

## Architecture Patterns

### System Architecture Diagram

```
ForestIR (from Phase 5 training)
         |
         v
[sylva-export Python module]
  - reads ForestIR JSON handle (via pyseam or direct Rust call)
  - maps ForestIR fields → ModelBuilder API calls
         |
         v
treelite.Model (in-memory Python object)
    /           \
   v             v
treelite.gtil   tl2cgen.generate_c_code()
  .predict()          |
(pure Python,         v
 CI primary)    tl2cgen.create_shared('msvc', ...)
                      |
                      v
               model.dll (Windows) / model.so (Linux)
                      |
                      v
              tl2cgen.Predictor.predict(X)
               (compiled CPU inference)

[Linux/WSL2 only]
treelite.Model → nvForest.load_model(model, device='gpu')
                 → fm.predict(X)   (GPU inference)
```

Data flow: ForestIR (read-only consumer) → Python ModelBuilder calls → treelite.Model → inference validation.

### Recommended Project Structure
```
python/
├── sylva/
│   ├── __init__.py
│   ├── _estimators.py      # sklearn-parity estimators (Phase 5)
│   └── export.py           # sylva-export: ForestIR → treelite.Model
tests/
├── test_export.py          # EXP-01/02 round-trip parity tests
└── test_wheel.py           # EXP-03 wheel import smoke (run in clean venv)
docs/
└── INSTALL.md              # CUDA prerequisites, platform notes
```

### Pattern 1: ForestIR → Treelite ModelBuilder (EXP-01)

**What:** Walk the ForestIR SoA arrays tree-by-tree, node-by-node, calling `ModelBuilder` to reconstruct the tree structure.

**ForestIR → Treelite field mapping:**

| ForestIR field | Treelite ModelBuilder call / parameter | Notes |
|---|---|---|
| `n_features` | `Metadata(num_feature=n_features)` | |
| `task` (Classification n_classes) | `task_type="kMultiClf"` or `"kBinaryClf"` | Multi-class: kMultiClf; binary: kBinaryClf |
| `task` (Regression) | `task_type="kRegressor"` | |
| `n_trees` | `TreeAnnotation(num_tree=n_trees)` | |
| classifier, n_classes leaves | `leaf_vector_shape=(1, n_classes)`, `class_id=[-1]` | Each tree outputs a class-probability vector |
| regressor scalar leaves | `leaf_vector_shape=(1, 1)`, `class_id=[0]` | |
| `feature_id[n]` | `numerical_test(feature_id=...)` | Per-node split feature |
| `threshold[n]` | `numerical_test(threshold=...)` | Per-node split threshold |
| `default_child[n] == left_child[n]` | `numerical_test(default_left=True/False)` | NaN direction |
| `left_child[n]` (tree-local) | `numerical_test(left_child_key=...)` | Must reindex to tree-local node IDs |
| `right_child[n]` (tree-local) | `numerical_test(right_child_key=...)` | |
| `"<"` (always — ExtraTrees/RF split) | `numerical_test(opname="<")` | Always `<` for numerical splits |
| `leaf_proba[offset*nc .. (offset+1)*nc]` | `leaf([p0, p1, ..., pn])` vector | Classifier leaf probabilities |
| `leaf_value[leaf_offset[n]]` | `leaf(v)` scalar | Regressor leaf value |
| `node_sample_count[n]` | `builder.data_count(count)` (optional metadata) | Can be set after `start_node()` |
| `node_weighted_count[n]` | `builder.sum_hess(v)` (optional metadata) | |
| `impurity[n]` | `builder.gain(-impurity)` (optional metadata) | Sign convention: gain = reduction in impurity; check Treelite convention |

**Key constraint:** ForestIR uses **global** node IDs across all trees. `ModelBuilder` uses **tree-local** node IDs. The export function must reindex: `local_id = global_id - tree_offsets[t]`.

**Classifier postprocessor:** sklearn ET/RF predict_proba outputs class probabilities directly (no sigmoid/softmax applied post-average). For sklearn parity with `average_tree_output=True`, use `postprocessor=PostProcessorFunc(name="identity")` with `kMultiClf` — each leaf already stores probabilities summing to 1, and Treelite averages them. Using `"softmax"` would re-normalize and break parity.

**Example (classifier):**
```python
# Source: treelite.readthedocs.io/en/4.1.2/tutorials/builder.html
from treelite.model_builder import (
    Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation
)

def forest_ir_to_treelite(ir: ForestIR) -> treelite.Model:
    n_classes = ir.n_classes()
    is_clf = isinstance(ir.task, Task.Classification)

    if is_clf:
        task_type = "kMultiClf" if n_classes > 2 else "kBinaryClf"
        # For sklearn parity: use vector leaves with identity postprocessor
        # Each tree leaf already holds averaged class probabilities
        leaf_vector_shape = (1, n_classes) if n_classes > 2 else (1, 1)
        postprocessor = "identity"   # [ASSUMED — verify against gtil output]
        class_id = [-1] * ir.n_trees if n_classes > 2 else [0] * ir.n_trees
        num_class = [n_classes] if n_classes > 2 else [1]
    else:
        task_type = "kRegressor"
        leaf_vector_shape = (1, 1)
        postprocessor = "identity"
        class_id = [0] * ir.n_trees
        num_class = [1]

    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=ir.n_features,
            task_type=task_type,
            average_tree_output=True,  # RF/ET: average across trees
            num_target=1,
            num_class=num_class,
            leaf_vector_shape=leaf_vector_shape,
        ),
        tree_annotation=TreeAnnotation(
            num_tree=ir.n_trees,
            target_id=[0] * ir.n_trees,
            class_id=class_id,
        ),
        postprocessor=PostProcessorFunc(name=postprocessor),
        base_scores=[0.0] * (num_class[0] if is_clf else 1),
    )

    for t in range(ir.n_trees):
        start = ir.tree_offsets[t]
        end = ir.tree_offsets[t + 1]
        builder.start_tree()
        for global_id in range(start, end):
            local_id = global_id - start
            builder.start_node(local_id)
            if ir.is_leaf[global_id]:
                if is_clf and n_classes > 2:
                    o = ir.leaf_offset[global_id]
                    proba = ir.leaf_proba[o * n_classes:(o + 1) * n_classes]
                    builder.leaf(list(proba))
                elif is_clf:  # binary
                    o = ir.leaf_offset[global_id]
                    builder.leaf(float(ir.leaf_proba[o * 2 + 1]))  # P(class=1)
                else:
                    o = ir.leaf_offset[global_id]
                    builder.leaf(float(ir.leaf_value[o]))
            else:
                lc_local = ir.left_child[global_id] - start
                rc_local = ir.right_child[global_id] - start
                default_left = (ir.default_child[global_id] == ir.left_child[global_id])
                builder.numerical_test(
                    feature_id=int(ir.feature_id[global_id]),
                    threshold=float(ir.threshold[global_id]),
                    default_left=default_left,
                    opname="<",
                    left_child_key=lc_local,
                    right_child_key=rc_local,
                )
                builder.data_count(int(ir.node_sample_count[global_id]))
                builder.sum_hess(float(ir.node_weighted_count[global_id]))
            builder.end_node()
        builder.end_tree()

    return builder.commit()
```

**[ASSUMED] items in the above example:**
- The exact postprocessor for sklearn-parity multi-class (identity vs softmax) — depends on whether ForestIR leaf_proba already sums to 1 (it does per ForestIR contract) and whether `gtil.predict` applies postprocessor before or after averaging
- Binary classifier leaf representation — whether `kBinaryClf` with scalar `P(class=1)` leaves + `postprocessor='sigmoid'` or `postprocessor='identity'` gives `predict_proba`-parity
- Whether `builder.data_count()` / `builder.sum_hess()` calls happen inside `start_node()/end_node()` or outside

These are open questions the planner must gate behind a spike task before the full export implementation.

### Pattern 2: Round-Trip CI Test (EXP-02)

**What:** After building a `treelite.Model`, validate that predictions match Sylva's native `predict`.

```python
# Primary path: treelite.gtil.predict (pure Python, works on Windows)
import treelite
import numpy as np

tl_model = forest_ir_to_treelite(ir)
X_test = np.random.default_rng(42).random((200, ir.n_features), dtype=np.float32)

# Sylva native predict
sylva_pred = sylva.ExtraTreesClassifier._from_ir(ir).predict_proba(X_test)

# Treelite GTIL predict
tl_pred = treelite.gtil.predict(tl_model, X_test)

np.testing.assert_allclose(sylva_pred, tl_pred, atol=1e-5)

# Secondary path (TL2cgen compiled, Windows)
import tl2cgen, tempfile, pathlib
with tempfile.TemporaryDirectory() as tmp:
    tl2cgen.generate_c_code(tl_model, tmp, params={})
    libpath = tl2cgen.create_shared("msvc", tmp)  # produces .dll on Windows
    pred = tl2cgen.Predictor(libpath).predict(
        tl2cgen.DMatrix(X_test)
    )
    np.testing.assert_allclose(sylva_pred, pred, atol=1e-5)
```

### Pattern 3: abi3 Windows Wheel Validation (EXP-03)

The Phase 1 proof (01-03-PLAN.md) is the direct template. Phase 9 generalizes it to the full package.

```bash
# Build the release wheel (same as Phase 1 proven pattern)
maturin build --release
# → target/wheels/sylva-*.cp310-abi3-win_amd64.whl

# Validate in a clean venv
py -m venv .venv-smoke-09
.venv-smoke-09\Scripts\pip install target\wheels\sylva-*.cp310-abi3-win_amd64.whl
.venv-smoke-09\Scripts\python -c "import sylva; print(sylva.__version__)"
# → should print version and exit 0 without CUDA error
# (dynamic-loading means CUDA driver resolved at runtime if available)
```

**CUDA as runtime prerequisite:** The wheel does NOT bundle CUDA DLLs. The user must have:
1. An NVIDIA GPU with a driver supporting CUDA 12.x+
2. CUDA Toolkit 12.x installed (for NVRTC at runtime — the cudarc `nvrtc` feature)
3. The CUDA dynamic libraries must be on PATH or in the system DLL search path

These must be documented in `INSTALL.md`.

### Anti-Patterns to Avoid
- **Passing global node IDs to ModelBuilder:** ModelBuilder expects tree-local IDs. Global IDs cause wrong tree topology. Always subtract `tree_offsets[t]`.
- **Using `postprocessor="softmax"` for RF classifier leaves that already store probabilities:** sklearn ET/RF leaves store per-class probabilities normalized to 1.0. Applying softmax again renormalizes and breaks parity.
- **Serializing ForestIR to 3.x JSON format and calling `import_from_json()`:** That function is gone in Treelite 4.x. Do not build a JSON string targeting the 3.x schema.
- **Bundling CUDA DLLs in the wheel:** cudarc `dynamic-loading` feature means CUDA resolves at runtime. Never statically embed CUDA libraries (license and size issues).
- **Testing FIL/nvForest in native Windows CI:** RAPIDS is Linux/WSL2 only. Gate behind `checkpoint:human-verify` and document as a Linux-only step.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tree model → compiled inference | Custom LLVM/C codegen | `tl2cgen.generate_c_code()` + `create_shared()` | TL2cgen handles SIMD, OpenMP, and platform differences with a known-good MSVC path |
| Tree model serving format | Custom JSON schema + parser | `treelite.ModelBuilder` + `treelite.Model` | Treelite is the standard; FIL, TL2cgen, and nvForest all consume it |
| Pure-Python tree inference | Custom NumPy traversal | `treelite.gtil.predict()` | gtil handles model metadata (task type, postprocessor, averaging) correctly |
| Wheel build + abi3 wrangling | Custom distutils extension | `maturin build --release` | Phase 1 proved this; maturin handles MSVC + abi3 + dynamic-loading |

**Key insight:** The Treelite `ModelBuilder` API is the correct abstraction point for custom-trained models. Every downstream consumer (TL2cgen, nvForest, Triton FIL) loads a `treelite.Model` — building to that API once enables all serving paths.

## Common Pitfalls

### Pitfall 1: Global vs. Tree-Local Node IDs
**What goes wrong:** ForestIR uses global node IDs across all trees (`tree_offsets` carves them). `ModelBuilder` uses IDs scoped to the current tree. Passing global IDs to `left_child_key`/`right_child_key` points to nodes in other trees.
**Root cause:** SoA layout is global for cache-friendliness; Treelite's builder is tree-scoped for encapsulation.
**How to avoid:** Always compute `local_id = global_id - tree_offsets[t]` before passing to `ModelBuilder`. Validate by checking `local_id >= 0 && local_id < (tree_offsets[t+1] - tree_offsets[t])`.
**Warning signs:** `treelite.gtil.predict` raises an index-out-of-bounds or returns NaN for all rows.

### Pitfall 2: Binary Classifier Leaf Semantics
**What goes wrong:** `kBinaryClf` with `sigmoid` postprocessor expects raw logit scores in leaves. Sylva's `leaf_proba` stores `[P(class=0), P(class=1)]` — already probabilities, not logits. Applying `sigmoid` again gives wrong predictions.
**Root cause:** sklearn ET/RF leaves store probabilities; gradient boosting leaves store logits. Treelite's `sigmoid` postprocessor assumes logits.
**How to avoid:** For binary classifiers, either use `kMultiClf` with 2-class vector leaves (stores `[P(class=0), P(class=1)]`) and `postprocessor="identity"`, or use `kBinaryClf` with `P(class=1)` scalar leaves and `postprocessor="identity"` (not sigmoid). The round-trip parity test (EXP-02) catches this immediately.
**Warning signs:** Binary classifier predictions are `sigmoid(P(class=1))` ≈ 0.6–0.7 for inputs that should be 0.9.

### Pitfall 3: FIL on Windows (Platform Mismatch)
**What goes wrong:** CI job attempts `from cuml.fil import ForestInference` on Windows — import fails because RAPIDS requires Linux or WSL2.
**Root cause:** RAPIDS does not ship Windows-native wheels.
**How to avoid:** Gate FIL tests behind a `checkpoint:human-verify` labeled "Linux/WSL2 only". Use `treelite.gtil.predict()` + TL2cgen for the Windows CI round-trip.
**Warning signs:** `ModuleNotFoundError: No module named 'cuml'` in Windows CI.

### Pitfall 4: TL2cgen MSVC Compilation Timeout
**What goes wrong:** `tl2cgen.create_shared("msvc", dirpath)` is a build step that invokes cl.exe — can be slow for large forests.
**Root cause:** MSVC compilation of generated C code is not instantaneous.
**How to avoid:** Use a small held-out test dataset (50 trees, max_depth=6) for the round-trip CI test, not the full production forest. The parity gate is on correctness, not speed.
**Warning signs:** CI job times out at the `create_shared()` call.

### Pitfall 5: postprocessor Interacts with average_tree_output
**What goes wrong:** Using `postprocessor="softmax"` with `average_tree_output=True` applies softmax AFTER averaging, not per-tree — different semantics than per-leaf softmax.
**Root cause:** Treelite applies postprocessor on the averaged output.
**How to avoid:** For sklearn-parity ET/RF classifiers whose leaves store class probabilities (already summing to 1): use `postprocessor="identity"` and `average_tree_output=True`. Let the parity test catch any deviation.

### Pitfall 6: Wheel Requires CUDA at Import Time (wrong cudarc feature)
**What goes wrong:** Using the `cuda-static` feature instead of `dynamic-loading` causes the wheel to fail import in a clean venv that has no CUDA runtime linked.
**Root cause:** `cuda-static` links CUDA libraries at compile time.
**How to avoid:** The Phase 1 proof locked the shipping config to `cudarc dynamic-loading`. Phase 9 uses the same feature. Verify in the wheel's CI: `python -c "import sylva"` must succeed even before any GPU is accessed (dynamic-loading defers CUDA init to first use).

## Code Examples

### Full Export Function Sketch
```python
# Source: treelite.readthedocs.io/en/4.1.2/tutorials/builder.html (ModelBuilder API)
# + research mapping from ForestIR to Treelite
import numpy as np
import treelite
from treelite.model_builder import (
    Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation
)

def export_to_treelite(ir_json: str) -> treelite.Model:
    """Convert a Sylva ForestIR JSON handle to a treelite.Model.
    
    [ASSUMED] binary classifier postprocessor needs empirical verification.
    """
    import json
    ir_dict = json.loads(ir_json)
    # ... parse ir_dict fields ...
    # (or accept a ForestIR Rust struct via pyseam)
    ...
```

### Round-Trip Parity Test
```python
# Source: treelite.readthedocs.io/en/latest (gtil.predict)
import treelite, numpy as np, pytest

def test_export_round_trip_classifier(trained_clf_ir):
    X_test = make_test_data(trained_clf_ir.n_features)
    sylva_proba = sylva_predict_proba(trained_clf_ir, X_test)
    
    tl_model = export_to_treelite(trained_clf_ir)
    tl_proba = treelite.gtil.predict(tl_model, X_test.astype(np.float32))
    
    np.testing.assert_allclose(sylva_proba, tl_proba, atol=1e-5,
        err_msg="Treelite round-trip failed: postprocessor mismatch?")

def test_tl2cgen_round_trip(trained_clf_ir, tmp_path):
    """TL2cgen compiled-CPU round-trip (Windows CI path)."""
    import tl2cgen
    tl_model = export_to_treelite(trained_clf_ir)
    tl2cgen.generate_c_code(tl_model, str(tmp_path), params={})
    libpath = tl2cgen.create_shared("msvc", str(tmp_path))
    pred = tl2cgen.Predictor(libpath).predict(
        tl2cgen.DMatrix(make_test_data(trained_clf_ir.n_features))
    )
    np.testing.assert_allclose(
        sylva_predict_proba(trained_clf_ir, make_test_data(trained_clf_ir.n_features)),
        pred, atol=1e-5
    )
```

### Clean Venv Import Smoke (EXP-03)
```bash
# Follows Phase 1 01-03-PLAN.md proven pattern exactly
maturin build --release
py -m venv .venv-smoke-09
.venv-smoke-09\Scripts\pip install target\wheels\sylva-*.cp310-abi3-win_amd64.whl
.venv-smoke-09\Scripts\python scripts\import_smoke_09.py
# script: import sylva; print(sylva.__version__); e = sylva.ExtraTreesClassifier(); print("OK")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `treelite.import_from_json()` JSON string | `treelite.ModelBuilder` Python API | Treelite 4.0 (2023) | No JSON intermediary needed; direct programmatic model construction |
| `cuml.fil.ForestInference` (FIL) | `nvforest.load_model()` (nvForest) | RAPIDS ~25.04 (2025) | FIL deprecated; nvForest is the forward path; both Linux-only |
| `treelite.model.compile()` → .so | `tl2cgen.generate_c_code()` + `create_shared()` | Treelite 4.0 (2023) | Code generation moved to separate `tl2cgen` package |
| `treelite_runtime.Predictor` | `tl2cgen.Predictor` | tl2cgen 1.0 (2024) | Runtime module renamed; same API pattern |

**Deprecated:**
- `treelite.import_from_json()`: Gone in 4.x. Do not target the 3.x JSON schema.
- `cuml.fil.ForestInference`: Deprecated; new code uses `nvForest`.
- `model.export_lib()` on treelite Model: Moved to `tl2cgen.export_lib()`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `import_from_json()` is gone in Treelite 4.x (inferred from absence in 4.x docs and frontend.py) | Critical Finding, EXP-01 | If it still exists in 4.7, we can use JSON as an intermediary — lower-risk approach, but MEDIUM confidence that 4.x docs are comprehensive |
| A2 | `postprocessor="identity"` with `kMultiClf` + vector leaves + `average_tree_output=True` gives `predict_proba` parity with sklearn ET/RF | Pattern 1 code example | If wrong, the round-trip parity test fails; the fix is a 1-line parameter change, caught immediately by EXP-02 |
| A3 | Binary classifier (`kBinaryClf`) should use `postprocessor="identity"` with scalar `P(class=1)` leaves (not `sigmoid` + logit leaves) for sklearn parity | Pattern 1 code example, Pitfall 2 | If wrong, binary classifier round-trip test fails; minor fix |
| A4 | `builder.data_count()`/`builder.sum_hess()` are called inside the `start_node()/end_node()` scope | Pattern 1 code example | If wrong, builder raises or ignores them; safe to omit optional metadata if it causes issues |
| A5 | `treelite.gtil.predict()` returns `(n_samples, n_classes)` float32 array matching `predict_proba` format | Pattern 2, test examples | If wrong, shape/format mismatch in parity assert; check gtil docs for exact output shape |
| A6 | TL2cgen `create_shared("msvc", ...)` works with MSVC 2022 build tools already installed | Pattern 2, Pitfall 4 | If cl.exe is not on PATH for the test runner, compilation fails; add a PATH setup step |
| A7 | FIL/nvForest absolutely requires Linux/WSL2 — no native Windows path | Environment Availability, Pitfall 3 | If wrong (nvForest adds Windows support), we can add a Windows GPU inference test; safe assumption for CI planning |
| A8 | The `sylva` package wheel covers sylva-core + sylva-cuda + Python estimators in one manifest | Pattern 3 | Depends on Phase 5 completing the estimator layer; plan must verify pyproject.toml consolidation |

**If this table is empty:** Not applicable — 8 ASSUMED claims exist; planner must gate A1 (the most critical) behind a mini-spike before committing to the full implementation.

## Open Questions (RESOLVED)

> Resolved by the orchestrator before planning; locked in the plans:
> - **Q1 RESOLVED:** target the Treelite 4.x `ModelBuilder` API; a Wave-0 spike (Plan 09-01) confirms whether a 3.x `import_from_json` JSON shim also exists in the installed Treelite — ModelBuilder is the path regardless. (Deviates from the roadmap's literal SC#1 wording — flagged for a later ROADMAP update.)
> - **Q2 RESOLVED:** classifier `task_type=kMultiClf` + `postprocessor="identity"` (leaves already store normalized `leaf_proba`); verified empirically by the Wave-0 spike + the gtil round-trip parity gate (Plan 09-01/09-02).
> - **Q3 RESOLVED:** GPU FIL/nvForest is Linux/WSL2-only → a manual `checkpoint:human-verify` benchmark; `treelite.gtil.predict()` is the Windows-portable primary CI gate, TL2cgen compiled-CPU secondary (Plan 09-02/09-03).
> - **Q4 RESOLVED:** the export glue + estimators ship in the `python/sylva/` package (per Phase 5); export consumes the ForestIR JSON handle via the existing pyseam contract — no new PyO3 binding.

1. **Does `import_from_json()` still exist in Treelite 4.x as a compatibility shim?**
   - What we know: The function does not appear in the 4.7.x API docs or `frontend.py` source (mainline branch). The 3.x tutorial pages at treelite.readthedocs.io/en/3.9.0/tutorials/json_import.html are versioned 3.x only.
   - What's unclear: Whether `import_from_json()` was removed in 4.0 or later, or merely undocumented.
   - Recommendation: The Wave 0 task should be a spike — `pip install treelite==4.7.0 && python -c "import treelite; treelite.import_from_json"` — to definitively confirm removal before designing the export format. If it exists, the 3.x JSON approach is still viable and simpler. If it's gone, `ModelBuilder` is the only path.

2. **Exact postprocessor for sklearn-parity classifier export**
   - What we know: sklearn ET/RF leaves store class probabilities already normalized to sum=1. Treelite `postprocessor="softmax"` would renormalize after averaging.
   - What's unclear: Whether `kMultiClf` + `postprocessor="identity"` + `average_tree_output=True` gives exactly `predict_proba` output, or whether a different task_type/postprocessor combination is needed.
   - Recommendation: The parity test (EXP-02) will catch this immediately. Design the spike to test multiple configurations against a sklearn reference and pick the matching one. Do not assume — verify empirically.

3. **Binary classifier representation (kBinaryClf vs kMultiClf with 2 classes)**
   - What we know: `kBinaryClf` with `postprocessor="sigmoid"` expects logit leaves. Sylva stores probabilities.
   - What's unclear: Whether `kBinaryClf` + `postprocessor="identity"` with `P(class=1)` leaves gives parity, or whether `kMultiClf` with `num_class=[2]` vector leaves is the correct path.
   - Recommendation: Test both in the parity spike. `kMultiClf` with 2-class vector leaves is probably the most explicit and correct option.

4. **FIL/nvForest throughput benchmark — Linux CI or document-only?**
   - What we know: FIL/nvForest is Linux/WSL2 only. The EXP-02 requirement says "Treelite/FIL" but is satisfiable with TL2cgen on Windows.
   - What's unclear: Whether the project should set up a WSL2 CI step for the FIL throughput benchmark, or note it as a manual benchmark.
   - Recommendation: Gate the FIL throughput benchmark behind `checkpoint:human-verify`. On Windows CI, use TL2cgen compiled CPU for the automated round-trip test. Document nvForest as the GPU serving path in INSTALL.md with a Linux/WSL2 note.

5. **Can the Python export module receive a ForestIR from Rust without JSON serialization?**
   - What we know: The current pyseam uses JSON as the FFI handle (a documented decision: "JSON string as ForestIR FFI handle across PyO3 boundary"). Python-side deserializes the JSON to get ForestIR fields.
   - What's unclear: Whether the export function should accept a JSON string (consistent with pyseam) or whether a direct PyO3 binding for the export is better.
   - Recommendation: Accept the JSON string from pyseam (or the estimator's internal `_ir_json` attribute). This avoids a new PyO3 binding and is consistent with the existing seam. The export function is pure-Python: `json.loads(ir_json) → ModelBuilder calls`.

6. **Wheel scope: single `sylva` package or separate `sylva-cuda` + `sylva` packages?**
   - What we know: Phase 1 built `sylva-cuda` as the wheel. Phase 5 adds Python estimators.
   - What's unclear: Whether Phase 9's "full package" means updating `sylva-cuda` to also expose estimators, or creating a new `sylva` package that imports `sylva-cuda` under the hood.
   - Recommendation: Phase 5 resolves this (it adds the full sklearn-parity estimator API). Phase 9 validates whatever structure Phase 5 establishes. The planner should note this dependency on Phase 5 completion.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.14 | Wheel build and tests | ✓ | 3.14.3 | — |
| maturin | Wheel build (EXP-03) | ✓ | 1.14.1 | — |
| Rust stable | Wheel build | ✓ | 1.96.0 | — |
| CUDA Toolkit (nvcc) | Kernel compile verification | ✓ | 13.2 (CUDA 13.2) | — |
| NVIDIA driver | GPU operations | ✓ | 595.79 | — |
| treelite 4.7.0 | EXP-01/02 (not yet installed) | ✗ | — | pip install treelite==4.7.0 |
| tl2cgen 1.0.0 | EXP-02 TL2cgen path (not yet installed) | ✗ | — | pip install tl2cgen==1.0.0 |
| MSVC (cl.exe) | TL2cgen `create_shared("msvc")` | ✓ (present from Phase 1) | VS 2022 | gcc via WSL2 |
| cuML / FIL / nvForest | FIL GPU inference (EXP-02 secondary) | ✗ | — | TL2cgen (Windows CI primary); manual WSL2 step |

**Missing dependencies with no fallback:** None that block execution. All missing packages installable via `pip install`.

**Missing dependencies with fallback:** FIL/nvForest (Linux/WSL2 only → TL2cgen on Windows).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (>=7) |
| Config file | `pytest.ini` or `pyproject.toml [tool.pytest]` |
| Quick run command | `pytest tests/test_export.py -x -q` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXP-01 | ForestIR → treelite.Model builds without error; `gtil.predict` returns array | unit | `pytest tests/test_export.py::test_export_builds -x` | ❌ Wave 0 |
| EXP-01 | Exported model has correct num_feature, task_type, n_trees metadata | unit | `pytest tests/test_export.py::test_export_metadata -x` | ❌ Wave 0 |
| EXP-02 | `gtil.predict` on exported model matches Sylva `predict_proba` (clf) within atol=1e-5 | integration | `pytest tests/test_export.py::test_gtil_round_trip_clf -x` | ❌ Wave 0 |
| EXP-02 | `gtil.predict` matches Sylva regression `predict` within atol=1e-5 | integration | `pytest tests/test_export.py::test_gtil_round_trip_reg -x` | ❌ Wave 0 |
| EXP-02 | TL2cgen compiled path matches Sylva predict within atol=1e-5 (Windows MSVC) | integration | `pytest tests/test_export.py::test_tl2cgen_round_trip -x -m "not slow"` | ❌ Wave 0 |
| EXP-02 | FIL/nvForest GPU inference (Linux CI) — MANUAL / checkpoint:human-verify | manual | `pytest tests/test_export.py::test_fil_round_trip -m linux` (Linux only) | ❌ Wave 0 |
| EXP-03 | `import sylva` succeeds in clean venv | smoke | `scripts/import_smoke_09.py` | ❌ Wave 0 |
| EXP-03 | `ExtraTreesClassifier()` instantiates without error in clean venv | smoke | `scripts/import_smoke_09.py` | ❌ Wave 0 |
| CBS | `gtil.predict` throughput vs Sylva native predict vs sklearn predict (rows/s) | benchmark | `pytest tests/test_export.py::test_inference_throughput --benchmark-only` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_export.py -x -q`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green + clean-venv smoke green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_export.py` — EXP-01/02 round-trip parity tests
- [ ] `scripts/import_smoke_09.py` — EXP-03 clean-venv import smoke
- [ ] `tests/conftest.py` — shared fixtures (trained ForestIR for clf + reg)
- [ ] Framework install: `pip install treelite==4.7.0 tl2cgen==1.0.0 pytest`

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (export fn input) | Validate ForestIR JSON handle at the export function boundary; reject malformed input with a clear error before calling ModelBuilder |
| V6 Cryptography | no | — |

### Known Threat Patterns for {stack}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed ForestIR JSON causes uncontrolled ModelBuilder calls | Tampering | Parse and validate ForestIR JSON before iterating; check node counts, array lengths, and valid feature IDs match ForestIR.validate_structure() |
| Wheel supply-chain: loading wrong .dll from tl2cgen create_shared | Tampering | Use `tmp_path` (pytest tmpdir) for compiled libraries; do not load arbitrary .dll paths in production |

## Project Constraints (from CLAUDE.md)

The following directives from `./CLAUDE.md` are binding on Phase 9:

| Directive | Impact on Phase 9 |
|-----------|-------------------|
| `serde_json` export | ForestIR JSON serialization already in place; the export module receives the IR as JSON from the pyseam |
| Validate against a Treelite round-trip in CI | EXP-02; use `gtil.predict` + TL2cgen on Windows as the CI-testable paths |
| `abi3` + `dynamic-loading` | Phase 9 wheel uses the same config proven in Phase 1; no change needed |
| `fallback="error"` / no silent fallback | Export function must raise explicitly if FIL is unavailable on Windows (no silent CPU fallback) |
| Apache-2.0 license | treelite (Apache 2.0) and tl2cgen (Apache 2.0) are compatible |
| TOOL-03/04 pattern from Phase 1 | Clean-venv import smoke follows the exact Phase 1 pattern |
| `numpy crate = 0.29 to match pyo3 0.29` | Confirmed in Cargo.toml; no change for Phase 9 |

## Sources

### Primary (MEDIUM confidence)
- [treelite.readthedocs.io/en/4.1.2/tutorials/builder.html](https://treelite.readthedocs.io/en/4.1.2/tutorials/builder.html) — ModelBuilder complete API with Python code examples including kMultiClf, kBinaryClf, kRegressor, vector/scalar leaves, numerical_test parameters
- [treelite.readthedocs.io/en/latest/treelite-api.html](https://treelite.readthedocs.io/en/latest/treelite-api.html) — Treelite 4.7 API reference; import_from_json() absent; ModelBuilder, dump_as_json, from_xgboost_json confirmed
- [tl2cgen.readthedocs.io/en/latest/api.html](https://tl2cgen.readthedocs.io/en/latest/api.html) — generate_c_code(), create_shared() (msvc/gcc/clang), Predictor class with Windows .dll support
- [tl2cgen.readthedocs.io/en/latest/treelite-migration.html](https://tl2cgen.readthedocs.io/en/latest/treelite-migration.html) — Treelite 4.0 migration: compile() → tl2cgen.generate_c_code()
- [docs.rapids.ai/install/](https://docs.rapids.ai/install/) — RAPIDS/cuML Linux-only; Windows requires WSL2
- [github.com/rapidsai/nvforest](https://github.com/rapidsai/nvforest) — nvForest replaces FIL; accepts treelite.Model; WSL2/Linux only
- `pip index versions treelite` → 4.7.0 (latest) [VERIFIED: local pip]
- `pip index versions tl2cgen` → 1.0.0 (latest) [VERIFIED: local pip]
- `maturin --version` → 1.14.1 [VERIFIED: local shell]

### Secondary (MEDIUM confidence, cross-checked)
- [treelite.readthedocs.io/en/3.9.0/tutorials/json_import.html](https://treelite.readthedocs.io/en/3.9.0/tutorials/json_import.html) — 3.x JSON schema (task_param, model_param, split_feature_id, comparison_op, leaf_value) — confirmed as 3.x only
- [docs.rapids.ai/api/cuml/stable/fil/](https://docs.rapids.ai/api/cuml/stable/fil/) — FIL deprecated; cuml.fil → nvForest
- `crates/sylva-core/src/ir.rs` — ForestIR SoA field set (verified in-codebase)

### Tertiary (LOW confidence, assumptions flagged)
- Training-data knowledge of postprocessor semantics for sklearn-parity classifiers — marked [ASSUMED] throughout

## Metadata

**Confidence breakdown:**
- Treelite 4.x ModelBuilder API: MEDIUM — confirmed against official docs and Python examples
- `import_from_json()` removal: MEDIUM — inferred from absence in 4.x docs; not confirmed from changelog directly
- FIL/nvForest Windows unavailability: MEDIUM — confirmed from RAPIDS install guide (WSL2 only)
- TL2cgen API: MEDIUM — confirmed against official docs
- postprocessor semantics for sklearn parity: LOW — [ASSUMED]; must be verified by spike
- Package versions: HIGH — verified locally via pip

**Research date:** 2026-06-27
**Valid until:** 2026-09-27 (90 days; Treelite stable, tl2cgen stable)
