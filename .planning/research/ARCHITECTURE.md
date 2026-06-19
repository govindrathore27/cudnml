# Architecture Research

**Domain:** GPU-native tree-ensemble library (Rust core + CUDA kernels + Python/sklearn API)
**Researched:** 2026-06-19
**Confidence:** MEDIUM-HIGH (histogram engine + determinism: HIGH; Rust↔CUDA layer + SHAP specifics: MEDIUM)

## Standard Architecture

The system is a **layered dispatch stack**. The load-bearing insight from the feasibility study governs every layer: this is a **histogram-build + scatter-partition engine, NOT a matmul/Tensor-Core engine**. There is no GEMM in the hot path. The hot path is `quantize → atomic-accumulate histograms → argmax over bins → scatter rows`. Component design must optimize for HBM bandwidth, shared-memory residency, and atomic-contention reduction — never for Tensor-Core occupancy.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  L5  PYTHON API LAYER  (sklearn-parity, pip-installable)               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ExtraTrees*   │ │RandomForest* │ │  TreeSHAP    │ │ ExportToFIL  │  │
│  │ Clf / Regr   │ │ Clf / Regr   │ │  Explainer   │ │ (Treelite)   │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘  │
│         │  fit/predict   │   device="auto|cuda|cpu", fallback="error" │
│         │   + execution_report_  (no silent fallback)                  │
├─────────┴────────────────┴───────────────┴───────────────┴────────────┤
│  L4  PyO3 BINDING SEAM  (maturin)  — thin marshalling, no logic        │
│  ndarray <-> TableView ; GIL release during fit ; error translation    │
├────────────────────────────────────────────────────────────────────────┤
│  L3  RUST ORCHESTRATION  (device-agnostic control plane)               │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Estimator state machine · param validation · RNG seed schedule  │   │
│  │ NodeScheduler (breadth-first frontier) · fit-scoped Arena        │   │
│  │ ExecutionReport builder · ForestIR owner                         │   │
│  └────────────────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────────────┤
│  L2  BACKEND TRAIT SEAM   ── trait Backend  (THE abstraction boundary) │
│  fn quantize · fn build_histograms · fn eval_splits · fn partition     │
│  fn predict · fn alloc/free  ── all device-neutral signatures          │
│  ┌──────────────────────────┐      ┌──────────────────────────────┐    │
│  │ CudaBackend (impl)        │      │ CpuBackend (impl, ORACLE)    │    │
│  │ cudarc + hand-written CUDA │      │ pure Rust, rayon, exact      │    │
│  │ [future: RocmBackend, etc] │      │ sklearn semantics            │    │
│  └────────────┬─────────────┘      └──────────────────────────────┘    │
├───────────────┴──────────────────────────────────────────────────────┤
│  L1  CUDA KERNELS  (.cu → PTX, loaded by cudarc)                       │
│  quantize · histogram(privatized,shared-mem) · split-eval(scan+argmax) │
│  ExtraTrees-fused-random-candidate · partition-scatter · predict       │
├────────────────────────────────────────────────────────────────────────┤
│  L0  GPU MEMORY  (stream-ordered pool: cudaMallocAsync / arena)        │
│  BinnedMatrix(SoA uint8/16) · histogram buffers (reused) · ForestIR    │
│  · row-index double buffers · RNG counters                             │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Python estimators (L5) | sklearn-parity surface (`fit`/`predict`/`predict_proba`/`feature_importances_`), param objects, `execution_report_` | Python classes over the PyO3 module; thin |
| PyO3 binding (L4) | Marshal `ndarray`↔`TableView`, release GIL during `fit`, translate Rust errors to Python exceptions | `#[pyclass]`/`#[pymethods]`, maturin build |
| Rust orchestration (L3) | Device-agnostic control plane: validation, RNG seed schedule, node scheduling, arena lifetime, IR ownership, report assembly | Pure Rust; calls Backend trait only |
| **Backend trait (L2)** | **The device-abstraction seam.** Defines every device operation as a neutral signature so a new backend is purely additive | Rust `trait Backend` |
| CudaBackend (L2) | Implements Backend on NVIDIA via cudarc: loads PTX, owns streams + pool, launches kernels | cudarc host code + hand-written CUDA C |
| CpuBackend (L2) | Correctness oracle + small-data path; exact sklearn semantics | Pure Rust + rayon |
| CUDA kernels (L1) | The actual histogram/scatter compute | `.cu` files compiled to PTX |
| Memory pool (L0) | Stream-ordered, fit-scoped allocation; reusable buffers | cudaMallocAsync pool / arena |

## Recommended Project Structure

```
sylva/                          # repo root (workspace)
├── crates/
│   ├── sylva-core/             # L3 orchestration — device-agnostic
│   │   ├── ir.rs               # ForestIR: SoA node arrays (the shared contract)
│   │   ├── table.rs            # TableView input abstraction
│   │   ├── scheduler.rs        # NodeScheduler: BF frontier, large/med/small strategy
│   │   ├── rng.rs              # counter-based stateless RNG (Philox-style) seed schedule
│   │   ├── arena.rs            # fit-scoped allocation lifetime
│   │   ├── report.rs           # ExecutionReport (dispatch + conversion log)
│   │   └── estimator.rs        # fit/predict state machine, param validation
│   ├── sylva-backend/          # L2 — trait + dispatch
│   │   └── lib.rs              # trait Backend { quantize, build_histograms, ... }
│   ├── sylva-cpu/              # L2 CpuBackend — the ORACLE (build FIRST)
│   ├── sylva-cuda/             # L2 CudaBackend
│   │   ├── host/               # cudarc context, streams, pool, PTX loading
│   │   ├── kernels/            # *.cu hand-written CUDA C
│   │   │   ├── quantize.cu
│   │   │   ├── histogram.cu    # privatized shared-mem + two-level reduction
│   │   │   ├── split_rf.cu     # scan(prefix-sum) + argmax
│   │   │   ├── split_extratrees.cu  # fused random-candidate eval
│   │   │   ├── partition.cu    # double-buffered scatter
│   │   │   └── predict.cu
│   │   └── build.rs            # nvcc → PTX, embed or load at runtime
│   ├── sylva-shap/             # exact tree SHAP — consumes ForestIR
│   └── sylva-export/           # ForestIR → Treelite-compatible representation
└── python/
    └── sylva/                  # L5 + L4 (PyO3 module compiled here via maturin)
        ├── _native.pyd/.so     # compiled Rust extension
        └── ensemble.py         # ExtraTreesClassifier, ... sklearn-parity wrappers
```

### Structure Rationale

- **`sylva-core/` holds `ir.rs` and is device-agnostic:** The Forest IR is the single contract that training *writes* and inference/SHAP/export *read*. Keeping it in the device-neutral crate prevents any backend from leaking into the IR.
- **`sylva-backend/` is a tiny crate that only defines the trait:** This is the seam. `sylva-cpu` and `sylva-cuda` depend on it; orchestration depends on it; neither knows about the other. A `sylva-rocm` crate later adds a file, changes nothing.
- **`sylva-cuda/kernels/` are hand-written `.cu`, not Rust-to-PTX:** See Pattern 2 — the research shows Rust→PTX (Rust-CUDA/cust) is still early and emits invalid PTX for common ops, while the workload needs hand-tuned shared-memory histogram kernels.
- **`sylva-shap/` and `sylva-export/` depend only on `sylva-core` (the IR), never on a backend:** They are pure IR consumers, so they can run on CPU first and gain a GPU path later without touching training.

## Architectural Patterns

### Pattern 1: Backend Trait as the Device-Abstraction Seam

**What:** A single `trait Backend` whose methods are the *only* way orchestration touches a device. Every method signature is device-neutral (takes/returns host-describable handles + the IR). CUDA-specific types (streams, device pointers) live *inside* `CudaBackend` and never cross the trait.

**When to use:** Always — it is the spine of the additive-backend requirement.

**Trade-offs:** Slight indirection cost (negligible — these are coarse per-node-wave calls, not per-element). Forces discipline: anything CUDA-shaped that wants to escape the trait is a design smell. Future ROCm/SYCL/CPU-SIMD backends are then purely additive.

**Example:**
```rust
pub trait Backend {
    type Buffer;                       // opaque device/host buffer handle
    fn quantize(&self, t: &TableView, edges: &BinEdges) -> Self::Buffer;        // → BinnedMatrix
    fn build_histograms(&self, bins: &Self::Buffer, frontier: &Frontier,
                        rows: &RowIndex) -> Histograms;
    fn eval_splits(&self, h: &Histograms, mode: SplitMode) -> Vec<SplitDecision>;
    fn partition(&self, bins: &Self::Buffer, splits: &[SplitDecision],
                 rows: &mut RowIndex) -> ChildRanges;       // double-buffered scatter
    fn predict(&self, ir: &ForestIR, t: &TableView) -> Predictions;
}
// CpuBackend and CudaBackend both impl this. Orchestration is generic over Backend.
```

### Pattern 2: cudarc + Hand-Written CUDA C (not Rust→PTX)

**What:** Author the histogram/scatter kernels in CUDA C (`.cu`), compile to PTX with `nvcc` in `build.rs`, and load + launch them from Rust via **cudarc** (`CudaContext::load_module`, builder-pattern launch).

**When to use:** This MVP. The kernels need hand-tuned shared-memory privatization, warp-synchronous reductions, and bank-conflict control — exactly where the pure-Rust path is weakest.

**Trade-offs:** Two languages in one crate (CUDA C + Rust) and `nvcc` in the build. But: cudarc is mature, dynamic-loads PTX (no CUDA libs needed at build time), and gives precise launch control. The alternative (Rust-CUDA/`cust` + `cuda_std`) was rebooted in 2025 and still emits invalid PTX for many common Rust operations — too risky for the single biggest technical risk PROJECT.md flags. **This is the call that resolves PROJECT.md's open "kernel-authoring path" decision: choose cudarc + CUDA C.**

**Example:**
```rust
// build.rs: nvcc -ptx kernels/histogram.cu -o histogram.ptx
let ptx = Ptx::from_file("histogram.ptx");
let module = ctx.load_module(ptx)?;
let f = module.load_function("build_histograms_privatized")?;
stream.launch_builder(&f).arg(&bins).arg(&out).launch(cfg)?;
```

### Pattern 3: SoA Forest IR — One Structure, Shared by Train / Infer / SHAP / Export

**What:** The trained forest is a Structure-of-Arrays of node records: `feature_id[]`, `threshold[]` (or `bin_threshold[]`), `left[]`, `right[]`, `default[]` (missing-direction), `leaf_value[]`, plus per-tree offsets. Training *appends* nodes wave-by-wave; inference, SHAP, and export *read* the same arrays.

**When to use:** Always. SoA is coalescing-friendly on GPU and cache-friendly on CPU, and a single representation eliminates train→infer translation bugs.

**Trade-offs:** Must pick the threshold representation up front (store *binned* thresholds for compactness + exact retraining-equivalence, with a parallel real-valued table for export). Predicting must replay the same binning to be bit-exact with training.

### Pattern 4: Determinism via Fixed Reduction Order + Stateless RNG

**What:** Determinism is a *cross-cutting* property threaded through three components, not a single switch.
1. **Stateless counter-based RNG** (Philox-style): tree/node/feature index → counter → deterministic draw. No sequential RNG state, so tree-wave parallelism cannot reorder draws. Same seed ⇒ same candidate splits regardless of scheduling.
2. **Fixed reduction order** in histogram + criterion reduction: integer bin *counts* use atomics safely (integer add is associative ⇒ already deterministic); **floating-point** gain/impurity sums must use a canonical warp-synchronous binary-tree reduction (no FP atomics) when `deterministic=True`.
3. **Fixed tie-breaking** in `argmax` over candidate splits: ties resolved by lowest `(feature_id, bin)` — a total order, never "whichever thread won."

**When to use:** `deterministic=True` mode. The non-deterministic fast path may use FP atomics for speed; the report documents which was used.

**Trade-offs:** Research shows ordered reductions retain ~95–98% throughput on modern GPUs — the documented perf cost is modest, mostly in the FP reduction, not the integer histogram. The win is bit-reproducible models (a stated differentiator).

### Pattern 5: Stream-Ordered Fit-Scoped Arena (RMM-equivalent)

**What:** Allocate one large device pool at `fit` start via **cudaMallocAsync / a stream-ordered pool** (cudarc exposes stream-ordered alloc). Sub-allocate histogram buffers, row-index double buffers, and the growing IR from a **fit-scoped arena**; reuse the histogram buffer across node waves rather than malloc/free per node. Free the whole arena at `fit` end.

**When to use:** All GPU training. Per-node `cudaMalloc` would dominate runtime; the arena removes allocation from the hot path.

**Trade-offs:** Must size the pool from `(n, d, n_bins, max_frontier)` up front (over-allocate slightly). Mirrors RMM's `pool_memory_resource` (coalescing best-fit) layered over `cuda_async_memory_resource`. Histogram buffers are the dominant reusable resource and should be allocated once at max-frontier size.

## Data Flow

### Training Flow (fit)

```
X (ndarray, dense f32)
   ↓  [L4 PyO3: marshal → TableView, release GIL]
TableView ──[Backend::quantize]──► BinnedMatrix (SoA uint8/uint16, on device via pool)
   ↓
Sampler (counter-based RNG, tree waves) ──► bootstrap row sets / feature subsets
   ↓                                           per tree, breadth-first frontier
┌─────────────── NodeScheduler loop (per wave) ──────────────────────────┐
│  Frontier nodes  ──[Backend::build_histograms]──► Histograms           │
│      (shared-mem privatization, two-level reduction, sibling subtract) │
│  Histograms ──[Backend::eval_splits]──► SplitDecision[]                │
│      RF: prefix-sum scan + argmax  |  ExtraTrees: fused random-candidate│
│  SplitDecision[] ──[Backend::partition]──► child row ranges           │
│      (double-buffered row indices, scatter)                            │
│  append nodes ──────────────────────────────► ForestIR (SoA, growing) │
└────────────────────────────────────────────────────────────────────────┘
   ↓  (optional: variance-based redundant-tree pruning, folded in)
ForestIR (complete)  +  ExecutionReport
```

### Inference / SHAP / Export Flow (all read the IR)

```
                      ┌──[Backend::predict]──► Predictions (replay binning)
ForestIR ─────────────┼──[sylva-shap]────────► exact tree SHAP values
  (SoA node arrays)   │       (path-dependent: uses leaf sample counts as background)
                      └──[sylva-export]──────► Treelite-compatible model → FIL serving
```

### Key Data Flows

1. **Quantize-once:** `TableView` is binned a single time into `BinnedMatrix`; every subsequent histogram pass reads bins (uint8/16), never the f32 matrix — this is the bandwidth saving that makes histograms shared-mem-resident.
2. **Sibling subtraction:** A child node's histogram is computed as `parent − sibling` for one child and built directly for the other — halving histogram work per split, the dominant cost.
3. **Double-buffered partition:** Row indices live in two buffers; `partition` scatters from buffer A to buffer B by split outcome, swapping each level — avoids in-place hazards and keeps the scatter coalesced.
4. **IR is write-once-by-training, read-many:** Inference, SHAP, and export are strict consumers; they never mutate the IR, so they parallelize trivially and can be added without touching the training path.

## Build Order (Dependency-Ordered — drives roadmap phasing)

This is the critical output. Each stage depends on the prior and is independently testable.

```
┌─ B0  TableView + ForestIR + Backend trait  (contracts, no compute)
│        → defines the shared IR and the seam everything plugs into
│
├─ B1  CpuBackend = ORACLE  (pure Rust, exact sklearn semantics)
│        depends: B0
│        → full fit/predict on CPU; differential-tested vs scikit-learn
│        → becomes the correctness oracle for ALL later GPU work
│
├─ B2  Quantization engine  (CPU first, then GPU kernel)
│        depends: B1
│        → BinnedMatrix (SoA uint8/16); GPU bins must match CPU bins bit-for-bit
│
├─ B3  Single-tree GPU path  (histogram → split-eval → partition)
│        depends: B2, cudarc+nvcc toolchain proven
│        → ExtraTrees first (random splits = no scan, simplest hot path),
│          validate one GPU tree == CPU oracle tree on fixed seed
│
├─ B4  Forest on GPU  (NodeScheduler waves, RNG schedule, sibling subtraction,
│        fit-scoped arena, RF prefix-sum split-eval added alongside ExtraTrees)
│        depends: B3
│        → full ExtraTrees + RandomForest, Clf + Regr; arena/pool memory model
│
├─ B5  Determinism mode  (fixed reductions, fixed tie-break, stateless RNG audit)
│        depends: B4
│        → deterministic=True bit-reproducible; execution_report_ wired; no silent fallback
│
├─ B6  Exact tree SHAP  (consumes ForestIR; CPU exact first, then GPU)
│        depends: B1 (IR exists) — can start in parallel with B4 on CPU
│        → path-dependent exact SHAP; high-depth path compression
│
└─ B7  Export → Treelite-compatible  (consumes ForestIR)
         depends: B1 (IR exists) — independent of GPU path
         → model serialization for CPU/FIL serving
```

**Why this order:**
- **CPU oracle before any GPU code** is non-negotiable: without a trusted reference, GPU correctness is unverifiable, and PROJECT.md mandates sklearn-semantic (not just accuracy-close) parity.
- **Quantizer before single-tree** because every GPU kernel reads bins; bins must be proven equal to CPU bins before building histograms on them.
- **ExtraTrees before RF** because random split thresholds delete the scan+argmax+atomic-heavy step — the simplest possible GPU hot path to validate the toolchain (PROJECT.md: "Extra Trees is the seam").
- **Single tree before forest** isolates kernel correctness from scheduler/RNG/memory concerns.
- **Determinism after the forest works** — you cannot make a thing reproducible before it exists; determinism is a constraint layered onto correct kernels.
- **SHAP and Export depend only on the IR (B1)**, so they fork off early and proceed in parallel with the GPU training build — they never block, and never block, training.

## Anti-Patterns

### Anti-Pattern 1: Designing for Tensor Cores / matmul

**What people do:** Reach for cuBLAS/Tensor-Core formulations because "GPUs are matmul machines."
**Why it's wrong:** Tree training has **no GEMM in the hot path** (PROJECT.md debunked premise). It is bandwidth- and atomic-contention-bound. Tensor Cores sit idle; chasing them wastes the whole effort.
**Do this instead:** Optimize for HBM bandwidth, shared-memory histogram residency, privatization to cut atomic conflicts, and coalesced scatter. Measure occupancy and memory throughput, not FLOPS.

### Anti-Pattern 2: Leaking CUDA types through the Backend trait

**What people do:** Put `CudaStream`/device pointers in trait signatures "for convenience."
**Why it's wrong:** It welds orchestration to CUDA and destroys the additive-backend property — a ROCm backend becomes a rewrite, not a new file.
**Do this instead:** Keep CUDA types inside `CudaBackend`. The trait speaks only IR + neutral handles.

### Anti-Pattern 3: Per-node device allocation

**What people do:** `cudaMalloc` a fresh histogram buffer for every node.
**Why it's wrong:** Allocation latency dominates; it serializes the stream and churns the pool.
**Do this instead:** One stream-ordered pool + fit-scoped arena; reuse a max-frontier-sized histogram buffer across waves (Pattern 5).

### Anti-Pattern 4: Silent CPU fallback

**What people do:** Quietly run on CPU when the GPU path can't handle an input.
**Why it's wrong:** It violates the core differentiator (`fallback="error"`, `execution_report_`) and hides correctness/perf cliffs.
**Do this instead:** Explicit dispatch; on any unsupported condition, error with a clear reason recorded in `execution_report_`.

### Anti-Pattern 5: Rust→PTX for the hot kernels (this MVP)

**What people do:** Write histogram kernels in pure Rust via Rust-CUDA/`cust` to "stay in one language."
**Why it's wrong:** The Rust→PTX backend is early-stage (rebooted 2025) and emits invalid PTX for many common operations — unacceptable for the single biggest technical risk.
**Do this instead:** Hand-written CUDA C compiled to PTX, loaded via cudarc (Pattern 2). Revisit pure-Rust kernels only after the engine is proven.

## Integration Points

### External Services / Toolchain

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| CUDA toolkit (nvcc) | `build.rs` compiles `.cu` → PTX | Windows-native or documented WSL fallback |
| cudarc crate | Host-side context, streams, pool, PTX load + launch | Mature; dynamic-loads PTX, no CUDA libs at build time |
| maturin / PyO3 | Build Rust extension into pip wheel | GIL released during `fit` |
| scikit-learn | Differential-test oracle (not a runtime dep) | Semantics parity gate |
| Treelite / FIL | Export target representation | Consumes ForestIR via `sylva-export` |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Python ↔ Rust (L5↔L4) | PyO3 calls; ndarray↔TableView | Thin marshalling only, no algorithm logic |
| Orchestration ↔ Backend (L3↔L2) | `trait Backend` method calls | THE seam; device-neutral; future backends additive here |
| Backend ↔ Kernels (L2↔L1) | cudarc PTX load + launch (CUDA); direct calls (CPU) | CUDA types confined to CudaBackend |
| Training ↔ {Infer, SHAP, Export} | Shared read-only ForestIR | Write-once-by-train, read-many; no back-coupling |

## Sources

- [cudarc::driver docs](https://docs.rs/cudarc/latest/cudarc/) and [cudarc on crates.io](https://crates.io/crates/cudarc) — PTX load, launch builder, stream-ordered alloc (MEDIUM)
- [Rust CUDA project reboot 2025](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot/) and [Rust-CUDA FAQ](https://github.com/Rust-GPU/Rust-CUDA/blob/main/guide/src/faq.md) — Rust→PTX maturity caveats (MEDIUM)
- [Exploiting GPUs for Efficient GBDT Training (TPDS'19)](https://readingxtra.github.io/docs/ml-gpu/wen_tpds19_gpugbdt.pdf) and [XGBoost GPU algorithm updates](https://xgboost.ai/2018/07/04/gpu-xgboost-update.html) — privatized shared-mem histograms, sibling subtraction (HIGH)
- [Impacts of FP non-associativity on reproducibility (arXiv 2408.05148)](https://arxiv.org/pdf/2408.05148) and [Deterministic Atomic Buffering](https://microarch.org/micro53/papers/738300a981.pdf) — atomics/reduction determinism, ~95-98% throughput retained (HIGH)
- [shap.GPUTreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.GPUTreeExplainer.html) — exact GPU tree SHAP, path-dependent vs interventional (MEDIUM)
- [Fast Flexible Allocation with RAPIDS Memory Manager (NVIDIA)](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/) and [RMM device memory resources](https://deepwiki.com/rapidsai/rmm/2.2.1-device-memory-resources) — stream-ordered pool / arena design (MEDIUM)

---
*Architecture research for: GPU-native tree-ensemble library (Rust + CUDA + Python)*
*Researched: 2026-06-19*
