# Phase 4: Single GPU ExtraTree - Research

**Researched:** 2026-06-26
**Domain:** GPU-native single ExtraTree training (privatized shared-memory histograms, breadth-first level-at-a-time build, fused random-candidate split, scatter-partition) with bit-exact parity to the Phase-2 CPU oracle, authored in hand-written CUDA C compiled at runtime via cudarc 0.19.8 + NVRTC on native Windows/MSVC (sm_89 / RTX 4060 Ti).
**Confidence:** HIGH (toolchain, kernel structure, parity mechanism, file layout — all grounded in already-shipped repo code from Phases 1–3); MEDIUM (the exact bit-exact reconciliation between the CPU's *raw-f32-range* threshold draw and a binned-histogram hot path — this is the single largest design decision and is flagged as an Open Question requiring a planner/user decision).

## Summary

Phase 4 is the project's **first GPU hot path**, and its entire success criterion is *negative-defined by correctness*: the GPU ExtraTree must match the Phase-2 CPU oracle **bit-exactly** on a fixed seed (GPU-02), with `compute-sanitizer` (racecheck + memcheck) clean against every kernel (GPU-02), built breadth-first / level-at-a-time with privatized shared-memory histograms and no global float atomics on the hot path (GPU-01, SC-4). The toolchain risk is already retired — Phase 1 proved cudarc 0.19.8 + NVRTC compiles a *representative privatized 256-bin histogram* for sm_89, launches it on the local RTX 4060 Ti, and passes all four sanitizer tools clean. Phase 2 shipped the CPU oracle (`fit_forest`/`predict_forest`), the SoA `ForestIR`, the Philox-4×32-10 RNG keyed by `(seed,tree,node,feature,draw)` with verified KAT vectors, and the `Backend`/`HistogramBackend` trait seam. Phase 3 (planned, partially executed — the `quantize/` module already exists on disk) defines the `BinnedMatrix` (column-major SoA uint8/uint16), `BinEdges` (CSR host buffer), and the `assign_bin` searchsorted contract the histogram kernel reads. So Phase 4 is overwhelmingly an **integration + correctness** phase against well-defined inputs, not a greenfield kernel research phase.

The dominant technical risk is **NOT** the histogram kernel mechanics (de-risked in Phase 1) — it is the **bit-exact reconciliation problem**. The Phase-2 CPU ExtraTree splitter (`split_et.rs`) draws its random threshold from the *raw f32 feature range* `[fmin, fmax]` of the node's rows (`threshold = fmin + u*(fmax-fmin)`), evaluates an exact `x[i,f] <= threshold` partition, and computes proxy impurity from exact class counts. A histogram-based GPU path operates on *binned* data — which produces **different split structure** unless the binned histogram is used only to *accelerate* a computation that is then made bit-identical to the CPU's exact-arithmetic result. Three architecturally distinct ways to resolve this are laid out in the Open Questions; the **recommended** path for Phase 4's "simplest possible GPU hot path" mandate is **Strategy A: the GPU replays the CPU's exact algorithm faithfully** (same Philox draw on the same raw f32 range, same `<=` partition, same fixed-order integer/Kahan-free accumulation), using the privatized-histogram kernel as the *count-accumulation engine* over rows, NOT as a binned-threshold-candidate engine. The binned-candidate prefix-sum approach is RandomForest's Phase-5 concern (GPU-04); ExtraTrees' random single threshold per feature does not need it.

**Primary recommendation:** Add a `CudaBackend` in `crates/sylva-cuda` implementing the existing `Backend` trait (`fit`/`predict`) for the single-tree ET case, authored as four NVRTC-compiled CUDA-C kernels (Philox-replay random-threshold + per-feature partition-count via privatized shared-memory integer histograms, fused split-score reduction, scatter-partition of row indices into children, and a leaf-stats kernel), driven by a host-side breadth-first level-at-a-time scheduler. Carry the Phase-3 "no `--use_fast_math`, no-FMA, IEEE-754 `<=`" metadata into the NVRTC compile options (`-fmad=false`), inline the *identical* Philox-4×32-10 + `u32_to_unit_f32` in CUDA C and verify it against the frozen KAT vectors on-device before any tree is built, and accumulate all float reductions in the *same fixed canonical order* the CPU uses (integer class counts are order-free; the f32 gini/entropy/MSE proxy must be computed from those integer counts with the identical f32 op sequence as `criterion.rs`). The bit-exact differential test against `fit_forest(...,n_estimators=1)` is the phase gate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GPU kernel authoring (histogram / split / scatter / leaf) | L1 CUDA-C via NVRTC (`crates/sylva-cuda`) | — | Tree training is bandwidth/atomic-bound; CUDA-C gives explicit shared-mem + warp + `atomicAdd` control. NVRTC = no nvcc-at-build (PROJECT.md, Phase-1 proof). |
| Breadth-first level-at-a-time scheduler (frontier, row-index ranges) | L3 host orchestration (Rust, `crates/sylva-cuda`) | drives L1 launches | The LOCKED breadth-first decision lives host-side: it manages the node frontier, per-node row ranges, and the per-level kernel-launch wave. Kernels are stateless per-launch. |
| Philox-4×32-10 RNG (bit-identical CPU↔GPU) | L1 CUDA-C (inlined) + L3 reference (`sylva-core::rng`) | — | Same `(seed,tree,node,feature,draw)` scheme; the CUDA copy must reproduce the frozen KAT vectors (ENG-06, Phase-2 deliverable). |
| Quantized input (`BinnedMatrix` / `BinEdges`) | L3 device-neutral (`sylva-core::quantize`, Phase 3) | uploaded to L1 | Histogram kernel reads the column-major `BinnedMatrix`; produced CPU-side in Phase 3, bit-parity already gated there. |
| ForestIR assembly from GPU results | L3 host (`sylva-cuda` → `sylva-core::ir::ForestIR`) | read by predict/SHAP/export | GPU returns node arrays (D2H); host assembles the *same* SoA `ForestIR` the CPU writes — single shared representation (ENG-02). |
| Bit-exact differential test vs CPU oracle | Test tier (Rust integration test) | compute-sanitizer gate | The phase gate (GPU-02); CPU `fit_forest(n_estimators=1)` is the byte-exact oracle. |
| Device dispatch / no-silent-fallback | L3 boundary (typed `CudaError` → `SylvaError`) | — | Already the Phase-1 pattern: every cudarc call is a `Result`, no `.unwrap()`. Full `device=`/`fallback=` API is Phase 6 — Phase 4 only needs an explicit CUDA entry. |

## Standard Stack

> Phase 4 introduces **no new external dependencies**. The stack is already pinned and proven in Phases 1–3. This table records what Phase 4 *uses*, with the in-repo source of truth.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cudarc | 0.19.8 | CUDA driver API + NVRTC runtime compile + launch | `[VERIFIED: crates/sylva-cuda/Cargo.toml + VERSIONS.md]` Proven in Phase 1: compiles CUDA-C for sm_89, launches, sanitizer-clean. Features `["std","driver","nvrtc","cuda-12080", <link-mode>]`. |
| CUDA Toolkit | 12.8 (NVRTC `nvrtc64_120_0.dll`) | Runtime kernel compilation + compute-sanitizer | `[VERIFIED: VERSIONS.md]` Installed, sanitizer at `…\CUDA\v12.8\compute-sanitizer\`. |
| sylva-core | (workspace) | `Backend`/`HistogramBackend` traits, `ForestIR`, Philox, `BinnedMatrix`/`BinEdges`, `TrainConfig`, criterion fns | `[VERIFIED: crates/sylva-core/src]` The device-neutral contract + CPU oracle Phase 4 matches bit-for-bit. |
| ndarray | 0.16.x | Host `ArrayView2<f32>` input to the `Backend::fit` signature | `[VERIFIED: backend.rs uses ArrayView2/ArrayView1]` Already the trait's host I/O type. |
| thiserror | 1.x | `CudaError` typed enum → `SylvaError`/PyErr | `[VERIFIED: nvrtc_launch.rs CudaError]` Established no-silent-fallback pattern. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| serde / serde_json | 1.x | `ForestIR` round-trip in the differential test (byte-compare path) | `[VERIFIED: ir.rs derives Serialize]` The bit-exact gate compares serialized IRs (the Phase-2 determinism-test idiom). |
| approx | 0.5.x | (Tolerance asserts ONLY for the sklearn *distributional* check — never for the CPU bit-exact gate) | `[VERIFIED: existing dev-dep]` Distributional parity uses tolerance; CPU parity uses `to_bits()` equality. |
| rayon | 1.x | (CPU side only — not used in GPU kernels) | The GPU path is single-stream this phase; rayon stays on the CPU oracle. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cudarc + hand-written CUDA-C (NVRTC) | CubeCL / Rust-CUDA `cust` | `[CITED: .claude/CLAUDE.md kernel-authoring table]` Rejected for MVP: CubeCL alpha (API churn); `cust` frozen/nightly. CubeCL is the deliberate M2 fallback behind the kernel trait, not now. |
| Privatized shared-mem integer histogram | Global `atomicAdd` histogram | LOCKED OUT (SC-4): global float atomics are non-deterministic + contention-bound. Integer atomics into private `__shared__` then a deterministic flush is the mandated baseline (Phase-1 proven). |
| GPU replays CPU exact algorithm (Strategy A) | GPU binned-candidate split via prefix-sum (Strategy B) | Strategy B is RandomForest's Phase-5 path (GPU-04 scan+argmax). For ET's single random threshold, Strategy A is simpler AND is what makes bit-exact-to-CPU achievable. See Open Question 1. |
| NVRTC runtime compile | nvcc AOT + bindgen | `[CITED: PROJECT.md]` Broken on native MSVC (cc-rs = GNU/Clang only). NVRTC is the whole reason the toolchain works on Windows. |

**Installation:** No new packages. The `crates/sylva-cuda/Cargo.toml` cudarc line and the `default=cuda-static` / `wheel=dynamic-loading` feature toggle are already in place (VERSIONS.md). Phase 4 adds *source files* under `crates/sylva-cuda/src/`, not dependencies.

**Version verification:** All versions are already verified and committed in `Cargo.lock` + `VERSIONS.md` from Phase 1 (cudarc 0.19.8, CUDA 12.8, sm_89, driver 595.79, Rust 1.96.0 stable). No registry re-check needed — these are locked toolchain pins, not new additions.

## Package Legitimacy Audit

> No external packages are added in Phase 4. All dependencies (cudarc, ndarray, thiserror, serde, rayon, approx) were vetted and pinned in Phases 1–2 and are in the committed `Cargo.lock`. The legitimacy gate is therefore **N/A for new packages**; the existing pins stand.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| cudarc | crates.io | active (0.19.8, 2026) | 300k+/version | github.com/coreylowman/cudarc | OK (pinned Phase 1) | Approved (in use) |
| ndarray / rayon / serde / thiserror / approx | crates.io | mature | very high | (canonical repos) | OK (pinned Phase 1–2) | Approved (in use) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
  HOST (Rust, crates/sylva-cuda)                          DEVICE (CUDA-C via NVRTC, sm_89)
  ──────────────────────────────                          ────────────────────────────────
  X: ArrayView2<f32>, y: ArrayView1<f32>, cfg: TrainConfig
        │
        ▼
  CudaBackend::fit  (impl Backend)
        │
        ├─ validate inputs at boundary  ──► CudaError (no silent fallback)
        │
        ├─ Phase-3 Quantizer::fit_transform(X) ──► BinnedMatrix (col-major u8/u16) + BinEdges
        │        (CPU-side this phase; bit-parity already gated in Phase 3)
        │
        ├─ H2D upload:  binned matrix, y (as labels), BinEdges, raw-f32 columns (Strategy A needs raw X)
        │        d_bins, d_y, d_x  (device buffers, allocated once / fit-scoped)
        │
        ├─ build NVRTC modules ONCE (compile_for_sm89 with -fmad=false, NO --use_fast_math)
        │        verify on-device Philox KAT before first tree  ──► [KAT check kernel]
        │
        ▼  ── BREADTH-FIRST LEVEL-AT-A-TIME LOOP (host scheduler) ──────────────────────
        │   frontier = [root node over all rows];   row_index buffer = [0..n_rows]
        │   while frontier not empty AND depth < max_depth:
        │     for each node in frontier (mapped to a block / block-group):
        │       ┌──────────────────────────────────────────────────────────────┐
        │  (1)  │ random_threshold + partition-count kernel                      │
        │       │   per candidate feature: Philox(seed,tree,node,feat,draw)      │
        │       │   threshold = fmin + u*(fmax-fmin)  [raw f32 range over rows]   │
        │       │   privatized __shared__ int hist of (left/right × class)        │
        │       │   counts via integer atomicAdd into shared, then det. flush     │
        │       └──────────────────────────────────────────────────────────────┘
        │       ┌──────────────────────────────────────────────────────────────┐
        │  (2)  │ fused split-score kernel: from counts compute gini/entropy/MSE │
        │       │   proxy_improvement in the EXACT f32 op order of criterion.rs; │
        │       │   pick best feature (tie-break lowest (feature, threshold_bits))│
        │       └──────────────────────────────────────────────────────────────┘
        │       ┌──────────────────────────────────────────────────────────────┐
        │  (3)  │ scatter-partition kernel: reorder this node's row-index range   │
        │       │   into [left_rows | right_rows] by x[row,best_feat] <= thr;     │
        │       │   NaN handled at predict only (training data clean, D-01)       │
        │       └──────────────────────────────────────────────────────────────┘
        │     append nodes to device node arrays; build next frontier
        │     emit leaves where stop-criteria met (depth/min_samples/pure) ──► (4) leaf-stats
        │   ─────────────────────────────────────────────────────────────────────────────
        │
        ├─ D2H: node arrays (feature_id, threshold, children, default_child,
        │        sample_count, impurity, leaf_proba/leaf_value, leaf_offset)
        │
        ▼
  assemble ForestIR  (the SAME SoA struct CPU writes — ENG-02)
        │
        ├──► CudaBackend::predict (GPU traversal kernel OR reuse CPU predict_forest)
        └──► serde → byte-compare vs CPU fit_forest(n_estimators=1)  [BIT-EXACT GATE]
```

### Recommended Project Structure
```
crates/sylva-cuda/src/
├── lib.rs                  # extend: pub mod cuda_backend; re-export CudaBackend
├── kernels.rs              # EXTEND: add real ET kernel source strings (keep the
│                           #   Phase-1 vector_add/histogram as a deletable spike or
│                           #   move them to a spike module; the privatized-histogram
│                           #   shape is the template for kernels::HISTOGRAM_*)
├── nvrtc_launch.rs         # EXTEND: compile_for_sm89 gains -fmad=false; add launch
│                           #   wrappers per kernel (reuse the CudaError pattern)
├── cuda_backend/
│   ├── mod.rs              # CudaBackend struct, impl Backend (fit/predict)
│   ├── scheduler.rs        # breadth-first level-at-a-time host loop + frontier/row-range mgmt
│   ├── device_buffers.rs   # fit-scoped device buffer alloc (bins, y, x, row_index, node arrays)
│   ├── philox_device.rs    # the CUDA-C Philox source string + on-device KAT verify launch
│   ├── histogram.rs        # launch wrapper for the privatized partition-count kernel
│   ├── split_score.rs      # launch wrapper for the fused split-score/argmax kernel
│   ├── partition.rs        # launch wrapper for the scatter-partition kernel
│   └── assemble.rs         # D2H node arrays → ForestIR (mirrors cpu/fit.rs assemble_forest)
└── kernels/                # OPTIONAL: .cu-as-string files via include_str! if kernels.rs grows
    ├── philox.cuh.str
    ├── histogram_et.cu.str
    ├── split_score.cu.str
    └── partition.cu.str

crates/sylva-cuda/tests/
├── parity_cpu_gpu.rs       # THE GATE: GPU ForestIR == CPU fit_forest(n_estimators=1) bit-exact
├── philox_device_kat.rs    # on-device Philox reproduces the frozen KAT vectors (ENG-06)
└── sanitizer_et_kernels.rs # standalone single-launch targets for racecheck/memcheck per kernel

python/tests/gpu_parity/    # the Comparative Baseline Study harness
├── test_cpu_gpu_bitexact.py   # GPU vs CPU oracle bit-exact (gate)
├── test_sklearn_distributional.py  # GPU vs sklearn single ExtraTree (distributional, informational)
└── test_single_tree_timing.py # transfer-inclusive single-tree GPU-vs-CPU timing (REPORTED, not gated)
```

**Structure rationale:** Mirrors the established `cpu/` module layout (one file per algorithmic concern, <400 lines, CLAUDE.md coding-style). The breadth-first scheduler is a *host* concern and gets its own file. Kernel source strings stay co-located with their launch wrappers (the Phase-1 `kernels.rs`/`nvrtc_launch.rs` precedent) unless they grow past ~800 lines, at which point `include_str!`-ing `.cu` strings keeps files small. `assemble.rs` deliberately parallels `cpu/fit.rs::assemble_forest` so the IR is constructed identically.

### Pattern 1: NVRTC compile-once, launch-many with no-fast-math flags
**What:** Compile each kernel's CUDA-C to PTX **once** at `fit` start (not per node/level), load the module, cache the `CudaFunction`, then launch it many times across the breadth-first waves. Carry the bit-parity compile flags.
**When to use:** Always — recompiling per launch would dominate runtime and is unnecessary.
**Example:**
```rust
// Source: extends crates/sylva-cuda/src/nvrtc_launch.rs (Phase-1 proven API)
fn compile_for_sm89(src: &str) -> Result<cudarc::nvrtc::Ptx, CudaError> {
    let opts = CompileOptions {
        arch: Some("sm_89"),
        // -lineinfo: sanitizer source attribution (Phase 1).
        // -fmad=false: disable fused multiply-add so device float ops match the
        //   CPU's separate mul/add sequence (bit-exact parity). NEVER --use_fast_math.
        options: vec!["-lineinfo".into(), "-fmad=false".into()],
        ..Default::default()
    };
    Ok(compile_ptx_with_opts(src, opts)?)
}
// Compile once in CudaBackend::fit, store module + functions, reuse every wave.
```

### Pattern 2: Breadth-first frontier with row-index ranges (the LOCKED build order)
**What:** Maintain a single device `row_index` buffer (a permutation of `0..n_rows`). Each node owns a contiguous `[start,end)` range into it. A *level* (frontier) is the set of nodes at the current depth. For each level: launch the count kernel over all frontier nodes (one block or block-group per node), score, then a scatter-partition rewrites each node's range in-place into `[left | right]` sub-ranges, producing the next level's node ranges. This is the classic level-synchronous tree build (the structure XGBoost/LightGBM GPU hist and cuML use; reimplemented from the algorithm, not copied).
**When to use:** This phase — it is the locked decision. Depth-first per-node is explicitly NOT used (it underutilizes the GPU and was ruled out).
**Example (host pseudocode):**
```rust
// Source: original (reimplemented from the level-synchronous histogram-tree algorithm)
struct NodeRange { start: usize, end: usize, depth: usize, node_id: u32 }
let mut frontier = vec![NodeRange { start: 0, end: n_rows, depth: 0, node_id: 0 }];
let mut next = Vec::new();
while !frontier.is_empty() {
    // (1) count kernel over all frontier nodes (random threshold per feature)
    // (2) split-score kernel → best (feature, threshold) per node
    launch_count_and_score(&frontier, ...)?;
    for node in &frontier {
        if is_leaf(node) { emit_leaf(node); continue; }
        // (3) scatter-partition rewrites row_index[node.start..node.end]
        let mid = launch_scatter_partition(node, best_split[node], &mut d_row_index)?;
        next.push(left_range(node, mid));
        next.push(right_range(node, mid));
    }
    frontier = std::mem::take(&mut next);
}
```

### Pattern 3: Privatized shared-memory integer histogram → deterministic flush (SC-4)
**What:** Each block zero-inits a private `__shared__ unsigned int` histogram (e.g. `[2 sides × n_classes]` for the partition-count, or `[n_bins × n_classes]` if a binned variant is used), `__syncthreads()`, accumulates with **integer** `atomicAdd` into shared only, `__syncthreads()`, then flushes to global. Integer counts are associative → order-free → deterministic. **No float `atomicAdd` anywhere** (SC-4, PITFALLS Pitfall 5). This is exactly the Phase-1 sanitizer-clean kernel shape, scaled to the ET partition-count.
**When to use:** The count step in every level. It is the mandated baseline.
**Example:** see Code Examples below; the Phase-1 `HISTOGRAM_PRIVATIZED_SRC` (kernels.rs) is the verified template — cooperative zero-init, `__syncthreads()` before use, integer `atomicAdd`, `__syncthreads()` before flush.

### Pattern 4: Philox replay for bit-identical CPU↔GPU randomness
**What:** Inline the *exact* Philox-4×32-10 (constants, round fn, key bump, `u32_to_unit_f32` = `(x>>8) * (1/16777216)`) from `sylva-core::rng` into CUDA C. Draw `philox_uniform(seed, tree, node, feature, draw)` device-side with the **same counter packing** `[tree,node,feature,draw]` and the same `DRAW_THRESHOLD=0` / `DRAW_FEATURE_SELECT=1` indices. A NVRTC-compiled KAT-check kernel must reproduce the three frozen KAT vectors (kat.rs) before any tree is built — this is the cheap on-device proof the streams match.
**When to use:** Every random draw (feature subset Fisher-Yates prefix + threshold). It is the spine of bit-exactness.
**Anti-pattern:** Using cuRAND's device API or a different uint→float conversion — would silently diverge from the CPU stream.

### Anti-Patterns to Avoid
- **Global float `atomicAdd` for impurity/leaf sums** — non-deterministic, contention-bound. Use integer count atomics into shared, compute floats from counts in fixed order. (SC-4 / PITFALLS Pitfall 5.)
- **`--use_fast_math` or default FMA on the parity path** — collapses mul+add into a single rounded FMA, diverging from CPU `a*b + c`. Compile with `-fmad=false`. (Phase-3 `assign.rs` parity note carries this forward.)
- **Recompiling kernels per node/level** — compile once per `fit`.
- **Per-node device allocation** — allocate fit-scoped buffers once (the full arena pool is Phase-5 GPU-05, but even this phase must not `cudaMalloc` per node; pre-size to the max node count or a level-capacity bound).
- **Drawing the GPU threshold from bin edges instead of the raw f32 range** — would NOT match the CPU oracle (which draws from `[fmin,fmax]` of raw values). See Open Question 1.
- **Parallelizing the feature-subset draw in a way that reorders Philox draws** — the Fisher-Yates prefix in `split_et.rs` draws `DRAW_FEATURE_SELECT` per swap index `i`; the GPU must reproduce the same swap sequence (or the same resulting candidate set) deterministically.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NVRTC compile + module load + launch | A new FFI layer | `crates/sylva-cuda/src/nvrtc_launch.rs` pattern (cudarc) | Proven Phase-1; just add `-fmad=false` and new kernel wrappers. |
| Counter-based RNG | cuRAND / a second RNG | The shipped `sylva-core::rng` Philox + its CUDA-C twin | Bit-identity is the contract; KAT vectors already frozen. |
| Quantile binning / `BinnedMatrix` | A GPU re-binner | `sylva-core::quantize::Quantizer` (Phase 3, CPU, bit-parity gated) | Phase-3 already produces the contract input; GPU only *reads* it. |
| SoA forest representation | A GPU-specific tree struct | `sylva-core::ir::ForestIR` (ENG-02) | Single shared representation; assemble GPU results into it. |
| Impurity formulas | A new GPU gini/entropy/MSE | Reproduce `criterion.rs` op order exactly in CUDA C | Bit-exactness requires identical f32 op sequence, not a "mathematically equal" rewrite. |
| Typed device errors | `.unwrap()` on cudarc calls | `CudaError` (thiserror) → `SylvaError` | No-silent-fallback contract (PROJECT.md differentiator). |
| Bit-exact comparison | `allclose` / tolerance | `serde_json` byte-compare of the two `ForestIR`s (or `to_bits()` per field) | The gate is bit-exact, not approximate (GPU-02). |

**Key insight:** Phase 4 hand-rolls *only the four CUDA-C kernels and the breadth-first scheduler*. Everything else (RNG, binning, IR, criterion math, error handling, the differential-test harness) already exists and must be *matched*, not rebuilt. The kernels are reimplemented from the level-synchronous histogram-tree algorithm (Apache-2.0 clean — never copy cuML/XGBoost/LightGBM/Snap ML source).

## Common Pitfalls

### Pitfall 1: The raw-range-vs-binned threshold mismatch (THE phase risk)
**What goes wrong:** The histogram kernel is built on `BinnedMatrix` (uint8 bins), but the CPU oracle draws the threshold from the *raw f32* `[fmin,fmax]` and partitions on raw `x[i,f] <= threshold`. If the GPU draws a *bin-edge* threshold or partitions on bin codes, the tree structure diverges → bit-exact gate fails, and no amount of kernel tuning fixes it.
**Why it happens:** Histogram tree builders normally split on bins for speed; ET's defining feature is a *random real-valued threshold*, which is a different contract.
**How to avoid:** Decide the strategy up front (Open Question 1). The recommended Strategy A keeps raw f32 X on device and reproduces the CPU's exact `[fmin,fmax]` draw + raw `<=` partition; the histogram is used only to *count* class membership per side (an exact integer operation), not to choose the threshold.
**Warning signs:** GPU tree differs from CPU at the very first split; thresholds land exactly on bin edges.

### Pitfall 2: Float non-associativity / FMA breaks bit-exactness
**What goes wrong:** GPU computes `gini = 1 - Σ p²` with FMA or a different summation order than `criterion.rs`'s sequential `fold`, producing a last-bit-different f32 → a different "best split" on near-ties → divergent tree.
**Why it happens:** NVRTC defaults to FMA contraction; warp reductions sum in tree order, not sequential order.
**How to avoid:** Compile `-fmad=false`. Compute impurity from **integer counts** (order-free) using the *exact* op sequence of `gini`/`entropy`/`mse`/`proxy_improvement` (`c as f32 / n`, `p*p`, accumulate in counts-slice order). For any unavoidable reduction, use a fixed canonical order, never a warp-shuffle tree-sum on the parity path.
**Warning signs:** Two runs agree with each other but disagree with CPU by 1 ULP in thresholds/impurity.

### Pitfall 3: Tie-breaking divergence
**What goes wrong:** When two candidate features give equal `improvement`, the CPU keeps the one with lower `(feature_id, threshold.to_bits())` (split_et.rs:185-191). A GPU argmax that keeps "first thread to write" or "highest improvement, arbitrary tie" picks a different feature → divergent tree.
**How to avoid:** Reproduce the exact total-order tie-break: `improvement > best || (improvement == best && (feat, thr_bits) < (best_feat, best_thr_bits))`. Implement as a deterministic reduction with the same comparator.
**Warning signs:** Divergence only on datasets with constant/duplicate features (more ties).

### Pitfall 4: Feature-subset draw order mismatch
**What goes wrong:** `split_et.rs` draws the candidate features via a Fisher-Yates prefix using `philox_uniform(seed,tree,node, i, DRAW_FEATURE_SELECT)` for swap index `i` in `0..k`, mapping `j = i + (u*(n_features-i)) as usize` clamped to `n_features-1`. A GPU that draws features differently selects a different candidate set → different split.
**How to avoid:** Reproduce the identical Fisher-Yates prefix and clamp on device (or compute the candidate set host-side per node and upload it — simpler and equally bit-exact, since the draw is a pure function of `(seed,tree,node,i)`).
**Warning signs:** Correct threshold math but the *chosen feature* differs.

### Pitfall 5: `compute-sanitizer` racecheck on the privatized histogram
**What goes wrong:** A real ET count kernel that privatizes per-warp instead of per-block, or that flushes to global before `__syncthreads()`, trips `racecheck` even though counts "look right" on one run.
**How to avoid:** Follow the Phase-1 proven shape exactly: cooperative zero-init → `__syncthreads()` → integer `atomicAdd` into shared → `__syncthreads()` → cooperative flush. Run racecheck + memcheck on each kernel as a standalone single-launch target (the Phase-1 `sanitizer_histogram.rs` pattern). Compile with `-lineinfo` for source attribution.
**Warning signs:** Intermittent count errors; racecheck reports a WAR/WAW on `sh[]`.

### Pitfall 6: NaN handling drift
**What goes wrong:** Adding NaN routing into the *training* kernels when the contract says training data is clean (D-01) — NaN is a *predict-time* concern routed via `default_child`. Over-engineering it into the build kernels risks divergence and wasted complexity.
**How to avoid:** Training kernels assume clean data (matching the CPU oracle, which never sees NaN in training). `default_child` is computed exactly as CPU does (higher sample-count child; tie→left). NaN routing lives only in predict (reuse `predict_forest` or mirror its `is_nan()`-first traverse).

### Pitfall 7: Over-claiming speed (fairness contract)
**What goes wrong:** Reporting the single-tree GPU-vs-CPU timing as evidence the project is "faster."
**How to avoid:** Phase 4 is foundational — **NO end-to-end speed claim** (STATE.md binding fairness rule). Single-tree timing is transfer-inclusive and labeled *informational only*; the bit-exact + sanitizer-clean + distributional-parity correctness is the gate. First real speed claim is Phase 5; authoritative crossover is Phase 7.

## Code Examples

### CUDA-C Philox-4×32-10 (must reproduce sylva-core::rng + KAT vectors)
```cuda
// Source: reimplemented from Random123 (DEShawResearch philox.h), bit-matching
// crates/sylva-core/src/rng/mod.rs. Apache-2.0 (not copied). MUST pass the three
// kat.rs KAT vectors on-device before any tree is built (ENG-06 parity proof).
__device__ __forceinline__ void mulhilo32(unsigned a, unsigned b, unsigned* hi, unsigned* lo) {
    unsigned long long p = (unsigned long long)a * (unsigned long long)b;
    *hi = (unsigned)(p >> 32); *lo = (unsigned)p;
}
__device__ void philox4x32_10(unsigned ctr[4], unsigned key[2], unsigned out[4]) {
    const unsigned M0=0xD2511F53u, M1=0xCD9E8D57u, W0=0x9E3779B9u, W1=0xBB67AE85u;
    unsigned c0=ctr[0],c1=ctr[1],c2=ctr[2],c3=ctr[3], k0=key[0],k1=key[1];
    for (int r=0; r<10; ++r) {
        if (r!=0) { k0+=W0; k1+=W1; }              // bump key on rounds 1..9
        unsigned hi0,lo0,hi1,lo1;
        mulhilo32(M0,c0,&hi0,&lo0); mulhilo32(M1,c2,&hi1,&lo1);
        unsigned n0=hi1^c1^k0, n1=lo1, n2=hi0^c3^k1, n3=lo0;
        c0=n0; c1=n1; c2=n2; c3=n3;
    }
    out[0]=c0; out[1]=c1; out[2]=c2; out[3]=c3;
}
__device__ __forceinline__ float u32_to_unit_f32(unsigned x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);   // identical to rng/mod.rs
}
__device__ float philox_uniform(unsigned long long seed, unsigned tree,
                                unsigned node, unsigned feature, unsigned draw) {
    unsigned key[2] = { (unsigned)seed, (unsigned)(seed>>32) };
    unsigned ctr[4] = { tree, node, feature, draw };   // pack_counter order
    unsigned out[4]; philox4x32_10(ctr, key, out);
    return u32_to_unit_f32(out[0]);
}
```

### Privatized partition-count kernel (ET, Strategy A — counts per side × class)
```cuda
// Source: original, extending the Phase-1 sanitizer-clean privatized histogram
// (crates/sylva-cuda/src/kernels.rs HISTOGRAM_PRIVATIZED_SRC). One block per node.
// Draws the random threshold from the RAW f32 range (matches split_et.rs), counts
// class membership on each side with integer atomics into shared. NO float atomics.
extern "C" __global__ void et_count_node(
    const float* __restrict__ x,        // raw f32, column-major: x[feat*n_rows + row]
    const float* __restrict__ y_label,  // class label as float (cast to int)
    const int*   __restrict__ row_index,// permutation; this node owns [start,end)
    int start, int end, int n_rows, int n_classes,
    unsigned long long seed, unsigned tree, unsigned node,
    int feat, float fmin, float fmax,   // raw range precomputed (or computed here)
    unsigned int* __restrict__ out_counts /* [2*n_classes]: left then right */) {
    extern __shared__ unsigned int sh[]; // size = 2*n_classes ints (dynamic shared)
    int t = threadIdx.x;
    for (int b=t; b<2*n_classes; b+=blockDim.x) sh[b]=0u;
    __syncthreads();
    float u = philox_uniform(seed, tree, node, (unsigned)feat, /*DRAW_THRESHOLD=*/0u);
    float thr = fmin + u * (fmax - fmin);     // identical to split_et.rs
    for (int p=start+t; p<end; p+=blockDim.x) {
        int row = row_index[p];
        float v = x[(size_t)feat*n_rows + row];
        int cls = (int)y_label[row];
        int side = (v <= thr) ? 0 : 1;        // raw <= compare, NO fast-math
        if (cls < n_classes) atomicAdd(&sh[side*n_classes + cls], 1u);
    }
    __syncthreads();
    for (int b=t; b<2*n_classes; b+=blockDim.x) atomicAdd(&out_counts[b], sh[b]);
}
// Host then computes gini/entropy/MSE proxy from out_counts in criterion.rs op order,
// picks best feature with the exact (feature, threshold_bits) tie-break, and launches
// the scatter-partition. (Score can also be a tiny on-device reduction with -fmad=false.)
```

### The bit-exact gate (Rust integration test)
```rust
// Source: extends the Phase-2 determinism test idiom (cpu/fit.rs seed_determinism).
#[test]
fn gpu_single_tree_matches_cpu_oracle_bit_exact() {
    let (x, y) = fixed_seed_dataset();          // medium dense, fixed seed
    let cfg = TrainConfig { n_estimators: 1, algo: Algo::ExtraTrees, seed: 42, /* .. */ };
    let cpu = CpuBackend.fit(x.view(), y.view(), &cfg).unwrap();
    let gpu = CudaBackend::new().unwrap().fit(x.view(), y.view(), &cfg).unwrap();
    let s_cpu = serde_json::to_string(&cpu).unwrap();
    let s_gpu = serde_json::to_string(&gpu).unwrap();
    assert_eq!(s_cpu, s_gpu, "GPU ExtraTree must equal CPU oracle byte-for-byte");
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nvcc AOT + bindgen FFI | cudarc + NVRTC runtime compile | settled Phase 1 | No nvcc-at-build; native Windows/MSVC works. |
| Depth-first per-node tree build on GPU | Breadth-first level-synchronous (frontier waves) | LOCKED (this phase) | GPU utilization; the modern hist-tree standard. |
| Global atomic histograms | Privatized shared-mem + deterministic flush | LOCKED (this phase) | Determinism + reduced contention; sanitizer-clean (Phase 1). |
| Float histogram accumulation | Integer counts → float impurity in fixed order | LOCKED (designed in now, hardened Phase 6) | Bit-reproducibility (DET-01 foundation). |

**Deprecated/outdated:**
- cudarc deprecated `memcpy_stod`/`memcpy_dtov` — use `clone_htod`/`clone_dtoh` (already adopted Phase 1).
- Do not reach for `--use_fast_math` or default FMA on the parity path.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Strategy A (GPU replays the CPU's raw-f32-range exact algorithm; histogram used only for integer counting) is the right path for a bit-exact single ExtraTree. The alternative (binned-candidate split) would NOT match the raw-range CPU oracle without also changing the CPU oracle. | Summary / Open Q1 | If the planner/user instead wants the CPU oracle changed to operate on bins (so GPU and CPU both bin-split), Phase 2's `split_et.rs` would need revision — a cross-phase change. This is the single biggest design fork; flagged as Open Question 1 for explicit decision. |
| A2 | `-fmad=false` (NVRTC option) is sufficient to make device float ops match the CPU's separate mul/add for the gini/entropy/MSE/proxy computations. | Pitfall 2 / Pattern 1 | If residual divergence remains (e.g. `log2` implementation differs for entropy), the parity test catches it; mitigation is to compute impurity host-side from device-returned integer counts (counts are exact), sidestepping device float entirely for the score. Recommend the host-side-score fallback be designed in. |
| A3 | The `BinnedMatrix` is needed at all for the single-ET bit-exact path. Under Strategy A the histogram counts class membership by raw `<=` partition, which does not strictly require the bins — raw X suffices. | Open Q2 | If bins are not used in Phase 4, the Phase-3 quantizer is still validated independently (its own gate) and becomes load-bearing in Phase 5 (forest/RF binned splits). Phase 4 may upload raw X only. Flagged Open Q2. |
| A4 | One block per frontier node is a workable mapping for the medium dataset (200k×50). For very wide frontiers or tiny nodes this underutilizes/overspills; the full arena + small-node CPU-finish cutover is Phase-5 (GPU-05/06), not required for correctness here. | Pattern 2 | Performance only, not correctness — Phase 4 makes no speed claim, so a simple mapping is acceptable. |
| A5 | The three Random123 KAT vectors frozen in `kat.rs` are correct (Phase 2 shipped + tested them). The CUDA Philox need only reproduce *those* vectors to be bit-identical to the CPU. | Pattern 4 | Already de-risked: `rng/mod.rs` tests pass against `kat.rs`. The on-device KAT check is the proof, not an assumption about the vectors themselves. |
| A6 | Entropy parity: CPU uses `p.log2()` (f32). Device `log2f` must produce the identical f32. | Pitfall 2 | `log2f` is the most likely single-ULP divergence point. Mitigation: gate the parity test on gini + MSE first; if entropy diverges, compute the entropy score host-side from device counts (A2 fallback). Recommend planner sequence gini/MSE before entropy. |

**If this table is empty:** it is not — A1 (the strategy fork) and A6 (entropy log2) are the two the planner/discuss-phase must resolve before locking the plan.

## Open Questions

1. **Bit-exact strategy: replay raw-range (A) vs binned-candidate (B) vs change the CPU oracle (C)?**
   - What we know: CPU `split_et.rs` draws `threshold = fmin + u*(fmax-fmin)` from the **raw f32** node range and partitions on raw `x[i,f] <= threshold`. The histogram literature splits on *bins*.
   - What's unclear: whether Phase 4 must produce a tree bit-identical to the *current* CPU oracle (→ Strategy A, GPU replays raw-range exactly; bins used only to count) or whether the project accepts a binned ExtraTree as the canonical algorithm (→ Strategy B + a matching change to the Phase-2 CPU oracle, a cross-phase edit).
   - Recommendation: **Strategy A** — it satisfies GPU-02 against the shipped oracle with the least churn and keeps "simplest possible GPU hot path." Confirm with the user; if they prefer binned splits as canonical, that is a Phase-2 oracle change and must be decided explicitly (and re-validated against Phase-2's distributional gate).

2. **Does Phase 4 upload the `BinnedMatrix` or raw X (or both)?**
   - What we know: Under Strategy A the count kernel needs raw X for the `<=` partition and the `[fmin,fmax]` range. The `BinnedMatrix` accelerates binned histograms (Strategy B / RF Phase 5).
   - Recommendation: Upload **raw X (column-major f32)** for the Strategy-A count/partition. Optionally also upload the `BinnedMatrix` to exercise the Phase-3 → Phase-4 contract end-to-end (de-risks Phase 5), but it is not required for the bit-exact ET. Decide based on whether the user wants the binning contract exercised on GPU now or in Phase 5.

3. **Score computation: on-device (with `-fmad=false`) or host-side from device counts?**
   - What we know: integer counts returned from the count kernel are exact; the only float work is the impurity/proxy.
   - Recommendation: Start with **host-side scoring from device counts** (guarantees the exact `criterion.rs` op order, zero device-float-parity risk) for the single-tree phase; move scoring on-device in Phase 5 when the per-level node count makes host round-trips a bottleneck. This is the lowest-risk path to the bit-exact gate.

4. **Frontier→block mapping and max node-count pre-sizing.**
   - What we know: one block per node is simple; node count is bounded by `2^(max_depth+1)-1` per tree.
   - Recommendation: pre-size device node arrays to a depth-derived bound; defer the stream-ordered arena pool (GPU-05) and small-node CPU cutover (GPU-06) to Phase 5. Document the bound.

5. **Predict on GPU or reuse CPU `predict_forest` for the timing study?**
   - Recommendation: For correctness, the gate is on the *trained IR* (structure), so predict can reuse the CPU `predict_forest`. A GPU predict kernel is optional this phase; if added, it must also be sanitizer-clean. The transfer-inclusive timing study should state which predict path it used.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU (sm_89) | All GPU kernels | ✓ | RTX 4060 Ti, driver 595.79 | none (CUDA-only MVP) |
| CUDA Toolkit + NVRTC | Runtime kernel compile | ✓ | 12.8 (`nvrtc64_120_0.dll`) | none |
| compute-sanitizer | racecheck/memcheck gate (GPU-02) | ✓ | CUDA 12.8 (full path, not on PATH) | none — required for the gate |
| Rust stable + MSVC v143 | Build sylva-cuda / link | ✓ | rustc 1.96.0, cl.exe 14.44 | none |
| cudarc 0.19.8 | driver+NVRTC bindings | ✓ | pinned in Cargo.lock | none |
| scikit-learn (Python) | Distributional baseline (informational) | likely (Phase-2 harness uses it) | TBD — pin in study | skip distributional check (gate is the CPU bit-exact, not sklearn) |

**Missing dependencies with no fallback:** none — the full GPU toolchain is proven present (Phase 1 VERSIONS.md).
**Missing dependencies with fallback:** scikit-learn pin should be recorded in the study; the distributional check is informational, so its absence does not block the gate.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `#[test]` / `#[cfg(test)]` + `tests/` integration (cargo-nextest); Python pytest for the baseline study |
| Config file | none for Rust (cargo built-in); `python/tests/` harness mirrors Phase-2 `python/tests/parity/` |
| Quick run command | `cargo test -p sylva-cuda --test parity_cpu_gpu` |
| Full suite command | `cargo test -p sylva-cuda` + `cargo test -p sylva-core` (oracle still green) + `pytest python/tests/gpu_parity/` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-01 | Single GPU ExtraTree (clf+reg) trains via privatized shared-mem histogram + fused random-split + scatter-partition, breadth-first level-at-a-time | integration | `cargo test -p sylva-cuda --test parity_cpu_gpu` | ❌ Wave 0 |
| GPU-01 | Histograms shared-mem-resident, integer atomics only — NO global float atomics on the hot path (SC-4) | unit + sanitizer | `cargo test -p sylva-cuda --test sanitizer_et_kernels` + racecheck | ❌ Wave 0 |
| GPU-02 | GPU ExtraTree matches CPU oracle **bit-exactly** on a fixed seed (clf + reg) | integration (byte-compare) | `cargo test -p sylva-cuda --test parity_cpu_gpu` | ❌ Wave 0 |
| GPU-02 | `compute-sanitizer` racecheck + memcheck clean against every kernel | sanitizer (standalone targets) | `compute-sanitizer --tool racecheck/memcheck <exe>` (Phase-1 procedure) | ❌ Wave 0 |
| ENG-06 (reuse) | On-device Philox reproduces the frozen KAT vectors | unit (device) | `cargo test -p sylva-cuda --test philox_device_kat` | ❌ Wave 0 |
| SC-5 | Distributional parity vs sklearn single ExtraTree (informational gate) | Python | `pytest python/tests/gpu_parity/test_sklearn_distributional.py` | ❌ Wave 0 |
| SC-6 | Single-tree GPU-vs-CPU timing (transfer-inclusive, REPORTED not gated) | Python (report) | `pytest python/tests/gpu_parity/test_single_tree_timing.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p sylva-cuda --test parity_cpu_gpu` (the bit-exact gate) + the relevant sanitizer target for any touched kernel.
- **Per wave merge:** full `cargo test -p sylva-cuda` + `cargo test -p sylva-core` (oracle regression) + all four sanitizer tools on each kernel.
- **Phase gate:** bit-exact parity (clf + reg) green AND all kernels sanitizer-clean (memcheck + racecheck + synccheck + initcheck, the Phase-1 four-tool standard) AND distributional parity reported before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/sylva-cuda/tests/parity_cpu_gpu.rs` — the bit-exact gate (GPU-01/GPU-02)
- [ ] `crates/sylva-cuda/tests/philox_device_kat.rs` — on-device Philox KAT (ENG-06)
- [ ] `crates/sylva-cuda/tests/sanitizer_et_kernels.rs` — standalone single-launch sanitizer targets per kernel (the Phase-1 `sanitizer_histogram.rs` pattern, generalized)
- [ ] `python/tests/gpu_parity/` — distributional (sklearn) + timing harness, mirroring `python/tests/parity/` (datasets.py, conftest version manifest)
- [ ] A `pyseam` entry point for GPU fit (mirror `crates/sylva-core/src/pyseam.rs` `fit_cpu`) so the Python harness can call the GPU path

*(No existing GPU-path test infrastructure beyond the Phase-1 toolchain smoke/sanitizer tests — all Phase-4 validation files are new.)*

## Security Domain

> `security_enforcement` is enabled (absent from config = enabled). This is a local GPU compute library with no network/auth surface; the relevant controls are memory safety, input validation, and license provenance.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (local library). |
| V3 Session Management | no | N/A. |
| V4 Access Control | no | N/A. |
| V5 Input Validation | yes | Boundary validation in `CudaBackend::fit` (shape, dtype, label range) → typed `CudaError`/`SylvaError` before any device launch (the Phase-1 V5 pattern: e.g. `bins[i] < BIN_COUNT`, length checks). An out-of-range index would be an OOB device read. |
| V6 Cryptography | no | Philox is a **non-cryptographic** statistical RNG (documented in `rng/mod.rs`); never used for secrets. |

### Known Threat Patterns for {Rust core + CUDA-C via NVRTC}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device memory access (kernel index past buffer) | Tampering / Info-disclosure | `if (i < n)` thread guards in every kernel; host-side bounds validation before launch; `compute-sanitizer memcheck` clean gate. |
| Shared-memory data race in privatized histogram | Tampering | Cooperative zero-init + `__syncthreads()` discipline; `racecheck` clean gate (Phase-1 proven shape). |
| Uninitialized shared/global read | Info-disclosure | `initcheck` clean gate; `alloc_zeros` for device output buffers. |
| Silent CUDA error swallow → wrong result | Tampering | No `.unwrap()` on cudarc calls; every call `Result`-propagated to `CudaError`; explicit `stream.synchronize()` to surface launch/exec errors (Phase-1 contract). |
| License contamination (copying cuML/XGBoost/LightGBM/Snap-ML kernel source) | (Legal/IP) | Apache-2.0 reimplementation discipline (`.claude/CLAUDE.md`): author kernels from the level-synchronous histogram-tree *algorithm*; document provenance; never copy GPL/closed source. |
| Untrusted `X` with NaN/Inf at train time | Tampering | Training contract is clean data (D-01); NaN is a predict-time routing concern via `default_child`. Validate and document the train-time clean-data assumption at the boundary. |

## Project Constraints (from CLAUDE.md)

- **cudarc 0.19.8 + hand-written CUDA-C via NVRTC** — the only sanctioned kernel-authoring path. No CubeCL (M2 fallback), no Rust-CUDA `cust`, no nvcc/cc/build.rs AOT, no wgpu.
- **Native Windows / MSVC, no WSL** for the MVP build. NVRTC compiles kernels at runtime so the broken cc+MSVC CUDA path is never exercised.
- **NO nvcc-at-build-time** — kernels are `.cu` strings compiled by NVRTC.
- **Apache-2.0 reimplementation discipline** — reimplement algorithms from papers/docs; never copy GPL / Snap ML / sklearn / cuML / XGBoost source.
- **No silent fallback** — every CUDA call is a `Result`; typed `CudaError`; no `.unwrap()`/`.expect()` on device calls; `unsafe` confined to the launch call with a `// SAFETY:` comment.
- **Stable Rust 1.83+** (project on 1.96.0) — never nightly.
- **`-D warnings`** (clippy) is the lint bar — use non-deprecated cudarc APIs (`clone_htod`/`clone_dtoh`).
- **float32 end-to-end (D-05)** and **integer/deterministic accumulation** designed in from this phase (hardened Phase 6).
- **Many small files** (200–400 lines, 800 max); organize by domain (mirror the `cpu/` module shape).
- **Comparative-study fairness binding** — Phase 4 is foundational: correctness-parity gate only, **no end-to-end speed claim**; single-tree timing is informational.

## Sources

### Primary (HIGH confidence)
- `crates/sylva-cuda/src/{kernels.rs,nvrtc_launch.rs}` — proven NVRTC compile/launch API, sanitizer-clean privatized histogram template, `CudaError` pattern.
- `crates/sylva-core/src/{ir.rs,backend.rs,config.rs,error.rs}` — `ForestIR` SoA, `Backend`/`HistogramBackend` traits, `TrainConfig`, `SylvaError`.
- `crates/sylva-core/src/cpu/{fit.rs,split_et.rs,split_rf.rs,predict.rs,criterion.rs}` — the exact CPU algorithm + float op order the GPU must match bit-for-bit; tie-break, Fisher-Yates feature draw, NaN-first predict, fixed-order impurity.
- `crates/sylva-core/src/rng/{mod.rs,kat.rs}` — Philox-4×32-10 constants, `u32_to_unit_f32`, counter packing, frozen KAT vectors.
- `crates/sylva-core/src/quantize/{mod.rs,binned_matrix.rs,assign.rs}` — `BinnedMatrix`/`BinEdges` contract, column-major SoA layout, `assign_bin` searchsorted, the no-FMA/no-fast-math parity note.
- `VERSIONS.md` — toolchain pins (cudarc 0.19.8, CUDA 12.8, sm_89, driver 595.79, link-mode toggle, four-tool sanitizer-clean evidence).
- `.planning/{ROADMAP.md,REQUIREMENTS.md,STATE.md,PROJECT.md}` — Phase-4 goal/SC, GPU-01/GPU-02, the four locked architecture decisions, the fairness contract, the kernel-authoring decision.
- `.claude/CLAUDE.md` — kernel-authoring table, Apache-2.0 discipline, Windows/MSVC/NVRTC constraints.

### Secondary (MEDIUM confidence)
- `.planning/phases/03-feature-quantizer-cpu-gpu-bit-parity/03-PATTERNS.md` — confirms the quantize contract + the "GPU literal `<=` compare, no `--use_fast_math`" forward-carry.
- `.planning/phases/01-toolchain-spike-gate-1/01-02-SUMMARY.md` / `01-03-SUMMARY.md` — sanitizer procedure, link-mode toggle, NVRTC launch sequence.
- docs.rs cudarc `LaunchConfig` (`shared_mem_bytes` for dynamic shared memory) — confirms the dynamic-shared-mem launch field for variable `2*n_classes` histograms. https://docs.rs/cudarc/latest/cudarc/driver/safe/struct.LaunchConfig.html

### Tertiary (LOW confidence)
- General level-synchronous GPU histogram-tree algorithm knowledge (XGBoost/LightGBM/cuML hist builders) — used only as *algorithmic* reference for the breadth-first + privatized-histogram pattern; reimplemented clean, never copied. `[ASSUMED]` for any specific intrinsic-level detail beyond the Phase-1-proven shape.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; everything pinned + proven in Phases 1–3.
- Architecture (kernels + breadth-first scheduler + privatized histogram): HIGH for the *shape* (Phase-1 proven primitive + locked decisions); MEDIUM for the bit-exact reconciliation strategy (Open Question 1 — the one real design fork).
- Pitfalls: HIGH — derived directly from the shipped CPU oracle's exact arithmetic + the Phase-1 sanitizer experience + the Phase-3 parity note.

**Research date:** 2026-06-26
**Valid until:** ~2026-07-26 (stable — toolchain pinned; the only volatile element is whichever bit-exact strategy the user picks, which is a decision not a moving external fact).
