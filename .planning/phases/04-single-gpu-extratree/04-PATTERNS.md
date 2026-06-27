# Phase 4: Single GPU ExtraTree — Pattern Map

**Mapped:** 2026-06-27
**Files analyzed:** 11 new files (8 src + 3 tests)
**Analogs found:** 11 / 11 (all have direct analogs; no greenfield patterns required)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `crates/sylva-cuda/src/lib.rs` | module-root / FFI boundary | request-response | `crates/sylva-cuda/src/lib.rs` (Phase 1) | EXTEND (same file) |
| `crates/sylva-cuda/src/kernels.rs` | kernel source strings | CUDA-C via NVRTC | `crates/sylva-cuda/src/kernels.rs` (Phase 1) | EXTEND (same file) |
| `crates/sylva-cuda/src/nvrtc_launch.rs` | compile + launch harness | request-response | `crates/sylva-cuda/src/nvrtc_launch.rs` (Phase 1) | EXTEND (same file) |
| `crates/sylva-cuda/src/cuda_backend/mod.rs` | service (impl Backend) | request-response | `crates/sylva-core/src/cpu/fit.rs` `fit_forest` | role-match |
| `crates/sylva-cuda/src/cuda_backend/scheduler.rs` | orchestrator | event-driven (frontier loop) | `crates/sylva-core/src/cpu/fit.rs` `build_tree` | role-match |
| `crates/sylva-cuda/src/cuda_backend/device_buffers.rs` | resource manager | CRUD (alloc/upload/free) | `crates/sylva-cuda/src/nvrtc_launch.rs` (H2D/alloc pattern) | role-match |
| `crates/sylva-cuda/src/cuda_backend/philox_device.rs` | utility (CUDA-C source + KAT launch) | request-response | `crates/sylva-core/src/rng/mod.rs` + `kat.rs` | exact (inline translation) |
| `crates/sylva-cuda/src/cuda_backend/histogram.rs` | kernel launch wrapper | CRUD | `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` | exact |
| `crates/sylva-cuda/src/cuda_backend/split_score.rs` | utility (host-side impurity from counts) | transform | `crates/sylva-core/src/cpu/criterion.rs` | exact |
| `crates/sylva-cuda/src/cuda_backend/partition.rs` | kernel launch wrapper | CRUD | `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` | role-match |
| `crates/sylva-cuda/src/cuda_backend/assemble.rs` | assembler (D2H → ForestIR) | transform | `crates/sylva-core/src/cpu/fit.rs` `assemble_forest` / `TreeFragment` | exact |
| `crates/sylva-cuda/tests/parity_cpu_gpu.rs` | integration test (bit-exact gate) | request-response | `crates/sylva-core/tests/determinism.rs` | exact |
| `crates/sylva-cuda/tests/philox_device_kat.rs` | integration test (on-device KAT) | request-response | `crates/sylva-core/src/rng/mod.rs` `philox_matches_kat_vectors` | exact |
| `crates/sylva-cuda/tests/sanitizer_et_kernels.rs` | integration test (sanitizer target) | request-response | `crates/sylva-cuda/tests/sanitizer_histogram.rs` | exact |

---

## Pattern Assignments

### `crates/sylva-cuda/src/kernels.rs` (EXTEND — add ET kernel source strings)

**Analog:** `crates/sylva-cuda/src/kernels.rs` (Phase 1), lines 1–69

**Module doc pattern** (lines 1–14): Copy the module-doc shape — one sentence on scope, explicit "no restrictively-licensed source" note, explicit "NVRTC not nvcc" statement.

**Kernel source string pattern** (lines 21–64):
```rust
// Copy this shape for each new kernel:
pub const ET_COUNT_NODE_SRC: &str = r#"
// [kernel CUDA-C source — see RESEARCH.md Code Examples for the full body]
// Strip: extern "C" __global__ void NAME(args...) { ... }
"#;

pub const ET_PHILOX_KAT_SRC: &str = r#"
// Philox-4x32-10 + u32_to_unit_f32 inlined; KAT-check kernel
"#;

pub const ET_SCATTER_PARTITION_SRC: &str = r#"
// scatter-partition kernel
"#;
```

**Named constant pattern** (line 68):
```rust
// One named constant per magic number — never inline literals at launch sites.
pub const BIN_COUNT: usize = 256;      // ← existing, keep
pub const ET_BLOCK_SIZE: u32 = 256;    // ← add: threads per block for ET kernels
```

**Compile flag — CRITICAL addition** (for `compile_for_sm89` in `nvrtc_launch.rs`, line 57–63):
```rust
// Phase 1 had only "-lineinfo". Phase 4 MUST add "-fmad=false":
let opts = CompileOptions {
    arch: Some(ARCH_SM_89),
    options: vec!["-lineinfo".to_string(), "-fmad=false".to_string()],
    ..Default::default()
};
// -fmad=false: disables fused-multiply-add so device float ops match the
// CPU's separate mul/add sequence. NEVER --use_fast_math. (RESEARCH Pitfall 2)
```

---

### `crates/sylva-cuda/src/nvrtc_launch.rs` (EXTEND — add per-kernel launch wrappers)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` (Phase 1), lines 1–168

**Error type pattern** (lines 34–50): `CudaError` with three variants — `Compile(#[from] CompileError)`, `Driver(#[from] DriverError)`, `InvalidInput(String)`. Copy verbatim; do NOT add new variants unless a new failure class appears.

**Launch function pattern** (lines 124–168 — `run_histogram`):
```rust
// Template for every new ET kernel wrapper:
pub fn launch_et_count_node(/* typed args */) -> Result<Vec<u32>, CudaError> {
    // 1. Host-side V5 boundary validation → CudaError::InvalidInput (before any device call)
    if n_classes == 0 { return Err(CudaError::InvalidInput("n_classes must be > 0".into())); }

    // 2. Compile ONCE (caller should cache the compiled module — pass it in for Phase 4)
    //    For the standalone spike, compile here; for CudaBackend::fit, compile in mod.rs::new()
    let ptx = compile_for_sm89(ET_COUNT_NODE_SRC)?;
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("et_count_node")?;
    let stream = ctx.default_stream();

    // 3. H2D uploads — use clone_htod (NOT deprecated memcpy_stod)
    let d_x     = stream.clone_htod(x_col_major)?;
    let d_y     = stream.clone_htod(y_labels)?;
    let d_rows  = stream.clone_htod(row_index)?;
    let mut d_counts = stream.alloc_zeros::<u32>(2 * n_classes)?;

    // 4. LaunchConfig — shared_mem_bytes for dynamic shared memory
    let cfg = LaunchConfig {
        grid_dim:  (1, 1, 1),          // one block per node (Phase 4 mapping)
        block_dim: (ET_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: (2 * n_classes * std::mem::size_of::<u32>()) as u32,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&d_x).arg(&d_y).arg(&d_rows)
           .arg(&(start as i32)).arg(&(end as i32)).arg(&(n_rows as i32))
           .arg(&(n_classes as i32))
           .arg(&seed).arg(&tree_id).arg(&node_id)
           .arg(&(feat as i32)).arg(&fmin).arg(&fmax)
           .arg(&mut d_counts);
    // SAFETY: args match et_count_node signature; every thread guards index with
    // if (p < end); sh is zero-inited before use; __syncthreads() before flush.
    // compute-sanitizer racecheck+memcheck is the proof gate (GPU-02).
    unsafe { builder.launch(cfg)?; }
    stream.synchronize()?;   // surface launch/exec errors — never skip

    Ok(stream.clone_dtoh(&d_counts)?)
}
```

**Key invariants to copy from Phase 1:**
- Every cudarc call propagated with `?` into `CudaError` — no `.unwrap()` / `.expect()`.
- `unsafe` confined to the single `builder.launch(cfg)?;` line with `// SAFETY:` comment.
- `stream.synchronize()?;` immediately after launch.
- `stream.clone_htod` / `stream.clone_dtoh` (not deprecated `memcpy_*` variants).
- `stream.alloc_zeros` for output buffers (satisfies `initcheck`).

---

### `crates/sylva-cuda/src/cuda_backend/mod.rs` (CudaBackend struct + impl Backend)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` lines 48–116 (`fit_forest`) + `crates/sylva-cuda/src/lib.rs` lines 17–37 (PyO3 error mapping)

**Struct shape:**
```rust
// Source: mirrors cpu/fit.rs's no-state CpuBackend + Phase-1 lib.rs CudaError mapping
use crate::nvrtc_launch::CudaError;
use sylva_core::{Backend, ForestIR, SylvaError, TrainConfig};

pub struct CudaBackend {
    // Compiled NVRTC modules cached here (compile-once in ::new() or ::fit())
    // For Phase 4: Option<CompiledModules> — None until first fit() call
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaError> { /* init + KAT verify */ Ok(Self { }) }
}

impl Backend for CudaBackend {
    fn fit(&self, x: ArrayView2<f32>, y: ArrayView1<f32>, cfg: &TrainConfig)
        -> Result<ForestIR, SylvaError> {
        // 1. Boundary validation (mirror cpu/fit.rs lines 54–69)
        cfg.validate()?;
        if x.nrows() == 0 { return Err(SylvaError::InvalidInput("X has 0 rows".into())); }
        // ...

        // 2. Phase-3 Quantizer::fit_transform(x) → BinnedMatrix + BinEdges
        //    (upload BinnedMatrix to exercise Phase-3→4 contract; raw X also uploaded
        //    for Strategy A count kernel)

        // 3. H2D uploads (via device_buffers.rs)

        // 4. Compile NVRTC kernels once (or use cached modules)
        //    Verify on-device Philox KAT (philox_device.rs) before first tree

        // 5. Breadth-first loop (scheduler.rs)

        // 6. D2H + assemble.rs → ForestIR

        // Map CudaError → SylvaError (never expose raw CudaError across Backend boundary)
    }

    fn predict(&self, ir: &ForestIR, x: ArrayView2<f32>) -> Result<Predictions, SylvaError> {
        // Phase 4: reuse sylva_core::cpu::predict_forest (CPU predict on GPU-trained IR)
        use sylva_core::cpu::CpuBackend;
        CpuBackend.predict(ir, x)
    }
}
```

---

### `crates/sylva-cuda/src/cuda_backend/scheduler.rs` (breadth-first frontier loop)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` lines 118–161 (`TreeFragment` + `build_tree` control flow) — the per-node dispatch pattern; reimplemented as a level-synchronous loop.

**Core pattern** (from RESEARCH.md Pattern 2 — original algorithm, not copied):
```rust
// Source: original level-synchronous histogram-tree algorithm
// (reimplemented from the standard GPU hist-tree description; never copied
//  from cuML/XGBoost/LightGBM/Snap ML source — Apache-2.0 discipline)
#[derive(Debug, Clone)]
pub struct NodeRange {
    pub start: usize,   // inclusive, into d_row_index
    pub end:   usize,   // exclusive
    pub depth: usize,
    pub node_id: u32,   // global node id in the device node arrays
}

pub struct Scheduler {
    pub frontier: Vec<NodeRange>,
    pub next:     Vec<NodeRange>,
}

impl Scheduler {
    pub fn new(n_rows: usize) -> Self {
        Self {
            frontier: vec![NodeRange { start: 0, end: n_rows, depth: 0, node_id: 0 }],
            next:     Vec::new(),
        }
    }

    /// Advance one level. Returns true if there are more levels to process.
    pub fn advance(&mut self) -> bool {
        self.frontier = std::mem::take(&mut self.next);
        !self.frontier.is_empty()
    }
}
```

**Stop criteria pattern** (mirror `build_node` in `cpu/fit.rs` lines 195–230):
```rust
// Mirror the CPU's exact stop conditions — must agree with the CPU oracle:
fn is_leaf(range: &NodeRange, n_classes: usize, counts: &[u32], cfg: &TrainConfig) -> bool {
    let n = (range.end - range.start) as u64;
    n < cfg.min_samples_split as u64
    || cfg.max_depth.map_or(false, |d| range.depth >= d)
    // || all rows same class (pure node) — check from counts
}
```

---

### `crates/sylva-cuda/src/cuda_backend/device_buffers.rs` (fit-scoped device buffer allocation)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 85–111 (alloc + H2D upload pattern)

**Pattern:**
```rust
// Source: extends nvrtc_launch.rs H2D pattern — allocate ONCE per fit, not per node
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use crate::nvrtc_launch::CudaError;

pub struct FitBuffers {
    pub d_x:         CudaSlice<f32>,  // raw X, column-major: x[feat*n_rows + row]
    pub d_y:         CudaSlice<f32>,  // class labels as f32
    pub d_bins:      CudaSlice<u8>,   // BinnedMatrix (column-major u8)
    pub d_row_index: CudaSlice<i32>,  // permutation buffer [0..n_rows]
    // Node arrays (pre-sized to depth-bound: 2^(max_depth+1)-1 nodes per tree)
    pub d_feature_id:  CudaSlice<i32>,
    pub d_threshold:   CudaSlice<f32>,
    pub d_left_child:  CudaSlice<i32>,
    pub d_right_child: CudaSlice<i32>,
    pub d_counts:      CudaSlice<u32>, // [2 * n_classes] reused each node
}

impl FitBuffers {
    pub fn allocate(stream: &CudaStream, n_rows: usize, n_features: usize,
                    n_classes: usize, max_nodes: usize, x_col: &[f32],
                    y: &[f32], bins: &[u8]) -> Result<Self, CudaError> {
        // H2D uploads: stream.clone_htod(slice)? for filled buffers
        // alloc_zeros: stream.alloc_zeros::<T>(len)? for output buffers
        Ok(Self {
            d_x:         stream.clone_htod(x_col)?,
            d_y:         stream.clone_htod(y)?,
            d_bins:      stream.clone_htod(bins)?,
            d_row_index: stream.clone_htod(&(0..n_rows as i32).collect::<Vec<_>>())?,
            d_feature_id:  stream.alloc_zeros::<i32>(max_nodes)?,
            d_threshold:   stream.alloc_zeros::<f32>(max_nodes)?,
            d_left_child:  stream.alloc_zeros::<i32>(max_nodes)?,
            d_right_child: stream.alloc_zeros::<i32>(max_nodes)?,
            d_counts:      stream.alloc_zeros::<u32>(2 * n_classes)?,
        })
    }
}
// Pre-size max_nodes = 2^(max_depth+1) - 1. Never cudaMalloc per node.
```

---

### `crates/sylva-cuda/src/cuda_backend/philox_device.rs` (CUDA-C Philox + on-device KAT launch)

**Analog:** `crates/sylva-core/src/rng/mod.rs` (lines 1–85) — the host Philox to translate to CUDA C; `crates/sylva-core/src/rng/kat.rs` (lines 1–33) — the three KAT vectors to reproduce on-device.

**CUDA-C source string** (from RESEARCH.md Code Examples, reproduce exactly these constants):
```rust
// Source: crates/sylva-core/src/rng/mod.rs constants (lines 15–24 for M0/M1/W0/W1)
// These constant values are the match target:
//   PHILOX_M0 = 0xD2511F53, PHILOX_M1 = 0xCD9E8D57
//   PHILOX_W0 = 0x9E3779B9, PHILOX_W1 = 0xBB67AE85
//   u32_to_unit_f32: (x >> 8) * (1.0f / 16777216.0f)  [line 67]
//   pack_counter order: [tree, node, feature, draw]     [line 74-76]
//   key split: [seed as u32, (seed >> 32) as u32]       [line 82-83]
pub const PHILOX_DEVICE_SRC: &str = r#"
// [Philox CUDA-C + u32_to_unit_f32 + philox_uniform — see RESEARCH.md Code Examples]
// Must reproduce KAT vectors frozen in crates/sylva-core/src/rng/kat.rs
"#;

pub const PHILOX_KAT_CHECK_SRC: &str = r#"
// KAT-check kernel: for each of the 3 KAT vectors, run philox4x32_10(ctr,key)
// and write result to d_out[i*4..(i+1)*4]. Host checks against frozen vectors.
extern "C" __global__ void philox_kat_check(unsigned* d_out) {
    // KAT_ZERO: ctr={0,0,0,0} key={0,0} → expected {0x6627e8d5,0xe169c58d,...}
    // KAT_ONES: ctr={0xFFFFFFFF×4} key={0xFFFFFFFF×2}
    // KAT_MIXED: ctr/key = π digits
    // [see kat.rs lines 14-32 for the exact hex values]
}
"#;
```

**KAT verify function pattern:**
```rust
// Host verifier: compile KAT kernel → launch → D2H → compare to kat.rs constants
pub fn verify_device_philox_kat(stream: &CudaStream, module: &CudaModule) -> Result<(), CudaError> {
    let func = module.load_function("philox_kat_check")?;
    let mut d_out = stream.alloc_zeros::<u32>(12)?; // 3 vectors × 4 words
    let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut d_out);
    // SAFETY: d_out has 12 u32 elements; kernel writes exactly 12 words.
    unsafe { builder.launch(cfg)?; }
    stream.synchronize()?;
    let got = stream.clone_dtoh(&d_out)?;
    // Compare against kat.rs constants (KAT_ZERO.2, KAT_ONES.2, KAT_MIXED.2)
    let expected = [/* KAT_ZERO.2 */ 0x6627e8d5u32, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8,
                    /* KAT_ONES.2 */ 0x408f276d,    0x41c83b0e, 0xa20bc7c6, 0x6d5451fd,
                    /* KAT_MIXED.2*/ 0xd16cfe09,    0x94fdcceb, 0x5001e420, 0x2412_6ea1];
    if got != expected {
        return Err(CudaError::InvalidInput(format!("on-device Philox KAT mismatch: {:?}", got)));
    }
    Ok(())
}
```

---

### `crates/sylva-cuda/src/cuda_backend/histogram.rs` (partition-count kernel launch wrapper)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 124–168 (`run_histogram`) — the privatized histogram launch; adapt from 256-bin static shared to `2*n_classes` dynamic shared.

**Key differences from `run_histogram` analog:**
```rust
// Phase 1 used static sh[256] → shared_mem_bytes: 0
// Phase 4 needs dynamic shared for variable n_classes:
let cfg = LaunchConfig {
    grid_dim:  (1, 1, 1),              // one block per node (Phase 4)
    block_dim: (ET_BLOCK_SIZE, 1, 1),
    // Dynamic shared: 2*n_classes unsigned ints (left counts + right counts)
    shared_mem_bytes: (2 * n_classes * std::mem::size_of::<u32>()) as u32,
};
// The kernel declares: extern __shared__ unsigned int sh[];
// (NOT __shared__ unsigned int sh[256]; — that was the spike's static version)
```

**Validation pattern** (mirror `run_histogram` lines 128–133):
```rust
// V5 boundary check before any device call:
if n_classes == 0 || n_classes > 65536 {
    return Err(CudaError::InvalidInput(format!("n_classes {n_classes} out of range")));
}
if end <= start { return Err(CudaError::InvalidInput("empty node range".into())); }
```

---

### `crates/sylva-cuda/src/cuda_backend/split_score.rs` (host-side impurity from device counts)

**Analog:** `crates/sylva-core/src/cpu/criterion.rs` — the EXACT f32 op sequence to reproduce.

**This is host-only code** (Open Question 3 resolution: score host-side from device integer counts — zero device-float-parity risk):
```rust
// Source: crates/sylva-core/src/cpu/criterion.rs (the entire file — 108 lines)
// Import and call directly — do NOT reimplement these functions:
use sylva_core::cpu::criterion::{gini, entropy, mse, proxy_improvement};

// From integer counts returned by the histogram kernel:
pub fn score_from_counts(
    left_counts:  &[u64],   // [n_classes] from device histogram (left side)
    right_counts: &[u64],   // [n_classes] from device histogram (right side)
    criterion:    Criterion,
    task:         Task,
) -> (f32, u64, u64) {
    // parent counts = left + right element-wise (integer — order-free)
    let parent_counts: Vec<u64> = left_counts.iter().zip(right_counts)
        .map(|(l, r)| l + r).collect();
    let n_left  = left_counts.iter().sum::<u64>();
    let n_right = right_counts.iter().sum::<u64>();
    let n_total = n_left + n_right;
    // Call the EXACT same functions criterion.rs calls — same f32 op order:
    let parent_imp = gini(&parent_counts, n_total);  // or entropy / mse
    let left_imp   = gini(left_counts, n_left);
    let right_imp  = gini(right_counts, n_right);
    let improvement = proxy_improvement(parent_imp, left_imp, right_imp, n_left, n_right);
    (improvement, n_left, n_right)
}

// Tie-break: lowest (feature_id, threshold.to_bits()) — EXACT match to split_et.rs lines 185-191:
// improvement > b.improvement
//     || (improvement == b.improvement
//         && (feat, threshold.to_bits()) < (b.feature_id, b.threshold.to_bits()))
```

---

### `crates/sylva-cuda/src/cuda_backend/partition.rs` (scatter-partition kernel launch wrapper)

**Analog:** `crates/sylva-cuda/src/nvrtc_launch.rs` `run_histogram` launch shape (lines 124–168).

**Kernel contract** (new kernel, analogous shape to the count kernel):
```rust
// The scatter-partition kernel rewrites d_row_index[start..end] in-place:
// rows with x[row, best_feat] <= best_thr go to [start..mid], others to [mid..end].
// Returns the split midpoint `mid` to the scheduler (D2H of a single u32).
// V5 boundary: validate best_feat < n_features, thr is finite, start < end.
// SAFETY comment identical structure to run_histogram's (line 159–164 of nvrtc_launch.rs).
```

---

### `crates/sylva-cuda/src/cuda_backend/assemble.rs` (D2H node arrays → ForestIR)

**Analog:** `crates/sylva-core/src/cpu/fit.rs` `TreeFragment` struct (lines 122–161) + `assemble_forest` function pattern.

**Pattern — mirrors `TreeFragment` field names exactly:**
```rust
// Source: cpu/fit.rs lines 122-161 (TreeFragment) — these field names MUST match ForestIR field names
// D2H the same named arrays, then call the same ForestIR constructor:
use sylva_core::ir::{ForestIR, LEAF_FEATURE, NO_CHILD};

pub fn assemble_from_device(
    // D2H'd arrays (same names as TreeFragment / ForestIR):
    feature_id:         Vec<i32>,   // LEAF_FEATURE (-1) for leaves
    threshold:          Vec<f32>,
    left_child:         Vec<i32>,   // NO_CHILD (-1) for leaves
    right_child:        Vec<i32>,
    default_child:      Vec<i32>,
    is_leaf:            Vec<bool>,
    node_sample_count:  Vec<u64>,
    node_weighted_count: Vec<f32>,
    impurity:           Vec<f32>,
    leaf_value:         Vec<f32>,
    leaf_proba:         Vec<f32>,
    leaf_offset:        Vec<i32>,
    // Forest metadata:
    n_features: usize, task: Task, criterion: Criterion, seed: u64,
) -> ForestIR {
    // Identical structure to assemble_forest in cpu/fit.rs —
    // single tree → tree_offsets = [0, node_count], tree_root = [0]
    ForestIR {
        feature_id, threshold, left_child, right_child,
        default_child, is_leaf, node_sample_count, node_weighted_count,
        impurity, leaf_value, leaf_proba, leaf_offset,
        tree_offsets: vec![0, feature_id.len()], // will be pre-computed
        tree_root: vec![0],
        n_trees: 1,
        n_features,
        task,
        criterion,
        seed,
    }
}
```

---

### `crates/sylva-cuda/tests/parity_cpu_gpu.rs` (THE bit-exact gate — GPU-01/GPU-02)

**Analog:** `crates/sylva-core/tests/determinism.rs` lines 1–117 (byte-identical JSON gate pattern).

**Fixture pattern** (copy `clf_data`, `reg_data`, `et_clf_cfg` helpers from `determinism.rs` lines 32–77):
```rust
// Source: crates/sylva-core/tests/determinism.rs lines 32-77
// Copy these helpers verbatim (or import from a shared test-common module):
fn clf_data() -> (Array2<f32>, Array1<f32>) { /* same as determinism.rs:32-41 */ }
fn et_clf_cfg(seed: u64) -> TrainConfig { /* same as determinism.rs:55-67 */ }
```

**Bit-exact gate pattern** (from `determinism.rs` lines 101–117 + RESEARCH.md Code Examples):
```rust
// Source: crates/sylva-core/tests/determinism.rs lines 101-117 (assert_same_seed_byte_identical)
// + RESEARCH.md Code Examples "The bit-exact gate"
#[test]
fn gpu_single_tree_matches_cpu_oracle_bit_exact_clf() {
    let (x, y) = clf_data();
    let cfg = TrainConfig { n_estimators: 1, algo: Algo::ExtraTrees, seed: 42,
                            bootstrap: false, /* ... */ };
    let cpu_ir = CpuBackend.fit(x.view(), y.view(), &cfg).unwrap();
    let gpu_ir = CudaBackend::new().unwrap().fit(x.view(), y.view(), &cfg).unwrap();

    // byte-compare via serde_json — NOT approx / allclose (RESEARCH "Don't Hand-Roll")
    let s_cpu = serde_json::to_string(&cpu_ir).expect("cpu serialize");
    let s_gpu = serde_json::to_string(&gpu_ir).expect("gpu serialize");
    assert_eq!(s_cpu, s_gpu,
        "GPU ExtraTree must equal CPU oracle byte-for-byte (seed 42, clf)");
}
// Repeat for reg (Criterion::Mse, Task::Regression) — must also pass.
```

---

### `crates/sylva-cuda/tests/philox_device_kat.rs` (on-device Philox KAT — ENG-06)

**Analog:** `crates/sylva-core/src/rng/mod.rs` lines 87–112 (`philox_matches_kat_vectors` test).

**Pattern:**
```rust
// Source: rng/mod.rs lines 87-112 — same KAT vectors, now verified on-device
// The three vectors from kat.rs:
use sylva_core::rng::kat::{KAT_ZERO, KAT_ONES, KAT_MIXED};

#[test]
fn device_philox_reproduces_kat_zero() {
    // Launch the philox_kat_check kernel (philox_device.rs::verify_device_philox_kat)
    // and assert the returned u32[12] matches KAT_ZERO.2 / KAT_ONES.2 / KAT_MIXED.2.
    // Uses to_bits() equality (not approx) — these are exact u32 comparisons.
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    // ... compile PHILOX_KAT_CHECK_SRC, launch, D2H, assert_eq!
    assert_eq!(&got[0..4],   &KAT_ZERO.2,  "all-zero KAT (on-device)");
    assert_eq!(&got[4..8],   &KAT_ONES.2,  "all-ones KAT (on-device)");
    assert_eq!(&got[8..12],  &KAT_MIXED.2, "mixed KAT (on-device)");
}
```

---

### `crates/sylva-cuda/tests/sanitizer_et_kernels.rs` (standalone sanitizer targets)

**Analog:** `crates/sylva-cuda/tests/sanitizer_histogram.rs` (lines 1–55) — exact structural match.

**Pattern** (copy doc-comment shape + const + single-launch test per kernel):
```rust
// Source: crates/sylva-cuda/tests/sanitizer_histogram.rs — copy this structure
// for each ET kernel (count_node, scatter_partition, philox_kat):

//! Standalone `compute-sanitizer` target for the ET partition-count kernel.
//! ...doc mirrors sanitizer_histogram.rs lines 1-24 with updated names...

const SANITIZER_N: usize = 100_000;  // same reasoning as sanitizer_histogram.rs:29

#[test]
fn sanitizer_et_count_node_single_launch() {
    // One fixed-input launch, assert correctness vs CPU reference.
    // The binary structure lets compute-sanitizer wrap it with:
    //   --tool racecheck <exe>  --tool memcheck <exe>
    //   --tool synccheck <exe>  --tool initcheck <exe>
    // All four tools must report ERROR SUMMARY: 0 errors (Phase-1 standard).
}
```

---

## Shared Patterns (apply to ALL new files)

### No-Silent-Fallback (every cudarc call)
**Source:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 9–16 (module doc) + lines 34–50 (`CudaError`)
```rust
// Copy to every new module's doc: "No .unwrap()/.expect() on device calls"
// Every cudarc call: someCall()?;  — propagates to CudaError
// unsafe only at builder.launch(cfg)?; with // SAFETY: comment
// stream.synchronize()?; immediately after launch (surfaces launch/exec errors)
```

### Boundary Validation Pattern (V5)
**Source:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 128–133 (`run_histogram` validation)
```rust
// Before any device call, validate host inputs → CudaError::InvalidInput
// Mirror the pattern: if condition { return Err(CudaError::InvalidInput(format!("..."))); }
```

### Float-Parity Compile Flags (apply to ALL new NVRTC compilations)
**Source:** `crates/sylva-cuda/src/nvrtc_launch.rs` lines 56–63 (`compile_for_sm89`)
```rust
// REQUIRED addition over Phase 1:
options: vec!["-lineinfo".to_string(), "-fmad=false".to_string()],
// NEVER: "--use_fast_math" (collapses mul+add → FMA → diverges from CPU)
```

### CUDA-C Kernel Guard Pattern (every kernel)
**Source:** `crates/sylva-cuda/src/kernels.rs` lines 22–27 (`vector_add`), `histogram_privatized` lines 47–64
```rust
// Every kernel thread: if (i < n) { ... }   ← OOB guard
// Privatized histogram shape: zero-init sh → __syncthreads() → atomicAdd(&sh[...], 1u)
//   → __syncthreads() → flush to global (NO float atomicAdd — integer only)
// Dynamic shared: extern __shared__ unsigned int sh[];  (not sh[256] — that's static)
```

### Byte-Exact Comparison (tests only — never approx for the gate)
**Source:** `crates/sylva-core/tests/determinism.rs` lines 101–117
```rust
// Gate comparison: assert_eq!(s_cpu, s_gpu, "...");  // exact string equality
// Never: assert_abs_diff_eq! / allclose / approx for the ForestIR parity gate
// approx is ONLY for the sklearn distributional check (informational, not the gate)
```

### ForestIR Field Names (assembler must match exactly)
**Source:** `crates/sylva-core/src/ir.rs` lines 23–65
```
// All field names are fixed — the ForestIR derives PartialEq + Serialize:
// feature_id, threshold, left_child, right_child, default_child, is_leaf,
// node_sample_count, node_weighted_count, impurity,
// leaf_value, leaf_proba, leaf_offset, tree_offsets, tree_root
// LEAF_FEATURE = -1 (ir.rs line 16), NO_CHILD = -1 (ir.rs line 18)
```

### Default-Child Rule (must match CPU oracle)
**Source:** `crates/sylva-core/src/cpu/split_et.rs` line 182
```rust
// default_left = n_left >= n_right  (tie → left)
// In ForestIR: default_child[node] = left_child[node] if default_left else right_child[node]
```

---

## No Analog Found

None. All files have direct analogs in the existing codebase. The four CUDA-C kernels
(`et_count_node`, `philox_kat_check`, `et_scatter_partition`, `et_leaf_stats`) are
**original code** authored from the level-synchronous histogram-tree algorithm, but their
**Rust launch-wrapper shape** maps exactly to the Phase-1 `run_histogram` analog, and their
**CUDA-C structure** maps exactly to `HISTOGRAM_PRIVATIZED_SRC` and the RESEARCH.md
Philox/kernel code examples.

---

## Analog Search Scope

**Directories searched:** `crates/sylva-cuda/src/`, `crates/sylva-core/src/`, `crates/sylva-core/tests/`, `crates/sylva-cuda/tests/`
**Files scanned:** 21 source files (all Rust files in the codebase)
**Pattern extraction date:** 2026-06-27
