//! Hand-written CUDA-C kernel source strings (THROWAWAY spike logic).
//!
//! These are trivial, original Apache-2.0 kernels (no copied restrictively-
//! licensed source — see `.claude/CLAUDE.md`). They are compiled at runtime by
//! NVRTC (`crate::nvrtc_launch`), never by `nvcc`/`cc`/`build.rs`. Authoring the
//! hot-path primitive (the privatized histogram) in CUDA C is the entire reason
//! cudarc + NVRTC was chosen over a Rust->PTX path: it rides the mature CUDA
//! toolchain (NVRTC + compute-sanitizer + Nsight).
//!
//! Phase-1 scope is exactly two kernels — one elementwise op (the "is the
//! toolchain alive" signal) and one representative shared-memory privatized
//! histogram (the "are the hard primitives debuggable" signal). No Extra Trees /
//! Random Forest / sibling-subtraction / multi-feature SoA logic lives here.

/// `vector_add`: `out[i] = a[i] + b[i]` over `n` `float`s.
///
/// The `if (i < n)` bounds guard is the V5 input-validation control from
/// `01-RESEARCH.md` (Security Domain): an out-of-bounds GPU read is silent
/// corruption, so every thread guards its index. `n` is `size_t` (device
/// `usize`), passed from the host as a `u64`.
pub const VECTOR_ADD_SRC: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

/// `histogram_privatized`: 256-bin shared-memory privatized histogram with a
/// two-level integer-`atomicAdd` reduction (mirrors the real Phase-4 hot path
/// per `ARCHITECTURE.md`).
///
/// Shape (the minimum faithful slice of the Phase-4 `histogram.cu`):
/// - one private `__shared__ unsigned int sh[256]` per block (1 KiB),
/// - cooperative zero-init of `sh` + `__syncthreads()` (so `initcheck` sees
///   `sh` written before read, and `racecheck` has a clean privatized target),
/// - `if (i < n) atomicAdd(&sh[bins[i]], 1u)` — INTEGER atomic only (counts are
///   associative/deterministic; float atomics are banned per PITFALLS Pitfall
///   5, even in the spike, to model the deterministic path),
/// - `__syncthreads()` before the flush (so `synccheck` is satisfied),
/// - cooperative `atomicAdd(&global_hist[b], sh[b])` merge to global.
///
/// `bins[i]` is a `unsigned char`, so it is always in `[0, 256)` — the bin index
/// can never index `sh` out of bounds. The Rust boundary additionally validates
/// `bins[i] < BIN_COUNT` before launch (V5 control, defence in depth).
pub const HISTOGRAM_PRIVATIZED_SRC: &str = r#"
extern "C" __global__ void histogram_privatized(const unsigned char* bins, unsigned int* global_hist, int n) {
    __shared__ unsigned int sh[256];
    int t = threadIdx.x;
    for (int b = t; b < 256; b += blockDim.x) {
        sh[b] = 0u;
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + t;
    if (i < n) {
        atomicAdd(&sh[bins[i]], 1u);
    }
    __syncthreads();
    for (int b = t; b < 256; b += blockDim.x) {
        atomicAdd(&global_hist[b], sh[b]);
    }
}
"#;

/// Number of histogram bins (uint8 bin domain). Named constant — no magic
/// numbers at the launch site or in the kernel-shape derivation.
pub const BIN_COUNT: usize = 256;
