//! NVRTC compile -> module load -> launch host code (THROWAWAY spike logic).
//!
//! This is the cudarc 0.19.8 host side of the walking-skeleton GPU link: it
//! takes a hand-written CUDA-C string (`crate::kernels`), compiles it to PTX at
//! runtime with NVRTC for `sm_89` (the local RTX 4060 Ti), loads the module, and
//! launches the kernel on device 0. No `nvcc`/`cc`/`build.rs` is involved.
//!
//! No-silent-fallback contract: every cudarc call returns a `Result`, and every
//! one is propagated via `?` into [`CudaError`]. There is NO `.unwrap()` /
//! `.expect()` on a device call anywhere in this module — a failed compile or
//! launch surfaces as an error to the caller, never a silent degrade.
//!
//! `unsafe` is confined to the single `launch` call (the FFI boundary), with a
//! `// SAFETY:` justification; `compute-sanitizer memcheck` (Plan 02 Task 3) is
//! the memory-safety control proving no OOB/UB across that boundary.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::kernels::{BIN_COUNT, HISTOGRAM_PRIVATIZED_SRC, VECTOR_ADD_SRC};

/// NVRTC target architecture for the local RTX 4060 Ti (compute capability 8.9,
/// Ada). Named constant so the literal `sm_89` lives in exactly one place
/// (D-03). `&'static str` to match `CompileOptions.arch`.
const ARCH_SM_89: &str = "sm_89";

/// Histogram launch block size (threads per block). One thread reads one
/// element; the block owns one private `__shared__ unsigned int sh[256]`.
const HISTOGRAM_BLOCK: u32 = 256;

/// Typed crate error. Maps every cudarc failure mode (NVRTC compile, driver
/// launch/alloc/copy) plus the Rust-boundary input-validation failures into one
/// enum, so callers get a `Result` and no device call is ever unwrapped.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// NVRTC failed to compile a CUDA-C source string to PTX.
    #[error("NVRTC compile failed: {0}")]
    Compile(#[from] cudarc::nvrtc::CompileError),

    /// A cudarc driver call failed (context, module load, alloc, H2D/D2H copy,
    /// kernel launch, or stream sync).
    #[error("CUDA driver call failed: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// A host input failed boundary validation before any device launch
    /// (V5 control / T-01-03): an out-of-range value would index device memory
    /// out of bounds, so it is rejected on the host.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Compile a CUDA-C source string to PTX for `sm_89` with `-lineinfo`.
///
/// `-lineinfo` (Pitfall 4) makes any `compute-sanitizer` hazard carry a kernel
/// source-line attribution — required for TOOL-02's debuggability proof.
fn compile_for_sm89(src: &str) -> Result<cudarc::nvrtc::Ptx, CudaError> {
    let opts = CompileOptions {
        arch: Some(ARCH_SM_89),
        options: vec!["-lineinfo".to_string()],
        ..Default::default()
    };
    Ok(compile_ptx_with_opts(src, opts)?)
}

/// `out[i] = a[i] + b[i]` on device 0 via NVRTC-compiled CUDA-C (TOOL-01).
///
/// Compiles [`VECTOR_ADD_SRC`] for `sm_89`, launches on the local GPU, and
/// returns the device result. `a` and `b` must have equal length; a mismatch is
/// a boundary-validation error (never a silent truncation).
///
/// Reachable from `lib.rs` so Plan 03's PyO3 entrypoint can call it.
pub fn run_vector_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
    if a.len() != b.len() {
        return Err(CudaError::InvalidInput(format!(
            "vector_add length mismatch: a.len()={} b.len()={}",
            a.len(),
            b.len()
        )));
    }
    let n = a.len();

    let ptx = compile_for_sm89(VECTOR_ADD_SRC)?;
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("vector_add")?;
    let stream = ctx.default_stream();

    let d_a = stream.clone_htod(a)?;
    let d_b = stream.clone_htod(b)?;
    let mut d_out = stream.alloc_zeros::<f32>(n)?;

    // `size_t n` on the device == host `u64` here (LP64 device ABI).
    let n_arg: u64 = n as u64;
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&d_a).arg(&d_b).arg(&mut d_out).arg(&n_arg);
    // SAFETY: kernel args match the `vector_add(const float*, const float*,
    // float*, size_t)` signature (two read slices of length n, one write slice
    // of length n allocated above, and the scalar n); the kernel guards every
    // index with `if (i < n)`, so no thread reads or writes out of bounds. This
    // is the only `unsafe` in the launch path; compute-sanitizer memcheck is the
    // proof it is sound.
    unsafe {
        builder.launch(cfg)?;
    }
    // Surface any launch/exec error before reading back (no silent fallback).
    stream.synchronize()?;

    Ok(stream.clone_dtoh(&d_out)?)
}

/// 256-bin shared-memory privatized histogram of `bins` on device 0
/// (the representative hard primitive; `compute-sanitizer` target for TOOL-02).
///
/// Compiles [`HISTOGRAM_PRIVATIZED_SRC`] for `sm_89`, launches with
/// `block = HISTOGRAM_BLOCK`, `grid = ceil(n / HISTOGRAM_BLOCK)`, and returns the
/// global histogram (`BIN_COUNT` counts). Every `bins[i]` is validated `<
/// BIN_COUNT` on the host before launch (V5 control / T-01-03), in addition to
/// the kernel's `if (i < n)` guard.
///
/// Reachable from `lib.rs` for Plan 03 and reused by the standalone sanitizer
/// target (`tests/sanitizer_histogram.rs`).
pub fn run_histogram(bins: &[u8]) -> Result<Vec<u32>, CudaError> {
    // V5 boundary validation: a bin index >= BIN_COUNT would index `sh[256]`
    // out of bounds on the device. (For u8 inputs with BIN_COUNT == 256 this is
    // always satisfied, but the check is explicit defence-in-depth and guards
    // against a future narrower BIN_COUNT.)
    if let Some(&bad) = bins.iter().find(|&&v| (v as usize) >= BIN_COUNT) {
        return Err(CudaError::InvalidInput(format!(
            "bin index {bad} out of range [0, {BIN_COUNT})"
        )));
    }
    let n = bins.len();

    let ptx = compile_for_sm89(HISTOGRAM_PRIVATIZED_SRC)?;
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("histogram_privatized")?;
    let stream = ctx.default_stream();

    let d_bins = stream.clone_htod(bins)?;
    let mut d_hist = stream.alloc_zeros::<u32>(BIN_COUNT)?;

    // `int n` on the device; n fits i32 for spike-scale inputs.
    let n_arg: i32 = n as i32;
    let num_blocks = (n as u32).div_ceil(HISTOGRAM_BLOCK).max(1);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (HISTOGRAM_BLOCK, 1, 1),
        shared_mem_bytes: 0, // sh[256] is statically sized in the kernel.
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&d_bins).arg(&mut d_hist).arg(&n_arg);
    // SAFETY: args match `histogram_privatized(const unsigned char*, unsigned
    // int*, int)`: a read slice of length n, a write slice of length BIN_COUNT
    // (allocated above and zeroed), and the scalar n. The kernel guards the
    // element read with `if (i < n)`, zero-inits `sh` before use, and
    // `__syncthreads()` before the flush, so there is no OOB access or
    // uninitialized read. compute-sanitizer (Task 3) proves this.
    unsafe {
        builder.launch(cfg)?;
    }
    stream.synchronize()?;

    Ok(stream.clone_dtoh(&d_hist)?)
}
