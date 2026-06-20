//! Fairness-encoded kernel-launch / vector-op microbench (SC-6, THROWAWAY spike).
//!
//! This is a **feasibility sanity check, NOT an algorithm speed claim**
//! (PITFALLS Pitfalls 1/13; ROADMAP Comparative Baseline Study). It measures two
//! things for the cudarc + NVRTC path on the local RTX 4060 Ti, so they can be
//! compared against the CuPy (or raw-CUDA-C) baseline in `scripts/cupy_baseline.py`:
//!
//!   1. **Per-launch overhead** (us/launch): a tight loop of near-empty
//!      `vector_add` launches on a 1-element buffer, timed with CUDA events,
//!      divided by the launch count. Isolates the host->device launch cost.
//!   2. **Throughput** (GB/s): `vector_add` over a FIXED 1e7-element f32 vector,
//!      bytes = 3 * 4 * N (two reads + one write), GB/s = bytes / median_time.
//!
//! Fairness rules baked in (the same ones the CuPy baseline obeys):
//!   - WARMUP: 50 discarded launches before any timing (excludes NVRTC JIT /
//!     first-launch / context-spin-up costs).
//!   - MEDIAN of >= 10 timed runs (NOT mean — robust to scheduler jitter).
//!   - Device-event timing (CUDA events), not host `Instant` (which misses the
//!     async launch/exec overlap).
//!   - CORRECTNESS FIRST: `max_abs_err(out, a+b) == 0` is asserted BEFORE any
//!     timing number is reported. A wrong kernel is a hard failure, not a slow one.
//!   - Same GPU / driver / CUDA as the baseline (recorded in VERSIONS.md).
//!
//! Run modes:
//!   - `cargo bench -p sylva-cuda --bench microbench`            -> full report
//!   - `cargo bench -p sylva-cuda --bench microbench -- --test`  -> smoke mode:
//!     warmup + correctness assert only, no timing loop (fast CI signal; this is
//!     the command in the plan's `<automated>` verify gate).
//!
//! Uses the `wheel`/`cuda-static` cudarc feature like the rest of the crate; run
//! it under the default `cuda-static` feature (toolkit-linked) for benchmarking.

use cudarc::driver::{sys, CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use sylva_cuda::kernels::VECTOR_ADD_SRC;

/// NVRTC arch for the local RTX 4060 Ti (compute capability 8.9). Named constant
/// so the `sm_89` literal lives in one place (mirrors `nvrtc_launch.rs`).
const ARCH_SM_89: &str = "sm_89";

/// Discarded warmup launches before any timing (excludes JIT / first-launch).
const WARMUP_LAUNCHES: usize = 50;

/// Number of timed runs; we report the MEDIAN (not the mean) of these.
const TIMED_RUNS: usize = 10;

/// Launches per per-launch-overhead measurement run. The overhead per launch is
/// `total_time / LAUNCHES_PER_OVERHEAD_RUN`, averaging out single-launch noise.
const LAUNCHES_PER_OVERHEAD_RUN: usize = 1_000;

/// Fixed throughput vector length (1e7 f32 elements) — the Comparative Baseline
/// Study dataset shape. Bytes moved per launch = 3 * 4 * N (two reads + one write).
const THROUGHPUT_N: usize = 10_000_000;

/// Bytes touched per `vector_add` over N f32 elements: a + b read, out written.
const BYTES_PER_F32: usize = 4;
const ARRAYS_TOUCHED: usize = 3;

/// A compiled-once, buffers-allocated-once `vector_add` launcher. Compiling PTX
/// and allocating device buffers are one-time setup costs that MUST be excluded
/// from per-launch timing — so they happen here, in the constructor, and the
/// timed loop only calls [`Self::launch`].
struct VectorAddBench {
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    func: cudarc::driver::CudaFunction,
}

impl VectorAddBench {
    /// Compile `vector_add` for sm_89 once and bind a stream. Returns an error
    /// (never a panic) on any NVRTC/driver failure — no silent fallback.
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let opts = CompileOptions {
            arch: Some(ARCH_SM_89),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(VECTOR_ADD_SRC, opts)?;
        let ctx = CudaContext::new(0)?;
        let module = ctx.load_module(ptx)?;
        let func = module.load_function("vector_add")?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, func })
    }

    /// Launch `vector_add` over `n` elements on the given device buffers. The
    /// single `unsafe` is the FFI launch boundary (args match the kernel
    /// signature; the kernel guards every index with `if (i < n)`).
    fn launch(
        &self,
        d_a: &cudarc::driver::CudaSlice<f32>,
        d_b: &cudarc::driver::CudaSlice<f32>,
        d_out: &mut cudarc::driver::CudaSlice<f32>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_arg: u64 = n as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut builder = self.stream.launch_builder(&self.func);
        builder.arg(d_a).arg(d_b).arg(d_out).arg(&n_arg);
        // SAFETY: args match `vector_add(const float*, const float*, float*,
        // size_t)` — two read slices and one write slice of length n, plus the
        // scalar n; the kernel bounds-guards with `if (i < n)`.
        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }

    /// Median elapsed-ms helper: returns the median of a slice of timings.
    fn median(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = samples.len() / 2;
        if samples.len() % 2 == 0 {
            (samples[mid - 1] + samples[mid]) / 2.0
        } else {
            samples[mid]
        }
    }
}

/// Assert `out == a + b` exactly (vector_add is exact for finite f32 sums of the
/// fixed test pattern) BEFORE any timing is reported. Correctness-first.
fn assert_vector_add_correct() -> Result<(), Box<dyn std::error::Error>> {
    let bench = VectorAddBench::new()?;
    let n = 4096usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();

    let d_a = bench.stream.clone_htod(&a)?;
    let d_b = bench.stream.clone_htod(&b)?;
    let mut d_out = bench.stream.alloc_zeros::<f32>(n)?;
    bench.launch(&d_a, &d_b, &mut d_out, n)?;
    bench.stream.synchronize()?;
    let out = bench.stream.clone_dtoh(&d_out)?;

    let max_abs_err = out
        .iter()
        .zip(a.iter().zip(b.iter()))
        .map(|(o, (x, y))| (o - (x + y)).abs())
        .fold(0.0f32, f32::max);
    if max_abs_err != 0.0 {
        return Err(format!("vector_add incorrect: max_abs_err={max_abs_err}").into());
    }
    println!("correctness: max_abs_err(out, a+b) = 0  [asserted before timing]");
    Ok(())
}

/// Measure per-launch overhead (us/launch): warmup, then TIMED_RUNS runs of a
/// tight `LAUNCHES_PER_OVERHEAD_RUN`-launch loop over a 1-element buffer, timed
/// with CUDA events; report the median run's per-launch cost.
fn measure_launch_overhead_us(
    bench: &VectorAddBench,
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = 1usize; // near-empty launch -> isolates launch overhead, not compute.
    let d_a = bench.stream.clone_htod(&[1.0f32])?;
    let d_b = bench.stream.clone_htod(&[2.0f32])?;
    let mut d_out = bench.stream.alloc_zeros::<f32>(n)?;

    for _ in 0..WARMUP_LAUNCHES {
        bench.launch(&d_a, &d_b, &mut d_out, n)?;
    }
    bench.stream.synchronize()?;

    let mut per_launch_us = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let start = bench
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..LAUNCHES_PER_OVERHEAD_RUN {
            bench.launch(&d_a, &d_b, &mut d_out, n)?;
        }
        let end = bench
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let elapsed_ms = start.elapsed_ms(&end)? as f64;
        // ms for the whole loop -> us per single launch.
        let us = (elapsed_ms * 1000.0) / LAUNCHES_PER_OVERHEAD_RUN as f64;
        per_launch_us.push(us);
    }
    Ok(VectorAddBench::median(per_launch_us))
}

/// Measure throughput (GB/s) of `vector_add` over THROUGHPUT_N f32 elements:
/// warmup, then TIMED_RUNS event-timed launches; report median GB/s.
fn measure_throughput_gbps(
    bench: &VectorAddBench,
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = THROUGHPUT_N;
    let a: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i + 1) % 1000) as f32).collect();
    let d_a = bench.stream.clone_htod(&a)?;
    let d_b = bench.stream.clone_htod(&b)?;
    let mut d_out = bench.stream.alloc_zeros::<f32>(n)?;

    for _ in 0..WARMUP_LAUNCHES {
        bench.launch(&d_a, &d_b, &mut d_out, n)?;
    }
    bench.stream.synchronize()?;

    let bytes = (ARRAYS_TOUCHED * BYTES_PER_F32 * n) as f64;
    let mut gbps = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let start = bench
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        bench.launch(&d_a, &d_b, &mut d_out, n)?;
        let end = bench
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let elapsed_s = (start.elapsed_ms(&end)? as f64) / 1000.0;
        gbps.push(bytes / elapsed_s / 1e9);
    }
    Ok(VectorAddBench::median(gbps))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // --test (smoke) mode: warmup + correctness assert only, no timing loop.
    let smoke = std::env::args().any(|a| a == "--test");

    println!("== Sylva microbench (feasibility sanity check, NO algorithm speed claim) ==");
    assert_vector_add_correct()?;

    if smoke {
        // Exercise one warmup launch of each path so the smoke run touches the
        // overhead+throughput code paths without the full timing loop.
        let bench = VectorAddBench::new()?;
        let d_a = bench.stream.clone_htod(&[1.0f32])?;
        let d_b = bench.stream.clone_htod(&[2.0f32])?;
        let mut d_out = bench.stream.alloc_zeros::<f32>(1)?;
        bench.launch(&d_a, &d_b, &mut d_out, 1)?;
        bench.stream.synchronize()?;
        // Keep the context alive across the smoke launch.
        let _ = &bench.ctx;
        println!("smoke mode (--test): correctness asserted, launch path exercised; skipping timing.");
        return Ok(());
    }

    let bench = VectorAddBench::new()?;
    let overhead_us = measure_launch_overhead_us(&bench)?;
    let gbps = measure_throughput_gbps(&bench)?;

    println!(
        "warmup={WARMUP_LAUNCHES} launches; median of {TIMED_RUNS} runs; \
         overhead loop={LAUNCHES_PER_OVERHEAD_RUN} launches/run; N={THROUGHPUT_N}"
    );
    println!("launch_overhead_us_per_launch = {overhead_us:.4}");
    println!("throughput_gbps = {gbps:.2}");
    println!("(feasibility sanity check, no algorithm speed claim)");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("microbench FAILED: {e}");
        std::process::exit(1);
    }
}
