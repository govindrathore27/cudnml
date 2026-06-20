//! Toolchain smoke test (Wave 0).
//!
//! Proves the crate compiles and links end-to-end through the MSVC toolchain
//! BEFORE any GPU launch is attempted. This is the first link of the
//! walking-skeleton chain (workspace builds -> crate links). The real GPU
//! launch proof is `nvrtc_launch_vector_add`, an ignored placeholder that
//! Plan 02 (Wave 1) fills with the actual NVRTC kernel launch + correctness
//! assertion.

/// The crate links and its `version()` probe returns a non-empty string.
///
/// A non-empty return is the end-to-end signal that the Rust -> cdylib/rlib
/// MSVC link path works (this test binary links `sylva-cuda` as an rlib).
#[test]
fn smoke_crate_links() {
    let v = sylva_cuda::version();
    assert!(
        !v.is_empty(),
        "version() probe returned empty -- crate build/link is broken"
    );
}

/// Fixed problem size for the TOOL-01 launch proof: 1e7 f32 elements (D-01).
/// Named const so the magnitude is not duplicated as an inline magic number.
const VECTOR_ADD_N: usize = 10_000_000;

/// TOOL-01 launch proof: a hand-written CUDA-C `vector_add` compiled by
/// cudarc 0.19.8 + NVRTC (arch `sm_89`) launches on the local RTX 4060 Ti and
/// returns `a[i] + b[i]` bit-exactly over a fixed 1e7-element f32 array.
///
/// Float addition of these fixed inputs is exact (no rounding), so the assertion
/// is exact equality, not a tolerance. A compile or launch failure surfaces as a
/// `CudaError` (propagated by `?`), never a silent pass.
#[test]
fn nvrtc_launch_vector_add() {
    // Deterministic, reproducible inputs (no RNG): a[i] = i mod 1024 (exact in
    // f32), b[i] = 2 * (i mod 512) (exact in f32). Their sum is exact in f32.
    let a: Vec<f32> = (0..VECTOR_ADD_N).map(|i| (i % 1024) as f32).collect();
    let b: Vec<f32> = (0..VECTOR_ADD_N).map(|i| ((i % 512) * 2) as f32).collect();

    let out = sylva_cuda::run_vector_add(&a, &b)
        .expect("NVRTC compile + launch of vector_add on sm_89 must succeed");

    assert_eq!(
        out.len(),
        VECTOR_ADD_N,
        "result length must match input length"
    );
    // Exact equality across all 1e7 elements (max-abs-error == 0).
    for i in 0..VECTOR_ADD_N {
        assert_eq!(
            out[i],
            a[i] + b[i],
            "vector_add mismatch at index {i}: got {}, expected {}",
            out[i],
            a[i] + b[i]
        );
    }
}

/// Fixed problem size for the histogram correctness proof. Chosen to span many
/// blocks (n / 256) so the two-level reduction and the privatized shared-mem
/// path are genuinely exercised.
const HISTOGRAM_N: usize = 1_000_000;

/// Task 2: the representative 256-bin shared-memory privatized histogram
/// compiled by NVRTC launches on the GPU and matches a trivial CPU reference
/// count exactly (integer equality across all 256 bins).
#[test]
fn histogram_privatized_matches_cpu() {
    // Deterministic bin indices in [0, 256): a simple mixing of the index so the
    // distribution is non-uniform (exercises uneven bin contention) yet exactly
    // reproducible without an RNG.
    let bins: Vec<u8> = (0..HISTOGRAM_N)
        .map(|i| ((i * 31 + 7) % 256) as u8)
        .collect();

    // CPU reference: a trivial count loop.
    let mut expected = vec![0u32; 256];
    for &b in &bins {
        expected[b as usize] += 1;
    }

    let got = sylva_cuda::run_histogram(&bins)
        .expect("NVRTC compile + launch of histogram_privatized on sm_89 must succeed");

    assert_eq!(got.len(), 256, "histogram must have 256 bins");
    assert_eq!(
        got, expected,
        "GPU histogram must equal the CPU reference exactly"
    );
}
