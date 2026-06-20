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

/// TOOL-01 launch proof -- ignored placeholder so the test NAME exists at
/// Wave 0 (closes the Nyquist coverage gap from 01-RESEARCH.md).
///
/// Plan 02 (Wave 1) replaces the body with a real NVRTC vector-add kernel
/// compiled by cudarc + NVRTC (arch `sm_89`), launched on the local
/// RTX 4060 Ti, asserting the result is numerically correct.
#[test]
#[ignore = "filled by Plan 02 nvrtc launch proof"]
fn nvrtc_launch_vector_add() {
    unimplemented!("Plan 02 (Wave 1) implements the NVRTC launch + correctness check");
}
