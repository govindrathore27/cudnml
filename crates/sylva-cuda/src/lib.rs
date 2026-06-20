//! sylva-cuda — Phase 1 toolchain-spike crate.
//!
//! This is the PyO3 `#[pymodule]` seam. In Phase 1 it exposes nothing beyond a
//! `version()` probe used by the toolchain smoke test to prove the MSVC link
//! path works end-to-end (Rust -> .pyd -> rlib link). Plan 03 fills this module
//! with the NVRTC kernel-launch entrypoint; no algorithm logic lives here.

use pyo3::prelude::*;

pub mod kernels;
pub mod nvrtc_launch;

// Re-export the GPU entrypoints so integration tests, the standalone sanitizer
// target, and Plan 03's PyO3 wrapper call them as `sylva_cuda::run_vector_add` /
// `sylva_cuda::run_histogram` without reaching into module paths.
pub use nvrtc_launch::{run_histogram, run_vector_add, CudaError};

/// Return the crate version, sourced from `CARGO_PKG_VERSION` at compile time.
///
/// No hardcoded version string (coding-style: no hardcoded values). A non-empty
/// return is the signal that the crate compiled and linked successfully.
///
/// `pub` so the `tests/toolchain_smoke.rs` integration test (a separate binary
/// linking this crate as an rlib) can call it to prove the MSVC link path.
#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `sylva_cuda` Python extension module.
///
/// The function name (`sylva_cuda`) must match `module-name` in
/// `[tool.maturin]` so the built `.pyd` imports as `sylva_cuda`.
#[pymodule]
fn sylva_cuda(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
