//! sylva-cuda — Phase 1 toolchain-spike crate.
//!
//! This is the PyO3 `#[pymodule]` seam. It exposes the throwaway spike
//! entrypoints — `version()` (the MSVC link probe) and the `py_run_vector_add` /
//! `py_run_histogram` thin wrappers that call the Rust NVRTC launch core from
//! Python — closing the final link of the walking-skeleton chain (Rust -> CUDA
//! via NVRTC -> Python via abi3). NO Extra Trees / Random Forest / SHAP /
//! estimator logic lives here: these wrappers only marshal a sequence across the
//! FFI boundary, validate it, and call the spike kernels.
//!
//! No-silent-fallback contract across the FFI: the Rust core returns a
//! `Result`; the wrapper maps every [`CudaError`] to a clean Python exception
//! (`PyValueError` for boundary/input errors, `PyRuntimeError` for device
//! failures). There is NO `.unwrap()`/`.expect()` across the boundary — a failed
//! compile or launch surfaces as a Python exception, never a silent degrade.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod kernels;
pub mod nvrtc_launch;

// Re-export the GPU entrypoints so integration tests, the standalone sanitizer
// target, and Plan 03's PyO3 wrapper call them as `sylva_cuda::run_vector_add` /
// `sylva_cuda::run_histogram` without reaching into module paths.
pub use nvrtc_launch::{run_histogram, run_vector_add, CudaError};

/// Map a [`CudaError`] to a clean Python exception (no silent fallback across the
/// FFI boundary). Boundary/input-validation failures become `ValueError`; device
/// (NVRTC compile / driver launch) failures become `RuntimeError`. Either way the
/// caller gets a typed Python exception, never a panic unwinding across FFI.
fn cuda_error_to_pyerr(err: CudaError) -> PyErr {
    match err {
        CudaError::InvalidInput(_) => PyValueError::new_err(err.to_string()),
        CudaError::Compile(_) | CudaError::Driver(_) => PyRuntimeError::new_err(err.to_string()),
    }
}

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

/// Python entrypoint for the spike vector-add kernel: `out[i] = a[i] + b[i]`.
///
/// Accepts two plain Python sequences of `f32` (numpy-free — the spike does not
/// pull rust-numpy yet, per D-03), validates equal length at the boundary, calls
/// the Rust [`run_vector_add`] (which compiles `vector_add` via NVRTC for sm_89
/// and launches on the local GPU), and returns the result as a Python list.
///
/// This is the entrypoint the clean-venv import smoke test (TOOL-03) calls to
/// prove the dynamic-loading wheel resolves the CUDA driver at runtime and the
/// full Rust -> CUDA -> Python path works from a fresh install.
///
/// Errors map to Python exceptions (no silent fallback): a length mismatch is a
/// `ValueError`; an NVRTC/driver failure is a `RuntimeError`.
#[pyfunction]
#[pyo3(name = "run_vector_add")]
fn py_run_vector_add(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    // Length validation also lives in the Rust core (CudaError::InvalidInput);
    // checking here keeps the boundary failure a ValueError with a clear message
    // before any device work, per the V5 input-validation control (T-01-06).
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "run_vector_add: length mismatch a.len()={} b.len()={}",
            a.len(),
            b.len()
        )));
    }
    run_vector_add(&a, &b).map_err(cuda_error_to_pyerr)
}

/// Python entrypoint for the spike privatized histogram kernel.
///
/// Accepts a Python sequence of `u8` bin indices, calls the Rust
/// [`run_histogram`] (256-bin shared-memory privatized histogram on the GPU),
/// and returns the `BIN_COUNT` counts as a Python list. Errors map to Python
/// exceptions (boundary -> `ValueError`, device -> `RuntimeError`).
///
/// Exposed so the wheel demonstrates the representative hard primitive is
/// reachable from Python too, but the smoke test only requires `run_vector_add`.
#[pyfunction]
#[pyo3(name = "run_histogram")]
fn py_run_histogram(bins: Vec<u8>) -> PyResult<Vec<u32>> {
    run_histogram(&bins).map_err(cuda_error_to_pyerr)
}

/// The `sylva_cuda` Python extension module.
///
/// The function name (`sylva_cuda`) must match `module-name` in
/// `[tool.maturin]` so the built `.pyd` imports as `sylva_cuda`.
#[pymodule]
fn sylva_cuda(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(py_run_vector_add, module)?)?;
    module.add_function(wrap_pyfunction!(py_run_histogram, module)?)?;
    Ok(())
}
