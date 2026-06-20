//! Typed error for sylva-core. Mirrors `sylva-cuda::CudaError`'s style: a
//! `thiserror` enum so boundary/config validation returns a typed error instead
//! of panicking (no silent fallback; no `.unwrap()` on fallible paths).

use thiserror::Error;

/// Errors surfaced across the sylva-core boundary.
#[derive(Debug, Error)]
pub enum SylvaError {
    /// Invalid caller input (wrong shape, disallowed value, mismatched lengths).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Invalid training configuration (nonsensical hyperparameters).
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// A structurally invalid `ForestIR` was detected.
    #[error("invalid ForestIR: {0}")]
    InvalidIr(String),
}
