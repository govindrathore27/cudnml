//! sylva-core — device-neutral contracts + the pure-Rust CPU correctness oracle.
//!
//! Phase 2 lays the foundation every later slice imports: the [`Backend`] /
//! [`HistogramBackend`] trait seam (ENG-01), the SoA [`ForestIR`] (ENG-02), the
//! training [`config`] types, and the stateless Philox-4×32-10 [`rng`] (ENG-06).
//! There is **no CUDA and no Python** in this crate — `CudaBackend` (Phase 4)
//! and the PyO3 estimator API (Phase 5) are additive layers on top of it.

pub mod backend;
pub mod config;
pub mod error;
pub mod ir;
pub mod rng;

pub use backend::{Backend, HistogramBackend, Predictions};
pub use config::{Algo, Criterion, MaxFeatures, Task, TrainConfig};
pub use error::SylvaError;
pub use ir::ForestIR;
