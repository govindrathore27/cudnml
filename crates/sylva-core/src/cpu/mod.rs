//! Pure-Rust `CpuBackend` — the device-neutral correctness oracle (ENG-03).
//!
//! Training uses ndarray for host-side data and rayon for tree-level
//! parallelism. Inference reads the same `ForestIR` that the future
//! `CudaBackend` (Phase 4) will share.
//!
//! Module layout:
//! - `criterion` — Gini / entropy / MSE impurity with fixed-order f32 sums.
//! - `split_et`  — ExtraTrees random-threshold splitter.
//! - `fit`       — Recursive node builder, rayon forest loop.
//! - `predict`   — NaN-safe forest traversal + aggregation.

pub mod bootstrap;
pub mod criterion;
pub mod fit;
pub mod predict;
pub mod split_et;
pub mod split_rf;

use ndarray::{ArrayView1, ArrayView2};

use crate::backend::{Backend, Predictions};
use crate::config::TrainConfig;
use crate::error::SylvaError;
use crate::ir::ForestIR;

/// The pure-Rust CPU correctness oracle. Implements `Backend` via recursive
/// exact splitting (no quantizer/histograms until Phase 3/4). Thread-safe.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn fit(
        &self,
        x: ArrayView2<f32>,
        y: ArrayView1<f32>,
        cfg: &TrainConfig,
    ) -> Result<ForestIR, SylvaError> {
        fit::fit_forest(x, y, cfg)
    }

    fn predict(&self, ir: &ForestIR, x: ArrayView2<f32>) -> Result<Predictions, SylvaError> {
        predict::predict_forest(ir, x)
    }
}
