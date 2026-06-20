//! Recursive forest builder (ENG-03).
//!
//! Stub — implementation added in Task 3.

use ndarray::{ArrayView1, ArrayView2};

use crate::config::TrainConfig;
use crate::error::SylvaError;
use crate::ir::ForestIR;

/// Stub — replaced in Task 3.
pub(crate) fn fit_forest(
    _x: ArrayView2<f32>,
    _y: ArrayView1<f32>,
    _cfg: &TrainConfig,
) -> Result<ForestIR, SylvaError> {
    Err(SylvaError::InvalidInput(
        "fit_forest not yet implemented".into(),
    ))
}
