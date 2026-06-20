//! NaN-safe forest traversal + prediction aggregation (ENG-05).
//!
//! Stub — implementation added in Task 3.

use ndarray::ArrayView2;

use crate::backend::Predictions;
use crate::error::SylvaError;
use crate::ir::ForestIR;

/// Stub — replaced in Task 3.
pub(crate) fn predict_forest(
    _ir: &ForestIR,
    _x: ArrayView2<f32>,
) -> Result<Predictions, SylvaError> {
    Err(SylvaError::InvalidInput(
        "predict_forest not yet implemented".into(),
    ))
}
