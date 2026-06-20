//! Device-neutral backend contracts (ENG-01).
//!
//! Two traits split the contract by execution shape:
//! * [`Backend`] — the high-level `fit`/`predict` the CPU oracle implements via
//!   recursive splitting (this phase) and the future CudaBackend implements too.
//! * [`HistogramBackend`] — the GPU histogram hot path (Phase 4). It names the
//!   remaining ENG-01 device ops (`quantize`/`build_histograms`/`eval_splits`/
//!   `partition`) so all five op-names live in the contract surface.
//!
//! Device neutrality (ENG-01, anti-pattern AP-2): no device pointer or stream
//! type ever appears in these signatures. The associated `type Bins` on
//! [`HistogramBackend`] is the seam — a CPU backend binds it to an in-memory
//! type, a future CUDA backend to its own device buffer — so a concrete device
//! type never crosses the boundary.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::config::TrainConfig;
use crate::error::SylvaError;
use crate::ir::ForestIR;

/// Prediction output, device-neutral.
#[derive(Debug, Clone, PartialEq)]
pub enum Predictions {
    /// Per-row class probabilities, shape `(n_rows, n_classes)`.
    ClassProba(Array2<f32>),
    /// Per-row regression outputs, shape `(n_rows,)`.
    Regression(Array1<f32>),
}

/// The high-level training/inference contract. Implemented by `CpuBackend`
/// (Plan 02-02/03) and, additively, the future `CudaBackend` (Phase 4).
pub trait Backend {
    /// Train a forest into a device-neutral [`ForestIR`].
    fn fit(
        &self,
        x: ArrayView2<f32>,
        y: ArrayView1<f32>,
        cfg: &TrainConfig,
    ) -> Result<ForestIR, SylvaError>;

    /// Predict from a trained [`ForestIR`].
    fn predict(&self, ir: &ForestIR, x: ArrayView2<f32>) -> Result<Predictions, SylvaError>;
}

// --- Device-neutral placeholder handle types ---------------------------------
// Minimal device-agnostic stubs so `HistogramBackend` compiles CUDA-free. Their
// Phase-4 internals are out of scope; they exist only to name the contract.

/// Per-feature bin edges produced by quantization.
#[derive(Debug, Clone, Default)]
pub struct BinEdges;
/// The set of nodes being grown in the current breadth-first wave.
#[derive(Debug, Clone, Default)]
pub struct Frontier;
/// A device-neutral handle to a set of row indices.
#[derive(Debug, Clone, Default)]
pub struct RowIndex;
/// Per-node, per-feature histograms.
#[derive(Debug, Clone, Default)]
pub struct Histograms;
/// Split-evaluation mode/parameters.
#[derive(Debug, Clone, Default)]
pub struct SplitMode;
/// A chosen split (feature + threshold/bin + default direction).
#[derive(Debug, Clone, Default)]
pub struct SplitDecision;
/// The row ranges of the two children after partitioning.
#[derive(Debug, Clone, Default)]
pub struct ChildRanges;

/// The histogram-training device contract (the GPU hot path, Phase 4). Defined
/// now so all five ENG-01 op-names exist in the contract surface. The CPU oracle
/// trains by recursive exact splitting and does **not** implement this trait.
pub trait HistogramBackend {
    /// Device-neutral binned feature matrix (CPU: in-memory; CUDA: device buffer).
    type Bins;

    fn quantize(&self, x: ArrayView2<f32>) -> Result<(Self::Bins, BinEdges), SylvaError>;

    fn build_histograms(
        &self,
        bins: &Self::Bins,
        frontier: &Frontier,
        rows: &RowIndex,
    ) -> Result<Histograms, SylvaError>;

    fn eval_splits(&self, hist: &Histograms, mode: &SplitMode)
        -> Result<SplitDecision, SylvaError>;

    fn partition(
        &self,
        bins: &Self::Bins,
        rows: &RowIndex,
        split: &SplitDecision,
    ) -> Result<ChildRanges, SylvaError>;
}
