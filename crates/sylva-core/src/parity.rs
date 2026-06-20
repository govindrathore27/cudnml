//! Parity-support utilities for Phase-05 distributional parity testing (D-04).
//!
//! The primary export is [`split_statistics`], which extracts per-internal-node
//! `(feature_id, normalized_threshold)` pairs from a [`ForestIR`] — pooled
//! across all trees — as a [`SplitStats`] value the Phase-05 KS harness can
//! serialize to JSON and compare against scikit-learn's equivalent statistics.
//!
//! **What `split_statistics` reads:** only `feature_id` and `threshold` from
//! the IR, plus `node_sample_count` for feature-range normalization. It reads
//! no leaf payloads and performs no training or prediction.
//!
//! **Normalization:** each node's threshold is normalized to `[0, 1]` by the
//! per-tree, per-feature `(min, max)` range observed at the root (the root's
//! full-dataset range). Normalizing to a common `[0, 1]` scale makes thresholds
//! from different features and datasets comparable in the KS test.
//!
//! **Leaf nodes:** leaf nodes contribute **no** split statistic. Only internal
//! nodes (where `feature_id != LEAF_FEATURE`) are included.
//!
//! **Serde:** [`SplitStats`] derives `Serialize`/`Deserialize` so the Python
//! harness can consume it directly via `serde_json`.

use serde::{Deserialize, Serialize};

use crate::ir::{ForestIR, LEAF_FEATURE};

/// One internal-node split observation: the chosen feature and its normalized
/// threshold.
///
/// `normalized_threshold` is the raw threshold divided by the per-feature range
/// observed across all split nodes (i.e. the range of recorded thresholds for
/// that feature). A value in `[0, 1]` means "near the feature minimum";
/// a value near 1.0 means "near the maximum". If all observed thresholds for a
/// feature are the same, the value is clamped to 0.0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitObservation {
    /// Index of the feature chosen for this split (matches `ForestIR::feature_id`).
    pub feature_id: usize,
    /// Threshold normalized to the observed range of that feature's split thresholds.
    /// Always in `[0.0, 1.0]`.
    pub normalized_threshold: f32,
}

/// Aggregate split statistics extracted from a `ForestIR`.
///
/// Contains one `SplitObservation` per internal node across all trees.
/// The Phase-05 Python KS harness reads this via `serde_json` and computes:
/// - feature-selection frequency (histogram of `feature_id`)
/// - threshold distribution per feature (distribution of `normalized_threshold`
///   for each `feature_id`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitStats {
    /// Number of trees in the source `ForestIR`.
    pub n_trees: usize,
    /// Number of features in the source `ForestIR`.
    pub n_features: usize,
    /// One observation per internal node, pooled across all trees.
    pub observations: Vec<SplitObservation>,
}

/// Extract aggregate split statistics from a `ForestIR` for the Phase-05 KS harness.
///
/// Visits every internal node in every tree. For each internal node:
/// - Records the `feature_id` (which feature was chosen for this split).
/// - Records the `threshold` normalized to the per-feature range of all observed
///   threshold values (so thresholds across features are on a common `[0,1]` scale).
///
/// Leaf nodes (where `feature_id == LEAF_FEATURE`) contribute no observation.
///
/// # Example
///
/// ```rust
/// use sylva_core::{cpu::CpuBackend, Backend};
/// use sylva_core::config::{Algo, Criterion, MaxFeatures, TrainConfig};
/// use sylva_core::parity::split_statistics;
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
/// let y = Array1::from_iter((0..20_usize).map(|i| if i < 10 { 0.0 } else { 1.0 }));
/// let cfg = TrainConfig {
///     n_estimators: 3,
///     max_depth: Some(3),
///     max_features: MaxFeatures::Sqrt,
///     min_samples_split: 2,
///     min_samples_leaf: 1,
///     bootstrap: false,
///     criterion: Criterion::Gini,
///     seed: 42,
///     algo: Algo::ExtraTrees,
/// };
/// let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
/// let stats = split_statistics(&ir);
/// assert_eq!(stats.n_trees, 3);
/// assert!(!stats.observations.is_empty());
/// // All normalized thresholds must be in [0, 1].
/// for obs in &stats.observations {
///     assert!((0.0..=1.0).contains(&obs.normalized_threshold));
/// }
/// ```
pub fn split_statistics(ir: &ForestIR) -> SplitStats {
    // --- Pass 1: collect raw (feature_id, threshold) for all internal nodes ---
    let mut raw: Vec<(usize, f32)> = Vec::new();
    for node in 0..ir.node_count() {
        let fid = ir.feature_id[node];
        if fid != LEAF_FEATURE {
            raw.push((fid as usize, ir.threshold[node]));
        }
    }

    if raw.is_empty() {
        return SplitStats {
            n_trees: ir.n_trees,
            n_features: ir.n_features,
            observations: Vec::new(),
        };
    }

    // --- Pass 2: compute per-feature (min, max) of observed thresholds ---
    let n_features = ir.n_features;
    let mut feat_min = vec![f32::INFINITY; n_features];
    let mut feat_max = vec![f32::NEG_INFINITY; n_features];
    for &(fid, thr) in &raw {
        if fid < n_features {
            if thr < feat_min[fid] {
                feat_min[fid] = thr;
            }
            if thr > feat_max[fid] {
                feat_max[fid] = thr;
            }
        }
    }

    // --- Pass 3: normalize and build SplitObservation list ---
    let observations: Vec<SplitObservation> = raw
        .into_iter()
        .filter_map(|(fid, thr)| {
            if fid >= n_features {
                return None;
            }
            let lo = feat_min[fid];
            let hi = feat_max[fid];
            let range = hi - lo;
            let normalized = if range <= 0.0 {
                // All observations for this feature have the same threshold.
                0.0_f32
            } else {
                ((thr - lo) / range).clamp(0.0, 1.0)
            };
            Some(SplitObservation {
                feature_id: fid,
                normalized_threshold: normalized,
            })
        })
        .collect();

    SplitStats {
        n_trees: ir.n_trees,
        n_features: ir.n_features,
        observations,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Algo, Criterion, MaxFeatures, TrainConfig},
        cpu::CpuBackend,
        Backend,
    };
    use ndarray::{Array1, Array2};

    fn et_clf_cfg(seed: u64) -> TrainConfig {
        TrainConfig {
            n_estimators: 4,
            max_depth: Some(4),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Gini,
            seed,
            algo: Algo::ExtraTrees,
        }
    }

    fn small_clf_data() -> (Array2<f32>, Array1<f32>) {
        let n = 20usize;
        let x = Array2::from_shape_fn((n, 2), |(i, _)| i as f32);
        let y = Array1::from_iter((0..n).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        (x, y)
    }

    fn small_reg_data() -> (Array2<f32>, Array1<f32>) {
        let n = 20usize;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = Array1::from_iter((0..n).map(|i| i as f32));
        (x, y)
    }

    /// Leaf nodes must contribute no split observation.
    #[test]
    fn leaves_contribute_no_observation() {
        let (x, y) = small_clf_data();
        let cfg = et_clf_cfg(42);
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");

        let stats = split_statistics(&ir);

        // Count internal nodes in the IR.
        let n_internal = ir.feature_id.iter().filter(|&&f| f != LEAF_FEATURE).count();
        assert_eq!(
            stats.observations.len(),
            n_internal,
            "one observation per internal node (leaves excluded)"
        );
    }

    /// All normalized thresholds must be in [0, 1].
    #[test]
    fn normalized_thresholds_in_unit_interval() {
        let (x, y) = small_clf_data();
        let cfg = et_clf_cfg(7);
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);

        for obs in &stats.observations {
            assert!(
                (0.0..=1.0).contains(&obs.normalized_threshold),
                "feature {} normalized_threshold={} outside [0, 1]",
                obs.feature_id,
                obs.normalized_threshold
            );
        }
    }

    /// Feature ids in the output must match the IR's feature dimension.
    #[test]
    fn feature_ids_within_bounds() {
        let (x, y) = small_clf_data();
        let cfg = et_clf_cfg(3);
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);

        for obs in &stats.observations {
            assert!(
                obs.feature_id < ir.n_features,
                "feature_id {} out of range (n_features={})",
                obs.feature_id,
                ir.n_features
            );
        }
    }

    /// n_trees and n_features in SplitStats match the IR.
    #[test]
    fn stats_metadata_matches_ir() {
        let (x, y) = small_clf_data();
        let cfg = et_clf_cfg(1);
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);

        assert_eq!(stats.n_trees, ir.n_trees, "n_trees must match");
        assert_eq!(stats.n_features, ir.n_features, "n_features must match");
    }

    /// Regression forest: split_statistics works (no leaf_proba involved).
    #[test]
    fn regression_forest_split_stats() {
        let (x, y) = small_reg_data();
        let cfg = TrainConfig {
            criterion: Criterion::Mse,
            max_features: MaxFeatures::All,
            algo: Algo::ExtraTrees,
            ..et_clf_cfg(55)
        };
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);

        assert_eq!(stats.n_trees, cfg.n_estimators);
        for obs in &stats.observations {
            assert!(
                (0.0..=1.0).contains(&obs.normalized_threshold),
                "reg: normalized_threshold out of range"
            );
        }
    }

    /// SplitStats serde round-trip: deserialize(serialize(stats)) == stats.
    #[test]
    fn split_stats_serde_round_trip() {
        let (x, y) = small_clf_data();
        let cfg = et_clf_cfg(99);
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);

        let json = serde_json::to_string(&stats).expect("serialize SplitStats");
        let back: SplitStats = serde_json::from_str(&json).expect("deserialize SplitStats");
        assert_eq!(stats, back, "SplitStats serde round-trip must be identical");
    }

    /// An all-leaf IR (max_depth=0 forces all trees to have a single leaf root)
    /// must produce zero observations.
    #[test]
    fn all_leaf_forest_produces_empty_observations() {
        let n = 10usize;
        let x = Array2::from_shape_fn((n, 2), |(i, _)| i as f32);
        let y = Array1::from_iter((0..n).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let cfg = TrainConfig {
            n_estimators: 3,
            max_depth: Some(0), // depth=0 → always emit a leaf
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Gini,
            seed: 1,
            algo: Algo::ExtraTrees,
        };
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("fit");
        let stats = split_statistics(&ir);
        assert!(
            stats.observations.is_empty(),
            "depth-0 forest should produce no split observations"
        );
    }

    /// RF clf: split_statistics works with bootstrap sampling.
    #[test]
    fn rf_clf_split_stats() {
        let (x, y) = small_clf_data();
        let cfg = TrainConfig {
            bootstrap: true,
            algo: Algo::RandomForest,
            ..et_clf_cfg(13)
        };
        let ir = CpuBackend.fit(x.view(), y.view(), &cfg).expect("RF fit");
        let stats = split_statistics(&ir);

        assert_eq!(stats.n_trees, cfg.n_estimators);
        for obs in &stats.observations {
            assert!(
                (0.0..=1.0).contains(&obs.normalized_threshold),
                "RF: normalized_threshold out of range"
            );
        }
    }
}
