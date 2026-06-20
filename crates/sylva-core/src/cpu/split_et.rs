//! ExtraTrees random-threshold splitter (ENG-03 ET semantics).
//!
//! Reimplemented from sklearn's RandomSplitter algorithm description
//! (Apache-2.0; NOT copied from sklearn source or any GPL code).
//!
//! Algorithm (per node):
//! 1. Draw `max_features` candidate features.
//! 2. For each candidate feature:
//!    - Compute `(fmin, fmax)` over the node's rows (local range).
//!    - Skip if `fmax <= fmin + FEATURE_THRESHOLD` (constant feature).
//!    - Draw one uniform threshold via Philox: `fmin + u * (fmax - fmin)`.
//!    - Partition rows `x[i,f] <= threshold -> left`, score with criterion.
//! 3. Return the best split. Tie-break: lowest `(feature_id, threshold)`.
//!
//! NaN policy: training data is clean (D-01); `is_nan()` is handled in
//! `predict.rs` at inference time.

use ndarray::ArrayView2;

use crate::config::{Criterion, Task};
use crate::cpu::criterion::{entropy, gini, mse, proxy_improvement};
use crate::rng::philox_uniform;

/// Feature range threshold below which a feature is considered constant and
/// is skipped. Matches sklearn's `FEATURE_THRESHOLD` ≈ 1e-7.
pub const FEATURE_THRESHOLD: f32 = 1e-7;

/// Philox draw index used when drawing the split threshold for a given feature.
/// Frozen bit-allocation (see `rng/mod.rs` counter packing).
pub const DRAW_THRESHOLD: u32 = 0;

/// Philox draw index used when drawing the feature subset permutation index.
pub const DRAW_FEATURE_SELECT: u32 = 1;

/// The result of evaluating a candidate split at a node.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Feature index used to split.
    pub feature_id: usize,
    /// Split threshold: `x[i, feature_id] <= threshold` → left child.
    pub threshold: f32,
    /// Row indices that go to the left child.
    pub left_rows: Vec<usize>,
    /// Row indices that go to the right child.
    pub right_rows: Vec<usize>,
    /// Sample count in the left child.
    pub n_left: u64,
    /// Sample count in the right child.
    pub n_right: u64,
    /// Which child gets NaN/missing rows (D-01): the child with the larger
    /// `n_right` / `n_left`; on a tie, the left child (`left_rows` side).
    pub default_left: bool,
    /// Proxy impurity improvement for this split.
    pub improvement: f32,
}

/// Context passed to the ET splitter for one node.
///
/// Three lifetime parameters keep the borrow checker happy:
/// - `'x` — lifetime of the feature matrix data.
/// - `'y` — lifetime of the label/target slice.
/// - `'r` — lifetime of the row-index slice (may be shorter than `'x`/`'y`
///   because it comes from a locally-owned `Vec<usize>`).
pub struct EtSplitCtx<'x, 'y, 'r> {
    /// Feature matrix, shape `(n_total_rows, n_features)`.
    pub x: ArrayView2<'x, f32>,
    /// Row indices for the current node (subset of `0..n_total_rows`).
    pub rows: &'r [usize],
    /// Target values for classification (class labels as 0-based integers cast
    /// to f32) or regression.
    pub y: &'y [f32],
    /// Number of classes for classification (1 for regression).
    pub n_classes: usize,
    /// Resolved number of candidate features to try (already clamped to ≥1).
    pub max_features: usize,
    /// Training criterion.
    pub criterion: Criterion,
    /// Task type (determines impurity function).
    pub task: Task,
    /// Minimum number of samples required in each child (from `min_samples_leaf`).
    pub min_samples_leaf: usize,
    /// Philox seed (= `TrainConfig::seed`).
    pub seed: u64,
    /// Tree index (0-based) — part of the Philox counter.
    pub tree_id: u32,
    /// Node index within the tree (0-based).
    pub node_id: u32,
}

/// Try to find a valid ET split for the current node. Returns `None` if no
/// valid split was found (all features constant or no candidate produces a
/// child meeting `min_samples_leaf`).
///
/// Candidate features are drawn by shuffling `n_features` positions with the
/// Philox RNG and taking the first `max_features`. Fixed-order float
/// accumulation is used throughout (determinism rule).
pub fn best_random_split(ctx: &EtSplitCtx<'_, '_, '_>) -> Option<SplitResult> {
    let n_rows = ctx.rows.len();
    let n_features = ctx.x.ncols();

    // Build the candidate feature list. We shuffle the range [0, n_features)
    // using Philox-based draws for the Fisher-Yates prefix. Take the first
    // `max_features` elements as candidates.
    let mut feature_order: Vec<usize> = (0..n_features).collect();
    // Fisher-Yates prefix of length `max_features`:
    // for i in 0..max_features: swap feature_order[i] with a random index in [i, n_features).
    let k = ctx.max_features.min(n_features);
    for i in 0..k {
        // Draw a random index in [i, n_features).
        let u = philox_uniform(
            ctx.seed,
            ctx.tree_id,
            ctx.node_id,
            i as u32,
            DRAW_FEATURE_SELECT,
        );
        // Map u to [i, n_features)
        let range = (n_features - i) as f32;
        let j = i + (u * range) as usize;
        let j = j.min(n_features - 1); // clamp to avoid edge-case at u~=1
        feature_order.swap(i, j);
    }
    let candidates = &feature_order[..k];

    // Compute parent node impurity and class/value stats (fixed order).
    let (parent_impurity, parent_targets) = node_stats(ctx);

    let mut best: Option<SplitResult> = None;

    for &feat in candidates {
        // Compute (fmin, fmax) over the node's rows — sequential, fixed order.
        let (fmin, fmax) = feature_range(ctx.x, ctx.rows, feat);

        // Skip constant features.
        if fmax <= fmin + FEATURE_THRESHOLD {
            continue;
        }

        // Draw one uniform threshold in (fmin, fmax) via Philox.
        let u = philox_uniform(
            ctx.seed,
            ctx.tree_id,
            ctx.node_id,
            feat as u32,
            DRAW_THRESHOLD,
        );
        let threshold = fmin + u * (fmax - fmin);

        // Partition rows: x[i, feat] <= threshold → left.
        let mut left_rows = Vec::with_capacity(n_rows / 2);
        let mut right_rows = Vec::with_capacity(n_rows / 2);
        // Sequential row order — fixed float accumulation order.
        for &row in ctx.rows.iter() {
            if ctx.x[[row, feat]] <= threshold {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }

        let n_left = left_rows.len() as u64;
        let n_right = right_rows.len() as u64;

        // Enforce min_samples_leaf on both sides.
        if (n_left as usize) < ctx.min_samples_leaf || (n_right as usize) < ctx.min_samples_leaf {
            continue;
        }

        // Compute child impurities (fixed order via slice iteration).
        let left_impurity = compute_impurity(ctx, &left_rows, &parent_targets);
        let right_impurity = compute_impurity(ctx, &right_rows, &parent_targets);

        let improvement = proxy_improvement(
            parent_impurity,
            left_impurity,
            right_impurity,
            n_left,
            n_right,
        );

        // D-01: default_child = higher sample count child; tie → left.
        let default_left = n_left >= n_right;

        // Keep the best split. Tie-break: (feature_id, threshold) total order.
        let is_better = match &best {
            None => true,
            Some(b) => {
                improvement > b.improvement
                    || (improvement == b.improvement
                        && (feat, threshold.to_bits()) < (b.feature_id, b.threshold.to_bits()))
            }
        };

        if is_better {
            best = Some(SplitResult {
                feature_id: feat,
                threshold,
                left_rows,
                right_rows,
                n_left,
                n_right,
                default_left,
                improvement,
            });
        }
    }

    best
}

/// Compute the node-level impurity and collect label/target info for children.
/// Returns (impurity, parent label/target vector for child impurity reuse).
///
/// Note: we return a `NodeStats` that child sides can reuse, so we only
/// compute the parent's counts/targets once.
fn node_stats(ctx: &EtSplitCtx<'_, '_, '_>) -> (f32, NodeTargets) {
    match ctx.task {
        Task::Classification { n_classes: _ } => {
            // Tally class counts (integers — can be in any order; we use fixed
            // slice order for consistency).
            let counts = class_counts(ctx);
            let total: u64 = counts.iter().sum();
            let imp = match ctx.criterion {
                Criterion::Gini => gini(&counts, total),
                Criterion::Entropy => entropy(&counts, total),
                Criterion::Mse => {
                    // MSE over label values as targets.
                    let targets: Vec<f32> = ctx.rows.iter().map(|&r| ctx.y[r]).collect();
                    mse(&targets)
                }
            };
            (imp, NodeTargets::Counts(counts, total))
        }
        Task::Regression => {
            // Collect targets in fixed row order.
            let targets: Vec<f32> = ctx.rows.iter().map(|&r| ctx.y[r]).collect();
            let imp = mse(&targets);
            (imp, NodeTargets::Values(targets))
        }
    }
}

/// Child impurity given the child's rows, reusing parent label type.
fn compute_impurity(ctx: &EtSplitCtx<'_, '_, '_>, rows: &[usize], _parent: &NodeTargets) -> f32 {
    match ctx.task {
        Task::Classification { n_classes } => {
            let mut counts = vec![0u64; n_classes];
            for &r in rows.iter() {
                let c = ctx.y[r] as usize;
                if c < n_classes {
                    counts[c] += 1;
                }
            }
            let total: u64 = counts.iter().sum();
            match ctx.criterion {
                Criterion::Gini => gini(&counts, total),
                Criterion::Entropy => entropy(&counts, total),
                Criterion::Mse => {
                    let targets: Vec<f32> = rows.iter().map(|&r| ctx.y[r]).collect();
                    mse(&targets)
                }
            }
        }
        Task::Regression => {
            // Sequential accumulation — row order of `rows` is the fixed order.
            let targets: Vec<f32> = rows.iter().map(|&r| ctx.y[r]).collect();
            mse(&targets)
        }
    }
}

/// Class counts over a node's rows. Integer accumulation — order-independent.
fn class_counts(ctx: &EtSplitCtx<'_, '_, '_>) -> Vec<u64> {
    let n_classes = match ctx.task {
        Task::Classification { n_classes } => n_classes,
        Task::Regression => 1,
    };
    let mut counts = vec![0u64; n_classes];
    for &r in ctx.rows.iter() {
        let c = ctx.y[r] as usize;
        if c < n_classes {
            counts[c] += 1;
        }
    }
    counts
}

/// Enum carrying what the parent already computed for quick child impurity.
/// The variant data is not currently read in the child impurity path — we
/// re-compute from the node's `ctx` directly — but the enum is kept as the
/// extension point for future split-stat caching.
#[allow(dead_code)]
enum NodeTargets {
    Counts(Vec<u64>, u64),
    Values(Vec<f32>),
}

/// Feature range (min, max) over a node's rows. Sequential scan, fixed order.
pub fn feature_range(x: ArrayView2<'_, f32>, rows: &[usize], feat: usize) -> (f32, f32) {
    let mut fmin = f32::INFINITY;
    let mut fmax = f32::NEG_INFINITY;
    for &r in rows.iter() {
        let v = x[[r, feat]];
        if v < fmin {
            fmin = v;
        }
        if v > fmax {
            fmax = v;
        }
    }
    (fmin, fmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Construct a small 2-feature / 2-class dataset where feature 0 perfectly
    // separates the classes and feature 1 is noise.
    fn binary_separable_ctx() -> (ndarray::Array2<f32>, Vec<f32>, Vec<usize>) {
        // 10 rows: rows 0-4 are class 0 (x0 in [0,4]), rows 5-9 are class 1
        // (x0 in [5,9]).  x1 is random noise within [0,1].
        let x = ndarray::Array2::from_shape_fn((10, 2), |(i, j)| {
            if j == 0 {
                i as f32
            } else {
                (i as f32 * 0.1) % 1.0
            }
        });
        let y: Vec<f32> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let rows: Vec<usize> = (0..10).collect();
        (x, y, rows)
    }

    #[test]
    fn et_split_finds_separating_split() {
        let (x, y, rows) = binary_separable_ctx();
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_random_split(&ctx).expect("should find a split");
        // Both children must be non-empty.
        assert!(split.n_left > 0 && split.n_right > 0);
        // Improvement must be positive (the feature separates the classes).
        assert!(split.improvement > 0.0, "improvement must be positive");
    }

    #[test]
    fn constant_feature_is_skipped() {
        // Build a dataset where feature 0 is constant and feature 1 separates.
        let x = ndarray::Array2::from_shape_fn((10, 2), |(i, j)| {
            if j == 0 {
                5.0 // constant
            } else {
                i as f32
            }
        });
        let y: Vec<f32> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let rows: Vec<usize> = (0..10).collect();
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2, // try both, feature 0 must be skipped
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_random_split(&ctx).expect("feature 1 provides a valid split");
        // The chosen feature must NOT be the constant one.
        assert_ne!(split.feature_id, 0, "constant feature 0 must not be chosen");
    }

    #[test]
    fn min_samples_leaf_respected() {
        // 4 rows total; min_samples_leaf = 3 forces no valid split.
        let x = array![[0.0_f32, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]];
        let y = vec![0.0_f32, 1.0, 0.0, 1.0];
        let rows = vec![0, 1, 2, 3];
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 3, // forces both children to need ≥3 rows each
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        // No split can satisfy n_left >= 3 AND n_right >= 3 with only 4 rows.
        assert!(best_random_split(&ctx).is_none());
    }

    #[test]
    fn threshold_in_local_feature_range() {
        let (x, y, rows) = binary_separable_ctx();
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 1,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 123,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_random_split(&ctx).expect("should find a split");
        let (fmin, fmax) = feature_range(x.view(), &rows, split.feature_id);
        // Threshold must be in the local (min, fmin+THRESHOLD, max) range.
        assert!(
            split.threshold >= fmin && split.threshold <= fmax,
            "threshold {:.4} not in [{:.4}, {:.4}]",
            split.threshold,
            fmin,
            fmax
        );
    }

    #[test]
    fn default_child_is_higher_count_side() {
        let (x, y, rows) = binary_separable_ctx();
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_random_split(&ctx).expect("split");
        // default_left should be true when n_left >= n_right.
        assert_eq!(split.default_left, split.n_left >= split.n_right);
    }

    #[test]
    fn deterministic_same_seed_same_result() {
        let (x, y, rows) = binary_separable_ctx();
        let ctx_a = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 77,
            tree_id: 3,
            node_id: 5,
        };
        let ctx_b = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 77,
            tree_id: 3,
            node_id: 5,
        };
        let a = best_random_split(&ctx_a).unwrap();
        let b = best_random_split(&ctx_b).unwrap();
        assert_eq!(a.feature_id, b.feature_id);
        assert_eq!(a.threshold.to_bits(), b.threshold.to_bits());
        assert_eq!(a.left_rows, b.left_rows);
        assert_eq!(a.right_rows, b.right_rows);
    }

    #[test]
    fn regression_split_finds_valid_split() {
        // Simple regression dataset: y = x0 (linear).
        let x =
            ndarray::Array2::from_shape_fn((10, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let rows: Vec<usize> = (0..10).collect();
        let ctx = EtSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 1,
            max_features: 2, // All features (reg default)
            criterion: Criterion::Mse,
            task: Task::Regression,
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_random_split(&ctx).expect("should find a split");
        assert!(split.improvement > 0.0);
        assert!(split.n_left > 0 && split.n_right > 0);
    }
}
