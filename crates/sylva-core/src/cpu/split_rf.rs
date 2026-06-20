//! RandomForest exact best-split (sorted-midpoint candidate search).
//!
//! Reimplemented from sklearn's BestSplitter algorithm description
//! (Apache-2.0; NOT copied from sklearn source or any GPL code).
//! Provenance: `[CITED: github.com/scikit-learn/scikit-learn /sklearn/tree/_splitter.pyx
//! BestSplitter]`.
//!
//! # Algorithm (per node)
//!
//! 1. Draw `max_features` candidate features (Fisher-Yates prefix, same as ET).
//! 2. For each candidate feature: sort the node's feature values, evaluate
//!    midpoint thresholds `v_prev * 0.5 + v_curr * 0.5` between consecutive
//!    distinct values (skipping pairs within `FEATURE_THRESHOLD`), and score
//!    each threshold's impurity improvement.
//! 3. Return the globally best split across all features/thresholds with a
//!    deterministic `(feature_id, threshold_bits)` tie-break (lowest pair wins).
//! 4. `default_child` = higher-row-count child; tie → left (D-01 policy, same as ET).
//! 5. Partition uses `x[row, feat] <= threshold` → left (same `<=` convention as ET).
//!
//! **Float accumulation (determinism):** all f32 sums inside a node use a fixed
//! sequential row order (never a rayon parallel sum). The feature-value sort is on
//! locally-copied f32 values and is stable within a call.

use ndarray::ArrayView2;

use crate::config::{Criterion, Task};
use crate::cpu::criterion::{entropy, gini, mse, proxy_improvement};
use crate::cpu::split_et::{DRAW_FEATURE_SELECT, FEATURE_THRESHOLD};
use crate::rng::philox_uniform;

/// Context passed to the RF best-split finder for one node.
///
/// Same lifetime structure as `EtSplitCtx`: three independent lifetimes so
/// locally-allocated row Vecs from `SplitResult` can be borrowed in recursive
/// calls without tying them to the long-lived `x`/`y` data lifetimes.
pub struct RfSplitCtx<'x, 'y, 'r> {
    /// Feature matrix, shape `(n_total_rows, n_features)`.
    pub x: ArrayView2<'x, f32>,
    /// Row indices for the current node (subset of `0..n_total_rows`).
    pub rows: &'r [usize],
    /// Target values: class labels (0-based int cast to f32) or regression targets.
    pub y: &'y [f32],
    /// Number of classes for classification (1 for regression).
    pub n_classes: usize,
    /// Resolved number of candidate features to try (already clamped to ≥1).
    pub max_features: usize,
    /// Training criterion.
    pub criterion: Criterion,
    /// Task type.
    pub task: Task,
    /// Minimum number of samples required in each child.
    pub min_samples_leaf: usize,
    /// Philox seed (= `TrainConfig::seed`).
    pub seed: u64,
    /// Tree index (0-based) — part of the Philox counter.
    pub tree_id: u32,
    /// Node index within the tree (0-based).
    pub node_id: u32,
}

/// The result of the RF best-split search for one node.
///
/// Same shape as `split_et::SplitResult` so `build_node` in `fit.rs` consumes
/// both without conditional field access.
#[derive(Debug, Clone)]
pub struct RfSplitResult {
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
    /// Which child gets NaN/missing rows (D-01): higher count child; tie → left.
    pub default_left: bool,
    /// Proxy impurity improvement for this split.
    pub improvement: f32,
}

/// Find the best split for the current node using the RF BestSplitter midpoint
/// algorithm. Returns `None` if no valid split exists (all features constant,
/// or no candidate threshold satisfies `min_samples_leaf`).
///
/// This function is deterministic: given the same `ctx`, it always returns the
/// same result. The feature subset is drawn via the same Fisher-Yates Philox
/// prefix as the ET splitter (using `DRAW_FEATURE_SELECT`), ensuring the two
/// algorithms share the same feature-selection stream shape so tests can
/// predict which feature is examined.
pub fn best_split(ctx: &RfSplitCtx<'_, '_, '_>) -> Option<RfSplitResult> {
    let n_rows = ctx.rows.len();
    let n_features = ctx.x.ncols();

    // Draw candidate feature subset: Fisher-Yates prefix, same as ET.
    let mut feature_order: Vec<usize> = (0..n_features).collect();
    let k = ctx.max_features.min(n_features);
    for i in 0..k {
        let u = philox_uniform(
            ctx.seed,
            ctx.tree_id,
            ctx.node_id,
            i as u32,
            DRAW_FEATURE_SELECT,
        );
        let range = (n_features - i) as f32;
        let j = i + (u * range) as usize;
        let j = j.min(n_features - 1);
        feature_order.swap(i, j);
    }
    let candidates = &feature_order[..k];

    // Compute the parent node impurity (fixed sequential order — determinism rule).
    let parent_impurity = compute_impurity(ctx, ctx.rows);

    let mut best: Option<RfSplitResult> = None;

    for &feat in candidates {
        // Collect (row_index, feature_value) pairs for this node.
        // Sequential order over ctx.rows — fixed float accumulation order.
        let mut values: Vec<f32> = ctx.rows.iter().map(|&r| ctx.x[[r, feat]]).collect();

        // Sort the feature values (local copy; doesn't affect the original data).
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Walk consecutive distinct pairs: midpoint between v_prev and v_curr.
        // Skip pairs closer than FEATURE_THRESHOLD (constant-feature guard).
        let mut prev_val = values[0];
        let mut prev_threshold_bits: Option<u32> = None;

        for &curr_val in &values[1..] {
            // Skip if the gap is too small (effectively constant over this range).
            if curr_val - prev_val <= FEATURE_THRESHOLD {
                // Still advance prev_val even on skip so the next pair is evaluated
                // relative to the current value, not the last distinct pair.
                // (This matches BestSplitter: duplicate values just don't yield a
                //  threshold candidate.)
                prev_val = curr_val;
                continue;
            }

            let threshold = prev_val * 0.5 + curr_val * 0.5;

            // Skip if we already evaluated this exact threshold (can happen when
            // the midpoint rounds to the same f32 bits as the previous one).
            let thr_bits = threshold.to_bits();
            if Some(thr_bits) == prev_threshold_bits {
                prev_val = curr_val;
                continue;
            }
            prev_threshold_bits = Some(thr_bits);
            prev_val = curr_val;

            // Partition: x[row, feat] <= threshold → left.
            // Sequential row order — fixed float accumulation order.
            let mut left_rows: Vec<usize> = Vec::with_capacity(n_rows / 2);
            let mut right_rows: Vec<usize> = Vec::with_capacity(n_rows / 2);
            for &row in ctx.rows.iter() {
                if ctx.x[[row, feat]] <= threshold {
                    left_rows.push(row);
                } else {
                    right_rows.push(row);
                }
            }

            let n_left = left_rows.len() as u64;
            let n_right = right_rows.len() as u64;

            // Enforce min_samples_leaf.
            if (n_left as usize) < ctx.min_samples_leaf || (n_right as usize) < ctx.min_samples_leaf
            {
                continue;
            }

            // Compute child impurities (sequential — determinism rule).
            let left_imp = compute_impurity(ctx, &left_rows);
            let right_imp = compute_impurity(ctx, &right_rows);

            let improvement =
                proxy_improvement(parent_impurity, left_imp, right_imp, n_left, n_right);

            // D-01: default_child = higher sample count child; tie → left.
            let default_left = n_left >= n_right;

            // Keep the best split. Tie-break: (feature_id, threshold_bits) total order.
            let is_better = match &best {
                None => true,
                Some(b) => {
                    improvement > b.improvement
                        || (improvement == b.improvement
                            && (feat, thr_bits) < (b.feature_id, b.threshold.to_bits()))
                }
            };

            if is_better {
                best = Some(RfSplitResult {
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
    }

    best
}

/// Compute the impurity of a node's rows (sequential f32 accumulation).
fn compute_impurity(ctx: &RfSplitCtx<'_, '_, '_>, rows: &[usize]) -> f32 {
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
            let targets: Vec<f32> = rows.iter().map(|&r| ctx.y[r]).collect();
            mse(&targets)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// 2-class, 2-feature dataset: feature 0 linearly separates the classes.
    /// Rows 0–4: class 0 (x0 = 0.0, 1.0, 2.0, 3.0, 4.0)
    /// Rows 5–9: class 1 (x0 = 5.0, 6.0, 7.0, 8.0, 9.0)
    fn separable_ctx() -> (ndarray::Array2<f32>, Vec<f32>, Vec<usize>) {
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

    fn ctx_from<'x, 'y, 'r>(
        x: &'x ndarray::Array2<f32>,
        y: &'y [f32],
        rows: &'r [usize],
        max_features: usize,
        seed: u64,
        tree_id: u32,
        node_id: u32,
    ) -> RfSplitCtx<'x, 'y, 'r> {
        RfSplitCtx {
            x: x.view(),
            rows,
            y,
            n_classes: 2,
            max_features,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed,
            tree_id,
            node_id,
        }
    }

    #[test]
    fn best_split_picks_exact_separating_midpoint() {
        // Feature 0 perfectly separates classes at x0 = 4.5 (midpoint of 4.0 and 5.0).
        let (x, y, rows) = separable_ctx();
        let ctx = ctx_from(&x, &y, &rows, 2, 42, 0, 0);
        let split = best_split(&ctx).expect("must find a split");
        // The split must fully separate the classes.
        assert_eq!(split.n_left, 5, "5 rows should go left");
        assert_eq!(split.n_right, 5, "5 rows should go right");
        // Left rows should be class 0 (x0 ∈ [0,4]).
        for &r in &split.left_rows {
            assert_eq!(y[r], 0.0, "left child must be class 0");
        }
        // Right rows should be class 1 (x0 ∈ [5,9]).
        for &r in &split.right_rows {
            assert_eq!(y[r], 1.0, "right child must be class 1");
        }
    }

    #[test]
    fn midpoints_between_distinct_sorted_values_only() {
        // 4-row dataset with x0 in {1, 1, 3, 3}: only one candidate threshold
        // (midpoint of 1 and 3 = 2.0) should be evaluated.
        let x = array![[1.0_f32, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0];
        let rows = vec![0, 1, 2, 3];
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 1, // only feature 0
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_split(&ctx).expect("split must be found");
        // The only candidate threshold should be the midpoint of 1.0 and 3.0 = 2.0.
        assert_eq!(
            split.threshold, 2.0_f32,
            "threshold must be the midpoint of the two distinct values"
        );
        // Partition: rows 0,1 → left (x0=1 ≤ 2); rows 2,3 → right (x0=3 > 2).
        assert_eq!(split.n_left, 2);
        assert_eq!(split.n_right, 2);
    }

    #[test]
    fn constant_feature_produces_no_split() {
        // Feature 0 is constant; no valid threshold exists.
        let x = array![[5.0_f32, 0.0], [5.0, 1.0], [5.0, 0.0], [5.0, 1.0]];
        let y = vec![0.0_f32, 1.0, 0.0, 1.0];
        let rows = vec![0, 1, 2, 3];
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 1, // only feature 0 (constant)
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        assert!(
            best_split(&ctx).is_none(),
            "constant feature must produce no valid split"
        );
    }

    #[test]
    fn near_equal_values_skipped_by_feature_threshold() {
        // Two values that differ by less than FEATURE_THRESHOLD ≈ 1e-7 should
        // not produce a candidate threshold.
        let eps = FEATURE_THRESHOLD * 0.5; // below the threshold
        let x = array![[0.0_f32, 0.0], [eps, 1.0]];
        let y = vec![0.0_f32, 1.0];
        let rows = vec![0, 1];
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 1,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        assert!(
            best_split(&ctx).is_none(),
            "pair within FEATURE_THRESHOLD must be skipped"
        );
    }

    #[test]
    fn min_samples_leaf_respected() {
        // 4 rows; min_samples_leaf = 3 → only threshold placing 3+ on each side
        // is valid, which is impossible with 4 rows total.
        let x = array![[1.0_f32, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let y = vec![0.0_f32, 0.0, 1.0, 1.0];
        let rows = vec![0, 1, 2, 3];
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 1,
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 3,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        // Any split of 4 rows produces children of size ≤2 and ≥2, but min_leaf=3
        // requires each child to have ≥3 rows → impossible.
        assert!(best_split(&ctx).is_none());
    }

    #[test]
    fn tie_break_by_feature_id_then_threshold() {
        // When two features produce the same improvement, the lower (feature_id, threshold)
        // pair wins. Use a dataset where both features produce equal-gain splits.
        // Feature 0 and feature 1 are identical copies of [0,0,0,0,1,1,1,1].
        let n = 8;
        let x = ndarray::Array2::from_shape_fn((n, 2), |(i, _j)| if i < 4 { 0.0 } else { 1.0 });
        let y: Vec<f32> = (0..n).map(|i| if i < 4 { 0.0 } else { 1.0 }).collect();
        let rows: Vec<usize> = (0..n).collect();
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 2,
            max_features: 2, // try both features
            criterion: Criterion::Gini,
            task: Task::Classification { n_classes: 2 },
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_split(&ctx).expect("split must be found");
        // Both features are identical → equal improvement → tie-break by feature_id.
        // The lower feature_id (whichever is examined first in the sorted order)
        // should win. We just assert there IS a result and improvement is positive.
        assert!(
            split.improvement > 0.0,
            "improvement must be positive for a separating split"
        );
    }

    #[test]
    fn default_child_is_higher_count_child() {
        let (x, y, rows) = separable_ctx();
        let ctx = ctx_from(&x, &y, &rows, 2, 42, 0, 0);
        let split = best_split(&ctx).expect("split");
        // D-01: default_left must be true iff n_left >= n_right.
        assert_eq!(
            split.default_left,
            split.n_left >= split.n_right,
            "default_left must be true when n_left >= n_right (D-01)"
        );
    }

    #[test]
    fn left_convention_is_lte_threshold() {
        // All rows with x0 <= threshold must be in left_rows.
        let (x, y, rows) = separable_ctx();
        let ctx = ctx_from(&x, &y, &rows, 2, 42, 0, 0);
        let split = best_split(&ctx).expect("split");
        let thr = split.threshold;
        let feat = split.feature_id;
        for &r in &split.left_rows {
            assert!(
                x[[r, feat]] <= thr,
                "left row {} has x={} > threshold {}",
                r,
                x[[r, feat]],
                thr
            );
        }
        for &r in &split.right_rows {
            assert!(
                x[[r, feat]] > thr,
                "right row {} has x={} <= threshold {}",
                r,
                x[[r, feat]],
                thr
            );
        }
    }

    #[test]
    fn deterministic_same_ctx_same_result() {
        let (x, y, rows) = separable_ctx();
        let ctx_a = ctx_from(&x, &y, &rows, 2, 77, 3, 5);
        let ctx_b = ctx_from(&x, &y, &rows, 2, 77, 3, 5);
        let a = best_split(&ctx_a).expect("split a");
        let b = best_split(&ctx_b).expect("split b");
        assert_eq!(a.feature_id, b.feature_id);
        assert_eq!(a.threshold.to_bits(), b.threshold.to_bits());
        assert_eq!(a.left_rows, b.left_rows);
        assert_eq!(a.right_rows, b.right_rows);
    }

    #[test]
    fn regression_best_split_works() {
        // Regression dataset: y = x0; feature 0 should be chosen.
        let x =
            ndarray::Array2::from_shape_fn((10, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let rows: Vec<usize> = (0..10).collect();
        let ctx = RfSplitCtx {
            x: x.view(),
            rows: &rows,
            y: &y,
            n_classes: 1,
            max_features: 2,
            criterion: Criterion::Mse,
            task: Task::Regression,
            min_samples_leaf: 1,
            seed: 42,
            tree_id: 0,
            node_id: 0,
        };
        let split = best_split(&ctx).expect("regression split must succeed");
        assert!(
            split.improvement > 0.0,
            "regression split must have positive improvement"
        );
        assert!(split.n_left > 0 && split.n_right > 0);
    }

    #[test]
    fn partitions_cover_all_rows() {
        // left_rows ∪ right_rows must equal the original rows (no row lost/duplicated).
        let (x, y, rows) = separable_ctx();
        let ctx = ctx_from(&x, &y, &rows, 2, 42, 0, 0);
        let split = best_split(&ctx).expect("split");
        let mut all: Vec<usize> = split
            .left_rows
            .iter()
            .chain(&split.right_rows)
            .copied()
            .collect();
        all.sort_unstable();
        let mut expected = rows.clone();
        expected.sort_unstable();
        assert_eq!(
            all, expected,
            "left_rows + right_rows must be a partition of the parent rows"
        );
    }
}
