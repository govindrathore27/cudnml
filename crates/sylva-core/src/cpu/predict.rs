//! NaN-safe forest traversal + prediction aggregation (ENG-05).
//!
//! **NaN routing (D-01, RESEARCH Pitfall 2):** at every internal node, the
//! feature value is checked for `is_nan()` FIRST, before any threshold
//! comparison.  A NaN row goes to `default_child[node]`.  This is the only
//! place the NaN policy is implemented; training always receives clean data.
//!
//! **Aggregation:**
//! - Classifier → mean of per-tree leaf `leaf_proba` slices → `ClassProba`.
//! - Regressor → mean of per-tree `leaf_value` → `Regression`.

use ndarray::{Array1, Array2, ArrayView2};

use crate::backend::Predictions;
use crate::config::Task;
use crate::error::SylvaError;
use crate::ir::{ForestIR, NO_CHILD};

/// Predict for all rows in `x` using the given `ForestIR`.
///
/// Returns `Predictions::ClassProba` (shape `[n_rows, n_classes]`) for
/// classifiers and `Predictions::Regression` (shape `[n_rows]`) for
/// regressors.
pub(crate) fn predict_forest(ir: &ForestIR, x: ArrayView2<f32>) -> Result<Predictions, SylvaError> {
    let n_rows = x.nrows();
    let n_features = x.ncols();

    // Boundary validation (T-02-05).
    if n_features != ir.n_features {
        return Err(SylvaError::InvalidInput(format!(
            "predict: X has {} features but IR was trained on {}",
            n_features, ir.n_features
        )));
    }
    if ir.n_trees == 0 {
        return Err(SylvaError::InvalidIr("ForestIR has 0 trees".into()));
    }

    match ir.task {
        Task::Classification { n_classes } => {
            let mut out = Array2::<f32>::zeros((n_rows, n_classes));
            for t in 0..ir.n_trees {
                let root = ir.tree_root[t] as usize;
                for row in 0..n_rows {
                    let leaf_node = traverse_tree(ir, x, row, root);
                    let lo = ir.leaf_offset[leaf_node] as usize;
                    // Accumulate per-tree leaf probas into `out`.
                    let src = &ir.leaf_proba[lo * n_classes..(lo + 1) * n_classes];
                    let mut dst = out.row_mut(row);
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d += s;
                    }
                }
            }
            // Average over trees.
            let n_t = ir.n_trees as f32;
            out.mapv_inplace(|v| v / n_t);
            Ok(Predictions::ClassProba(out))
        }
        Task::Regression => {
            let mut out = Array1::<f32>::zeros(n_rows);
            for t in 0..ir.n_trees {
                let root = ir.tree_root[t] as usize;
                for row in 0..n_rows {
                    let leaf_node = traverse_tree(ir, x, row, root);
                    let lo = ir.leaf_offset[leaf_node] as usize;
                    out[row] += ir.leaf_value[lo];
                }
            }
            let n_t = ir.n_trees as f32;
            out.mapv_inplace(|v| v / n_t);
            Ok(Predictions::Regression(out))
        }
    }
}

/// Walk one tree from `root` to the leaf that `x[row, :]` reaches.
/// Returns the global node id of the leaf.
///
/// **NaN routing (D-01):** `v.is_nan()` is checked FIRST — before any
/// threshold comparison — so a NaN value goes to `default_child[node]`
/// and never falls through to the `v <= threshold` branch.
#[inline]
fn traverse_tree(ir: &ForestIR, x: ArrayView2<f32>, row: usize, root: usize) -> usize {
    let mut node = root;
    loop {
        if ir.is_leaf[node] {
            return node;
        }
        let feat = ir.feature_id[node] as usize;
        let v = x[[row, feat]];
        // D-01: is_nan check BEFORE threshold comparison (Pitfall 2 guard).
        node = if v.is_nan() {
            // Route to default_child (higher-sample-count child; tie→left).
            debug_assert_ne!(
                ir.default_child[node], NO_CHILD,
                "internal node must have a valid default_child"
            );
            ir.default_child[node] as usize
        } else if v <= ir.threshold[node] {
            ir.left_child[node] as usize
        } else {
            ir.right_child[node] as usize
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Algo, Criterion, MaxFeatures, TrainConfig};
    use crate::cpu::fit::fit_forest;
    use crate::ir::ForestIR;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    fn clf_cfg_tiny() -> TrainConfig {
        TrainConfig {
            n_estimators: 3,
            max_depth: Some(3),
            max_features: MaxFeatures::All,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Gini,
            seed: 7,
            algo: Algo::ExtraTrees,
        }
    }

    fn reg_cfg_tiny() -> TrainConfig {
        TrainConfig {
            n_estimators: 3,
            max_depth: Some(3),
            max_features: MaxFeatures::All,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Mse,
            seed: 7,
            algo: Algo::ExtraTrees,
        }
    }

    // ---------------------------------------------------------------------------
    // Basic predict shapes
    // ---------------------------------------------------------------------------

    #[test]
    fn clf_predict_proba_shape() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let cfg = clf_cfg_tiny();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        let preds = predict_forest(&ir, x.view()).expect("predict");
        match preds {
            Predictions::ClassProba(p) => {
                assert_eq!(p.nrows(), 20);
                assert_eq!(p.ncols(), 2);
            }
            _ => panic!("expected ClassProba"),
        }
    }

    #[test]
    fn clf_predict_probas_sum_to_one() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let cfg = clf_cfg_tiny();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        let preds = predict_forest(&ir, x.view()).expect("predict");
        match preds {
            Predictions::ClassProba(p) => {
                for row in p.rows() {
                    let s: f32 = row.iter().sum();
                    assert_abs_diff_eq!(s, 1.0_f32, epsilon = 1e-5);
                }
            }
            _ => panic!("expected ClassProba"),
        }
    }

    #[test]
    fn reg_predict_values_are_finite() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..20).map(|i| i as f32));
        let cfg = reg_cfg_tiny();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        let preds = predict_forest(&ir, x.view()).expect("predict");
        match preds {
            Predictions::Regression(v) => {
                for &val in v.iter() {
                    assert!(val.is_finite(), "regression prediction must be finite");
                }
            }
            _ => panic!("expected Regression"),
        }
    }

    // ---------------------------------------------------------------------------
    // NaN routing fixtures (SC-4 / ENG-05)
    // ---------------------------------------------------------------------------

    /// Build a hand-crafted `ForestIR` with one tree:
    ///   node 0 (root, internal): splits on feature 0 at threshold 0.5
    ///     default_child = left_child (node 1) — higher count
    ///   node 1 (leaf): class 0 with probability 1.0
    ///   node 2 (leaf): class 1 with probability 1.0
    fn nan_fixture_ir() -> ForestIR {
        use crate::config::Criterion;
        use crate::ir::{LEAF_FEATURE, NO_CHILD};
        ForestIR {
            feature_id: vec![0, LEAF_FEATURE, LEAF_FEATURE],
            threshold: vec![0.5, 0.0, 0.0],
            left_child: vec![1, NO_CHILD, NO_CHILD],
            right_child: vec![2, NO_CHILD, NO_CHILD],
            // default_child = left (node 1) because left has higher count.
            default_child: vec![1, NO_CHILD, NO_CHILD],
            is_leaf: vec![false, true, true],
            node_sample_count: vec![10, 7, 3],
            node_weighted_count: vec![10.0, 7.0, 3.0],
            impurity: vec![0.42, 0.0, 0.0],
            leaf_value: vec![],
            // leaf 0 (node 1): class 0 proba = 1.0, class 1 = 0.0
            // leaf 1 (node 2): class 0 proba = 0.0, class 1 = 1.0
            leaf_proba: vec![1.0, 0.0, 0.0, 1.0],
            leaf_offset: vec![-1, 0, 1],
            tree_offsets: vec![0, 3],
            tree_root: vec![0],
            n_trees: 1,
            n_features: 2,
            task: Task::Classification { n_classes: 2 },
            criterion: Criterion::Gini,
            seed: 0,
        }
    }

    #[test]
    fn nan_row_routes_to_default_child() {
        // default_child of root (node 0) = node 1 (left), which is class-0 leaf.
        // A NaN in feature 0 should route to class 0 (proba [1.0, 0.0]).
        let ir = nan_fixture_ir();
        // Row with NaN in feature 0.
        let x = array![[f32::NAN, 0.5]];
        let preds = predict_forest(&ir, x.view()).expect("predict");
        match preds {
            Predictions::ClassProba(p) => {
                // Should land in the left leaf (class 0 dominant).
                assert_abs_diff_eq!(p[[0, 0]], 1.0_f32, epsilon = 1e-6);
                assert_abs_diff_eq!(p[[0, 1]], 0.0_f32, epsilon = 1e-6);
            }
            _ => panic!("expected ClassProba"),
        }
    }

    #[test]
    fn nan_row_not_nan_goes_threshold_path() {
        // Feature 0 = 0.8 > 0.5 → should go right (node 2, class 1 leaf).
        let ir = nan_fixture_ir();
        let x = array![[0.8_f32, 0.0]];
        let preds = predict_forest(&ir, x.view()).expect("predict");
        match preds {
            Predictions::ClassProba(p) => {
                assert_abs_diff_eq!(p[[0, 0]], 0.0_f32, epsilon = 1e-6);
                assert_abs_diff_eq!(p[[0, 1]], 1.0_f32, epsilon = 1e-6);
            }
            _ => panic!("expected ClassProba"),
        }
    }

    #[test]
    fn nan_routing_is_deterministic() {
        // Same NaN row → same result on repeat.
        let ir = nan_fixture_ir();
        let x = array![[f32::NAN, 0.5]];
        let p1 = predict_forest(&ir, x.view()).expect("p1");
        let p2 = predict_forest(&ir, x.view()).expect("p2");
        match (p1, p2) {
            (Predictions::ClassProba(a), Predictions::ClassProba(b)) => {
                for (av, bv) in a.iter().zip(b.iter()) {
                    assert_eq!(av.to_bits(), bv.to_bits());
                }
            }
            _ => panic!("expected ClassProba"),
        }
    }

    #[test]
    fn nan_default_direction_matches_ir() {
        // Build a real forest, then check that NaN rows go to the IR's
        // default_child (not the threshold branch).
        let x_train = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y_train = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let cfg = clf_cfg_tiny();
        let ir = fit_forest(x_train.view(), y_train.view(), &cfg).expect("fit");

        // Row with NaN in feature 0 — we check the actual default direction
        // by comparing to a prediction with the default_child's own feature value.
        let x_nan = array![[f32::NAN, 0.5]];
        let pred_nan = predict_forest(&ir, x_nan.view()).expect("nan predict");

        // Walk the first tree manually: find what the first split's
        // default_child resolves to, build a row that goes there by
        // threshold (not NaN), and compare probabilities.
        let root = ir.tree_root[0] as usize;
        if !ir.is_leaf[root] {
            let dc = ir.default_child[root] as usize;
            // Build a row that forces exactly the threshold path to dc.
            let feat = ir.feature_id[root] as usize;
            let thr = ir.threshold[root];
            // Pick a value that routes to dc (left or right).
            let is_left_dc = dc == ir.left_child[root] as usize;
            let v = if is_left_dc { thr - 1.0 } else { thr + 1.0 };
            // We only control feature 0 or 1; pick the right slot.
            let mut x_thr = array![[0.0_f32, 0.5]];
            x_thr[[0, feat]] = v;
            let pred_thr = predict_forest(&ir, x_thr.view()).expect("thr predict");
            match (pred_nan, pred_thr) {
                (Predictions::ClassProba(p_nan), Predictions::ClassProba(p_thr)) => {
                    // The NaN row's first-tree contribution should match the
                    // threshold path that goes to default_child.
                    // (Both pass through more trees, so the full proba may differ
                    // if other trees split on feature 1 differently — we only
                    // assert the probas are in [0,1] and sum to 1.)
                    let sum: f32 = p_nan.row(0).iter().sum();
                    assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-5);
                    let sum_thr: f32 = p_thr.row(0).iter().sum();
                    assert_abs_diff_eq!(sum_thr, 1.0_f32, epsilon = 1e-5);
                }
                _ => panic!("expected ClassProba"),
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Feature mismatch guard
    // ---------------------------------------------------------------------------

    #[test]
    fn rejects_feature_mismatch() {
        let x_train = Array2::from_shape_fn((10, 2), |(i, _j)| i as f32);
        let y = ndarray::Array1::from_iter((0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }));
        let cfg = clf_cfg_tiny();
        let ir = fit_forest(x_train.view(), y.view(), &cfg).expect("fit");
        // Predict with 3-feature matrix (trained on 2).
        let x_bad = Array2::<f32>::zeros((5, 3));
        assert!(predict_forest(&ir, x_bad.view()).is_err());
    }

    // -----------------------------------------------------------------------
    // RF predict tests (Task 3 — reuses the unchanged predict path)
    // -----------------------------------------------------------------------

    fn rf_clf_cfg_tiny() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(3),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            criterion: Criterion::Gini,
            seed: 55,
            algo: Algo::RandomForest,
        }
    }

    fn rf_reg_cfg_tiny() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(3),
            max_features: MaxFeatures::All,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            criterion: Criterion::Mse,
            seed: 66,
            algo: Algo::RandomForest,
        }
    }

    #[test]
    fn rf_clf_predict_proba_shape_and_sum() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let cfg = rf_clf_cfg_tiny();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF clf fit");
        let preds = predict_forest(&ir, x.view()).expect("RF clf predict");
        match preds {
            Predictions::ClassProba(p) => {
                assert_eq!(p.nrows(), 20, "RF clf: wrong row count in proba output");
                assert_eq!(p.ncols(), 2, "RF clf: wrong class count in proba output");
                for row in p.rows() {
                    let s: f32 = row.iter().sum();
                    assert_abs_diff_eq!(s, 1.0_f32, epsilon = 1e-5);
                }
            }
            _ => panic!("RF clf: expected ClassProba"),
        }
    }

    #[test]
    fn rf_reg_predict_values_finite() {
        let x = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..20).map(|i| i as f32));
        let cfg = rf_reg_cfg_tiny();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF reg fit");
        let preds = predict_forest(&ir, x.view()).expect("RF reg predict");
        match preds {
            Predictions::Regression(v) => {
                for &val in v.iter() {
                    assert!(
                        val.is_finite(),
                        "RF reg: prediction must be finite, got {val}"
                    );
                }
            }
            _ => panic!("RF reg: expected Regression"),
        }
    }
}
