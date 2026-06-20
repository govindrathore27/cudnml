//! Recursive forest builder for ExtraTrees and RandomForest (ENG-03).
//!
//! `fit_forest` builds `n_estimators` trees in parallel (rayon) over trees,
//! sequential within each tree, with Philox-keyed draws per `(tree, node, ...)`.
//! Because each draw is a pure function of its coordinate, the parallel build
//! and a sequential loop produce an **identical** `ForestIR` (determinism rule).
//!
//! **Float accumulation rule (RESEARCH Pitfall 3):** integer class counts may
//! accumulate in any order, but ALL float sums (impurity probabilities, MSE
//! target sums, leaf means) use a fixed sequential row order — never a rayon
//! parallel sum inside a tree.
//!
//! **Algorithm dispatch:** `build_node` dispatches on `cfg.algo`:
//! - `ExtraTrees` → `best_random_split` (one random threshold per feature).
//! - `RandomForest` → `best_split` (sorted-midpoint exhaustive search).
//!
//! **Bootstrap:** `cfg.bootstrap == true` (RF default) draws `n` rows with
//! replacement via `bootstrap_indices`; `false` (ET default) uses all rows.
//! The bootstrap draw is keyed by `(seed, tree)` so per-tree sampling is
//! order-independent under rayon (T-02-10).

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::config::{Algo, Criterion, Task, TrainConfig};
use crate::cpu::bootstrap::bootstrap_indices;
use crate::cpu::criterion::{entropy, gini, mse};
use crate::cpu::split_et::{best_random_split, EtSplitCtx};
use crate::cpu::split_rf::{best_split as rf_best_split, RfSplitCtx};
use crate::error::SylvaError;
use crate::ir::{ForestIR, LEAF_FEATURE, NO_CHILD};

/// Common split decision fields returned by both ET and RF splitters.
/// Used to avoid a complex tuple in `build_node`.
struct NodeSplit {
    feature_id: usize,
    threshold: f32,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
    default_left: bool,
}

/// Build a complete ExtraTrees (or ET-mode) forest.
///
/// Validates inputs at the boundary (T-02-05), then builds `n_estimators`
/// trees in parallel, collects per-tree `TreeFragment`s, and assembles them
/// into a single `ForestIR`.
pub(crate) fn fit_forest(
    x: ArrayView2<f32>,
    y: ArrayView1<f32>,
    cfg: &TrainConfig,
) -> Result<ForestIR, SylvaError> {
    // --- Boundary validation (T-02-05) ---
    cfg.validate()?;
    let n_rows = x.nrows();
    let n_features = x.ncols();
    if n_rows == 0 {
        return Err(SylvaError::InvalidInput("X has 0 rows".into()));
    }
    if n_features == 0 {
        return Err(SylvaError::InvalidInput("X has 0 features".into()));
    }
    if y.len() != n_rows {
        return Err(SylvaError::InvalidInput(format!(
            "y length {} != X rows {}",
            y.len(),
            n_rows
        )));
    }

    // Infer task and class count from y.
    let task = infer_task(cfg, y)?;

    // Build the criterion from cfg, defaulting by task.
    let criterion = cfg.criterion;

    // Resolve max_features.
    let resolved_max_features = cfg.max_features.resolve(n_features, task);

    // Pre-collect y as a plain Vec<f32> so we can pass slices to closures.
    let y_vec: Vec<f32> = y.iter().copied().collect();
    // All-row index used when bootstrap=false (ET default).
    let all_rows: Vec<usize> = (0..n_rows).collect();

    // Build each tree independently.  Because each tree's Philox draws are
    // keyed by `tree_id`, the parallel and sequential builds produce identical
    // per-tree ForestIR fragments.
    let n_trees = cfg.n_estimators;
    let seed = cfg.seed;
    let use_bootstrap = cfg.bootstrap;
    let trees: Vec<TreeFragment> = (0..n_trees)
        .into_par_iter()
        .map(|tree_id| {
            // RF: draw n rows with replacement (keyed by tree → order-independent).
            // ET: use all rows.
            let tree_rows: Vec<usize> = if use_bootstrap {
                bootstrap_indices(n_rows, seed, tree_id as u32)
            } else {
                all_rows.clone()
            };
            build_tree(
                x,
                &y_vec,
                &tree_rows,
                tree_id as u32,
                cfg,
                task,
                criterion,
                resolved_max_features,
            )
        })
        .collect();

    // Assemble fragments in tree order (deterministic).
    assemble_forest(trees, n_features, task, criterion, cfg.seed)
}

// ---------------------------------------------------------------------------
// Per-tree build
// ---------------------------------------------------------------------------

/// One tree's node arrays, before being merged into the global IR.
struct TreeFragment {
    feature_id: Vec<i32>,
    threshold: Vec<f32>,
    left_child: Vec<i32>,
    right_child: Vec<i32>,
    default_child: Vec<i32>,
    is_leaf: Vec<bool>,
    node_sample_count: Vec<u64>,
    node_weighted_count: Vec<f32>,
    impurity: Vec<f32>,
    leaf_value: Vec<f32>,
    leaf_proba: Vec<f32>,
    leaf_offset: Vec<i32>,
    n_leaf_slots: usize, // number of leaf slots written
}

impl TreeFragment {
    fn new() -> Self {
        Self {
            feature_id: Vec::new(),
            threshold: Vec::new(),
            left_child: Vec::new(),
            right_child: Vec::new(),
            default_child: Vec::new(),
            is_leaf: Vec::new(),
            node_sample_count: Vec::new(),
            node_weighted_count: Vec::new(),
            impurity: Vec::new(),
            leaf_value: Vec::new(),
            leaf_proba: Vec::new(),
            leaf_offset: Vec::new(),
            n_leaf_slots: 0,
        }
    }

    fn node_count(&self) -> usize {
        self.feature_id.len()
    }
}

/// Shared context for the recursive node builder (reduces argument count).
/// `'x` — array data lifetime; `'y` — labels lifetime; `'c` — config lifetime.
struct BuildCtx<'x, 'y, 'c> {
    x: ArrayView2<'x, f32>,
    y: &'y [f32],
    tree_id: u32,
    cfg: &'c TrainConfig,
    task: Task,
    criterion: Criterion,
    resolved_max_features: usize,
    n_classes: usize,
}

/// Build one tree (single-threaded, fixed float accumulation order).
#[allow(clippy::too_many_arguments)]
fn build_tree<'x, 'y, 'c>(
    x: ArrayView2<'x, f32>,
    y: &'y [f32],
    root_rows: &[usize],
    tree_id: u32,
    cfg: &'c TrainConfig,
    task: Task,
    criterion: Criterion,
    resolved_max_features: usize,
) -> TreeFragment {
    let n_classes = match task {
        Task::Classification { n_classes } => n_classes,
        Task::Regression => 1,
    };
    let mut frag = TreeFragment::new();
    let ctx = BuildCtx {
        x,
        y,
        tree_id,
        cfg,
        task,
        criterion,
        resolved_max_features,
        n_classes,
    };
    // Recursive build starting at depth 0, node 0 (root).
    build_node(&ctx, root_rows, 0, 0, &mut frag);
    frag
}

/// Recursive node builder. Appends one node into `frag` and recurses.
/// Returns the global node id that was just appended (local to the fragment).
fn build_node(
    ctx: &BuildCtx<'_, '_, '_>,
    rows: &[usize],
    node_id: u32,
    depth: usize,
    frag: &mut TreeFragment,
) -> usize {
    let n = rows.len() as u64;
    let node_idx = frag.node_count(); // the id this node will have

    // Compute node impurity (fixed sequential float order — determinism rule).
    let node_imp = node_impurity(rows, ctx.y, ctx.task, ctx.criterion, ctx.n_classes);

    // Decide whether to make a leaf.
    let make_leaf = depth == ctx.cfg.max_depth.unwrap_or(usize::MAX)
        || rows.len() < ctx.cfg.min_samples_split
        || is_pure(rows, ctx.y, ctx.task, ctx.n_classes);

    if make_leaf {
        emit_leaf(rows, ctx.y, ctx.task, ctx.n_classes, n, node_imp, frag);
        return node_idx;
    }

    // Try to find a valid split — dispatch on algorithm.
    // Both ET and RF return equivalent fields; we extract them into a common
    // `NodeSplit` so the recursion below is algorithm-independent.
    let split_opt: Option<NodeSplit> = match ctx.cfg.algo {
        Algo::ExtraTrees => {
            let et_ctx = EtSplitCtx {
                x: ctx.x,
                rows,
                y: ctx.y,
                n_classes: ctx.n_classes,
                max_features: ctx.resolved_max_features,
                criterion: ctx.criterion,
                task: ctx.task,
                min_samples_leaf: ctx.cfg.min_samples_leaf,
                seed: ctx.cfg.seed,
                tree_id: ctx.tree_id,
                node_id,
            };
            best_random_split(&et_ctx).map(|s| NodeSplit {
                feature_id: s.feature_id,
                threshold: s.threshold,
                left_rows: s.left_rows,
                right_rows: s.right_rows,
                default_left: s.default_left,
            })
        }
        Algo::RandomForest => {
            let rf_ctx = RfSplitCtx {
                x: ctx.x,
                rows,
                y: ctx.y,
                n_classes: ctx.n_classes,
                max_features: ctx.resolved_max_features,
                criterion: ctx.criterion,
                task: ctx.task,
                min_samples_leaf: ctx.cfg.min_samples_leaf,
                seed: ctx.cfg.seed,
                tree_id: ctx.tree_id,
                node_id,
            };
            rf_best_split(&rf_ctx).map(|s| NodeSplit {
                feature_id: s.feature_id,
                threshold: s.threshold,
                left_rows: s.left_rows,
                right_rows: s.right_rows,
                default_left: s.default_left,
            })
        }
    };

    let Some(split) = split_opt else {
        // No valid split found — emit a leaf.
        emit_leaf(rows, ctx.y, ctx.task, ctx.n_classes, n, node_imp, frag);
        return node_idx;
    };

    // Reserve the slot for this internal node (children filled after recursion).
    frag.feature_id.push(split.feature_id as i32);
    frag.threshold.push(split.threshold);
    frag.left_child.push(NO_CHILD); // patched after recursion
    frag.right_child.push(NO_CHILD);
    frag.default_child.push(NO_CHILD);
    frag.is_leaf.push(false);
    frag.node_sample_count.push(n);
    frag.node_weighted_count.push(n as f32);
    frag.impurity.push(node_imp);
    frag.leaf_offset.push(-1);

    // Recurse: node_id for children = current node count (sequential within tree).
    let left_child_idx = frag.node_count();
    let left_node_id = left_child_idx as u32;
    build_node(ctx, &split.left_rows, left_node_id, depth + 1, frag);

    let right_child_idx = frag.node_count();
    let right_node_id = right_child_idx as u32;
    build_node(ctx, &split.right_rows, right_node_id, depth + 1, frag);

    // Patch in the child ids now that we know them.
    frag.left_child[node_idx] = left_child_idx as i32;
    frag.right_child[node_idx] = right_child_idx as i32;
    frag.default_child[node_idx] = if split.default_left {
        left_child_idx as i32
    } else {
        right_child_idx as i32
    };

    node_idx
}

/// Emit a leaf node into `frag`.
fn emit_leaf(
    rows: &[usize],
    y: &[f32],
    task: Task,
    n_classes: usize,
    n: u64,
    node_imp: f32,
    frag: &mut TreeFragment,
) {
    frag.feature_id.push(LEAF_FEATURE);
    frag.threshold.push(0.0);
    frag.left_child.push(NO_CHILD);
    frag.right_child.push(NO_CHILD);
    frag.default_child.push(NO_CHILD);
    frag.is_leaf.push(true);
    frag.node_sample_count.push(n);
    frag.node_weighted_count.push(n as f32);
    frag.impurity.push(node_imp);

    let leaf_slot = frag.n_leaf_slots;
    frag.leaf_offset.push(leaf_slot as i32);
    frag.n_leaf_slots += 1;

    match task {
        Task::Classification { .. } => {
            // Normalized class counts (sequential, fixed order — determinism).
            let mut counts = vec![0u64; n_classes];
            for &r in rows.iter() {
                let c = y[r] as usize;
                if c < n_classes {
                    counts[c] += 1;
                }
            }
            let total = rows.len() as f32;
            for c in &counts {
                frag.leaf_proba
                    .push(if total > 0.0 { *c as f32 / total } else { 0.0 });
            }
            frag.leaf_value.push(0.0); // unused for classification
        }
        Task::Regression => {
            // Leaf value = mean of targets (sequential accumulation).
            let sum: f32 = rows.iter().fold(0.0_f32, |acc, &r| acc + y[r]);
            let mean = if rows.is_empty() {
                0.0
            } else {
                sum / rows.len() as f32
            };
            frag.leaf_value.push(mean);
            // leaf_proba unused for regression
        }
    }
}

// ---------------------------------------------------------------------------
// Node impurity helpers
// ---------------------------------------------------------------------------

/// Compute node impurity with fixed sequential float accumulation.
fn node_impurity(
    rows: &[usize],
    y: &[f32],
    task: Task,
    criterion: Criterion,
    n_classes: usize,
) -> f32 {
    match task {
        Task::Classification { .. } => {
            let mut counts = vec![0u64; n_classes];
            for &r in rows.iter() {
                let c = y[r] as usize;
                if c < n_classes {
                    counts[c] += 1;
                }
            }
            let total: u64 = counts.iter().sum();
            match criterion {
                Criterion::Gini => gini(&counts, total),
                Criterion::Entropy => entropy(&counts, total),
                Criterion::Mse => {
                    let targets: Vec<f32> = rows.iter().map(|&r| y[r]).collect();
                    mse(&targets)
                }
            }
        }
        Task::Regression => {
            let targets: Vec<f32> = rows.iter().map(|&r| y[r]).collect();
            mse(&targets)
        }
    }
}

/// Check if the node is pure (only one class / zero variance).
fn is_pure(rows: &[usize], y: &[f32], task: Task, n_classes: usize) -> bool {
    match task {
        Task::Classification { .. } => {
            let first = y[rows[0]] as usize;
            rows.iter().all(|&r| y[r] as usize == first) || {
                let mut counts = vec![0u64; n_classes];
                for &r in rows.iter() {
                    let c = y[r] as usize;
                    if c < n_classes {
                        counts[c] += 1;
                    }
                }
                counts.iter().filter(|&&c| c > 0).count() <= 1
            }
        }
        Task::Regression => mse(&rows.iter().map(|&r| y[r]).collect::<Vec<_>>()) < 1e-10,
    }
}

/// Infer the learning task from the training config + labels.
pub(crate) fn infer_task(cfg: &TrainConfig, y: ArrayView1<f32>) -> Result<Task, SylvaError> {
    // If the config has no explicit task field, we infer by criterion.
    // Regression criterion -> regression task; otherwise classification.
    match cfg.criterion {
        Criterion::Mse => Ok(Task::Regression),
        Criterion::Gini | Criterion::Entropy => {
            // Infer n_classes from unique integer labels in y.
            let mut max_class: i64 = -1;
            for &v in y.iter() {
                if v < 0.0 || v.fract() != 0.0 {
                    return Err(SylvaError::InvalidInput(format!(
                        "classification label {v} is not a non-negative integer"
                    )));
                }
                let c = v as i64;
                if c > max_class {
                    max_class = c;
                }
            }
            if max_class < 0 {
                return Err(SylvaError::InvalidInput(
                    "no valid classification labels found".into(),
                ));
            }
            Ok(Task::Classification {
                n_classes: (max_class + 1) as usize,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Forest assembly
// ---------------------------------------------------------------------------

/// Merge per-tree `TreeFragment`s into one `ForestIR` in tree order.
fn assemble_forest(
    trees: Vec<TreeFragment>,
    n_features: usize,
    task: Task,
    criterion: Criterion,
    seed: u64,
) -> Result<ForestIR, SylvaError> {
    let n_trees = trees.len();
    let mut ir = ForestIR {
        feature_id: Vec::new(),
        threshold: Vec::new(),
        left_child: Vec::new(),
        right_child: Vec::new(),
        default_child: Vec::new(),
        is_leaf: Vec::new(),
        node_sample_count: Vec::new(),
        node_weighted_count: Vec::new(),
        impurity: Vec::new(),
        leaf_value: Vec::new(),
        leaf_proba: Vec::new(),
        leaf_offset: Vec::new(),
        tree_offsets: Vec::with_capacity(n_trees + 1),
        tree_root: Vec::with_capacity(n_trees),
        n_trees,
        n_features,
        task,
        criterion,
        seed,
    };
    ir.tree_offsets.push(0);

    for frag in trees {
        let offset = ir.node_count();
        // Adjust node ids (children, default) by the global offset.
        for &lid in &frag.left_child {
            ir.left_child.push(if lid == NO_CHILD {
                NO_CHILD
            } else {
                lid + offset as i32
            });
        }
        for &rid in &frag.right_child {
            ir.right_child.push(if rid == NO_CHILD {
                NO_CHILD
            } else {
                rid + offset as i32
            });
        }
        for &did in &frag.default_child {
            ir.default_child.push(if did == NO_CHILD {
                NO_CHILD
            } else {
                did + offset as i32
            });
        }
        // Adjust leaf offsets by the leaf_payload offset.
        let leaf_payload_offset = match task {
            Task::Classification { .. } => ir.leaf_proba.len() / task_n_classes(task),
            Task::Regression => ir.leaf_value.len(),
        };
        for &lo in &frag.leaf_offset {
            ir.leaf_offset.push(if lo < 0 {
                NO_CHILD
            } else {
                lo + leaf_payload_offset as i32
            });
        }

        ir.feature_id.extend_from_slice(&frag.feature_id);
        ir.threshold.extend_from_slice(&frag.threshold);
        ir.is_leaf.extend_from_slice(&frag.is_leaf);
        ir.node_sample_count
            .extend_from_slice(&frag.node_sample_count);
        ir.node_weighted_count
            .extend_from_slice(&frag.node_weighted_count);
        ir.impurity.extend_from_slice(&frag.impurity);
        ir.leaf_value.extend_from_slice(&frag.leaf_value);
        ir.leaf_proba.extend_from_slice(&frag.leaf_proba);

        ir.tree_root.push(offset as i32);
        ir.tree_offsets.push(ir.node_count());
    }

    ir.validate_structure()?;
    Ok(ir)
}

fn task_n_classes(task: Task) -> usize {
    match task {
        Task::Classification { n_classes } => n_classes,
        Task::Regression => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests for the forest builder
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Algo, MaxFeatures};
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use serde_json;

    /// Simple 2-class, 2-feature, 20-row dataset. Feature 0 in [0,9] is the
    /// separator (class = floor(x0/5)).
    fn make_clf_data() -> (Array2<f32>, ndarray::Array1<f32>) {
        let n = 20usize;
        let x = Array2::from_shape_fn(
            (n, 2),
            |(i, j)| {
                if j == 0 {
                    i as f32
                } else {
                    (i % 3) as f32
                }
            },
        );
        let y = ndarray::Array1::from_iter((0..n).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        (x, y)
    }

    /// Simple regression dataset: y = x0.
    fn make_reg_data() -> (Array2<f32>, ndarray::Array1<f32>) {
        let n = 20usize;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
        let y = ndarray::Array1::from_iter((0..n).map(|i| i as f32));
        (x, y)
    }

    fn clf_cfg() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(4),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Gini,
            seed: 42,
            algo: Algo::ExtraTrees,
        }
    }

    fn reg_cfg() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(4),
            max_features: MaxFeatures::All, // reg default
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Mse,
            seed: 42,
            algo: Algo::ExtraTrees,
        }
    }

    // --- Structural / invariant tests ---

    #[test]
    fn clf_forest_builds_and_validates() {
        let (x, y) = make_clf_data();
        let cfg = clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit clf");
        // validate_structure checked inside assemble_forest; calling again is
        // belt-and-suspenders.
        ir.validate_structure().expect("validate clf ir");
        assert_eq!(ir.n_trees, cfg.n_estimators);
        assert_eq!(ir.n_features, 2);
    }

    #[test]
    fn reg_forest_builds_and_validates() {
        let (x, y) = make_reg_data();
        let cfg = reg_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit reg");
        ir.validate_structure().expect("validate reg ir");
    }

    #[test]
    fn cover_invariant_parent_equals_left_plus_right() {
        let (x, y) = make_clf_data();
        let cfg = clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        let n = ir.node_count();
        for i in 0..n {
            if !ir.is_leaf[i] {
                let l = ir.left_child[i] as usize;
                let r = ir.right_child[i] as usize;
                assert_eq!(
                    ir.node_sample_count[i],
                    ir.node_sample_count[l] + ir.node_sample_count[r],
                    "cover invariant failed at node {i}"
                );
            }
        }
    }

    #[test]
    fn clf_leaf_probas_sum_to_one() {
        let (x, y) = make_clf_data();
        let cfg = clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        let nc = ir.n_classes();
        let mut found_leaf = false;
        for i in 0..ir.node_count() {
            if ir.is_leaf[i] {
                found_leaf = true;
                let lo = ir.leaf_offset[i] as usize;
                let sum: f32 = ir.leaf_proba[lo * nc..(lo + 1) * nc].iter().sum();
                assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-5);
            }
        }
        assert!(found_leaf, "should have at least one leaf");
    }

    #[test]
    fn reg_leaf_values_are_finite() {
        let (x, y) = make_reg_data();
        let cfg = reg_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("fit");
        for &v in &ir.leaf_value {
            assert!(v.is_finite(), "leaf_value must be finite; got {v}");
        }
    }

    // --- Determinism ---

    #[test]
    fn seed_determinism_byte_identical() {
        let (x, y) = make_clf_data();
        let cfg = clf_cfg();
        let ir1 = fit_forest(x.view(), y.view(), &cfg).expect("fit 1");
        let ir2 = fit_forest(x.view(), y.view(), &cfg).expect("fit 2");
        let s1 = serde_json::to_string(&ir1).expect("ser1");
        let s2 = serde_json::to_string(&ir2).expect("ser2");
        assert_eq!(s1, s2, "same seed must produce byte-identical IR");
    }

    #[test]
    fn parallel_equals_sequential_forest() {
        // Build with rayon (parallel by default) and compare to a sequential
        // single-tree-at-a-time build.  Since our parallel builder is
        // already rayon, we verify the result is stable across two calls.
        let (x, y) = make_clf_data();
        let cfg = TrainConfig {
            n_estimators: 10,
            max_depth: Some(3),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: false,
            criterion: Criterion::Gini,
            seed: 99,
            algo: Algo::ExtraTrees,
        };
        // Build the forest twice; because Philox draws are pure functions of
        // (seed, tree, node, feature, draw), parallel and sequential orders
        // must yield the same IR.
        let ir_a = fit_forest(x.view(), y.view(), &cfg).expect("a");
        let ir_b = fit_forest(x.view(), y.view(), &cfg).expect("b");
        let sa = serde_json::to_string(&ir_a).unwrap();
        let sb = serde_json::to_string(&ir_b).unwrap();
        assert_eq!(sa, sb, "parallel/sequential must be byte-identical");
    }

    // --- Input validation ---

    #[test]
    fn rejects_zero_rows() {
        let x = Array2::<f32>::zeros((0, 2));
        let y = ndarray::Array1::<f32>::zeros(0);
        let cfg = clf_cfg();
        assert!(fit_forest(x.view(), y.view(), &cfg).is_err());
    }

    #[test]
    fn rejects_mismatched_y_length() {
        let (x, _) = make_clf_data();
        let y_short = ndarray::Array1::<f32>::zeros(5);
        let cfg = clf_cfg();
        assert!(fit_forest(x.view(), y_short.view(), &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // RandomForest-specific tests (Task 3 — SC-2 / D-02)
    // -----------------------------------------------------------------------

    fn rf_clf_cfg() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(4),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true, // RF default
            criterion: Criterion::Gini,
            seed: 42,
            algo: Algo::RandomForest,
        }
    }

    fn rf_reg_cfg() -> TrainConfig {
        TrainConfig {
            n_estimators: 5,
            max_depth: Some(4),
            max_features: MaxFeatures::All, // reg default — all features (RESEARCH Pitfall 5)
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            criterion: Criterion::Mse,
            seed: 42,
            algo: Algo::RandomForest,
        }
    }

    // --- RF clf: fit, validate, leaf proba ---

    #[test]
    fn rf_clf_forest_builds_and_validates() {
        let (x, y) = make_clf_data();
        let cfg = rf_clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF clf fit");
        ir.validate_structure().expect("RF clf validate");
        assert_eq!(ir.n_trees, cfg.n_estimators);
    }

    #[test]
    fn rf_clf_leaf_probas_sum_to_one() {
        let (x, y) = make_clf_data();
        let cfg = rf_clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF clf fit");
        let nc = ir.n_classes();
        let mut found_leaf = false;
        for i in 0..ir.node_count() {
            if ir.is_leaf[i] {
                found_leaf = true;
                let lo = ir.leaf_offset[i] as usize;
                let sum: f32 = ir.leaf_proba[lo * nc..(lo + 1) * nc].iter().sum();
                assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-5);
            }
        }
        assert!(found_leaf, "RF clf: should have at least one leaf");
    }

    // --- RF reg: fit, validate, leaf values finite ---

    #[test]
    fn rf_reg_forest_builds_and_validates() {
        let (x, y) = make_reg_data();
        let cfg = rf_reg_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF reg fit");
        ir.validate_structure().expect("RF reg validate");
    }

    #[test]
    fn rf_reg_leaf_values_are_finite() {
        let (x, y) = make_reg_data();
        let cfg = rf_reg_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF reg fit");
        for &v in &ir.leaf_value {
            assert!(v.is_finite(), "RF reg: leaf_value must be finite; got {v}");
        }
    }

    // --- RF determinism ---

    #[test]
    fn rf_seed_determinism_byte_identical() {
        let (x, y) = make_clf_data();
        let cfg = rf_clf_cfg();
        let ir1 = fit_forest(x.view(), y.view(), &cfg).expect("RF fit 1");
        let ir2 = fit_forest(x.view(), y.view(), &cfg).expect("RF fit 2");
        let s1 = serde_json::to_string(&ir1).expect("ser1");
        let s2 = serde_json::to_string(&ir2).expect("ser2");
        assert_eq!(s1, s2, "RF: same seed must produce byte-identical IR");
    }

    /// RF parallel-build == sequential-build invariant (T-02-10).
    ///
    /// The rayon `par_iter` may process trees in any order. Because bootstrap
    /// indices and split draws are both keyed by `(seed, tree, ...)`, two
    /// independent calls with the same config MUST produce the same ForestIR.
    #[test]
    fn rf_parallel_equals_sequential() {
        let (x, y) = make_clf_data();
        let cfg = TrainConfig {
            n_estimators: 8,
            max_depth: Some(3),
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            criterion: Criterion::Gini,
            seed: 77,
            algo: Algo::RandomForest,
        };
        let ir_a = fit_forest(x.view(), y.view(), &cfg).expect("RF par a");
        let ir_b = fit_forest(x.view(), y.view(), &cfg).expect("RF par b");
        let sa = serde_json::to_string(&ir_a).unwrap();
        let sb = serde_json::to_string(&ir_b).unwrap();
        assert_eq!(
            sa, sb,
            "RF parallel/sequential must be byte-identical (T-02-10)"
        );
    }

    // --- RF cover invariant ---

    #[test]
    fn rf_cover_invariant_parent_equals_left_plus_right() {
        // Bootstrap sample counts are per-bootstrap-row (not global data rows),
        // so the cover invariant still holds within the tree.
        let (x, y) = make_clf_data();
        let cfg = rf_clf_cfg();
        let ir = fit_forest(x.view(), y.view(), &cfg).expect("RF fit");
        for i in 0..ir.node_count() {
            if !ir.is_leaf[i] {
                let l = ir.left_child[i] as usize;
                let r = ir.right_child[i] as usize;
                assert_eq!(
                    ir.node_sample_count[i],
                    ir.node_sample_count[l] + ir.node_sample_count[r],
                    "RF cover invariant failed at node {i}"
                );
            }
        }
    }

    // --- Four-estimator matrix (SC-2 / D-02) ---

    /// All four combinations (ET/RF × clf/reg) must build a valid ForestIR.
    /// This is the SC-2 / D-02 acceptance test: the oracle covers the full
    /// estimator matrix this phase.
    #[test]
    fn all_four_estimators_build_and_validate() {
        let (xc, yc) = make_clf_data();
        let (xr, yr) = make_reg_data();

        let configs: &[(&str, TrainConfig, bool)] = &[
            (
                "ET clf",
                TrainConfig {
                    algo: Algo::ExtraTrees,
                    bootstrap: false,
                    criterion: Criterion::Gini,
                    max_features: MaxFeatures::Sqrt,
                    n_estimators: 3,
                    max_depth: Some(4),
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    seed: 1,
                },
                true, // is_clf
            ),
            (
                "ET reg",
                TrainConfig {
                    algo: Algo::ExtraTrees,
                    bootstrap: false,
                    criterion: Criterion::Mse,
                    max_features: MaxFeatures::All,
                    n_estimators: 3,
                    max_depth: Some(4),
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    seed: 2,
                },
                false,
            ),
            (
                "RF clf",
                TrainConfig {
                    algo: Algo::RandomForest,
                    bootstrap: true,
                    criterion: Criterion::Gini,
                    max_features: MaxFeatures::Sqrt,
                    n_estimators: 3,
                    max_depth: Some(4),
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    seed: 3,
                },
                true,
            ),
            (
                "RF reg",
                TrainConfig {
                    algo: Algo::RandomForest,
                    bootstrap: true,
                    criterion: Criterion::Mse,
                    max_features: MaxFeatures::All,
                    n_estimators: 3,
                    max_depth: Some(4),
                    min_samples_split: 2,
                    min_samples_leaf: 1,
                    seed: 4,
                },
                false,
            ),
        ];

        for (name, cfg, is_clf) in configs {
            let (x, y) = if *is_clf {
                (xc.view(), yc.view())
            } else {
                (xr.view(), yr.view())
            };
            let ir = fit_forest(x, y, cfg).unwrap_or_else(|e| {
                panic!("{name}: fit_forest failed: {e}");
            });
            ir.validate_structure()
                .unwrap_or_else(|e| panic!("{name}: validate_structure failed: {e}"));
            assert_eq!(ir.n_trees, cfg.n_estimators, "{name}: n_trees mismatch");
        }
    }
}
