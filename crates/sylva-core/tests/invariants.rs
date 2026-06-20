//! Property-based invariants over ET/RF clf+reg (EST-07 property side).
//!
//! Uses `proptest` to generate small random `(TrainConfig, X, y)` cases across
//! all four estimators (ExtraTrees/RandomForest × clf/reg). For each generated
//! case the `CpuBackend` is fit and the resulting `ForestIR` is validated for
//! structural, leaf, sample-count-cover, and serialization invariants.
//!
//! **Runtime budget:** shapes are bounded (n≤100, d≤8, trees≤6, depth≤5) and
//! proptest is configured to ≤32 cases per test to stay well within the
//! ~90s latency target in CI.

use ndarray::{Array1, Array2};
use proptest::prelude::*;
use sylva_core::{
    config::{Algo, Criterion, MaxFeatures, Task, TrainConfig},
    cpu::CpuBackend,
    ir::{ForestIR, LEAF_FEATURE, NO_CHILD},
    Backend,
};

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

/// Small classification dataset: rows 0..n/2 are class 0; n/2..n are class 1.
fn make_clf_dataset(n: usize, d: usize, seed_shift: u64) -> (Array2<f32>, Array1<f32>) {
    // Use the row index + seed_shift as a simple deterministic source.
    let x = Array2::from_shape_fn((n, d), |(i, j)| {
        ((i as u64 * 7 + j as u64 * 13 + seed_shift * 31) % 100) as f32 * 0.1
    });
    let y = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
    (x, y)
}

/// Small regression dataset: y = feature-0 value.
fn make_reg_dataset(n: usize, d: usize, seed_shift: u64) -> (Array2<f32>, Array1<f32>) {
    let x = Array2::from_shape_fn((n, d), |(i, j)| {
        ((i as u64 * 11 + j as u64 * 17 + seed_shift * 29) % 100) as f32 * 0.1
    });
    let y = x.column(0).to_owned();
    (x, y)
}

/// Small 3-class dataset (label = row index mod 3).
fn make_clf3_dataset(n: usize, d: usize, seed_shift: u64) -> (Array2<f32>, Array1<f32>) {
    let x = Array2::from_shape_fn((n, d), |(i, j)| {
        ((i as u64 * 5 + j as u64 * 19 + seed_shift * 37) % 100) as f32 * 0.1
    });
    let y = Array1::from_iter((0..n).map(|i| (i % 3) as f32));
    (x, y)
}

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

fn clf_config(algo: Algo, n_trees: usize, depth: usize, seed: u64, bootstrap: bool) -> TrainConfig {
    TrainConfig {
        n_estimators: n_trees,
        max_depth: Some(depth),
        max_features: MaxFeatures::Sqrt,
        min_samples_split: 2,
        min_samples_leaf: 1,
        bootstrap,
        criterion: Criterion::Gini,
        seed,
        algo,
    }
}

fn reg_config(algo: Algo, n_trees: usize, depth: usize, seed: u64, bootstrap: bool) -> TrainConfig {
    TrainConfig {
        n_estimators: n_trees,
        max_depth: Some(depth),
        max_features: MaxFeatures::All,
        min_samples_split: 2,
        min_samples_leaf: 1,
        bootstrap,
        criterion: Criterion::Mse,
        seed,
        algo,
    }
}

// ---------------------------------------------------------------------------
// Invariant assertion helpers
// ---------------------------------------------------------------------------

/// Cover partition: `node_sample_count[node] == L + R` for all internal nodes.
/// This is the algebraic form of "left ∪ right == parent, left ∩ right == ∅"
/// on clean (no-NaN) training data.
fn assert_cover_partition(ir: &ForestIR) {
    for i in 0..ir.node_count() {
        if !ir.is_leaf[i] {
            let l = ir.left_child[i] as usize;
            let r = ir.right_child[i] as usize;
            assert_eq!(
                ir.node_sample_count[i],
                ir.node_sample_count[l] + ir.node_sample_count[r],
                "cover-partition failed at internal node {i}: \
                 parent={} left={} right={}",
                ir.node_sample_count[i],
                ir.node_sample_count[l],
                ir.node_sample_count[r],
            );
        }
    }
}

/// Leaf probabilities (clf): every leaf's proba slice ∈ [0,1] and sums to 1.
fn assert_leaf_proba_valid(ir: &ForestIR) {
    let nc = ir.n_classes();
    for i in 0..ir.node_count() {
        if ir.is_leaf[i] {
            let lo = ir.leaf_offset[i] as usize;
            let slice = &ir.leaf_proba[lo * nc..(lo + 1) * nc];
            for (idx, &p) in slice.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&p),
                    "leaf {i} proba[{idx}]={p} outside [0, 1]"
                );
            }
            let sum: f32 = slice.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "leaf {i} proba sum={sum} != 1.0 (eps=1e-4)"
            );
        }
    }
}

/// Leaf values (reg): every leaf value is finite.
fn assert_leaf_values_finite(ir: &ForestIR) {
    for (idx, &v) in ir.leaf_value.iter().enumerate() {
        assert!(v.is_finite(), "leaf_value[{idx}]={v} is not finite");
    }
}

/// All sample counts ≥ 1 (no zero-sample node should exist).
fn assert_sample_counts_positive(ir: &ForestIR) {
    for (i, &sc) in ir.node_sample_count.iter().enumerate() {
        assert!(sc >= 1, "node {i} has node_sample_count=0");
    }
}

/// `default_child` correctness: must point to the higher-count child; tie → left.
fn assert_default_child(ir: &ForestIR) {
    for i in 0..ir.node_count() {
        if !ir.is_leaf[i] {
            let l = ir.left_child[i] as usize;
            let r = ir.right_child[i] as usize;
            let d = ir.default_child[i] as usize;
            let expected_default = if ir.node_sample_count[l] >= ir.node_sample_count[r] {
                l
            } else {
                r
            };
            assert_eq!(
                d, expected_default,
                "default_child at node {i}: got {d}, expected {expected_default} \
                 (left_count={} right_count={})",
                ir.node_sample_count[l], ir.node_sample_count[r],
            );
        }
    }
}

/// `deserialize(serialize(ir)) == ir` (structural equality).
fn assert_serde_round_trip(ir: &ForestIR) {
    let json = serde_json::to_string(ir).expect("serialize ForestIR");
    let back: ForestIR = serde_json::from_str(&json).expect("deserialize ForestIR");
    assert_eq!(
        ir, &back,
        "serde round-trip: deserialized value differs from original"
    );
}

/// Leaf sentinels: leaves must have NO_CHILD children and LEAF_FEATURE feature_id.
fn assert_leaf_sentinels(ir: &ForestIR) {
    for i in 0..ir.node_count() {
        if ir.is_leaf[i] {
            assert_eq!(
                ir.feature_id[i], LEAF_FEATURE,
                "leaf {i} feature_id != LEAF_FEATURE"
            );
            assert_eq!(
                ir.left_child[i], NO_CHILD,
                "leaf {i} left_child != NO_CHILD"
            );
            assert_eq!(
                ir.right_child[i], NO_CHILD,
                "leaf {i} right_child != NO_CHILD"
            );
        }
    }
}

/// Full invariant suite for a classification ForestIR.
fn check_clf_invariants(ir: &ForestIR) {
    ir.validate_structure()
        .expect("validate_structure must pass for clf ForestIR");
    assert_cover_partition(ir);
    assert_leaf_proba_valid(ir);
    assert_sample_counts_positive(ir);
    assert_default_child(ir);
    assert_serde_round_trip(ir);
    assert_leaf_sentinels(ir);
}

/// Full invariant suite for a regression ForestIR.
fn check_reg_invariants(ir: &ForestIR) {
    ir.validate_structure()
        .expect("validate_structure must pass for reg ForestIR");
    assert_cover_partition(ir);
    assert_leaf_values_finite(ir);
    assert_sample_counts_positive(ir);
    assert_default_child(ir);
    assert_serde_round_trip(ir);
    assert_leaf_sentinels(ir);
}

// ---------------------------------------------------------------------------
// Proptest-driven invariant tests
// ---------------------------------------------------------------------------

proptest! {
    // 32 cases per test: fast CI, meaningful coverage with proptest's shrinking.
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// ExtraTrees classifier: full structural + leaf + cover + round-trip invariants.
    #[test]
    fn et_clf_invariants(
        n     in 20usize..=100,
        d     in 2usize..=8,
        trees in 2usize..=6,
        seed  in 0u64..=9999,
    ) {
        let backend = CpuBackend;
        let (x, y) = make_clf_dataset(n, d, seed);
        let cfg = clf_config(Algo::ExtraTrees, trees, 5, seed, false);
        let ir = backend.fit(x.view(), y.view(), &cfg)
            .expect("ET clf fit must succeed");
        check_clf_invariants(&ir);
    }

    /// ExtraTrees regressor: full structural + leaf + cover + round-trip invariants.
    #[test]
    fn et_reg_invariants(
        n     in 20usize..=80,
        d     in 2usize..=6,
        trees in 2usize..=5,
        seed  in 0u64..=9999,
    ) {
        let backend = CpuBackend;
        let (x, y) = make_reg_dataset(n, d, seed);
        let cfg = reg_config(Algo::ExtraTrees, trees, 5, seed, false);
        let ir = backend.fit(x.view(), y.view(), &cfg)
            .expect("ET reg fit must succeed");
        check_reg_invariants(&ir);
    }

    /// RandomForest classifier: full structural + leaf + cover + round-trip invariants.
    #[test]
    fn rf_clf_invariants(
        n     in 20usize..=80,
        d     in 2usize..=6,
        trees in 2usize..=5,
        seed  in 0u64..=9999,
    ) {
        let backend = CpuBackend;
        let (x, y) = make_clf_dataset(n, d, seed);
        let cfg = clf_config(Algo::RandomForest, trees, 5, seed, true);
        let ir = backend.fit(x.view(), y.view(), &cfg)
            .expect("RF clf fit must succeed");
        check_clf_invariants(&ir);
    }

    /// RandomForest regressor: full structural + leaf + cover + round-trip invariants.
    #[test]
    fn rf_reg_invariants(
        n     in 20usize..=80,
        d     in 2usize..=6,
        trees in 2usize..=5,
        seed  in 0u64..=9999,
    ) {
        let backend = CpuBackend;
        let (x, y) = make_reg_dataset(n, d, seed);
        let cfg = reg_config(Algo::RandomForest, trees, 5, seed, true);
        let ir = backend.fit(x.view(), y.view(), &cfg)
            .expect("RF reg fit must succeed");
        check_reg_invariants(&ir);
    }

    /// Multi-class (3-class) ET clf: leaf_proba slices must sum to 1 per class.
    #[test]
    fn et_multiclass_clf_invariants(
        n     in 30usize..=80,
        d     in 2usize..=6,
        trees in 2usize..=4,
        seed  in 0u64..=999,
    ) {
        let backend = CpuBackend;
        let (x, y) = make_clf3_dataset(n, d, seed);
        let cfg = clf_config(Algo::ExtraTrees, trees, 4, seed, false);
        let ir = backend.fit(x.view(), y.view(), &cfg)
            .expect("ET multiclass clf fit must succeed");
        check_clf_invariants(&ir);
    }

    /// Seed determinism (byte-identical) across all four estimators.
    #[test]
    fn seed_determinism_all_estimators(
        algo_idx in 0usize..4,
        n        in 20usize..=60,
        seed     in 0u64..=999,
    ) {
        let backend = CpuBackend;
        let (algo, bootstrap, is_reg) = match algo_idx {
            0 => (Algo::ExtraTrees,   false, false),
            1 => (Algo::ExtraTrees,   false, true),
            2 => (Algo::RandomForest, true,  false),
            _ => (Algo::RandomForest, true,  true),
        };
        let (cfg, x, y) = if is_reg {
            let (x, y) = make_reg_dataset(n, 3, seed);
            (reg_config(algo, 3, 4, seed, bootstrap), x, y)
        } else {
            let (x, y) = make_clf_dataset(n, 3, seed);
            (clf_config(algo, 3, 4, seed, bootstrap), x, y)
        };
        let ir1 = backend.fit(x.view(), y.view(), &cfg).expect("fit 1");
        let ir2 = backend.fit(x.view(), y.view(), &cfg).expect("fit 2");
        let s1 = serde_json::to_string(&ir1).expect("ser 1");
        let s2 = serde_json::to_string(&ir2).expect("ser 2");
        assert_eq!(
            s1, s2,
            "algo_idx={algo_idx} seed={seed}: same-seed fits must be byte-identical"
        );
    }
}

// ---------------------------------------------------------------------------
// Fixed (non-proptest) sanity tests
// ---------------------------------------------------------------------------

/// Deterministic ET clf fixture: confirms all invariants on a known dataset.
#[test]
fn et_clf_fixed_invariants() {
    let n = 30usize;
    let x = Array2::from_shape_fn(
        (n, 2),
        |(i, j)| if j == 0 { i as f32 } else { (i % 3) as f32 },
    );
    let y = Array1::from_iter((0..n).map(|i| if i < 15 { 0.0 } else { 1.0 }));
    let cfg = clf_config(Algo::ExtraTrees, 4, 4, 42, false);
    let ir = CpuBackend
        .fit(x.view(), y.view(), &cfg)
        .expect("ET clf fixed fit");
    check_clf_invariants(&ir);
}

/// Deterministic RF reg fixture: confirms all invariants on a known dataset.
#[test]
fn rf_reg_fixed_invariants() {
    let n = 30usize;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { i as f32 } else { 0.5 });
    let y = Array1::from_iter((0..n).map(|i| i as f32));
    let cfg = reg_config(Algo::RandomForest, 4, 4, 99, true);
    let ir = CpuBackend
        .fit(x.view(), y.view(), &cfg)
        .expect("RF reg fixed fit");
    check_reg_invariants(&ir);
}

/// Verify that the task stored in the IR matches the configured criterion.
#[test]
fn ir_task_matches_criterion() {
    let n = 20usize;
    let x = Array2::from_shape_fn((n, 2), |(i, _)| i as f32);
    let y_clf = Array1::from_iter((0..n).map(|i| (i % 2) as f32));
    let y_reg = Array1::from_iter((0..n).map(|i| i as f32));
    let base_clf = clf_config(Algo::ExtraTrees, 2, 3, 1, false);
    let base_reg = reg_config(Algo::ExtraTrees, 2, 3, 1, false);
    let ir_clf = CpuBackend
        .fit(x.view(), y_clf.view(), &base_clf)
        .expect("clf fit");
    let ir_reg = CpuBackend
        .fit(x.view(), y_reg.view(), &base_reg)
        .expect("reg fit");
    assert!(
        matches!(ir_clf.task, Task::Classification { .. }),
        "clf task must be Classification, got {:?}",
        ir_clf.task
    );
    assert_eq!(ir_reg.task, Task::Regression, "reg task must be Regression");
}
