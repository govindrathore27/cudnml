//! Byte-identical seed determinism + rayon parallel==sequential order-independence
//! (EST-07 determinism side; ENG-04 "exact" bit-level contract).
//!
//! These tests assert:
//!
//! 1. **Same-seed byte-identity:** two `CpuBackend::fit` runs with the same
//!    `(seed, cfg, data)` produce serialized `ForestIR` JSON that is byte-equal
//!    (exact string equality, NOT `approx`/allclose).
//!
//! 2. **Parallel == sequential:** a forest built with the default rayon parallel
//!    path and a second built via a 1-thread rayon pool (forcing sequential tree
//!    order) produce byte-identical JSON — proving that Philox counter-keying makes
//!    tree order irrelevant to the result.
//!
//! Both invariants hold for all four estimators: ET/RF × clf/reg.
//!
//! These tests guard the Phase-4 GPU contract: the CUDA backend will be held to
//! produce byte-identical serialized ForestIR for any given seed + config.

use ndarray::{Array1, Array2};
use sylva_core::{
    config::{Algo, Criterion, MaxFeatures, TrainConfig},
    cpu::CpuBackend,
    Backend,
};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// A 2-class, 3-feature, 40-row dataset with separable structure.
fn clf_data() -> (Array2<f32>, Array1<f32>) {
    let n = 40usize;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => i as f32,
        1 => (i % 5) as f32,
        _ => (i % 7) as f32 * 0.5,
    });
    let y = Array1::from_iter((0..n).map(|i| if i < 20 { 0.0 } else { 1.0 }));
    (x, y)
}

/// A regression dataset: y = x0 + 0.1 * x1.
fn reg_data() -> (Array2<f32>, Array1<f32>) {
    let n = 40usize;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => i as f32 * 0.25,
        1 => (i % 7) as f32,
        _ => 1.0,
    });
    let y = Array1::from_iter((0..n).map(|i| i as f32 * 0.25 + (i % 7) as f32 * 0.1));
    (x, y)
}

fn et_clf_cfg(seed: u64) -> TrainConfig {
    TrainConfig {
        n_estimators: 10,
        max_depth: Some(5),
        max_features: MaxFeatures::Sqrt,
        min_samples_split: 2,
        min_samples_leaf: 1,
        bootstrap: false,
        criterion: Criterion::Gini,
        seed,
        algo: Algo::ExtraTrees,
    }
}

fn et_reg_cfg(seed: u64) -> TrainConfig {
    TrainConfig {
        criterion: Criterion::Mse,
        max_features: MaxFeatures::All,
        algo: Algo::ExtraTrees,
        bootstrap: false,
        ..et_clf_cfg(seed)
    }
}

fn rf_clf_cfg(seed: u64) -> TrainConfig {
    TrainConfig {
        bootstrap: true,
        algo: Algo::RandomForest,
        ..et_clf_cfg(seed)
    }
}

fn rf_reg_cfg(seed: u64) -> TrainConfig {
    TrainConfig {
        criterion: Criterion::Mse,
        max_features: MaxFeatures::All,
        bootstrap: true,
        algo: Algo::RandomForest,
        ..et_clf_cfg(seed)
    }
}

// ---------------------------------------------------------------------------
// Helper: fit twice and assert byte-identical JSON
// ---------------------------------------------------------------------------

fn assert_same_seed_byte_identical(
    x: &Array2<f32>,
    y: &Array1<f32>,
    cfg: &TrainConfig,
    label: &str,
) {
    let backend = CpuBackend;
    let ir1 = backend.fit(x.view(), y.view(), cfg).expect("fit 1");
    let ir2 = backend.fit(x.view(), y.view(), cfg).expect("fit 2");
    let s1 = serde_json::to_string(&ir1).expect("ser 1");
    let s2 = serde_json::to_string(&ir2).expect("ser 2");
    assert_eq!(
        s1, s2,
        "{label}: same-seed fits must produce byte-identical serialized ForestIR \
         (exact string equality — NOT allclose)"
    );
}

// ---------------------------------------------------------------------------
// Helper: fit with rayon default vs forced 1-thread pool, assert byte-identical
// ---------------------------------------------------------------------------

/// Build a forest using `CpuBackend` (normal rayon parallel) vs. building the
/// same forest inside a rayon 1-thread pool. Because all draws are keyed by
/// `(seed, tree, node, feature, draw)` via the stateless Philox counter, the
/// resulting ForestIR must be byte-identical regardless of tree scheduling order.
fn assert_parallel_equals_sequential(
    x: &Array2<f32>,
    y: &Array1<f32>,
    cfg: &TrainConfig,
    label: &str,
) {
    let backend = CpuBackend;

    // Normal rayon build (may execute trees in any order).
    let ir_par = backend.fit(x.view(), y.view(), cfg).expect("parallel fit");

    // Forced 1-thread pool: no inter-tree parallelism, trees built in strict
    // index order. If Philox keying is correct, the result is identical.
    let ir_seq = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread pool")
        .install(|| {
            backend
                .fit(x.view(), y.view(), cfg)
                .expect("sequential fit")
        });

    let s_par = serde_json::to_string(&ir_par).expect("ser parallel");
    let s_seq = serde_json::to_string(&ir_seq).expect("ser sequential");
    assert_eq!(
        s_par, s_seq,
        "{label}: parallel (rayon) and sequential (1-thread pool) must be \
         byte-identical — Philox per-tree keying makes ordering irrelevant"
    );
}

// ---------------------------------------------------------------------------
// Test 1: Same-seed byte-identity for all four estimators
// ---------------------------------------------------------------------------

#[test]
fn et_clf_same_seed_byte_identical() {
    let (x, y) = clf_data();
    assert_same_seed_byte_identical(&x, &y, &et_clf_cfg(42), "ET clf");
}

#[test]
fn et_reg_same_seed_byte_identical() {
    let (x, y) = reg_data();
    assert_same_seed_byte_identical(&x, &y, &et_reg_cfg(42), "ET reg");
}

#[test]
fn rf_clf_same_seed_byte_identical() {
    let (x, y) = clf_data();
    assert_same_seed_byte_identical(&x, &y, &rf_clf_cfg(99), "RF clf");
}

#[test]
fn rf_reg_same_seed_byte_identical() {
    let (x, y) = reg_data();
    assert_same_seed_byte_identical(&x, &y, &rf_reg_cfg(99), "RF reg");
}

/// Different seeds must produce different ForestIRs (guards against trivial
/// "always equal" regressions in the determinism assertion logic).
#[test]
fn different_seeds_produce_different_irs() {
    let (x, y) = clf_data();
    let backend = CpuBackend;
    let ir_a = backend
        .fit(x.view(), y.view(), &et_clf_cfg(1))
        .expect("seed 1");
    let ir_b = backend
        .fit(x.view(), y.view(), &et_clf_cfg(2))
        .expect("seed 2");
    let sa = serde_json::to_string(&ir_a).expect("ser a");
    let sb = serde_json::to_string(&ir_b).expect("ser b");
    // Distinct seeds should almost certainly produce distinct forests;
    // this assertion protects against a trivial "always the same" bug.
    assert_ne!(
        sa, sb,
        "different seeds must produce different serialized ForestIRs"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Parallel == sequential (rayon order-independence) for all four
// ---------------------------------------------------------------------------

#[test]
fn et_clf_parallel_equals_sequential() {
    let (x, y) = clf_data();
    assert_parallel_equals_sequential(&x, &y, &et_clf_cfg(7), "ET clf");
}

#[test]
fn et_reg_parallel_equals_sequential() {
    let (x, y) = reg_data();
    assert_parallel_equals_sequential(&x, &y, &et_reg_cfg(7), "ET reg");
}

#[test]
fn rf_clf_parallel_equals_sequential() {
    let (x, y) = clf_data();
    assert_parallel_equals_sequential(&x, &y, &rf_clf_cfg(13), "RF clf");
}

#[test]
fn rf_reg_parallel_equals_sequential() {
    let (x, y) = reg_data();
    assert_parallel_equals_sequential(&x, &y, &rf_reg_cfg(13), "RF reg");
}

// ---------------------------------------------------------------------------
// Test 3: Multiple seeds — byte-identity holds across seeds
// ---------------------------------------------------------------------------

#[test]
fn seed_determinism_across_multiple_seeds() {
    let seeds: &[u64] = &[0, 1, 7, 42, 100, 999, u64::MAX / 2];
    let (x, y) = clf_data();
    let backend = CpuBackend;
    for &seed in seeds {
        let cfg = et_clf_cfg(seed);
        let ir1 = backend.fit(x.view(), y.view(), &cfg).expect("fit 1");
        let ir2 = backend.fit(x.view(), y.view(), &cfg).expect("fit 2");
        let s1 = serde_json::to_string(&ir1).expect("ser 1");
        let s2 = serde_json::to_string(&ir2).expect("ser 2");
        assert_eq!(s1, s2, "seed={seed}: same-seed must be byte-identical");
    }
}

/// Regression: multiple seeds + RF.
#[test]
fn rf_reg_seed_determinism_across_multiple_seeds() {
    let seeds: &[u64] = &[3, 17, 55, 200];
    let (x, y) = reg_data();
    let backend = CpuBackend;
    for &seed in seeds {
        let cfg = rf_reg_cfg(seed);
        let ir1 = backend.fit(x.view(), y.view(), &cfg).expect("fit 1");
        let ir2 = backend.fit(x.view(), y.view(), &cfg).expect("fit 2");
        let s1 = serde_json::to_string(&ir1).expect("ser 1");
        let s2 = serde_json::to_string(&ir2).expect("ser 2");
        assert_eq!(
            s1, s2,
            "RF reg seed={seed}: same-seed must be byte-identical"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Entropy criterion — byte-identical determinism
// ---------------------------------------------------------------------------

#[test]
fn entropy_clf_same_seed_byte_identical() {
    let (x, y) = clf_data();
    let cfg = TrainConfig {
        criterion: Criterion::Entropy,
        ..et_clf_cfg(77)
    };
    assert_same_seed_byte_identical(&x, &y, &cfg, "ET entropy clf");
}
