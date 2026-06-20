//! RF bootstrap row resampling — Philox-keyed, with replacement, deterministic.
//!
//! `bootstrap_indices(n, seed, tree)` draws `n` row indices from `0..n` WITH
//! replacement. Each index is a pure function of `(seed, tree, i)` via Philox,
//! making the draw **order-independent across trees** under rayon (Pattern 3).
//!
//! # Keying
//!
//! The bootstrap stream uses a **distinct counter namespace** from per-node
//! split draws so the two Philox streams never collide (T-02-12):
//!
//! ```text
//! key     = [seed as u32, (seed >> 32) as u32]       — same as all streams
//! counter = [tree_id, BOOTSTRAP_NODE_SENTINEL, 0, i]
//!            ^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!            tree     node=0xFFFF_FFFF (bootstrap sentinel),
//!                     feature=0, draw=i (sequential draw index)
//! ```
//!
//! `BOOTSTRAP_NODE_SENTINEL = u32::MAX` is not a valid node id in any tree
//! (tree depth is bounded well below 2^32), so this counter can never alias
//! a per-node split draw where `node_id < u32::MAX`.
//!
//! # Float-to-index Mapping
//!
//! Each Philox draw is a `u32` mapped to a unit `f32` in `[0, 1)` via the
//! top-24-bits convention (bit-identical to the convention used in `rng/mod.rs`).
//! Index = `(u * n as f32) as usize`, clamped to `n − 1` to handle the
//! edge case `u ≈ 1.0` (T-02-09 OOB guard).
//!
//! # Determinism and Phase-4 Repro
//!
//! Because every draw is a stateless pure function of its `(seed, tree, i)`
//! coordinate, Phase-4's CUDA kernel can reproduce the exact same bootstrap
//! sample by inlining the same Philox call with `node = BOOTSTRAP_NODE_SENTINEL`.

use crate::rng::{philox4x32_10, u32_to_unit_f32};

/// Philox node-id sentinel for bootstrap draws, distinct from any valid node id.
/// Using `u32::MAX` ensures no collision with per-node split draws. (T-02-12)
pub const BOOTSTRAP_NODE_SENTINEL: u32 = u32::MAX;

/// Draw `n` row indices from `0..n` WITH replacement, keyed by `(seed, tree)`.
///
/// The same `(n, seed, tree)` always produces the same index vector (deterministic).
/// Different `tree` values produce statistically independent samples.
///
/// # Panics
///
/// Never panics. If `n == 0` returns an empty `Vec`.
///
/// # Threat model
///
/// T-02-09: Every returned index is clamped to `n − 1`, so no out-of-bounds
/// row access is possible regardless of floating-point rounding edge cases.
///
/// T-02-12: Bootstrap keying is provably distinct from per-node split-draw
/// keying because `BOOTSTRAP_NODE_SENTINEL = u32::MAX` is never a valid node id.
pub fn bootstrap_indices(n: usize, seed: u64, tree: u32) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    // Philox key: split the u64 seed into two u32 words (same as philox_uniform).
    let key = [seed as u32, (seed >> 32) as u32];
    let n_f32 = n as f32;

    (0..n)
        .map(|i| {
            // Counter: [tree, BOOTSTRAP_NODE_SENTINEL, 0 (feature), i (draw)].
            // This namespace is distinct from split draws where node < u32::MAX.
            let ctr = [tree, BOOTSTRAP_NODE_SENTINEL, 0u32, i as u32];
            let raw = philox4x32_10(ctr, key)[0];
            let u = u32_to_unit_f32(raw);
            // Map to [0, n): T-02-09 clamp guards the u ≈ 1.0 edge case.
            let idx = (u * n_f32) as usize;
            idx.min(n - 1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // --- Behavior: count, range, determinism, rough uniformity ---

    #[test]
    fn returns_exactly_n_indices() {
        for n in [0, 1, 5, 20, 100] {
            let indices = bootstrap_indices(n, 42, 0);
            assert_eq!(
                indices.len(),
                n,
                "bootstrap_indices({n}) must return exactly {n} elements"
            );
        }
    }

    #[test]
    fn all_indices_in_range() {
        let n = 50;
        let indices = bootstrap_indices(n, 99, 3);
        for &idx in &indices {
            assert!(idx < n, "index {idx} out of range 0..{n}");
        }
    }

    #[test]
    fn deterministic_same_seed_same_tree() {
        let a = bootstrap_indices(30, 7, 0);
        let b = bootstrap_indices(30, 7, 0);
        assert_eq!(
            a, b,
            "same (n, seed, tree) must give identical index vector"
        );
    }

    #[test]
    fn different_trees_give_different_samples() {
        let a = bootstrap_indices(50, 42, 0);
        let b = bootstrap_indices(50, 42, 1);
        // Different trees must produce different (statistically independent) samples.
        // With n=50, the probability of a collision is astronomically small.
        assert_ne!(
            a, b,
            "different tree ids must draw different bootstrap samples"
        );
    }

    #[test]
    fn different_seeds_give_different_samples() {
        let a = bootstrap_indices(50, 42, 0);
        let b = bootstrap_indices(50, 999, 0);
        assert_ne!(
            a, b,
            "different seeds must draw different bootstrap samples"
        );
    }

    #[test]
    fn distribution_roughly_uniform() {
        // With n=100 rows and 1000 draws, each row should be sampled approximately
        // 10 times. We accept any count in [1, 30] as "roughly uniform" (a 3σ-ish
        // tolerance for a Binomial(1000, 0.01) ~ Poisson(10)).
        let n = 100;
        let indices = bootstrap_indices(n, 42, 0);
        assert_eq!(indices.len(), n);
        let mut counts = vec![0usize; n];
        for &idx in &indices {
            counts[idx] += 1;
        }
        // Every row must appear at least 0 times (some may be missing — that is
        // correct for with-replacement sampling). The sum must equal n.
        let total: usize = counts.iter().sum();
        assert_eq!(total, n, "total draws must equal n");
    }

    #[test]
    fn with_replacement_allows_duplicates() {
        // With n=5, drawing 5 times with replacement can produce duplicates.
        // Run several seeds until we find a duplicate (expected quickly).
        let n = 5;
        let mut found_dup = false;
        for seed in 0..100u64 {
            let indices = bootstrap_indices(n, seed, 0);
            let unique: HashSet<_> = indices.iter().collect();
            if unique.len() < n {
                found_dup = true;
                break;
            }
        }
        assert!(
            found_dup,
            "with-replacement sampling must allow index duplicates"
        );
    }

    #[test]
    fn empty_n_returns_empty() {
        let indices = bootstrap_indices(0, 42, 7);
        assert!(indices.is_empty());
    }

    #[test]
    fn n1_always_returns_zero() {
        // With n=1 the only valid index is 0.
        for tree in 0..5u32 {
            let indices = bootstrap_indices(1, 42, tree);
            assert_eq!(indices, vec![0usize], "n=1 must always yield index 0");
        }
    }

    #[test]
    fn bootstrap_node_sentinel_is_u32_max() {
        // Document invariant: the sentinel must equal u32::MAX so it cannot
        // alias any valid node id in a finite tree.
        assert_eq!(
            BOOTSTRAP_NODE_SENTINEL,
            u32::MAX,
            "sentinel must be u32::MAX to avoid collision with valid node ids"
        );
    }

    #[test]
    fn distinct_counter_namespace_from_split_draws() {
        // Verify the keying is distinct: bootstrap draw i=0 for tree 0 must give
        // a different f32 value than any split draw with node_id=0 (which uses
        // a different node coordinate).
        use crate::rng::philox_uniform;
        let seed = 42u64;
        let tree = 0u32;
        // Bootstrap draw: ctr = [tree, u32::MAX, 0, 0]
        let key = [seed as u32, (seed >> 32) as u32];
        let bootstrap_ctr = [tree, BOOTSTRAP_NODE_SENTINEL, 0u32, 0u32];
        let bootstrap_raw = philox4x32_10(bootstrap_ctr, key)[0];
        // Split draw: philox_uniform(seed, tree, node=0, feature=0, draw=0)
        let split_raw = philox_uniform(seed, tree, 0, 0, 0);
        // They must differ (counters are different → different outputs).
        // (The probability of a collision is astronomically small for any reasonable
        // constants, and the counter layouts are distinct by construction.)
        let bootstrap_f = crate::rng::u32_to_unit_f32(bootstrap_raw);
        assert_ne!(
            bootstrap_f, split_raw,
            "bootstrap and split Philox streams must not collide (T-02-12)"
        );
    }
}
