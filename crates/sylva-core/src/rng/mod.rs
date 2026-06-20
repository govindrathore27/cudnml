//! Stateless Philox-4×32-10 (ENG-06) — the Random123 counter-based RNG,
//! reimplemented from the published algorithm (Apache-2.0, NOT copied source).
//!
//! Keyed by `(seed, tree, node, feature, draw)` so CPU tree-parallelism is
//! order-independent (each draw is a pure function of its coordinate) and the
//! Phase-4 CUDA reimplementation is bit-verifiable against the frozen KAT
//! vectors in [`kat`].
//!
//! NON-cryptographic: a statistical RNG, never to be mistaken for secure
//! randomness.

pub mod kat;

/// Philox multiply constant applied to counter word 0.
pub const PHILOX_M0: u32 = 0xD251_1F53;
/// Philox multiply constant applied to counter word 2.
pub const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Weyl increment for key word 0 (fractional golden ratio).
pub const PHILOX_W0: u32 = 0x9E37_79B9;
/// Weyl increment for key word 1 (fractional √3 − 1).
pub const PHILOX_W1: u32 = 0xBB67_AE85;
/// Number of Philox rounds — the "10" in 4×32-10.
pub const PHILOX_ROUNDS: usize = 10;

/// 32×32→64 multiply, split into the high and low 32-bit halves.
#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32) // (hi, lo)
}

/// A single Philox-4×32 round (Random123 `_philox4x32round`).
#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo32(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo32(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Bump the key by the Weyl increments (wrapping; Random123 `_philox4x32bumpkey`).
#[inline]
fn bump_key(key: [u32; 2]) -> [u32; 2] {
    [
        key[0].wrapping_add(PHILOX_W0),
        key[1].wrapping_add(PHILOX_W1),
    ]
}

/// Run all 10 Philox rounds. Round 0 uses the original key; rounds 1..10 bump
/// the key first (the Random123 `philox4x32_R` schedule).
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    for r in 0..PHILOX_ROUNDS {
        if r != 0 {
            k = bump_key(k);
        }
        c = round(c, k);
    }
    c
}

/// Map a `u32` to a uniform `f32` in `[0, 1)` using its top 24 bits. Matches the
/// future CUDA conversion; `0xFFFFFFFF` maps to just below `1.0`.
#[inline]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Pack a draw coordinate into a Philox counter. Frozen bit-allocation: one
/// 32-bit word each for tree, node, feature, and draw index — the
/// parity-contract layout Phase 4 reproduces.
#[inline]
pub fn pack_counter(tree: u32, node: u32, feature: u32, draw: u32) -> [u32; 4] {
    [tree, node, feature, draw]
}

/// One uniform `f32` in `[0, 1)` for the given draw coordinate, keyed by the
/// `u64` seed split into the two key words.
#[inline]
pub fn philox_uniform(seed: u64, tree: u32, node: u32, feature: u32, draw: u32) -> f32 {
    let key = [seed as u32, (seed >> 32) as u32];
    let ctr = pack_counter(tree, node, feature, draw);
    u32_to_unit_f32(philox4x32_10(ctr, key)[0])
}

#[cfg(test)]
mod tests {
    use super::kat::{KAT_MIXED, KAT_ONES, KAT_ZERO};
    use super::*;
    use std::collections::HashSet;

    /// A from-spec implementation reproducing the published KAT outputs is the
    /// mutual validation (correct constants ⇄ authoritative vectors).
    #[test]
    fn philox_matches_kat_vectors() {
        assert_eq!(
            philox4x32_10(KAT_ZERO.0, KAT_ZERO.1),
            KAT_ZERO.2,
            "all-zero KAT"
        );
        assert_eq!(
            philox4x32_10(KAT_ONES.0, KAT_ONES.1),
            KAT_ONES.2,
            "all-ones KAT"
        );
        assert_eq!(
            philox4x32_10(KAT_MIXED.0, KAT_MIXED.1),
            KAT_MIXED.2,
            "mixed KAT"
        );
    }

    #[test]
    fn unit_f32_in_range() {
        assert!((0.0..1.0).contains(&u32_to_unit_f32(0)));
        let top = u32_to_unit_f32(0xFFFF_FFFF);
        assert!((0.0..1.0).contains(&top), "0xFFFFFFFF must map below 1.0");
    }

    #[test]
    fn counter_packing_injective() {
        let mut seen = HashSet::new();
        for tree in 0..4u32 {
            for node in 0..4u32 {
                for feature in 0..4u32 {
                    for draw in 0..4u32 {
                        assert!(
                            seen.insert(pack_counter(tree, node, feature, draw)),
                            "distinct (tree,node,feature,draw) must give distinct counters"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn philox_uniform_is_deterministic() {
        let a = philox_uniform(7, 1, 2, 3, 4);
        let b = philox_uniform(7, 1, 2, 3, 4);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }
}
