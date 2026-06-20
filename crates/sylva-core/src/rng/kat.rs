//! Frozen Philox-4×32-10 known-answer-test (KAT) vectors — the bit-match oracle
//! that Phase 4's CUDA Philox must reproduce exactly.
//!
//! Provenance: the canonical Random123 `tests/kat_vectors.txt`
//! (DEShawResearch/random123), the `philox4x32 10` rows. Transcribed from the
//! published algorithm/vectors, not copied source (Apache-2.0). The three
//! literal output triples are confirmed against the authoritative reference at
//! the Plan 02-01 Task 4 human-verify checkpoint before being frozen.
//!
//! Each entry is `(counter, key, expected_output)`; 32-bit words are listed
//! low → high.

/// `ctr = {0,0,0,0}`, `key = {0,0}`.
pub const KAT_ZERO: ([u32; 4], [u32; 2], [u32; 4]) = (
    [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000],
    [0x0000_0000, 0x0000_0000],
    [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
);

/// `ctr = {0xffffffff × 4}`, `key = {0xffffffff × 2}`.
pub const KAT_ONES: ([u32; 4], [u32; 2], [u32; 4]) = (
    [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
    [0xFFFF_FFFF, 0xFFFF_FFFF],
    [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
);

/// `ctr`/`key` = digits of π (the Random123 mixed-input KAT row).
pub const KAT_MIXED: ([u32; 4], [u32; 2], [u32; 4]) = (
    [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
    [0xa409_3822, 0x299f_31d0],
    [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
);
