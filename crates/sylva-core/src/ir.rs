//! The SoA `ForestIR` — the single shared representation (ENG-02).
//!
//! Written by training, read read-only by inference (this phase), tree-SHAP
//! (Phase 8), and Treelite export (Phase 6). Designed for ALL known consumers
//! now (D-03) so it is never rewritten. All numeric fields are `f32` (D-05) so
//! the Phase-4 `GPU == CPU oracle` bit-exact gate is reachable.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::config::{Criterion, Task};
use crate::error::SylvaError;

/// Sentinel `feature_id` marking a leaf node.
pub const LEAF_FEATURE: i32 = -1;
/// Sentinel child id marking "no child" (a leaf's children).
pub const NO_CHILD: i32 = -1;

/// Structure-of-arrays forest. Each per-node array is indexed by a global node
/// id; `tree_offsets` carves the node arrays into per-tree ranges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForestIR {
    // --- per-node arrays (indexed by global node id) ---
    /// Split feature, or [`LEAF_FEATURE`] (`-1`) for leaves.
    pub feature_id: Vec<i32>,
    /// Split threshold: go left when `x[feature] <= threshold`.
    pub threshold: Vec<f32>,
    /// Left child id, or [`NO_CHILD`] for leaves.
    pub left_child: Vec<i32>,
    /// Right child id, or [`NO_CHILD`] for leaves.
    pub right_child: Vec<i32>,
    /// Missing/NaN routing target (D-01): one of the two children.
    pub default_child: Vec<i32>,
    /// Leaf flag (redundant with `feature_id == LEAF_FEATURE`, kept for clarity).
    pub is_leaf: Vec<bool>,
    /// Unweighted samples reaching the node — tree-SHAP **cover** (Treelite
    /// `data_count`) and the D-01 default-direction tie source.
    pub node_sample_count: Vec<u64>,
    /// Weighted samples reaching the node — tree-SHAP cover / future
    /// `sample_weight` (Treelite `sum_hess`).
    pub node_weighted_count: Vec<f32>,
    /// Node impurity — export gain / diagnostics.
    pub impurity: Vec<f32>,
    // --- leaf payloads ---
    /// Regressor leaf outputs, indexed by `leaf_offset`.
    pub leaf_value: Vec<f32>,
    /// Classifier leaf probabilities, flattened with stride `n_classes`; the
    /// block for a leaf at `leaf_offset[node] = o` is
    /// `leaf_proba[o*n_classes .. (o+1)*n_classes]`.
    pub leaf_proba: Vec<f32>,
    /// Per node: index into the leaf payload block, or `-1` for internal nodes.
    pub leaf_offset: Vec<i32>,
    // --- per-tree / forest ---
    /// Node-range boundaries, length `n_trees + 1`, non-decreasing,
    /// `tree_offsets[n_trees] == node_count`.
    pub tree_offsets: Vec<usize>,
    /// Per-tree root node id.
    pub tree_root: Vec<i32>,
    pub n_trees: usize,
    pub n_features: usize,
    pub task: Task,
    pub criterion: Criterion,
    pub seed: u64,
}

impl ForestIR {
    /// Total node count across all trees.
    pub fn node_count(&self) -> usize {
        self.feature_id.len()
    }

    /// Number of classes (`1` for regression).
    pub fn n_classes(&self) -> usize {
        match self.task {
            Task::Classification { n_classes } => n_classes,
            Task::Regression => 1,
        }
    }

    /// Global node-id range for tree `t`.
    pub fn tree_node_range(&self, t: usize) -> Range<usize> {
        self.tree_offsets[t]..self.tree_offsets[t + 1]
    }

    /// Validate the structural invariants every reader (predict, SHAP, export)
    /// relies on. Returns a typed error rather than panicking.
    pub fn validate_structure(&self) -> Result<(), SylvaError> {
        let n = self.node_count();
        let lens: [(&str, usize); 9] = [
            ("threshold", self.threshold.len()),
            ("left_child", self.left_child.len()),
            ("right_child", self.right_child.len()),
            ("default_child", self.default_child.len()),
            ("is_leaf", self.is_leaf.len()),
            ("node_sample_count", self.node_sample_count.len()),
            ("node_weighted_count", self.node_weighted_count.len()),
            ("impurity", self.impurity.len()),
            ("leaf_offset", self.leaf_offset.len()),
        ];
        for (name, len) in lens {
            if len != n {
                return Err(SylvaError::InvalidIr(format!(
                    "array {name} length {len} != node_count {n}"
                )));
            }
        }
        if self.tree_offsets.len() != self.n_trees + 1 {
            return Err(SylvaError::InvalidIr(
                "tree_offsets length must be n_trees + 1".into(),
            ));
        }
        if self.tree_offsets.first() != Some(&0) || self.tree_offsets.last() != Some(&n) {
            return Err(SylvaError::InvalidIr(
                "tree_offsets must start at 0 and end at node_count".into(),
            ));
        }
        for w in self.tree_offsets.windows(2) {
            if w[1] < w[0] {
                return Err(SylvaError::InvalidIr(
                    "tree_offsets must be non-decreasing".into(),
                ));
            }
        }
        for i in 0..n {
            if self.is_leaf[i] {
                if self.left_child[i] != NO_CHILD || self.right_child[i] != NO_CHILD {
                    return Err(SylvaError::InvalidIr(format!("leaf node {i} has children")));
                }
            } else {
                let (l, r) = (self.left_child[i], self.right_child[i]);
                if l < 0 || r < 0 || l as usize >= n || r as usize >= n {
                    return Err(SylvaError::InvalidIr(format!(
                        "internal node {i} has out-of-range children"
                    )));
                }
                let d = self.default_child[i];
                if d != l && d != r {
                    return Err(SylvaError::InvalidIr(format!(
                        "internal node {i} default_child must be one of its children"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Criterion, Task};

    /// A tiny one-tree binary classifier: node 0 splits on feature 0 at 0.5
    /// into leaf 1 (class 0) and leaf 2 (class 1).
    fn tiny_clf_ir() -> ForestIR {
        ForestIR {
            feature_id: vec![0, LEAF_FEATURE, LEAF_FEATURE],
            threshold: vec![0.5, 0.0, 0.0],
            left_child: vec![1, NO_CHILD, NO_CHILD],
            right_child: vec![2, NO_CHILD, NO_CHILD],
            default_child: vec![2, NO_CHILD, NO_CHILD],
            is_leaf: vec![false, true, true],
            node_sample_count: vec![10, 4, 6],
            node_weighted_count: vec![10.0, 4.0, 6.0],
            impurity: vec![0.48, 0.0, 0.0],
            leaf_value: vec![],
            leaf_proba: vec![1.0, 0.0, 0.0, 1.0],
            leaf_offset: vec![-1, 0, 1],
            tree_offsets: vec![0, 3],
            tree_root: vec![0],
            n_trees: 1,
            n_features: 1,
            task: Task::Classification { n_classes: 2 },
            criterion: Criterion::Gini,
            seed: 42,
        }
    }

    #[test]
    fn serde_round_trip() {
        let ir = tiny_clf_ir();
        let s = serde_json::to_string(&ir).expect("serialize");
        let back: ForestIR = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(ir, back);
    }

    #[test]
    fn validate_accepts_well_formed() {
        assert!(tiny_clf_ir().validate_structure().is_ok());
    }

    #[test]
    fn validate_rejects_bad_default_child() {
        let mut ir = tiny_clf_ir();
        ir.default_child[0] = 0; // not one of node 0's children (1 or 2)
        assert!(ir.validate_structure().is_err());
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let mut ir = tiny_clf_ir();
        ir.threshold.push(0.0); // now longer than node_count
        assert!(ir.validate_structure().is_err());
    }

    #[test]
    fn classifier_leaf_proba_sums_to_one() {
        let ir = tiny_clf_ir();
        let nc = ir.n_classes();
        for i in 0..ir.node_count() {
            if ir.is_leaf[i] {
                let o = ir.leaf_offset[i] as usize;
                let s: f32 = ir.leaf_proba[o * nc..(o + 1) * nc].iter().sum();
                assert!((s - 1.0).abs() < 1e-6, "leaf {i} proba must sum to 1");
            }
        }
    }
}
