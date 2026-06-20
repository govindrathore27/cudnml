//! Impurity criterion functions (Gini / entropy / MSE) for the CPU oracle.
//!
//! Reimplemented from standard definitions (Apache-2.0; NOT copied from sklearn
//! or any GPL source). All arithmetic is f32 (D-05).
//!
//! **Determinism rule (RESEARCH Pitfall 3/5):** class COUNTS (integers) may be
//! tallied in any order, but every FLOAT sum (probability sums for gini/entropy,
//! target sum/mean and squared-deviation sum for MSE) MUST accumulate in a fixed
//! sequential row order — never a rayon `par_iter().sum()`. This keeps the CPU
//! oracle bit-reproducible across runs.

/// Compute the Gini impurity given class counts.
///
/// `counts[k]` is the number of samples with class `k` in the current node.
/// `total` must equal `counts.iter().sum()`.
///
/// Gini = 1 − Σ_k p_k² where p_k = counts[k] / total.
///
/// Pure node (one class): 0.0.
/// Balanced binary: 0.5.
/// Uses sequential float accumulation (determinism rule).
pub fn gini(counts: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f32;
    // Sequential accumulation — float order is fixed (counts slice order).
    let sum_sq: f32 = counts.iter().fold(0.0_f32, |acc, &c| {
        let p = c as f32 / n;
        acc + p * p
    });
    1.0 - sum_sq
}

/// Compute the entropy impurity given class counts.
///
/// `counts[k]` is the number of samples with class `k`.
/// `total` must equal `counts.iter().sum()`.
///
/// Entropy = −Σ_k p_k · log₂(p_k) (base-2 bits).
///
/// Pure node: 0.0.  Balanced binary: 1.0.
/// Uses sequential float accumulation (determinism rule).
pub fn entropy(counts: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f32;
    // Sequential accumulation — float order is fixed (counts slice order).
    let h: f32 = counts.iter().fold(0.0_f32, |acc, &c| {
        if c == 0 {
            acc
        } else {
            let p = c as f32 / n;
            acc - p * p.log2()
        }
    });
    h.max(0.0) // guard against tiny negative from f32 rounding
}

/// Compute the MSE (mean squared error / population variance) of target values.
///
/// `targets` is the slice of f32 target values for the current node's rows, in
/// a **fixed row order** — the order used here for the float sums is the same
/// slice order, making the result bit-reproducible (determinism rule).
///
/// MSE = (1/n) Σ (y_i − ȳ)²  =  Σ y_i² / n  −  (Σ y_i / n)²
///
/// Constant targets: 0.0.
pub fn mse(targets: &[f32]) -> f32 {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }
    // Sequential accumulation — slice order is the fixed reduction order.
    let (sum, sum_sq) = targets
        .iter()
        .fold((0.0_f32, 0.0_f32), |(s, s2), &y| (s + y, s2 + y * y));
    let mean = sum / n as f32;
    let var = sum_sq / n as f32 - mean * mean;
    var.max(0.0) // guard against tiny negative from f32 rounding
}

/// Proxy impurity improvement for a candidate split (weighted child impurity
/// decrease). Used by the ET splitter to rank candidate splits.
///
/// `parent_impurity` is the impurity of the node before the split.
/// `left_impurity` / `right_impurity` are the impurities of the two children.
/// `n_left` / `n_right` are the sample counts.
///
/// Proxy = parent − (n_left·left + n_right·right) / (n_left + n_right)
///
/// Returns `f32::NEG_INFINITY` if the split has 0 total samples (degenerate).
pub fn proxy_improvement(
    parent_impurity: f32,
    left_impurity: f32,
    right_impurity: f32,
    n_left: u64,
    n_right: u64,
) -> f32 {
    let n_total = n_left + n_right;
    if n_total == 0 {
        return f32::NEG_INFINITY;
    }
    let n = n_total as f32;
    let weighted = (n_left as f32 * left_impurity + n_right as f32 * right_impurity) / n;
    parent_impurity - weighted
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Gini tests ---

    #[test]
    fn gini_pure_node_is_zero() {
        // All samples in class 0.
        assert_eq!(gini(&[10, 0], 10), 0.0);
    }

    #[test]
    fn gini_balanced_binary_is_half() {
        // 50/50 split: 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
        assert_abs_diff_eq!(gini(&[5, 5], 10), 0.5_f32, epsilon = 1e-6);
    }

    #[test]
    fn gini_multiclass() {
        // 3 classes, equal: 1 - 3*(1/3)² = 1 - 1/3 = 2/3 ≈ 0.6667
        let g = gini(&[4, 4, 4], 12);
        assert_abs_diff_eq!(g, 2.0_f32 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn gini_empty_is_zero() {
        assert_eq!(gini(&[], 0), 0.0);
    }

    #[test]
    fn gini_deterministic() {
        let a = gini(&[7, 3], 10);
        let b = gini(&[7, 3], 10);
        assert_eq!(a.to_bits(), b.to_bits(), "byte-identical on repeat");
    }

    // --- Entropy tests ---

    #[test]
    fn entropy_pure_node_is_zero() {
        assert_eq!(entropy(&[10, 0], 10), 0.0);
    }

    #[test]
    fn entropy_balanced_binary_is_one() {
        // H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
        assert_abs_diff_eq!(entropy(&[5, 5], 10), 1.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn entropy_multiclass() {
        // 3 equal classes: H = -3*(1/3)*log2(1/3) = log2(3) ≈ 1.585
        let h = entropy(&[4, 4, 4], 12);
        let expected = (3.0_f32).log2();
        assert_abs_diff_eq!(h, expected, epsilon = 1e-5);
    }

    #[test]
    fn entropy_empty_is_zero() {
        assert_eq!(entropy(&[], 0), 0.0);
    }

    #[test]
    fn entropy_deterministic() {
        let a = entropy(&[7, 3], 10);
        let b = entropy(&[7, 3], 10);
        assert_eq!(a.to_bits(), b.to_bits(), "byte-identical on repeat");
    }

    // --- MSE tests ---

    #[test]
    fn mse_constant_targets_is_zero() {
        let targets = vec![3.0_f32; 5];
        assert_abs_diff_eq!(mse(&targets), 0.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn mse_population_variance() {
        // targets = [1, 2, 3, 4, 5]; mean = 3; var = (1+0+1+4+4+1+0+1+4)/5? Wait.
        // Var = sum((y-mean)^2) / n = ((-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2)/5
        //     = (4+1+0+1+4)/5 = 10/5 = 2.0
        let targets = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(mse(&targets), 2.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn mse_empty_is_zero() {
        assert_eq!(mse(&[]), 0.0);
    }

    #[test]
    fn mse_deterministic() {
        let targets = vec![1.0_f32, 2.0, 3.0];
        let a = mse(&targets);
        let b = mse(&targets);
        assert_eq!(a.to_bits(), b.to_bits(), "byte-identical on repeat");
    }

    // --- proxy_improvement ---

    #[test]
    fn proxy_improvement_perfect_split() {
        // parent impurity = 0.5 (balanced), left pure = 0.0, right pure = 0.0
        let imp = proxy_improvement(0.5, 0.0, 0.0, 5, 5);
        assert_abs_diff_eq!(imp, 0.5_f32, epsilon = 1e-6);
    }

    #[test]
    fn proxy_improvement_no_gain() {
        // Same impurity everywhere → improvement should be ≈ 0
        let imp = proxy_improvement(0.5, 0.5, 0.5, 5, 5);
        assert_abs_diff_eq!(imp, 0.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn proxy_improvement_degenerate_returns_neg_inf() {
        assert_eq!(proxy_improvement(0.5, 0.0, 0.0, 0, 0), f32::NEG_INFINITY);
    }
}
