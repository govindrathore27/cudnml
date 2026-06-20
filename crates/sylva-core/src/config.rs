//! Device-neutral training configuration and the enums that parameterize a fit.
//! Validated at the boundary (`TrainConfig::validate`) so bad params surface as
//! a typed `SylvaError`, never a panic (V5 input-validation control).

use serde::{Deserialize, Serialize};

use crate::error::SylvaError;

/// Split-quality criterion. `Mse` is the regression criterion; `Gini`/`Entropy`
/// are classification criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Criterion {
    Gini,
    Entropy,
    Mse,
}

/// Forest algorithm: ExtraTrees (random split thresholds) or RandomForest
/// (impurity-best split + bootstrap sampling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algo {
    ExtraTrees,
    RandomForest,
}

/// Learning task. Classification carries the class count, which fixes the
/// leaf-probability layout in `ForestIR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    Classification { n_classes: usize },
    Regression,
}

/// How many features are considered per split.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaxFeatures {
    /// `floor(sqrt(n_features))` — the classifier default (RESEARCH Pitfall 5).
    Sqrt,
    /// A fraction of `n_features`, in `(0, 1]`.
    Fraction(f32),
    /// All features — the regressor default.
    All,
    /// An explicit count.
    Count(usize),
}

impl MaxFeatures {
    /// Resolve to a concrete, clamped feature count for a node. The task is
    /// accepted so callers can branch the *default* choice (clf → `Sqrt`,
    /// reg → `All`) before constructing the config; here we resolve whatever
    /// explicit variant was chosen.
    pub fn resolve(&self, n_features: usize, _task: Task) -> usize {
        let k = match self {
            MaxFeatures::Sqrt => (n_features as f64).sqrt().floor() as usize,
            MaxFeatures::Fraction(f) => ((*f as f64) * n_features as f64).floor() as usize,
            MaxFeatures::All => n_features,
            MaxFeatures::Count(c) => *c,
        };
        k.clamp(1, n_features.max(1))
    }
}

/// The full, device-neutral training configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub max_features: MaxFeatures,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub bootstrap: bool,
    pub criterion: Criterion,
    pub seed: u64,
    pub algo: Algo,
}

impl TrainConfig {
    /// Reject nonsensical hyperparameters with a typed error (no panic).
    pub fn validate(&self) -> Result<(), SylvaError> {
        if self.n_estimators == 0 {
            return Err(SylvaError::InvalidConfig(
                "n_estimators must be >= 1".into(),
            ));
        }
        if self.min_samples_leaf == 0 {
            return Err(SylvaError::InvalidConfig(
                "min_samples_leaf must be >= 1".into(),
            ));
        }
        if self.min_samples_split < 2 {
            return Err(SylvaError::InvalidConfig(
                "min_samples_split must be >= 2".into(),
            ));
        }
        match self.max_features {
            MaxFeatures::Fraction(f) if !(f > 0.0 && f <= 1.0) => {
                return Err(SylvaError::InvalidConfig(
                    "max_features fraction must be in (0, 1]".into(),
                ));
            }
            MaxFeatures::Count(0) => {
                return Err(SylvaError::InvalidConfig(
                    "max_features count must be >= 1".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}
