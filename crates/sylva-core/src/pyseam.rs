//! Minimal test-only PyO3 seam — Phase-05 parity harness entry point.
//!
//! This module is compiled **only** when the `pyseam` Cargo feature is enabled:
//!
//! ```text
//! cargo build -p sylva-core --features pyseam
//! maturin develop --features pyseam   (from the pyseam pyproject.toml)
//! ```
//!
//! It is **strictly below** the full sklearn-parity estimator API (EST-02),
//! which is deferred to Phase 5. The three functions exposed here — `fit_cpu`,
//! `predict_cpu`, and `split_statistics` — are the minimum needed to drive the
//! Phase-05 calibration and distributional parity harness from Python.
//!
//! ## FFI safety contract (mirrors `sylva-cuda::cuda_error_to_pyerr`)
//!
//! * Every fallible Rust call crosses the boundary via `map_err` to a typed
//!   Python exception — `ValueError` for caller errors, `RuntimeError` for
//!   internal failures. There is **no** `.unwrap()` / `.expect()` across FFI.
//! * Array shape and dtype are validated at entry; a bad input raises
//!   `ValueError` before any training work begins.
//! * The IR handle is an opaque JSON string — no pointer crosses the boundary.
//!
//! ## NOT exported by this seam (Phase 5 / EST-02, deferred)
//!
//! `predict_proba`, `get_params`, `set_params`, `check_estimator`, `fit`
//! with a sklearn-compatible signature, `classes_`, `feature_importances_`, etc.

#![cfg(feature = "pyseam")]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backend::{Backend, Predictions};
use crate::config::{Algo, Criterion, MaxFeatures, TrainConfig};
use crate::error::SylvaError;
use crate::ir::ForestIR;
use crate::parity;

// ---------------------------------------------------------------------------
// Error mapping — mirrors sylva-cuda::cuda_error_to_pyerr (no silent fallback)
// ---------------------------------------------------------------------------

fn sylva_error_to_pyerr(err: SylvaError) -> PyErr {
    match err {
        SylvaError::InvalidInput(_) | SylvaError::InvalidConfig(_) => {
            PyValueError::new_err(err.to_string())
        }
        SylvaError::InvalidIr(_) => PyRuntimeError::new_err(err.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Config parsing from Python dict
// ---------------------------------------------------------------------------

/// Parse a Python `cfg_dict` to `TrainConfig`.
///
/// Expected keys (all optional with documented defaults):
/// - `n_estimators`    (int, default 200)
/// - `max_depth`       (int or None, default None)
/// - `max_features`    (`"sqrt"` | `"all"` | float fraction | int count, default `"sqrt"`)
/// - `min_samples_split` (int, default 2)
/// - `min_samples_leaf`  (int, default 1)
/// - `bootstrap`       (bool, default False for ET, True for RF)
/// - `criterion`       (`"gini"` | `"entropy"` | `"mse"`, default `"gini"`)
/// - `seed`            (int, default 42)
/// - `algo`            (`"et"` | `"extra_trees"` | `"rf"` | `"random_forest"`, default `"et"`)
fn parse_config(cfg_dict: Bound<'_, PyDict>) -> PyResult<TrainConfig> {
    let get_int = |key: &str, default: usize| -> PyResult<usize> {
        match cfg_dict.get_item(key)? {
            Some(v) => v.extract::<usize>().map_err(|_| {
                PyValueError::new_err(format!("cfg_dict[{key:?}] must be a non-negative int"))
            }),
            None => Ok(default),
        }
    };
    let get_u64 = |key: &str, default: u64| -> PyResult<u64> {
        match cfg_dict.get_item(key)? {
            Some(v) => v.extract::<u64>().map_err(|_| {
                PyValueError::new_err(format!("cfg_dict[{key:?}] must be a non-negative int"))
            }),
            None => Ok(default),
        }
    };
    let get_bool = |key: &str, default: bool| -> PyResult<bool> {
        match cfg_dict.get_item(key)? {
            Some(v) => v
                .extract::<bool>()
                .map_err(|_| PyValueError::new_err(format!("cfg_dict[{key:?}] must be a bool"))),
            None => Ok(default),
        }
    };

    // --- algo (controls bootstrap default) ---
    let algo_str = match cfg_dict.get_item("algo")? {
        Some(v) => v.extract::<String>().unwrap_or_else(|_| "et".to_string()),
        None => "et".to_string(),
    };
    let algo = match algo_str.to_lowercase().as_str() {
        "rf" | "random_forest" | "randomforest" => Algo::RandomForest,
        "et" | "extra_trees" | "extratrees" => Algo::ExtraTrees,
        other => {
            return Err(PyValueError::new_err(format!(
                "cfg_dict['algo'] must be 'et' or 'rf', got {other:?}"
            )));
        }
    };

    // Bootstrap default: True for RF, False for ET (mirrors sklearn)
    let bootstrap_default = matches!(algo, Algo::RandomForest);

    // --- criterion ---
    let criterion_str = match cfg_dict.get_item("criterion")? {
        Some(v) => v.extract::<String>().unwrap_or_else(|_| "gini".to_string()),
        None => "gini".to_string(),
    };
    let criterion = match criterion_str.to_lowercase().as_str() {
        "gini" => Criterion::Gini,
        "entropy" => Criterion::Entropy,
        "mse" | "squared_error" => Criterion::Mse,
        other => {
            return Err(PyValueError::new_err(format!(
                "cfg_dict['criterion'] must be 'gini', 'entropy', or 'mse', got {other:?}"
            )));
        }
    };

    // --- max_features ---
    let max_features = match cfg_dict.get_item("max_features")? {
        Some(v) => {
            // Try string first
            if let Ok(s) = v.extract::<String>() {
                match s.to_lowercase().as_str() {
                    "sqrt" => MaxFeatures::Sqrt,
                    "all" | "none" => MaxFeatures::All,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "cfg_dict['max_features'] string must be 'sqrt' or 'all', got {other:?}"
                        )));
                    }
                }
            } else if let Ok(f) = v.extract::<f64>() {
                // float fraction
                if f <= 0.0 || f > 1.0 {
                    return Err(PyValueError::new_err(
                        "cfg_dict['max_features'] fraction must be in (0, 1]",
                    ));
                }
                MaxFeatures::Fraction(f as f32)
            } else if let Ok(n) = v.extract::<usize>() {
                // explicit count
                MaxFeatures::Count(n)
            } else {
                return Err(PyValueError::new_err(
                    "cfg_dict['max_features'] must be 'sqrt', 'all', a float fraction, or int count",
                ));
            }
        }
        None => MaxFeatures::Sqrt, // sklearn classifier default
    };

    // --- max_depth (None = unlimited) ---
    let max_depth = match cfg_dict.get_item("max_depth")? {
        Some(v) => {
            if v.is_none() {
                None
            } else {
                Some(v.extract::<usize>().map_err(|_| {
                    PyValueError::new_err(
                        "cfg_dict['max_depth'] must be None or a non-negative int",
                    )
                })?)
            }
        }
        None => None,
    };

    let cfg = TrainConfig {
        n_estimators: get_int("n_estimators", 200)?,
        max_depth,
        max_features,
        min_samples_split: get_int("min_samples_split", 2)?,
        min_samples_leaf: get_int("min_samples_leaf", 1)?,
        bootstrap: get_bool("bootstrap", bootstrap_default)?,
        criterion,
        seed: get_u64("seed", 42)?,
        algo,
    };

    cfg.validate().map_err(sylva_error_to_pyerr)?;
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// PyO3 functions
// ---------------------------------------------------------------------------

/// Train a forest using the pure-Rust `CpuBackend` and return an opaque IR
/// handle (a JSON string).
///
/// **TEST-ONLY seam.** Strictly below the Phase-5 estimator API (EST-02).
///
/// Parameters
/// ----------
/// X : numpy ndarray, shape (n_samples, n_features), dtype float32
///     Training feature matrix. Must be C-contiguous.
/// y : numpy ndarray, shape (n_samples,), dtype float32
///     Target vector. For classification, values must be non-negative integer
///     class labels cast to f32 (e.g. 0.0, 1.0, 2.0).
/// cfg_dict : dict
///     Training configuration (see `parse_config` for keys and defaults).
///
/// Returns
/// -------
/// str
///     An opaque JSON-encoded `ForestIR` handle. Pass to `predict_cpu` and
///     `split_statistics`. The format is internal and not part of any public API.
///
/// Raises
/// ------
/// ValueError
///     Bad array shapes/dtypes or invalid config.
/// RuntimeError
///     Internal training error.
#[pyfunction]
#[pyo3(name = "fit_cpu")]
fn py_fit_cpu(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray1<f32>,
    cfg_dict: Bound<'_, PyDict>,
) -> PyResult<String> {
    // Parse config first (cheap, surfaces bad input early).
    let cfg = parse_config(cfg_dict)?;

    // Validate array shapes (boundary security control T-02-16).
    let x_shape = x.shape();
    let y_shape = y.shape();
    if x_shape[0] != y_shape[0] {
        return Err(PyValueError::new_err(format!(
            "fit_cpu: X has {} rows but y has {} elements",
            x_shape[0], y_shape[0]
        )));
    }
    if x_shape[0] == 0 {
        return Err(PyValueError::new_err("fit_cpu: X must have at least 1 row"));
    }
    if x_shape[1] == 0 {
        return Err(PyValueError::new_err(
            "fit_cpu: X must have at least 1 feature",
        ));
    }

    // Zero-copy borrow into ndarray views.
    let x_view = x.as_array();
    let y_view = y.as_array();

    // Release the GIL for the training work (pyo3 0.29: allow_threads → detach).
    let ir_json = py.detach(|| -> Result<String, SylvaError> {
        let backend = crate::cpu::CpuBackend;
        let ir = backend.fit(x_view, y_view, &cfg)?;
        serde_json::to_string(&ir)
            .map_err(|e| SylvaError::InvalidIr(format!("ForestIR serialization failed: {e}")))
    });

    ir_json.map_err(sylva_error_to_pyerr)
}

/// Predict from a `CpuBackend`-trained forest.
///
/// **TEST-ONLY seam.** Strictly below the Phase-5 estimator API (EST-02).
///
/// For classification, returns **class probabilities** (shape `(n_samples, n_classes)`).
/// For regression, returns **predicted values** (shape `(n_samples,)`) — returned
/// as a 2-D array of shape `(n_samples, 1)` for a uniform interface.
///
/// Parameters
/// ----------
/// ir_handle : str
///     Opaque handle returned by `fit_cpu`.
/// X : numpy ndarray, shape (n_samples, n_features), dtype float32
///
/// Returns
/// -------
/// numpy ndarray of float32
///     Classification: shape (n_samples, n_classes) — predicted probabilities.
///     Regression: shape (n_samples, 1) — predicted values.
///
/// Raises
/// ------
/// ValueError
///     Invalid handle or bad array.
/// RuntimeError
///     Internal prediction error.
#[pyfunction]
#[pyo3(name = "predict_cpu")]
fn py_predict_cpu<'py>(
    py: Python<'py>,
    ir_handle: &str,
    x: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Deserialize the handle.
    let ir: ForestIR = serde_json::from_str(ir_handle)
        .map_err(|e| PyValueError::new_err(format!("predict_cpu: invalid ir_handle: {e}")))?;

    let x_view = x.as_array();
    if x_view.ncols() != ir.n_features {
        return Err(PyValueError::new_err(format!(
            "predict_cpu: X has {} features but model has {}",
            x_view.ncols(),
            ir.n_features
        )));
    }

    // Release the GIL for the prediction work (pyo3 0.29: allow_threads → detach).
    let preds = py.detach(|| -> Result<Predictions, SylvaError> {
        let backend = crate::cpu::CpuBackend;
        backend.predict(&ir, x_view)
    });

    let preds = preds.map_err(sylva_error_to_pyerr)?;

    match preds {
        Predictions::ClassProba(arr) => {
            // arr is Array2<f32> with shape (n_samples, n_classes).
            Ok(arr.into_pyarray(py))
        }
        Predictions::Regression(arr) => {
            // Wrap regression output as (n_samples, 1) for uniform interface.
            let n = arr.len();
            let arr2 = arr
                .into_shape_with_order((n, 1))
                .map_err(|e| PyRuntimeError::new_err(format!("reshape error: {e}")))?;
            Ok(arr2.into_pyarray(py))
        }
    }
}

/// Extract aggregate split statistics from a trained forest as a JSON string.
///
/// **TEST-ONLY seam.** The returned JSON encodes a `SplitStats` value:
/// `{ "n_trees": ..., "n_features": ..., "observations": [{ "feature_id": ...,
/// "normalized_threshold": ... }, ...] }`.
///
/// The Phase-05 calibration and KS harness use this to compare feature-selection
/// frequency and normalized threshold distributions between Sylva and sklearn,
/// strictly like-for-like (ET-vs-ET, RF-vs-RF — never crossed).
///
/// Parameters
/// ----------
/// ir_handle : str
///     Opaque handle returned by `fit_cpu`.
///
/// Returns
/// -------
/// str
///     JSON-encoded `SplitStats`.
///
/// Raises
/// ------
/// ValueError
///     Invalid handle.
#[pyfunction]
#[pyo3(name = "split_statistics")]
fn py_split_statistics(ir_handle: &str) -> PyResult<String> {
    let ir: ForestIR = serde_json::from_str(ir_handle)
        .map_err(|e| PyValueError::new_err(format!("split_statistics: invalid ir_handle: {e}")))?;
    let stats = parity::split_statistics(&ir);
    serde_json::to_string(&stats)
        .map_err(|e| PyRuntimeError::new_err(format!("split_statistics: serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// PyO3 module registration
// ---------------------------------------------------------------------------

/// Test-only PyO3 module for the Phase-05 distributional parity harness.
///
/// Module name: `sylva_core_pyseam` (set by `[tool.maturin] module-name` in
/// the parity `pyproject.toml`).
///
/// Exposes: `fit_cpu`, `predict_cpu`, `split_statistics`.
///
/// **NOT the Phase-5 estimator API (EST-02).** Does not expose `predict_proba`,
/// `get_params`, `check_estimator`, `classes_`, or any sklearn-compatible
/// attribute. Those are deferred to Phase 5.
#[pymodule]
pub fn sylva_core_pyseam(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_fit_cpu, module)?)?;
    module.add_function(wrap_pyfunction!(py_predict_cpu, module)?)?;
    module.add_function(wrap_pyfunction!(py_split_statistics, module)?)?;
    Ok(())
}
