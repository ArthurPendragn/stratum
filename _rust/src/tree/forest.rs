use std::sync::Arc;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::builder::{build_tree, BuildParams};
use super::exact::ExactSplitFinder;
use super::model::TreeModel;
use super::rng::NumpyRng;

// Immutable serial forest sharing the compact tree representation and traversal.
pub(crate) struct ForestModel {
    trees: Vec<TreeModel>,
    n_classes: usize,
    n_features: usize,
}

#[pyclass(name = "_ForestModelHandle", frozen)]
pub(crate) struct ForestModelHandle {
    model: Arc<ForestModel>,
}

impl ForestModel {
    fn predict(&self, x: &[f32], n_rows: usize) -> Vec<f64> {
        let mut probabilities = vec![0.0; n_rows * self.n_classes];
        let scale = 1.0 / self.trees.len() as f64;
        for tree in &self.trees {
            for (row_idx, row) in x.chunks_exact(self.n_features).enumerate() {
                let leaf = tree.leaf_for_row(row);
                let source = &tree.values[leaf * self.n_classes..(leaf + 1) * self.n_classes];
                let target = &mut probabilities
                    [row_idx * self.n_classes..(row_idx + 1) * self.n_classes];
                for class in 0..self.n_classes {
                    target[class] += source[class] * scale;
                }
            }
        }
        probabilities
    }
}

// Fit trees serially. Bootstrap samples are represented by multiplicity
// weights, so duplicate source rows are never copied or materialized.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn forest_fit_exact(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    y: PyReadonlyArray1<'_, i64>,
    tree_seeds: PyReadonlyArray1<'_, i64>,
    n_classes: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: usize,
    min_impurity_decrease: f64,
    max_leaf_nodes: Option<usize>,
    bootstrap: bool,
    n_bootstrap: usize,
) -> PyResult<Py<ForestModelHandle>> {
    let shape = x.shape();
    let n_rows = shape[0];
    let n_features = shape[1];
    let x = x.as_slice()?;
    let y_raw = y.as_slice()?;
    let seeds = tree_seeds.as_slice()?;
    if y_raw.len() != n_rows || seeds.is_empty() {
        return Err(PyValueError::new_err("training arrays or tree seeds are empty/inconsistent"));
    }
    let labels: Result<Vec<usize>, _> = y_raw
        .iter()
        .map(|&label| {
            if label < 0 || label as usize >= n_classes {
                Err(PyValueError::new_err("encoded class is outside n_classes"))
            } else {
                Ok(label as usize)
            }
        })
        .collect();
    let labels = labels?;
    let params = BuildParams {
        n_features,
        n_classes,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        max_leaf_nodes,
    };
    let result = py.detach(|| {
        let mut trees = Vec::with_capacity(seeds.len());
        let mut weights = vec![1.0; n_rows];
        for &seed in seeds {
            let seed = seed as u32;
            weights.fill(if bootstrap { 0.0 } else { 1.0 });
            if bootstrap {
                let mut bootstrap_rng = NumpyRng::new(seed);
                for _ in 0..n_bootstrap {
                    weights[bootstrap_rng.interval((n_rows - 1) as u32) as usize] += 1.0;
                }
            }
            // sklearn initializes the splitter's xorshift state from an
            // independent RandomState seeded with the estimator's tree seed.
            let split_seed = NumpyRng::new(seed).interval(i32::MAX as u32 - 1);
            let mut finder = ExactSplitFinder::new(
                n_features,
                split_seed,
                max_features,
                min_samples_leaf,
                0.0,
            );
            trees.push(build_tree(x, &labels, &weights, params, &mut finder)?);
        }
        Ok::<_, String>(ForestModel {
            trees,
            n_classes,
            n_features,
        })
    }).map_err(PyValueError::new_err)?;
    Py::new(py, ForestModelHandle { model: Arc::new(result) })
}

#[pyfunction]
pub(crate) fn forest_predict(
    py: Python<'_>,
    model: PyRef<'_, ForestModelHandle>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let shape = x.shape();
    if shape[1] != model.model.n_features {
        return Err(PyValueError::new_err(format!(
            "X has {} features, but the forest expects {}",
            shape[1], model.model.n_features
        )));
    }
    let n_rows = shape[0];
    let x = x.as_slice()?;
    let model = Arc::clone(&model.model);
    let probabilities = py.detach(|| model.predict(x, n_rows));
    let output = Array2::from_shape_vec((n_rows, model.n_classes), probabilities)
        .expect("forest prediction output shape is exact");
    Ok(Py::from(output.into_pyarray(py).to_owned()))
}

#[pyfunction]
pub(crate) fn forest_model_info(
    py: Python<'_>,
    model: PyRef<'_, ForestModelHandle>,
) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    let n_nodes: usize = model.model.trees.iter().map(|tree| tree.children_left.len()).sum();
    let model_bytes: usize = model.model.trees.iter().map(|tree| {
        tree.children_left.len() * (3 * size_of::<i64>() + 4 * size_of::<f64>() + size_of::<bool>())
            + tree.values.len() * size_of::<f64>()
            + tree.feature_importances.len() * size_of::<f64>()
    }).sum();
    let mut importances = vec![0.0; model.model.n_features];
    for tree in &model.model.trees {
        for (target, source) in importances.iter_mut().zip(&tree.feature_importances) {
            *target += source / model.model.trees.len() as f64;
        }
    }
    dict.set_item("n_estimators", model.model.trees.len())?;
    dict.set_item("n_nodes", n_nodes)?;
    dict.set_item("model_bytes", model_bytes)?;
    dict.set_item(
        "root_n_node_samples",
        PyArray1::from_vec(
            py,
            model.model.trees.iter().map(|tree| tree.n_node_samples[0]).collect(),
        ),
    )?;
    dict.set_item(
        "root_weighted_n_node_samples",
        PyArray1::from_vec(
            py,
            model
                .model
                .trees
                .iter()
                .map(|tree| tree.weighted_n_node_samples[0])
                .collect(),
        ),
    )?;
    dict.set_item("feature_importances", PyArray1::from_vec(py, importances))?;
    Ok(dict.into_any().unbind())
}
