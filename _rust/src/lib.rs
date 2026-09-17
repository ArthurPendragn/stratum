use pyo3::prelude::*;
use pyo3::types::PyModule;

mod csr;
mod fd;
mod hashing;
mod one_hot_encoder;
mod tfidf;
mod threads;
mod timing;
mod tokenize;
mod truncated_svd;

#[pymodule]
fn _rust_backend_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<tfidf::TfidfModelHandle>()?;
    m.add_class::<fd::FdEmbedModel>()?;
    m.add_class::<truncated_svd::TruncatedSvdModel>()?;
    m.add_function(wrap_pyfunction!(tfidf::hashing_tfidf_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::hashing_tfidf_csr_with_idf, m)?)?;
    m.add_function(wrap_pyfunction!(fd::fd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fd::fd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd::truncated_svd_fit_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(truncated_svd::truncated_svd_transform_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::ohe_transform_csr, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot_encoder::csr_to_dense, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::tfidf_fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf::tfidf_transform_csr, m)?)?;
    Ok(())
}
