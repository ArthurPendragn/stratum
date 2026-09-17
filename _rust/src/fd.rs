use ndarray::{Array2, ArrayView2, Axis, s};
use ndarray_linalg::{SVDInto, SVD};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::ThreadPool;
use rayon::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

use crate::threads::get_thread_pool;
use crate::timing::{print_timing, start_timing};

/// Python-owned fitted Frequent-Directions state.
#[pyclass(name = "_FdEmbedModelHandle", frozen)]
pub(crate) struct FdEmbedModel {
    n_cols: usize,
    k: usize,
    oversample: usize,
    // Projection matrix P: (m × k) where m = k + oversample
    projection: Arc<Array2<f32>>,
    // Random matrix Ω: (n_cols × m) stored column-major for efficient CSR matmul
    // omega[j * m + t] = Ω[j, t]
    omega: Arc<Vec<f32>>,
    m: usize,  // m = k + oversample, width of reduced space
}

// Simple Frequent Directions (FD) implementation for a tall matrix Y (n x m), where
// m is small (≈ k+p). Maintain a sketch B (l x m) with l = 2k (or slightly larger), then shrinks.
pub fn fd_reduce(y: ArrayView2<f32>, k: usize, pool_ref: Option<&ThreadPool>) -> Result<Array2<f32>, PyErr> {
    let n = y.nrows();
    let m = y.ncols();
    if k == 0 || k > m {
        return Err(PyErr::new::<PyValueError, _>("Invalid k"));
    }

    // Use ell = 8*k to reduce shrink frequency. With k=30, this means:
    // - ell = 240, shrink every ~210 rows
    // Each shrink does a full SVD on the sketch matrix.
    let ell = (8 * k).max(k + 1);

    // The sketch
    let mut b = Array2::<f32>::zeros((ell, m));
    let mut filled = 0usize;

    // Stream rows of Y to B till filled
    for i in 0..n {
        if filled < ell {
            b.slice_mut(s![filled, ..]).assign(&y.slice(s![i, ..]));
            filled += 1;
            continue;
        }
        // B is full. Shrink.
        shrink(&mut b, k)?;
        // After shrink, k rows remain. Append ith row at position k
        b.slice_mut(s![k, ..]).assign(&y.slice(s![i, ..]));
        filled = k + 1;
    }

    // Final shrink
    if filled > k {
        shrink(&mut b, k)?;
    }

    // At this point, top-k rows of B span principal directions in the reduced space (m-dim).
    // Project Y -> Z (n x k): Z = Y · (V_k), but here rows of B already align; use the least squares:
    // We compute R = qr(B_top) and use its columns as basis; for simplicity, do SVD of B_top.
    let b_top = b.slice(s![0..k, ..]).to_owned();
    let (u_opt, s_vec, vt_opt) = b_top
        .svd_into(true, true)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("SVD failed: {e}")))?;
    let _u = u_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD: U missing"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD: VT missing"))?;
    let p = vt.t().to_owned();

    // Z = Y.P -> (n x m) . (m x k) = (n x k)
    // n is large and m, k are small. Parallelization is better than BLAS/MKL (prefers big blocks)
    let mut z = Array2::<f32>::zeros((n, k));
    let mut do_work = || {
        z.axis_iter_mut(Axis(0)) //mutable view of each row
            .into_par_iter() //convert to parallel iterator
            .zip(y.axis_iter(Axis(0))) //combine z.row_mut and y.row
            .for_each(|(mut zrow, yrow)| {
                for r in 0..k {
                    let mut sum = 0.0f32;
                    for c in 0..m {
                        sum += yrow[c] * p[(c, r)];
                    }
                    zrow[r] = sum;
                }
            });
    };
    match pool_ref {
        Some(pool) => pool.install(do_work), //use custom threadpool
        None => do_work() //use global threadpool
    }

    Ok(z)
}

/// FD fit: computes both the projection Z and the projection matrix P.
/// Returns (Z, P) where:
/// - Z: (n × k) the reduced embeddings
/// - P: (m × k) the projection matrix to transform new data
pub fn fd_fit(y: ArrayView2<f32>, k: usize, pool_ref: Option<&ThreadPool>) -> Result<(Array2<f32>, Array2<f32>), PyErr> {
    let n = y.nrows();
    let m = y.ncols();
    if k == 0 || k > m {
        return Err(PyErr::new::<PyValueError, _>("Invalid k"));
    }

    // Use ell = 8*k to reduce shrink frequency. With k=30, this means:
    // - ell = 240, shrink every ~210 rows
    // Each shrink does a full SVD on the sketch matrix.
    let ell = (8 * k).max(k + 1);

    // The sketch
    let mut b = Array2::<f32>::zeros((ell, m));
    let mut filled = 0usize;

    // Stream rows of Y to B till filled
    for i in 0..n {
        if filled < ell {
            b.slice_mut(s![filled, ..]).assign(&y.slice(s![i, ..]));
            filled += 1;
            continue;
        }
        // B is full. Shrink.
        shrink(&mut b, k)?;
        // After shrink, k rows remain. Append ith row at position k
        b.slice_mut(s![k, ..]).assign(&y.slice(s![i, ..]));
        filled = k + 1;
    }

    // At this point, top-k rows of B span principal directions in the reduced space (m-dim).
    // Project Y -> Z (n x k): Z = Y · (V_k), but here rows of B already align; use the least squares:
    let b_filled = b.slice(s![0..filled, ..]);
    let (_, _, vt_opt) = b_filled
        .svd(false, true) // We only need VT, so we skip computing U
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("Final SVD failed: {e}")))?;

    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("Final SVD: VT missing"))?;

    // P (m x k) is the transpose of the first k rows of VT
    let p = vt.slice(s![0..k, ..]).t().to_owned();

    // Z = Y.P -> (n x m) . (m x k) = (n x k)
    let mut z = Array2::<f32>::zeros((n, k));
    ndarray::linalg::general_mat_mul(1.0, &y, &p, 0.0, &mut z);

    Ok((z, p))
}

// Shrink step. Do SVD of B (ell x m).
fn shrink(b: &mut Array2<f32>, k: usize) -> Result<(), PyErr> {
    // 1. Compute SVD. Note: b.view().svd() returns owned U and VT matrices.
    let (u_opt, s_vec, vt_opt) = b.view()
        .svd(true, true)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("SVD (shrink) failed: {e}")))?;

    let mut u = u_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): U missing"))?;
    let vt = vt_opt.ok_or_else(|| PyErr::new::<PyValueError, _>("SVD (shrink): VT missing"))?;

    // 2. Compute delta (s_k^2)
    // s_vec is already sorted in descending order by ndarray-linalg.
    let s_k = s_vec.get(k.saturating_sub(1)).copied().unwrap_or(0.0);
    let delta = s_k * s_k;

    // 3. Scale U in-place
    // Instead of allocating a new 'u_scaled' matrix, we modify the columns of 'u' directly.
    let r = s_vec.len();
    for j in 0..r {
        let s_sq = s_vec[j] * s_vec[j];
        let s_shrunk = if s_sq > delta { (s_sq - delta).sqrt() } else { 0.0 };

        // Scale column j of U by s_shrunk.
        // If s_shrunk is 0, this effectively zeros out directions beyond the top singular values.
        let mut col = u.column_mut(j);
        col *= s_shrunk;
    }

    // 4. Direct Recomposition into 'b'
    // We use views of the modified U and the original VT to avoid any '.to_owned()' calls.
    let u_view = u.slice(s![.., 0..r]);
    let vt_view = vt.slice(s![0..r, ..]);

    // general_mat_mul(alpha, A, B, beta, C) computes C = alpha*A*B + beta*C.
    // By setting beta to 0.0, we overwrite the contents of 'b' without an intermediate allocation.
    ndarray::linalg::general_mat_mul(1.0, &u_view, &vt_view, 0.0, b);

    Ok(())
}

// Helper: CSR × Omega matmul: X @ Ω -> Y
// Omega is stored column-major: omega[j * m + t] = Ω[j, t]
// Result Y is (n_rows × m)
fn csr_matmul_omega(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    omega: &[f32],
    m: usize,
    pool_ref: Option<&rayon::ThreadPool>,
) -> Array2<f32> {
    let mut y = Array2::<f32>::zeros((n_rows, m));
    let mut build_y = || {
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut yrow)| {
                let start = indptr[row] as usize;
                let end = indptr[row + 1] as usize;
                for t in 0..m {
                    let mut acc = 0.0f32;
                    for p in start..end {
                        let j = indices[p] as usize;
                        let v = data[p];
                        acc += v * omega[j * m + t];
                    }
                    yrow[t] = acc;
                }
            });
    };
    match pool_ref {
        Some(p) => p.install(build_y),
        None => build_y(),
    }
    y
}

fn compute_fd_embed(data: &[f32], indices: &[i32], indptr: &[i64],
    n_rows: usize, n_cols: usize, k: usize, oversample: usize, seed: Option<u64>) -> Result<Array2<f32>, PyErr>
{
    // Step 2: Gather the parameters
    let out_w = k + oversample; //k+p
    let s = seed.unwrap_or(0xC0FFEE); //I love coffee :)

    // Step 3: Build Ω (d x out_w), but don't store full Ω. Generate on the fly per-column.
    // We pre-allocate Ω^T as Vec<Vec<f32>>; width is small (<= 128).
    // Do all heavy work without the GIL (detach closure)
    // TODO: Avoid materializing omega. Stream random f32 numbers in during building Y
    let mut rng = StdRng::seed_from_u64(s);
    let mut omega_t: Vec<Vec<f32>> = Vec::with_capacity(out_w);
    for _ in 0..out_w {
        let mut col: Vec<f32> = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            let r: f32 = if rng.random::<bool>() { 1.0 } else { -1.0 };
            col.push(r); //col is a vector of 1s and -1s
        }
        omega_t.push(col);
    }

    // Get rayon thread pool
    let pool = get_thread_pool();

    // Step 4: Compute Y = X · Ω  (n x out_w) in a single pass over CSR rows
    // TODO: Move this to CSR utility module
    let t0 = start_timing();
    let mut y = Array2::<f32>::zeros((n_rows, out_w)); //dense y
    let mut build_y = || {
        y.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut yrow)| {
                let start = indptr[row] as usize;
                let end   = indptr[row + 1] as usize;
                for t in 0..out_w {
                    let mut acc = 0.0f32;
                    for p in start..end {
                        let j = indices[p] as usize;
                        let v = data[p];
                        acc += v * omega_t[t][j];
                    }
                    yrow[t] = acc;
                }
            });
    };
    match pool {
        Some(p) => p.install(build_y), //use custom threadpool
        None => build_y() //use global threadpool
    }
    print_timing("build y", t0);

    // Step 5: Run FD on Y (n x out_w) -> Z (n x k)
    // FD operates on small width (out_w), making it cheap
    let t0 = start_timing();
    let z = fd_reduce(y.view(), k, pool)?;
    print_timing("fd_reduce", t0);
    Ok(z)
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, k, oversample=16, seed=None))]
fn fd_embed_from_csr(py: Python<'_>, data: Bound<PyArray1<f32>>, indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>, n_rows: usize, n_cols: usize, k: usize,
    oversample: usize, seed: Option<u64>) -> PyResult<Py<PyArray2<f32>>>
{
    // Step 1: Zero-copy view of NumPy arrays
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let z = py.detach(||
        compute_fd_embed(data, indices, indptr, n_rows, n_cols, k, oversample, seed))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_embed failed: {e}")))?;

    // Step 6: Return NumPy (zero-copy)
    let py_z = z.into_pyarray(py).to_owned();
    Ok(Py::from(py_z))
}

fn compute_fd_fit(
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
    k: usize,
    oversample: usize,
    seed: Option<u64>,
) -> Result<(FdEmbedModel, Array2<f32>), PyErr> {
    let m = k + oversample;
    let s = seed.unwrap_or(0xC0FFEE);

    // Generate Ω matrix (n_cols × m) in column-major format
    let mut rng = StdRng::seed_from_u64(s);
    let mut omega = Vec::<f32>::with_capacity(n_cols * m);
    for _ in 0..(n_cols * m) {
        let r: f32 = if rng.random::<bool>() { 1.0 } else { -1.0 };
        omega.push(r);
    }

    // Get rayon thread pool
    let pool = get_thread_pool();

    // Compute Y = X @ Ω (n_rows × m)
    let t0 = start_timing();
    let y = csr_matmul_omega(data, indices, indptr, n_rows, n_cols, &omega, m, pool);
    print_timing("build y (fd_fit)", t0);

    // Run FD to get projection matrix P and reduced embeddings Z
    let t0 = start_timing();
    let (z, projection) = fd_fit(y.view(), k, pool)?;
    print_timing("fd_fit", t0);

    // The model is allocated and remains in the Rust heap
    let model = FdEmbedModel {
        n_cols,
        k,
        oversample,
        projection: Arc::new(projection),
        omega: Arc::new(omega),
        m,
    };

    Ok((model, z))
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, n_rows, n_cols, k, oversample=16, seed=None))]
pub(crate) fn fd_fit_from_csr(
    py: Python<'_>,
    data: Bound<PyArray1<f32>>,
    indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>,
    n_rows: usize,
    n_cols: usize,
    k: usize,
    oversample: usize,
    seed: Option<u64>,
) -> PyResult<(Py<FdEmbedModel>, Py<PyArray2<f32>>)> {
    // Zero-copy view of NumPy arrays
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };

    let (model, z) = py
        .detach(|| compute_fd_fit(data, indices, indptr, n_rows, n_cols, k, oversample, seed))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_fit failed: {e}")))?;

    // The fitted state (wrapper) is owned by Python and released with its last reference.
    let model = Py::new(py, model)?;
    let py_z = z.into_pyarray(py).to_owned();
    Ok((model, Py::from(py_z)))
}

fn compute_fd_transform(
    projection: Arc<Array2<f32>>,
    omega: Arc<Vec<f32>>,
    model_n_cols: usize,
    m: usize,
    data: &[f32],
    indices: &[i32],
    indptr: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Array2<f32>, PyErr> {
    // Validate cols match
    if n_cols != model_n_cols {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "n_cols mismatch: input n_cols={} but model expects {}",
            n_cols, model_n_cols
        )));
    }

    let pool = get_thread_pool();

    // Compute Y_new = X_new @ Ω (n_rows × m)
    let t0 = start_timing();
    let y_new = csr_matmul_omega(data, indices, indptr, n_rows, n_cols, &omega, m, pool);
    print_timing("build y_new (fd_transform)", t0);

    // Apply projection Z_new = Y_new @ P (n_rows × k)
    let t0 = start_timing();
    let k = projection.ncols();
    let mut z_new = Array2::<f32>::zeros((n_rows, k));
    let mut apply_projection = || {
        z_new
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(y_new.axis_iter(Axis(0)))
            .for_each(|(mut zrow, yrow)| {
                for r in 0..k {
                    let mut sum = 0.0f32;
                    for c in 0..m {
                        sum += yrow[c] * projection[(c, r)];
                    }
                    zrow[r] = sum;
                }
            });
    };
    match pool {
        Some(p) => p.install(apply_projection),
        None => apply_projection(),
    }
    print_timing("apply projection (fd_transform)", t0);

    Ok(z_new)
}

#[pyfunction]
#[pyo3(signature = (model_id, data, indices, indptr, n_rows, n_cols))]
pub(crate) fn fd_transform_from_csr(
    py: Python<'_>,
    model_id: PyRef<'_, FdEmbedModel>,
    data: Bound<PyArray1<f32>>,
    indices: Bound<PyArray1<i32>>,
    indptr: Bound<PyArray1<i64>>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let data = unsafe { data.as_slice()? };
    let indices = unsafe { indices.as_slice()? };
    let indptr = unsafe { indptr.as_slice()? };
    let projection = Arc::clone(&model_id.projection);
    let omega = Arc::clone(&model_id.omega);
    let model_n_cols = model_id.n_cols;
    let m = model_id.m;
    drop(model_id);

    let z = py
        .detach(|| {
            compute_fd_transform(
                projection,
                omega,
                model_n_cols,
                m,
                data,
                indices,
                indptr,
                n_rows,
                n_cols,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fd_transform failed: {e}")))?;

    let py_z = z.into_pyarray(py).to_owned();
    Ok(Py::from(py_z))
}
