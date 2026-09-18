use anyhow::Result;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator};
use rayon::prelude::*;
use crate::{tokenize, hashing};
use crate::csr::CsrParts;
use numpy::ndarray::Array1;
use ahash::AHasher;
use ahash::AHashMap as HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Python-owned fitted TF-IDF state.
#[pyclass(name = "_TfidfModelHandle", frozen)]
pub(crate) struct TfidfModelHandle {
    model: Arc<TfidfModel>,
}

#[derive(Debug)]
pub enum Error {
    InvalidAnalyzer,
    InvalidNgramRange,
    Internal,
}

#[derive(Clone)]
pub struct TfidfModel {
    pub analyzer: tokenize::Analyzer,
    pub nmin: usize,
    pub nmax: usize,
    pub vocab: HashMap<String, i32>, // token -> col_id
    pub idf: Vec<f32>,               // len = vocab_size
    pub n_cols: usize,               // vocab_size (cache)
}

// TODO (refactor): Move fit and transform to the same struct
impl TfidfModel {
    // Transform docs using the stored vocabulary + stored IDF.
    // Returns CSR parts (data, indices, indptr) for shape (n_rows, n_cols).
    // TODO: consider sublinear_tf, alternate normalizations, etc.
    pub fn transform_csr(&self, docs: &[String]) -> Result<(Vec<f32>, Vec<i32>, Vec<i64>), Error> {
        let n_rows = docs.len();
        let mut data: Vec<f32> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        indptr.push(0);

        // Temporary buffers (reused per document)
        let mut tokens_char: Vec<&str> = Vec::new();
        let mut tokens_wb: Vec<String> = Vec::new();

        for s in docs {
            tokens_char.clear();
            tokens_wb.clear();

            match self.analyzer {
                tokenize::Analyzer::Char => {
                    tokenize::char_ngrams(s, self.nmin, self.nmax, &mut tokens_char);
                }
                tokenize::Analyzer::Char_wb => {
                    // FIXME (perf): Allocation per string
                    tokenize::char_wb_ngrams(s, self.nmin, self.nmax, &mut tokens_wb);
                }
            }

            // Count term frequencies over *in-vocab* features only.
            // Key = column index (usize), Value = count
            let mut tf_map: HashMap<usize, u32> = HashMap::new();

            match self.analyzer {
                tokenize::Analyzer::Char => {
                    for t in tokens_char.iter() {
                        // Need owned String to look up in vocab.
                        // TODO(perf): avoid allocation by changing vocab to hashed keys or using Cow.
                        let key = t.to_string();
                        if let Some(&col_i32) = self.vocab.get(&key) {
                            let col = col_i32 as usize;
                            if col < self.n_cols {
                                *tf_map.entry(col).or_insert(0) += 1;
                            }
                        }
                    }
                }
                tokenize::Analyzer::Char_wb => {
                    for t in tokens_wb.iter() {
                        if let Some(&col_i32) = self.vocab.get(t) {
                            let col = col_i32 as usize;
                            if col < self.n_cols {
                                *tf_map.entry(col).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }

            // Convert to sorted list by column for stable CSR
            let mut items: Vec<(usize, u32)> = tf_map.into_iter().collect();
            //items.sort_by_key(|(col, _)| *col);

            // Append row entries and compute norm
            let row_start = data.len();
            let mut norm_sq = 0.0f32;

            for (col, cnt) in items.iter() {
                let tf = *cnt as f32;
                // idf length is n_cols; bounds-safe
                let idf = self.idf.get(*col).copied().unwrap_or(0.0);
                let v = tf * idf;

                indices.push(*col as i32);
                data.push(v);
                norm_sq += v * v;
            }

            // L2 normalize row
            let row_end = data.len();
            if norm_sq > 0.0 {
                let scale = 1.0 / norm_sq.sqrt();
                for v in &mut data[row_start..row_end] {
                    *v *= scale;
                }
            }

            indptr.push(row_end as i64);
        }

        Ok((data, indices, indptr))
    }
}


pub(crate) struct Builder {
    analyzer: tokenize::Analyzer,
    nmin: usize,
    nmax: usize,
    nfeat: usize,
}

impl Builder {
    pub fn new(analyzer: &str, nmin: usize, nmax: usize, nfeat: usize) -> Result<Self, Error> {
        let a = match analyzer {
            "char" => tokenize::Analyzer::Char,
            "char_wb" => tokenize::Analyzer::Char_wb,
            _ => return Err(Error::InvalidAnalyzer)
        };
        if nmin == 0 || nmin > nmax {
            return Err(Error::InvalidNgramRange);
        }
        Ok(Self {
            analyzer: a,
            nmin,
            nmax,
            nfeat
        })
    }

    pub fn build_csr(&self, docs: &[String]) -> Result<(Vec<f32>, Vec<i32>, Vec<i64>, Array1<f32>), Error> {
        let n_docs = docs.len();
        let mut df = vec![0u32; self.nfeat];

        // Step1: Create per-doc hashmaps and document frequency (DF)
        let mut per_doc: Vec<HashMap<usize, u32>> = Vec::with_capacity(n_docs); //list of token frequencies
        for s in docs {
            let mut tokens_char: Vec<&str> = Vec::new();
            let mut tokens_wb: Vec<String> = Vec::new(); //FIXME: Avoid allocating Strings
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    tokenize::char_ngrams(s, self.nmin, self.nmax, &mut tokens_char);
                }
                tokenize::Analyzer::Char_wb => {
                    tokenize::char_wb_ngrams(s, self.nmin, self.nmax, &mut tokens_wb);
                }
            }

            let mut seen: HashMap<usize, u32> = HashMap::new(); // bucket -> count
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    for t in tokens_char.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
                tokenize::Analyzer::Char_wb => {
                    for t in tokens_wb.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
            }

            // update DF once per present feature
            for (&feat, _) in seen.iter() {
                df[feat] += 1;
            }
            per_doc.push(seen);
        }

        // Step2: Compute smoothed IDF: log((1 + n_docs) / (1 + df)) + 1
        let n_docs_f = n_docs as f32;
        let mut idf = Array1::<f32>::zeros(self.nfeat);
        for j in 0..self.nfeat {
            let dfj = df[j] as f32;
            idf[j] = ((1.0 + n_docs_f) / (1.0 + dfj)).ln() + 1.0;
        }

        // Step3: Build CSR with TF*IDF values and per-row L2 normalization
        let mut csr = CsrParts::with_capacity(n_docs, 0);
        let mut nnz_cum = 0i64;

        for doc_map in per_doc.into_iter() {
            // Convert to sorted vector for stable indices (optional)
            let mut items: Vec<(usize, u32)> = doc_map.into_iter().collect();
            items.sort_by_key(|&(k, _)| k);

            // L2 norm accumulator
            let row_start = csr.data.len();
            let mut norm_sq: f32 = 0.0;
            for (feat, cnt) in items.iter() {
                let tf = *cnt as f32; // TODO: match sklearn's sublinear_tf/norm options
                let val = tf * idf[*feat];
                csr.indices.push(*feat as i32);
                csr.data.push(val);
                norm_sq += val * val;
            }

            let row_end = csr.data.len();
            if norm_sq > 0.0 {
                let scale = 1.0 / norm_sq.sqrt();
                for v in &mut csr.data[row_start..row_end] {
                    *v *= scale;
                }
            }

            // Update cumulative nnz and indptr
            let row_nnz = (row_end - row_start) as i64;
            nnz_cum += row_nnz;
            csr.indptr.push(nnz_cum);
        }

        // Return to get the Python integration wired
        Ok((csr.data, csr.indices, csr.indptr, idf))
    }

    /// Build CSR with pre-computed IDF (for transform).
    /// This skips DF/IDF computation and uses the provided IDF vector.
    pub fn build_csr_with_idf(&self, docs: &[String], idf: &[f32]) -> Result<(Vec<f32>, Vec<i32>, Vec<i64>), Error> {
        if idf.len() != self.nfeat {
            return Err(Error::Internal);
        }
        let n_docs = docs.len();

        // Step1: Create per-doc hashmaps (TF only, no DF counting)
        let mut per_doc: Vec<HashMap<usize, u32>> = Vec::with_capacity(n_docs);
        for s in docs {
            let mut tokens_char: Vec<&str> = Vec::new();
            let mut tokens_wb: Vec<String> = Vec::new();
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    tokenize::char_ngrams(s, self.nmin, self.nmax, &mut tokens_char);
                }
                tokenize::Analyzer::Char_wb => {
                    tokenize::char_wb_ngrams(s, self.nmin, self.nmax, &mut tokens_wb);
                }
            }

            let mut seen: HashMap<usize, u32> = HashMap::new();
            match self.analyzer {
                tokenize::Analyzer::Char => {
                    for t in tokens_char.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
                tokenize::Analyzer::Char_wb => {
                    for t in tokens_wb.iter() {
                        let b = hashing::bucket_id(t, self.nfeat);
                        *seen.entry(b).or_insert(0) += 1;
                    }
                }
            }
            per_doc.push(seen);
        }

        // Step2: Build CSR with TF*IDF values using pre-computed IDF and per-row L2 normalization
        let mut csr = CsrParts::with_capacity(n_docs, 0);
        let mut nnz_cum = 0i64;

        for doc_map in per_doc.into_iter() {
            // Convert to sorted vector for stable indices
            let mut items: Vec<(usize, u32)> = doc_map.into_iter().collect();
            items.sort_by_key(|&(k, _)| k);

            // L2 norm accumulator
            let row_start = csr.data.len();
            let mut norm_sq: f32 = 0.0;
            for (feat, cnt) in items.iter() {
                let tf = *cnt as f32;
                let val = tf * idf[*feat];
                csr.indices.push(*feat as i32);
                csr.data.push(val);
                norm_sq += val * val;
            }

            let row_end = csr.data.len();
            if norm_sq > 0.0 {
                let scale = 1.0 / norm_sq.sqrt();
                for v in &mut csr.data[row_start..row_end] {
                    *v *= scale;
                }
            }

            // Update cumulative nnz and indptr
            let row_nnz = (row_end - row_start) as i64;
            nnz_cum += row_nnz;
            csr.indptr.push(nnz_cum);
        }

        Ok((csr.data, csr.indices, csr.indptr))
    }
}

pub struct VocabBuilder {
    pub analyzer: tokenize::Analyzer,
    pub nmin: usize,
    pub nmax: usize,
}

impl VocabBuilder {
    pub fn new(analyzer: &str, nmin: usize, nmax: usize) -> Result<Self, Error> {
        let a = match analyzer {
            "char" => tokenize::Analyzer::Char,
            "char_wb" => tokenize::Analyzer::Char_wb,
            _ => return Err(Error::InvalidAnalyzer)
        };
        if nmin == 0 || nmin > nmax {
            return Err(Error::InvalidNgramRange);
        }
        Ok(Self {
            analyzer: a,
            nmin,
            nmax
        })
    }
}


// Helper: Hash a token string to u64
fn hash_token(token: &str) -> u64 {
    let mut hasher = AHasher::default();
    token.hash(&mut hasher);
    hasher.finish()
}

impl VocabBuilder {
    pub fn fit_csr(&self, docs: &[String]) -> Result<(TfidfModel, Vec<f32>, Vec<i32>, Vec<i64>), Error> {
        let n_docs = docs.len();

        let mut vocab: HashMap<String, i32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new(); // grows with vocab
        let mut per_doc: Vec<HashMap<i32, u32>> = Vec::with_capacity(n_docs);

        // Parallel tokenize + per-doc token counts (by token string)
        let t0 = crate::timing::start_timing();
        let per_doc_tokens: Vec<HashMap<String, u32>> = docs
            .par_iter()
            .map(|s| {
                // Tokenize into tokens (char uses &str slices; char_wb returns owned Strings)
                let mut tokens_char: Vec<&str> = Vec::new();
                let mut tokens_wb: Vec<String> = Vec::new();

                match self.analyzer {
                    tokenize::Analyzer::Char => tokenize::char_ngrams(s, self.nmin, self.nmax, &mut tokens_char),
                    tokenize::Analyzer::Char_wb => tokenize::char_wb_ngrams(s, self.nmin, self.nmax, &mut tokens_wb),
                }

                // Local term counts by token string (no shared state => parallel-safe)
                let mut counts_tok: HashMap<String, u32> = HashMap::new();

                match self.analyzer {
                    tokenize::Analyzer::Char => {
                        for &t in tokens_char.iter() {
                            // unavoidable allocation here if we want parallelism without shared vocab
                            *counts_tok.entry(t.to_owned()).or_insert(0) += 1;
                        }
                    }
                    tokenize::Analyzer::Char_wb => {
                        for t in tokens_wb.iter() {
                            *counts_tok.entry(t.clone()).or_insert(0) += 1;
                        }
                    }
                }

                counts_tok
            })
            .collect();


        // Sequential merge into shared vocab/df and produce per_doc column-counts
        for counts_tok in per_doc_tokens {
            let mut counts: HashMap<i32, u32> = HashMap::with_capacity(counts_tok.len());

            for (tok, cnt) in counts_tok {
                // Entry API handles "get or insert" in a single hash/lookup
                let col = *vocab.entry(tok).or_insert_with(|| {
                    let new_id = df.len() as i32;
                    df.push(0);
                    new_id
                });

                let e = counts.entry(col).or_insert(0);
                if *e == 0 {
                    df[col as usize] += 1; // Increment DF directly here
                }
                *e += cnt;
            }
            per_doc.push(counts);
        }
        crate::timing::print_timing("tfidf tokenize and DF update", t0);

        // Compute IDF
        let n_cols = df.len();
        let n_docs_f = n_docs as f32;

        let mut idf: Vec<f32> = vec![0.0; n_cols];
        for j in 0..n_cols {
            let dfj = df[j] as f32;
            idf[j] = ((1.0 + n_docs_f) / (1.0 + dfj)).ln() + 1.0;
        }

        // Build CSR (values already TF-IDF and L2 normalized)
        let mut csr = CsrParts::with_capacity(n_docs, 0);
        let mut nnz_cum = 0i64;

        let t0 = crate::timing::start_timing();
        for doc_map in per_doc.into_iter() {
            let mut items: Vec<(i32, u32)> = doc_map.into_iter().collect();
            //items.sort_by_key(|&(k, _)| k);

            let row_start = csr.data.len();
            let mut norm_sq = 0.0f32;

            for (col, cnt) in items.iter() {
                let tf = *cnt as f32;
                let val = tf * idf[*col as usize];
                csr.indices.push(*col);
                csr.data.push(val);
                norm_sq += val * val;
            }

            let row_end = csr.data.len();
            if norm_sq > 0.0 {
                let scale = 1.0 / norm_sq.sqrt();
                for v in &mut csr.data[row_start..row_end] { *v *= scale; }
            }

            nnz_cum += (row_end - row_start) as i64;
            csr.indptr.push(nnz_cum);
        }
        crate::timing::print_timing("tfidf build CSR output", t0);

        let model = TfidfModel {
            analyzer: self.analyzer,
            nmin: self.nmin,
            nmax: self.nmax,
            vocab,
            idf,
            n_cols,
        };

        Ok((model, csr.data, csr.indices, csr.indptr))
    }
}


fn get_or_insert_col(vocab: &mut HashMap<String, i32>, df: &mut Vec<u32>, token: &str) -> i32 {
    if let Some(&col) = vocab.get(token) {
        return col;
    }
    let col = df.len() as i32;
    vocab.insert(token.to_string(), col);
    df.push(0);
    col
}

fn get_or_insert_col_owned(vocab: &mut HashMap<String, i32>, df: &mut Vec<u32>, token: &String) -> i32 {
    if let Some(&col) = vocab.get(token) {
        return col;
    }
    let col = df.len() as i32;
    vocab.insert(token.clone(), col);
    df.push(0);
    col
}

// Simple mapping from domain error to PyErr
fn to_pyerr(err: Error) -> PyErr {
    use Error::*;
    let msg = match err {
        InvalidAnalyzer => "Invalid analyzer".to_string(),
        InvalidNgramRange => "Invalid ngram_range".to_string(),
        Internal => "Internal error".to_string()
    };
    PyErr::new::<PyValueError, _>(msg)
}

#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max, n_features))]
pub(crate) fn hashing_tfidf_csr(
    py: Python<'_>,
    seq: Bound<PyAny>,    //iterable of strings (empty for nulls)
    analyzer: &str, //"char"/"char_wb"
    ngram_min: usize, ngram_max: usize, n_features: usize
) -> PyResult<(
    Py<PyArray1<f32>>,  //data
    Py<PyArray1<i32>>,  //indices
    Py<PyArray1<i64>>,  //indptr
    usize,              //n_rows
    usize,              //n_cols (n_features)
    Py<PyArray1<f32>>   //idf (length of n_features)
)> {
    // Collect input into a vector. TODO: zero-copy
    let mut docs: Vec<String> = Vec::new();
    let iter = PyIterator::from_object(&seq)?;
    for item in iter {
        let obj = item?;
        // Treat none as empty string. Python pre-fill should already do this.
        let s: String = if obj.is_none() {String::new()} else {obj.extract()?};
        docs.push(s);
    }
    let n_rows = docs.len();

    // Work buffers to be produced by tfidf::build_csr
    // Compute-intensive work without the GIL. TODO: multi-threading.
    let (data, indices, indptr, idf) = py.detach(|| {
        let builder = Builder::new(analyzer, ngram_min, ngram_max, n_features)?;
        let out = builder.build_csr(&docs); //(data, indices, indptr, idf)
        out
    }).map_err(to_pyerr)?;

    // Convert to NumPy without copying where possible. from_vec is zero-copy but from_array is not.
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();
    let py_idf = idf.into_pyarray(py).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_features, Py::from(py_idf)))

}

#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max, n_features, idf))]
pub(crate) fn hashing_tfidf_csr_with_idf(
    py: Python<'_>,
    seq: Bound<PyAny>,    //iterable of strings (empty for nulls)
    analyzer: &str, //"char"/"char_wb"
    ngram_min: usize, ngram_max: usize, n_features: usize,
    idf: PyReadonlyArray1<f32>  //pre-computed IDF vector
) -> PyResult<(
    Py<PyArray1<f32>>,  //data
    Py<PyArray1<i32>>,  //indices
    Py<PyArray1<i64>>,  //indptr
    usize,              //n_rows
    usize,              //n_cols (n_features)
)> {
    // Collect input into a vector. TODO: zero-copy
    let mut docs: Vec<String> = Vec::new();
    let iter = PyIterator::from_object(&seq)?;
    for item in iter {
        let obj = item?;
        // Treat none as empty string. Python pre-fill should already do this.
        let s: String = if obj.is_none() {String::new()} else {obj.extract()?};
        docs.push(s);
    }
    let n_rows = docs.len();

    // Get IDF slice (zero-copy read)
    let idf_slice = idf.as_slice()?;
    if idf_slice.len() != n_features {
        return Err(PyErr::new::<PyValueError, _>(
            format!("IDF length {} does not match n_features {}", idf_slice.len(), n_features)
        ));
    }

    // Work buffers to be produced by tfidf::build_csr_with_idf
    // Compute-intensive work without the GIL.
    let (data, indices, indptr) = py.detach(|| {
        let builder = Builder::new(analyzer, ngram_min, ngram_max, n_features)?;
        let out = builder.build_csr_with_idf(&docs, idf_slice);
        out
    }).map_err(to_pyerr)?;

    // Convert to NumPy without copying where possible. from_vec is zero-copy but from_array is not.
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_features))

}

// ---- Fit TF-IDF vocabulary + return CSR ----
#[pyfunction]
#[pyo3(signature = (seq, analyzer, ngram_min, ngram_max))]
pub(crate) fn tfidf_fit_csr(
    py: Python<'_>,
    seq: Vec<String>, //Fixme: reference (&PyList) instead of copying
    analyzer: &str,
    ngram_min: usize,
    ngram_max: usize,
) -> PyResult<(
    Py<TfidfModelHandle>, // Python-owned fitted model
    Py<PyArray1<f32>>, // data
    Py<PyArray1<i32>>, // indices
    Py<PyArray1<i64>>, // indptr
    usize,             // n_rows
    usize,             // n_cols (vocab size)
)> {
    let docs = seq;
    let n_rows = docs.len();

    let (model, data, indices, indptr) = py
        .detach(|| {
            let builder = VocabBuilder::new(analyzer, ngram_min, ngram_max)?;
            builder.fit_csr(&docs)
        })
        .map_err(to_pyerr)?;

    let n_cols = model.n_cols;
    let model = Py::new(
        py,
        TfidfModelHandle {
            model: Arc::new(model),
        },
    )?;

    // Convert to NumPy (Vec -> NumPy is zero-copy for from_vec)
    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((model, Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_cols))
}

// ---- Transform using Python-owned vocab/idf + return CSR ----
#[pyfunction]
#[pyo3(signature = (model_id, seq))]
pub(crate) fn tfidf_transform_csr(
    py: Python<'_>,
    model_id: PyRef<'_, TfidfModelHandle>,
    seq: Vec<String>,
) -> PyResult<(
    Py<PyArray1<f32>>, // data
    Py<PyArray1<i32>>, // indices
    Py<PyArray1<i64>>, // indptr
    usize,             // n_rows
    usize,             // n_cols
)> {
    let docs = seq;
    let n_rows = docs.len();

    // Clone only the Arc before releasing the GIL. Concurrent transforms share
    // the immutable fitted vocabulary and IDF without cloning their contents.
    let model = Arc::clone(&model_id.model);
    let n_cols = model.n_cols;
    drop(model_id);

    let (data, indices, indptr) = py
        .detach(|| model.transform_csr(&docs))
        .map_err(to_pyerr)?;

    let py_data = PyArray1::<f32>::from_vec(py, data).to_owned();
    let py_indices = PyArray1::<i32>::from_vec(py, indices).to_owned();
    let py_indptr = PyArray1::<i64>::from_vec(py, indptr).to_owned();

    Ok((Py::from(py_data), Py::from(py_indices), Py::from(py_indptr), n_rows, n_cols))
}
