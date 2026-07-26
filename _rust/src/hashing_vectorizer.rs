//! sklearn `HashingVectorizer` (word analyzer) — the stateless hashing trick.
//!
//! Documents are processed in parallel chunks. Each chunk reuses its token-count
//! scratch map across rows and emits one flat CSR block, avoiding the maps and
//! small vectors previously retained for every document. The blocks are then
//! concatenated into the final CSR arrays without per-row copies.

use ahash::AHashMap as HashMap;
use rayon::prelude::*;

use crate::hashing::signed_bucket;
use crate::threads::get_thread_pool;
use crate::tokenize::{for_each_word_ngram, word_tokens};

/// Target number of chunks per Rayon worker.
const CHUNKS_PER_THREAD: usize = 4;
/// Lower bound on documents per chunk (also the size used for tiny corpora).
const CHUNK_DOCS_MIN: usize = 256;
/// Upper bound on documents per chunk to keep CSR blocks bounded.
const CHUNK_DOCS_MAX: usize = 4096;

/// Options mirroring the sklearn parameters we support on the fast path.
#[derive(Debug)]
pub struct HashingOptions {
    pub n_features: usize,
    pub nmin: usize,
    pub nmax: usize,
    pub binary: bool,
    pub alternate_sign: bool,
    pub lowercase: bool,
    /// `norm="l2"` when true; `norm=None` when false. (l1/max fall back in Python.)
    pub norm_l2: bool,
}

/// Flat CSR block for one document chunk. `indptr` is local and starts at zero.
struct CsrChunk {
    data: Vec<f64>,
    indices: Vec<i32>,
    indptr: Vec<i64>,
}

/// Reusable scratch for bucket accumulation and canonical column ordering.
struct ChunkScratch {
    counts: HashMap<usize, f64>,
    features: Vec<(usize, f64)>,
}

impl ChunkScratch {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            features: Vec::new(),
        }
    }
}

/// Transform `documents` into CSR parts `(data, indices, indptr)` of shape
/// `(documents.len(), n_features)`.
///
/// `AsRef<str>` lets the PyO3 boundary pass `PyBackedStr` values, so the kernel
/// reads Python's UTF-8 buffers directly instead of copying every document into
/// an owned Rust `String`.
pub fn transform<S: AsRef<str> + Sync>(
    documents: &[S],
    options: &HashingOptions,
) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
    let t0 = crate::util::start_timing();
    let chunks = map_document_chunks(documents, |chunk| transform_chunk(chunk, options));
    crate::util::print_timing("hv map_chunks", t0);

    let t1 = crate::util::start_timing();
    let out = assemble_chunks(chunks, documents.len());
    crate::util::print_timing("hv assemble_csr", t1);
    out
}

/// Documents per chunk: `n_docs / (threads * CHUNKS_PER_THREAD)`, clamped to
/// `[CHUNK_DOCS_MIN, CHUNK_DOCS_MAX]`, and never larger than `n_docs`.
fn chunk_size(n_docs: usize) -> usize {
    if n_docs == 0 {
        return 1;
    }
    let target = (n_docs / worker_threads().max(1).saturating_mul(CHUNKS_PER_THREAD)).max(1);
    target.clamp(CHUNK_DOCS_MIN, CHUNK_DOCS_MAX).min(n_docs)
}

fn worker_threads() -> usize {
    match get_thread_pool() {
        Some(pool) => pool.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}

fn map_document_chunks<S, T, F>(documents: &[S], map: F) -> Vec<T>
where
    S: Sync,
    T: Send,
    F: Fn(&[S]) -> T + Sync + Send,
{
    if documents.is_empty() {
        return Vec::new();
    }
    let size = chunk_size(documents.len());
    let run = || documents.par_chunks(size).map(&map).collect();
    match get_thread_pool() {
        Some(pool) => pool.install(run),
        None => run(),
    }
}

fn transform_chunk<S: AsRef<str>>(documents: &[S], options: &HashingOptions) -> CsrChunk {
    let mut scratch = ChunkScratch::new();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(documents.len() + 1);
    indptr.push(0);

    for document in documents {
        transform_document_into(
            document.as_ref(),
            options,
            &mut scratch,
            &mut data,
            &mut indices,
        );
        indptr.push(indices.len() as i64);
    }

    CsrChunk {
        data,
        indices,
        indptr,
    }
}

fn transform_document_into(
    document: &str,
    options: &HashingOptions,
    scratch: &mut ChunkScratch,
    data: &mut Vec<f64>,
    indices: &mut Vec<i32>,
) {
    scratch.counts.clear();
    accumulate_document(document, options, &mut scratch.counts);

    scratch.features.clear();
    scratch.features.extend(scratch.counts.drain());
    scratch.features.sort_unstable_by_key(|&(column, _)| column);

    let row_start = data.len();
    for &(column, value) in &scratch.features {
        indices.push(column as i32);
        data.push(if options.binary { 1.0 } else { value });
    }

    if options.norm_l2 {
        normalize_l2(&mut data[row_start..]);
    }
}

fn accumulate_document(document: &str, options: &HashingOptions, counts: &mut HashMap<usize, f64>) {
    if !options.lowercase {
        accumulate_text(document, options, counts);
        return;
    }

    // Lowercase ASCII is the dominant path in the benchmark and needs no
    // allocation. Uppercase ASCII can use the cheaper byte-wise conversion;
    // non-ASCII falls back to full Unicode lowercasing only when necessary.
    let mut non_ascii = false;
    let mut has_ascii_upper = false;
    for &byte in document.as_bytes() {
        if !byte.is_ascii() {
            non_ascii = true;
            break;
        }
        has_ascii_upper |= byte.is_ascii_uppercase();
    }

    if non_ascii {
        if document.chars().any(changes_when_lowercased) {
            let lowered = document.to_lowercase();
            accumulate_text(&lowered, options, counts);
        } else {
            accumulate_text(document, options, counts);
        }
    } else if has_ascii_upper {
        let lowered = document.to_ascii_lowercase();
        accumulate_text(&lowered, options, counts);
    } else {
        accumulate_text(document, options, counts);
    }
}

fn accumulate_text(text: &str, options: &HashingOptions, counts: &mut HashMap<usize, f64>) {
    let mut tokens = Vec::with_capacity(16);
    word_tokens(text, &mut tokens);
    for_each_word_ngram(&tokens, options.nmin, options.nmax, |ngram| {
        let (column, sign) =
            signed_bucket(ngram.as_bytes(), options.n_features, options.alternate_sign);
        *counts.entry(column).or_insert(0.0) += sign;
    });
}

#[inline]
fn changes_when_lowercased(character: char) -> bool {
    let mut lowercase = character.to_lowercase();
    lowercase.next() != Some(character) || lowercase.next().is_some()
}

fn normalize_l2(data: &mut [f64]) {
    let norm_squared = data.iter().map(|value| value * value).sum::<f64>();
    if norm_squared > 0.0 {
        let scale = norm_squared.sqrt().recip();
        for value in data {
            *value *= scale;
        }
    }
}

fn assemble_chunks(chunks: Vec<CsrChunk>, document_count: usize) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
    let nonzero_count = chunks.iter().map(|chunk| chunk.indices.len()).sum();
    let mut data = Vec::with_capacity(nonzero_count);
    let mut indices = Vec::with_capacity(nonzero_count);
    let mut indptr = Vec::with_capacity(document_count + 1);
    indptr.push(0);

    let mut nonzero_offset = 0i64;
    for chunk in chunks {
        for &local_end in &chunk.indptr[1..] {
            indptr.push(nonzero_offset + local_end);
        }
        nonzero_offset += chunk.indices.len() as i64;
        data.extend(chunk.data);
        indices.extend(chunk.indices);
    }

    debug_assert_eq!(indptr.len(), document_count + 1);
    (data, indices, indptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_default(n_features: usize) -> HashingOptions {
        HashingOptions {
            n_features,
            nmin: 1,
            nmax: 1,
            binary: false,
            alternate_sign: true,
            lowercase: true,
            norm_l2: true,
        }
    }

    /// Reference implementation of the original per-document code path.
    fn transform_reference(
        documents: &[String],
        options: &HashingOptions,
    ) -> (Vec<f64>, Vec<i32>, Vec<i64>) {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for document in documents {
            let lowered;
            let text = if options.lowercase {
                lowered = document.to_lowercase();
                lowered.as_str()
            } else {
                document.as_str()
            };

            let mut counts = HashMap::new();
            accumulate_text(text, options, &mut counts);
            let mut features: Vec<_> = counts.into_iter().collect();
            features.sort_unstable_by_key(|&(column, _)| column);

            let row_start = data.len();
            for (column, value) in features {
                indices.push(column as i32);
                data.push(if options.binary { 1.0 } else { value });
            }
            if options.norm_l2 {
                normalize_l2(&mut data[row_start..]);
            }
            indptr.push(indices.len() as i64);
        }
        (data, indices, indptr)
    }

    fn assert_matches_reference(documents: &[String], options: &HashingOptions) {
        let got = transform(documents, options);
        let expected = transform_reference(documents, options);
        assert_eq!(got.0, expected.0);
        assert_eq!(got.1, expected.1);
        assert_eq!(got.2, expected.2);
    }

    #[test]
    fn empty_docs_produce_empty_rows() {
        let documents = vec![String::new(), "a".to_owned()];
        let (data, indices, indptr) = transform(&documents, &opts_default(1 << 10));
        assert_eq!(indptr, vec![0, 0, 0]);
        assert!(data.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn rows_are_l2_normalized() {
        let documents = vec!["the quick brown fox".to_owned()];
        let (data, _indices, indptr) = transform(&documents, &opts_default(1 << 18));
        assert_eq!(indptr.len(), 2);
        let norm = data.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "row norm = {norm}");
    }

    #[test]
    fn lowercase_folds_case() {
        let options = opts_default(1 << 18);
        let lower = transform(&["hello world"], &options);
        let upper = transform(&["HELLO WORLD"], &options);
        assert_eq!(lower.1, upper.1);
    }

    #[test]
    fn optimized_lowercasing_matches_reference() {
        let documents = [
            "the quick brown fox",
            "The Quick BROWN Fox",
            "HELLO, WORLD! hello.",
            "MIXED123 under_score X_Y a b cc",
            "café CAFÉ Ünïcodé test",
            "İstanbul TİTLE dotted",
            "straße STRASSE eszett",
            "Δoxa ΔΟΞΑ delta greek",
            "ﬁle ﬂour ligature",
            "中文 混合 English WORDS",
            "",
            "a",
        ]
        .map(str::to_owned);

        for &(nmin, nmax) in &[(1, 1), (1, 2), (2, 3)] {
            for &binary in &[false, true] {
                for &alternate_sign in &[false, true] {
                    for &norm_l2 in &[false, true] {
                        let options = HashingOptions {
                            n_features: 1 << 16,
                            nmin,
                            nmax,
                            binary,
                            alternate_sign,
                            lowercase: true,
                            norm_l2,
                        };
                        assert_matches_reference(&documents, &options);
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_boundaries_preserve_rows() {
        let documents: Vec<String> = (0..(CHUNK_DOCS_MIN * 3 + 17))
            .map(|row| format!("document number {row} repeated repeated"))
            .collect();
        let options = HashingOptions {
            nmin: 1,
            nmax: 2,
            ..opts_default(1 << 18)
        };
        assert_matches_reference(&documents, &options);
    }

    #[test]
    fn chunk_size_is_bounded() {
        assert_eq!(chunk_size(0), 1);
        assert_eq!(chunk_size(1), 1);
        assert!(chunk_size(100_000) >= CHUNK_DOCS_MIN);
        assert!(chunk_size(100_000) <= CHUNK_DOCS_MAX);
    }
}
