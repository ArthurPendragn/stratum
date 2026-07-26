use anyhow::{bail, Result};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    Char,
    Char_wb,
}

#[allow(dead_code)]
pub fn parse_analyzer(s: &str) -> Result<Analyzer> {
    match s {
        "char" => Ok(Analyzer::Char),
        "char_wb" => Ok(Analyzer::Char_wb),
        _ => bail!("Unsupported Analyzer: {}", s),
    }
}

pub fn char_ngrams<'a>(s: &'a str, nmin: usize, nmax: usize, buf: &mut Vec<&'a str>) {
    // Collect byte offsets at char boundaries (safe UTF-8 slicing)
    let mut idx: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    idx.push(s.len());

    // Gather all n-grams for all n between nmin and nmax
    for n in nmin..=nmax {
        if n == 0 || idx.len() < n + 1 {
            //coherence check
            continue;
        }
        for start in 0..=idx.len().saturating_sub(n + 1) {
            let end = start + n;
            let i = idx[start];
            let j = idx[end];
            buf.push(&s[i..j]); // slice is valid UTF-8 by construction
        }
    }
}

/// Fast word-character predicate for the adapter's ASCII-only native path.
///
/// Direct native callers may pass Unicode, but Rust's `is_alphanumeric` is not
/// guaranteed to match Python's `re.UNICODE` `\w` classification.
#[inline(always)]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Word tokenizer replicating sklearn's default `token_pattern=r"(?u)\b\w\w+\b"`:
/// emit every maximal run of word-characters whose length is >= 2 code points.
/// Slices borrow from `text`, so the caller should lowercase up front (matching
/// sklearn's `lowercase=True`, which lowercases the whole doc before tokenizing).
pub fn word_tokens<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut start: Option<usize> = None;
    let mut nchars: usize = 0;
    for (i, c) in text.char_indices() {
        if is_word_char(c) {
            if start.is_none() {
                start = Some(i);
                nchars = 0;
            }
            nchars += 1;
        } else if let Some(s) = start.take() {
            if nchars >= 2 {
                out.push(&text[s..i]);
            }
        }
    }
    if let Some(s) = start {
        if nchars >= 2 {
            out.push(&text[s..]);
        }
    }
}

/// Iterate word n-grams over pre-tokenized `tokens` for `nmin..=nmax`, matching
/// sklearn's `_word_ngrams` (unigrams kept verbatim when `nmin == 1`, higher-order
/// grams are consecutive tokens joined by a single space). Each n-gram string is
/// passed to `f`; a scratch `String` is reused across joins to avoid churn.
pub fn for_each_word_ngram<F: FnMut(&str)>(tokens: &[&str], nmin: usize, nmax: usize, mut f: F) {
    let ntok = tokens.len();
    let start = nmin.max(1);
    let end = nmax.min(ntok);
    if start > end {
        return;
    }

    let mut buf = String::new();
    for n in start..=end {
        if n == 1 {
            for &t in tokens {
                f(t);
            }
        } else {
            for i in 0..=ntok - n {
                buf.clear();
                for (j, &t) in tokens[i..i + n].iter().enumerate() {
                    if j > 0 {
                        buf.push(' ');
                    }
                    buf.push_str(t);
                }
                f(&buf);
            }
        }
    }
}

pub fn char_wb_ngrams(str: &str, nmin: usize, nmax: usize, buf: &mut Vec<String>) {
    // Pad with spaces at word boundaries and then extract like char_ngrams
    // TODO: Implement proper word-boundary padding
    let padded = format!(" {} ", str);
    for n in nmin..=nmax {
        if n == 0 || n > padded.len() {
            continue;
        }
        for i in 0..=padded.len().saturating_sub(n) {
            let j = i + n;
            buf.push(padded[i..j].to_string()); //FIXME (perf): allocations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::for_each_word_ngram;

    fn collect_ngrams(tokens: &[&str], nmin: usize, nmax: usize) -> Vec<String> {
        let mut ngrams = Vec::new();
        for_each_word_ngram(tokens, nmin, nmax, |ngram| {
            ngrams.push(ngram.to_owned());
        });
        ngrams
    }

    #[test]
    fn empty_tokens_ignore_unbounded_nmax() {
        assert!(collect_ngrams(&[], 1, usize::MAX).is_empty());
    }

    #[test]
    fn one_token_ignores_unbounded_nmax() {
        assert_eq!(collect_ngrams(&["token"], 1, usize::MAX), vec!["token"]);
    }

    #[test]
    fn nmin_above_token_count_produces_no_ngrams() {
        assert!(collect_ngrams(&["one", "two"], 3, usize::MAX).is_empty());
    }

    #[test]
    fn ordinary_word_ngram_ranges_are_unchanged() {
        let tokens = ["one", "two", "three"];
        assert_eq!(collect_ngrams(&tokens, 1, 1), vec!["one", "two", "three"]);
        assert_eq!(
            collect_ngrams(&tokens, 1, 2),
            vec!["one", "two", "three", "one two", "two three"]
        );
        assert_eq!(
            collect_ngrams(&tokens, 2, 3),
            vec!["one two", "two three", "one two three"]
        );
    }
}
