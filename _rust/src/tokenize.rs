use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analyzer {
    Char,
    Char_wb
}

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
        if n == 0 || idx.len() < n+1 { //coherence check
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

/// sklearn default `\w`: alphanumeric or underscore.
#[inline(always)]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Tokenize like sklearn's default `token_pattern=r"(?u)\b\w\w+\b"`.
/// Tokens borrow from `text` (no per-token allocation).
pub fn word_tokens<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut start = None;
    let mut char_count = 0usize;

    for (offset, ch) in text.char_indices() {
        if is_word_char(ch) {
            if start.is_none() {
                start = Some(offset);
                char_count = 0;
            }
            char_count += 1;
        } else if let Some(token_start) = start.take() {
            if char_count >= 2 {
                out.push(&text[token_start..offset]);
            }
        }
    }

    if let Some(token_start) = start {
        if char_count >= 2 {
            out.push(&text[token_start..]);
        }
    }
}

/// Word n-grams in sklearn order. Unigrams are borrowed; higher-order grams
/// are joined with a shared scratch buffer.
pub fn for_each_word_ngram<F>(tokens: &[&str], nmin: usize, nmax: usize, mut visit: F)
where
    F: FnMut(&str),
{
    let token_count = tokens.len();
    let start = nmin.max(1);
    let end = nmax.min(token_count);
    if start > end {
        return;
    }

    let mut joined = String::new();

    for n in start..=end {
        if n == 1 {
            for &token in tokens {
                visit(token);
            }
            continue;
        }

        for start in 0..=token_count - n {
            joined.clear();
            for (position, &token) in tokens[start..start + n].iter().enumerate() {
                if position != 0 {
                    joined.push(' ');
                }
                joined.push_str(token);
            }
            visit(&joined);
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
    use super::*;

    #[test]
    fn default_word_pattern_drops_single_character_tokens() {
        let mut tokens = Vec::new();
        word_tokens("a fox 42 _ x_y café", &mut tokens);
        assert_eq!(tokens, vec!["fox", "42", "x_y", "café"]);
    }

    #[test]
    fn word_ngrams_follow_sklearn_order() {
        let tokens = vec!["one", "two", "three"];
        let mut grams = Vec::new();
        for_each_word_ngram(&tokens, 1, 2, |gram| grams.push(gram.to_owned()));
        assert_eq!(grams, vec!["one", "two", "three", "one two", "two three"]);
    }

    #[test]
    fn word_ngrams_support_ordinary_ranges_and_empty_ranges() {
        let tokens = vec!["one", "two", "three"];

        let mut unigrams = Vec::new();
        for_each_word_ngram(&tokens, 1, 1, |gram| unigrams.push(gram.to_owned()));
        assert_eq!(unigrams, vec!["one", "two", "three"]);

        let mut higher_order = Vec::new();
        for_each_word_ngram(&tokens, 2, 3, |gram| higher_order.push(gram.to_owned()));
        assert_eq!(higher_order, vec!["one two", "two three", "one two three"]);

        let mut beyond_token_count = Vec::new();
        for_each_word_ngram(&tokens, 4, 8, |gram| {
            beyond_token_count.push(gram.to_owned())
        });
        assert!(beyond_token_count.is_empty());
    }

    #[test]
    fn word_ngrams_cap_extreme_bounds_at_the_token_count() {
        let mut empty_output = Vec::new();
        for_each_word_ngram(&[], 1, usize::MAX, |gram| {
            empty_output.push(gram.to_owned())
        });
        assert!(empty_output.is_empty());

        let mut one_token_output = Vec::new();
        for_each_word_ngram(&["one"], 1, usize::MAX, |gram| {
            one_token_output.push(gram.to_owned())
        });
        assert_eq!(one_token_output, vec!["one"]);

        let mut range_above_tokens = Vec::new();
        for_each_word_ngram(&["one", "two"], 3, usize::MAX, |gram| {
            range_above_tokens.push(gram.to_owned())
        });
        assert!(range_above_tokens.is_empty());
    }
}
