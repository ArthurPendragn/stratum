// Small deterministic xorshift RNG that mirrors sklearn's feature sampling.
#[derive(Clone, Copy)]
pub(crate) struct SklearnRng(u32);

impl SklearnRng {
    // Seed the RNG with the raw sklearn-provided tree seed.
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }

    // Return a bounded index using the same xorshift-style update every call.
    pub(crate) fn bounded(&mut self, low: usize, high: usize) -> usize {
        debug_assert!(low < high);
        let mut value = if self.0 == 0 { 1 } else { self.0 };
        value ^= value.wrapping_shl(13);
        value ^= value.wrapping_shr(17);
        value ^= value.wrapping_shl(5);
        self.0 = value;
        low + ((value % 0x8000_0000) as usize % (high - low))
    }
}

// Regression test for the deterministic feature-selection sequence.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_sklearn_xorshift_sequence() {
        let mut rng = SklearnRng::new(209_652_396);
        assert_eq!(rng.bounded(0, 5), 2);
        assert_eq!(rng.bounded(0, 4), 2);
    }
}
