use ahash::AHasher; //fast non-cryptographic hasher
use std::hash::{Hash, Hasher};

pub fn bucket_id(token: &str, n_features: usize) -> usize {
    let mut hasher = AHasher::default();
    token.hash(&mut hasher);
    (hasher.finish() as usize) % n_features
}

// ---------------------------------------------------------------------------
// MurmurHash3 (x86_32) — bit-exact port of the canonical Austin Appleby
// implementation that scikit-learn vendors in `src/MurmurHash3.cpp` and calls
// via `murmurhash3_bytes_s32`. Matching this exactly is what lets the Rust
// HashingVectorizer produce the *same* buckets/signs as sklearn's FeatureHasher.
// ---------------------------------------------------------------------------

#[inline(always)]
fn rotl32(x: u32, r: u32) -> u32 {
    x.rotate_left(r)
}

#[inline(always)]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// MurmurHash3_x86_32 over `data` with `seed` (little-endian block reads).
#[inline]
pub fn murmurhash3_x86_32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();
    let nblocks = len / 4;
    let mut h1 = seed;
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;

    // body — process 4-byte blocks
    for i in 0..nblocks {
        let j = i * 4;
        let mut k1 = u32::from_le_bytes([data[j], data[j + 1], data[j + 2], data[j + 3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = rotl32(k1, 15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    // tail — remaining 1..=3 bytes
    let tail = &data[nblocks * 4..];
    let mut k1: u32 = 0;
    match len & 3 {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = rotl32(k1, 15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = rotl32(k1, 15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = rotl32(k1, 15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        _ => {}
    }

    // finalization
    h1 ^= len as u32;
    fmix32(h1)
}

/// sklearn's `murmurhash3_bytes_s32(key, seed)` — the 32-bit hash reinterpreted
/// as a signed int32. The sign bit drives `alternate_sign` in FeatureHasher.
#[inline]
pub fn murmurhash3_bytes_s32(data: &[u8], seed: u32) -> i32 {
    murmurhash3_x86_32(data, seed) as i32
}

/// Map a token to a (bucket, sign) pair exactly as sklearn's FeatureHasher does:
/// `index = abs(h) % n_features`, `sign = +1 if h >= 0 else -1`.
///
/// sklearn handles `h == i32::MIN` as a magnitude of `2^31` before taking the
/// modulo. Widening to `i64` makes that absolute value well-defined and matches
/// sklearn for every valid `n_features`.
#[inline]
pub fn signed_bucket(token: &[u8], n_features: usize, alternate_sign: bool) -> (usize, f64) {
    let h = murmurhash3_bytes_s32(token, 0);
    let index = (h as i64).unsigned_abs() as usize % n_features;
    let sign = if alternate_sign && h < 0 { -1.0 } else { 1.0 };
    (index, sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scikit-learn:
    //   from sklearn.utils.murmurhash import murmurhash3_bytes_s32
    //   murmurhash3_bytes_s32(b"...", 0)
    #[test]
    fn murmur_matches_sklearn_reference() {
        assert_eq!(murmurhash3_bytes_s32(b"", 0), 0);
        assert_eq!(murmurhash3_bytes_s32(b"hello", 0), 613153351);
        assert_eq!(murmurhash3_bytes_s32(b"foo", 0), -156908512);
        assert_eq!(murmurhash3_bytes_s32(b"the", 0), -1132748958);
        assert_eq!(murmurhash3_bytes_s32(b"quick", 0), 771291085);
    }
}
