use once_cell::sync::Lazy;
use std::time::Instant;

// Thread-safe check on the first use
static NUM_THREADS: Lazy<usize> = Lazy::new(|| {
    match std::env::var("SKRUB_RUST_THREADS") {
        Ok(num) => num.parse().unwrap_or(0),
        _ => 0,
    }
});

#[inline]
pub fn debug_enabled() -> bool {
    // Read in each call to allow dynamic change
    let debug: Lazy<bool> = Lazy::new(|| {
        std::env::var("SKRUB_RUST_DEBUG_TIMING")
            .map(|v| matches!(v.to_lowercase().as_str(), "1"))
            .unwrap_or(false)
    });

    *debug
}

#[inline]
pub fn get_num_threads() -> usize {
    *NUM_THREADS
}

#[inline]
pub fn start_timing() -> Option<Instant> {
    if debug_enabled() {
        Some(Instant::now())
    }
    else { None }
}

#[inline]
pub fn print_timing(msg: &str, start: Option<Instant>) {
    match start {
        Some(t0) => {
            // Fractional ms so sub-millisecond stages (e.g. hv assemble_csr) are visible.
            eprintln!("[rust] {msg}: {:.3}ms", t0.elapsed().as_secs_f64() * 1000.0);
        }
        None => { /*do nothing*/ }
    }
}