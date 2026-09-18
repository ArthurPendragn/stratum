use std::time::Instant;

#[inline]
pub fn debug_enabled() -> bool {
    // Read in each call to allow dynamic change
    let debug: once_cell::sync::Lazy<bool> = once_cell::sync::Lazy::new(|| {
        std::env::var("SKRUB_RUST_DEBUG_TIMING")
            .map(|v| matches!(v.to_lowercase().as_str(), "1"))
            .unwrap_or(false)
    });

    *debug
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
        Some(t0) => eprintln!("[rust] {msg}: {}ms", t0.elapsed().as_millis()),
        None => { /*do nothing*/ }
    }
}
