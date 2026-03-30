use rand::seq::index::sample;
use rand::Rng;

/// Return up to `max` random indices from `0..n`.
/// If `n <= max`, returns all indices in order.
pub fn subsample_indices(n: usize, max: usize, rng: &mut impl Rng) -> Vec<usize> {
    if n <= max {
        (0..n).collect()
    } else {
        let mut indices = sample(rng, n, max).into_vec();
        indices.sort_unstable();
        indices
    }
}

/// Build a histogram with `n_bins` equally spaced bins over the data range.
/// Returns (counts, edges) where edges has length n_bins + 1.
pub fn histogram(values: &[f64], n_bins: usize) -> (Vec<u64>, Vec<f64>) {
    if values.is_empty() || n_bins == 0 {
        return (vec![0; n_bins], vec![0.0; n_bins + 1]);
    }

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in values {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    // Handle degenerate case where all values are identical
    if (max_val - min_val).abs() < f64::EPSILON {
        max_val = min_val + 1.0;
    }

    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges.push(min_val + i as f64 * bin_width);
    }

    for &v in values {
        let bin = ((v - min_val) / bin_width) as usize;
        let bin = bin.min(n_bins - 1); // clamp last edge into final bin
        counts[bin] += 1;
    }

    (counts, edges)
}
