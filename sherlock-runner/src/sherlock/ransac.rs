use log::info;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use serde::Serialize;

use crate::dataset::DatasetState;
use super::subsample::subsample_indices;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct RansacResult {
    pub slope: f64,
    pub intercept: f64,
    pub inlier_count: usize,
    pub outlier_count: usize,
    pub total: usize,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub display_inlier: Vec<bool>,
}

#[derive(Debug, Serialize)]
pub struct PreciseAutocorrelationResult {
    pub slope: f64,
    pub intercept: f64,
    pub inlier_count: usize,
    pub outlier_count: usize,
    pub total: usize,
    pub peaks_near_line: usize,
    pub peaks_used: usize,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub display_inlier: Vec<bool>,
}

/// Simple least-squares linear regression: y = slope * x + intercept
fn least_squares(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return (0.0, sy / n);
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

// ---------------------------------------------------------------------------
// Rough fit helpers: bin the raw data with quadratic spacing, find local
// maxima, filter, then RANSAC to get an approximate autocorrelation line.
// This mirrors the original Sherlock "rough autocorrelation" pipeline.
// ---------------------------------------------------------------------------

/// 1-D sliding maximum using a monotonic deque. O(n).
fn sliding_max_1d(input: &[f64], output: &mut [f64], kernel: usize) {
    let n = input.len();
    if n == 0 {
        return;
    }
    let half = kernel / 2;
    let mut deque = std::collections::VecDeque::<usize>::with_capacity(kernel);
    let mut fed = 0usize;

    for c in 0..n {
        let right = (c + half).min(n - 1);
        while fed <= right {
            while let Some(&back) = deque.back() {
                if input[back] <= input[fed] {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(fed);
            fed += 1;
        }
        let left = c.saturating_sub(half);
        while let Some(&front) = deque.front() {
            if front < left {
                deque.pop_front();
            } else {
                break;
            }
        }
        output[c] = input[*deque.front().unwrap()];
    }
}

/// 2-D separable maximum filter on a row-major (n_rows x n_cols) buffer.
/// Returns a flat row-major Vec<f64>.
fn maximum_filter_2d_f64(buf: &[f64], n_rows: usize, n_cols: usize, kernel: usize) -> Vec<f64> {
    // Pass 1: row-wise
    let mut intermediate = vec![0.0f64; n_rows * n_cols];
    for row in 0..n_rows {
        let src = &buf[row * n_cols..(row + 1) * n_cols];
        let dst = &mut intermediate[row * n_cols..(row + 1) * n_cols];
        sliding_max_1d(src, dst, kernel);
    }
    // Pass 2: column-wise
    let mut result = vec![0.0f64; n_rows * n_cols];
    let mut col_buf = vec![0.0f64; n_rows];
    let mut col_out = vec![0.0f64; n_rows];
    for col in 0..n_cols {
        for row in 0..n_rows {
            col_buf[row] = intermediate[row * n_cols + col];
        }
        sliding_max_1d(&col_buf, &mut col_out, kernel);
        for row in 0..n_rows {
            result[row * n_cols + col] = col_out[row];
        }
    }
    result
}

/// Try a candidate pair (a, b), returning (slope, intercept, inlier_count).
/// Returns None if the pair is degenerate (dx ~ 0).
#[inline]
fn try_pair(
    tof: &[f64],
    swim: &[f64],
    a: usize,
    b: usize,
    tolerance: f64,
) -> Option<(f64, f64, usize)> {
    let dx = tof[b] - tof[a];
    if dx.abs() < f64::EPSILON {
        return None;
    }
    let sl = (swim[b] - swim[a]) / dx;
    let ic = swim[a] - sl * tof[a];
    let cnt = tof
        .iter()
        .zip(swim.iter())
        .filter(|(&t, &s)| (s - (sl * t + ic)).abs() < tolerance)
        .count();
    Some((sl, ic, cnt))
}

/// Call sklearn RANSACRegressor via Python subprocess for exact production parity.
/// Falls back to native Rust RANSAC if Python is unavailable.
fn ransac_fit_sklearn(
    tof: &[f64],
    swim: &[f64],
    tolerance: f64,
    seed: u64,
) -> (f64, f64, usize) {
    use std::process::{Command, Stdio};
    use std::io::Write as IoWrite;

    // Find the bridge script — check Lambda path first, then local paths
    let bridge_paths = [
        std::path::PathBuf::from("/opt/ransac_bridge.py"),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ransac_bridge.py"),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ransac_bridge.py"),
        std::path::PathBuf::from("ransac_bridge.py"),
    ];

    let bridge = bridge_paths.iter().find(|p| p.exists());
    if bridge.is_none() {
        info!("[ransac] sklearn bridge not found, falling back to native RANSAC");
        return ransac_fit_native(tof, swim, tolerance, 1000, seed);
    }

    let req = serde_json::json!({
        "tof": tof,
        "swim": swim,
        "residual_threshold": tolerance,
        "random_state": seed,
        "stop_n_inliers": 50,
    });

    let result = Command::new("python3")
        .arg(bridge.unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(ref mut stdin) = child.stdin {
                let _ = stdin.write_all(req.to_string().as_bytes());
            }
            child.wait_with_output()
        });

    match result {
        Ok(output) if output.status.success() => {
            if let Ok(resp) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                let slope = resp["slope"].as_f64().unwrap_or(1.0);
                let intercept = resp["intercept"].as_f64().unwrap_or(0.0);
                let inliers = resp["inlier_count"].as_u64().unwrap_or(0) as usize;
                info!("[ransac] sklearn bridge: slope={:.6} intercept={:.4} inliers={}", slope, intercept, inliers);
                return (slope, intercept, inliers);
            }
            info!("[ransac] sklearn bridge: failed to parse response, falling back");
            ransac_fit_native(tof, swim, tolerance, 1000, seed)
        }
        _ => {
            info!("[ransac] sklearn bridge: process failed, falling back to native");
            ransac_fit_native(tof, swim, tolerance, 1000, seed)
        }
    }
}

/// Primary RANSAC entry point.
/// When `use_sklearn` is true, uses the Python sklearn bridge for production parity.
/// When false, uses the native Rust implementation.
fn ransac_fit(
    tof: &[f64],
    swim: &[f64],
    tolerance: f64,
    iterations: usize,
    seed: u64,
    use_sklearn: bool,
) -> (f64, f64, usize) {
    if use_sklearn {
        ransac_fit_sklearn(tof, swim, tolerance, seed)
    } else {
        ransac_fit_native(tof, swim, tolerance, iterations, seed)
    }
}

/// Native Rust RANSAC fallback.
/// For small N (<=500), exhaustively enumerates all pairs for deterministic results.
/// For larger N, falls back to random sampling with `iterations` trials.
fn ransac_fit_native(
    tof: &[f64],
    swim: &[f64],
    tolerance: f64,
    iterations: usize,
    seed: u64,
) -> (f64, f64, usize) {
    let n = tof.len();
    if n < 2 {
        return (1.0, 0.0, n);
    }

    let mut best_slope = 1.0;
    let mut best_intercept = 0.0;
    let mut best_count = 0usize;

    // Exhaustively try all C(n,2) pairs — deterministic, no RNG needed.
    // N=500 gives 124,750 pairs × N inlier checks ≈ 62M ops, under 1 second.
    // This eliminates cross-language RANSAC divergence entirely.
    const EXHAUSTIVE_THRESHOLD: usize = 500;

    if n <= EXHAUSTIVE_THRESHOLD {
        for a in 0..n {
            for b in (a + 1)..n {
                if let Some((sl, ic, cnt)) = try_pair(tof, swim, a, b, tolerance) {
                    if cnt > best_count {
                        best_slope = sl;
                        best_intercept = ic;
                        best_count = cnt;
                    }
                }
            }
        }
    } else {
        // Large N: random sampling (RNG may differ from Python, but rough fit
        // tolerances absorb the small divergence).
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..iterations {
            let idx = sample(&mut rng, n, 2).into_vec();
            let (a, b) = (idx[0], idx[1]);
            if let Some((sl, ic, cnt)) = try_pair(tof, swim, a, b, tolerance) {
                if cnt > best_count {
                    best_slope = sl;
                    best_intercept = ic;
                    best_count = cnt;
                }
            }
        }
    }

    // Refit via least-squares on inliers
    let inlier_tof: Vec<f64> = tof
        .iter()
        .zip(swim.iter())
        .filter(|(&t, &s)| (s - (best_slope * t + best_intercept)).abs() < tolerance)
        .map(|(&t, _)| t)
        .collect();
    let inlier_swim: Vec<f64> = tof
        .iter()
        .zip(swim.iter())
        .filter(|(&t, &s)| (s - (best_slope * t + best_intercept)).abs() < tolerance)
        .map(|(_, &s)| s)
        .collect();

    if inlier_tof.len() >= 2 {
        let (sl, ic) = least_squares(&inlier_tof, &inlier_swim);
        (sl, ic, inlier_tof.len())
    } else {
        (best_slope, best_intercept, best_count)
    }
}

/// Perform the rough fit on binned raw data. Returns (slope, intercept).
///
/// Algorithm (mirrors original Sherlock rough autocorrelation pipeline):
///   1. Bin the TOF axis with quadratic spacing (5000 bins)
///   2. Find local maxima in the (n_swim, 5000) binned array
///   3. Filter to swim_range=[400,1000] and tof_range=[400,1000]
///   4. Amplitude threshold: mean + 0.5*std of binned data
///   5. Remove stripes (fixed threshold of 10)
///   6. Remove peaks outside linear band: 0.8*tof-20 < swim < 1.2*tof+20
///   7. RANSAC fit on remaining peaks
fn rough_binned_fit(ds: &DatasetState, tolerance: f64, use_sklearn: bool) -> Result<(f64, f64), String> {
    let data = ds
        .data
        .as_ref()
        .ok_or("No data loaded — cannot do rough binned fit")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No TOF coords")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coords")?;

    let n_swim = data.nrows();
    let n_tof_orig = data.ncols();

    let tof_min = tof_coords[0];
    let tof_max = tof_coords[n_tof_orig - 1];

    // -----------------------------------------------------------------------
    // 1. Build quadratic bin edges: tof_min + (t/n_bins)^2 * (tof_max - tof_min)
    // -----------------------------------------------------------------------
    let n_bins: usize = 5000;
    let tof_range = tof_max - tof_min;
    let bin_edges: Vec<f64> = (0..=n_bins)
        .map(|i| {
            let t = i as f64 / n_bins as f64;
            tof_min + t * t * tof_range
        })
        .collect();

    // -----------------------------------------------------------------------
    // 2. Bin the data: for each swim row, sum across columns in each bin
    //    Result shape: (n_swim, n_bins)
    // -----------------------------------------------------------------------
    let data_raw = data.as_slice().unwrap();

    // Precompute column-to-bin mapping
    let mut col_to_bin = vec![0usize; n_tof_orig];
    for col in 0..n_tof_orig {
        let t = tof_coords[col];
        // Binary search: find rightmost edge <= t
        let b = bin_edges.partition_point(|&e| e <= t);
        let bin_idx = if b == 0 { 0 } else { (b - 1).min(n_bins - 1) };
        col_to_bin[col] = bin_idx;
    }

    let mut binned = vec![0.0f64; n_swim * n_bins];
    for row in 0..n_swim {
        let row_data = &data_raw[row * n_tof_orig..(row + 1) * n_tof_orig];
        for col in 0..n_tof_orig {
            let bin_idx = col_to_bin[col];
            binned[row * n_bins + bin_idx] += row_data[col] as f64;
        }
    }

    // Compute bin centres (midpoint of each bin edge pair in TOF mass units)
    let bin_centres: Vec<f64> = (0..n_bins)
        .map(|i| (bin_edges[i] + bin_edges[i + 1]) / 2.0)
        .collect();

    info!(
        "[ransac] rough: binned data to ({}, {}) with quadratic spacing",
        n_swim, n_bins
    );

    // -----------------------------------------------------------------------
    // 3. Extract subset within range BEFORE peak finding (matching original).
    //    The original calls get_binned(api, n_bins).sel(swim_range, tof_range)
    //    THEN get_peaks on the subset. This means the max filter only sees
    //    data within the range, not the full binned array.
    // -----------------------------------------------------------------------
    let swim_lo = 400.0;
    let swim_hi = 1000.0;
    let tof_lo = 400.0;
    let tof_hi = 1000.0;

    // Find swim rows and tof bins within range
    let swim_rows_in: Vec<usize> = (0..n_swim)
        .filter(|&r| swim_coords[r] >= swim_lo && swim_coords[r] <= swim_hi)
        .collect();
    // Use bin edges for range selection: include bins whose interval overlaps [tof_lo, tof_hi].
    // Matches xarray sel(slice(400, 1000)) which selects by interval overlap, not bin centre.
    let tof_bins_in: Vec<usize> = (0..n_bins)
        .filter(|&b| bin_edges[b + 1] > tof_lo && bin_edges[b] < tof_hi)
        .collect();

    let sub_n_swim = swim_rows_in.len();
    let sub_n_tof = tof_bins_in.len();

    // The original transposes to (tof, swim) before local_maximum.
    // Build subset in (tof, swim) order to match.
    let mut subset = vec![0.0f64; sub_n_tof * sub_n_swim];
    for (ti, &bin_idx) in tof_bins_in.iter().enumerate() {
        for (si, &swim_row) in swim_rows_in.iter().enumerate() {
            subset[ti * sub_n_swim + si] = binned[swim_row * n_bins + bin_idx];
        }
    }

    info!(
        "[ransac] rough: subset ({}, {}) within range [{},{}]×[{},{}]",
        sub_n_tof, sub_n_swim, tof_lo, tof_hi, swim_lo, swim_hi
    );

    // Find local maxima in the subset
    let kernel = 5;
    let max_filt = maximum_filter_2d_f64(&subset, sub_n_tof, sub_n_swim, kernel);

    // Precompute mean original tof index per bin (matches production's binned_tof_idx)
    let mut bin_mean_tof_idx = vec![0i64; n_bins];
    let mut bin_col_count = vec![0usize; n_bins];
    for col in 0..n_tof_orig {
        let b = col_to_bin[col];
        bin_mean_tof_idx[b] += col as i64;
        bin_col_count[b] += 1;
    }
    for b in 0..n_bins {
        if bin_col_count[b] > 0 {
            bin_mean_tof_idx[b] /= bin_col_count[b] as i64;
        }
    }

    let mut filt_tof = Vec::new();
    let mut filt_swim = Vec::new();
    let mut filt_amps = Vec::new();
    // Use ORIGINAL dataset indices for stripe counting (matching production's to_idx_array)
    let mut filt_tof_idx = Vec::new();
    let mut filt_swim_idx = Vec::new();

    for ti in 0..sub_n_tof {
        for si in 0..sub_n_swim {
            let idx = ti * sub_n_swim + si;
            let val = subset[idx];
            if val > 0.0 && val == max_filt[idx] {
                filt_tof.push(bin_centres[tof_bins_in[ti]]);
                filt_swim.push(swim_coords[swim_rows_in[si]]);
                filt_amps.push(val);
                // Original dataset indices — production uses (tof_idx, swim_idx)
                filt_tof_idx.push(bin_mean_tof_idx[tof_bins_in[ti]]);
                filt_swim_idx.push(swim_rows_in[si] as i64);
            }
        }
    }

    eprintln!("[rough_binned_fit] subset shape: ({}, {})", sub_n_tof, sub_n_swim);
    eprintln!("[rough_binned_fit] local maxima in subset: {}", filt_tof.len());

    // -----------------------------------------------------------------------
    // 5. Amplitude threshold: mean + 0.5 * std of the SUBSET (not full binned array).
    //    Original: get_threshold_from_std(binned_dataset, 0.5) where binned_dataset
    //    is already range-selected.
    // -----------------------------------------------------------------------
    let total_elements = subset.len() as f64;
    let mean: f64 = subset.iter().sum::<f64>() / total_elements;
    let variance: f64 =
        subset.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / total_elements;
    let std_dev = variance.sqrt();
    let amp_threshold = mean + 0.5 * std_dev;

    let mut keep = Vec::new();
    for i in 0..filt_tof.len() {
        if filt_amps[i] > amp_threshold {
            keep.push(i);
        }
    }
    let filt_tof: Vec<f64> = keep.iter().map(|&i| filt_tof[i]).collect();
    let filt_swim: Vec<f64> = keep.iter().map(|&i| filt_swim[i]).collect();
    let _filt_amps: Vec<f64> = keep.iter().map(|&i| filt_amps[i]).collect(); // no longer used after keep_tallest removal
    let filt_tof_idx: Vec<i64> = keep.iter().map(|&i| filt_tof_idx[i]).collect();
    let filt_swim_idx: Vec<i64> = keep.iter().map(|&i| filt_swim_idx[i]).collect();

    eprintln!("[rough_binned_fit] peaks after amplitude threshold: {} (threshold={:.2}, mean={:.2}, std={:.2})",
        filt_tof.len(), amp_threshold, mean, std_dev);
    // Diagnostic: how many unique tof/swim indices?
    {
        let unique_tof: std::collections::HashSet<i64> = filt_tof_idx.iter().copied().collect();
        let unique_swim: std::collections::HashSet<i64> = filt_swim_idx.iter().copied().collect();
        eprintln!("[rough_binned_fit] unique tof_idx: {}, unique swim_idx: {} (production: 48 tof, 114 swim)",
            unique_tof.len(), unique_swim.len());
        eprintln!("[rough_binned_fit] tof_idx range: [{}, {}]",
            filt_tof_idx.iter().min().unwrap_or(&0), filt_tof_idx.iter().max().unwrap_or(&0));
    }

    // -----------------------------------------------------------------------
    // 6. Optimal stripe threshold search + stripe removal.
    //    Matching get_optimal_stripe_threshold(peaks, min=5, max=70, n_tests=200)
    //    then remove_stripes(peaks, threshold, keep_tallest=True).
    //    NOTE: keep_tallest is a no-op due to a production bug (see comment below).
    // -----------------------------------------------------------------------

    // Use LOCAL array indices for stripe counting, matching original's
    // filter_dense_lines(peaks.to_idx_array(), ...) which uses (tof_idx, swim_idx).
    let int_tof = &filt_tof_idx;
    let int_swim = &filt_swim_idx;

    // Helper: count peaks surviving filter_dense_lines at a given threshold.
    // Production bug: filter_dense_lines computes keep_tallest but returns
    // the base mask (x_mask & y_mask) WITHOUT the tallest peaks added back.
    // We must match this behaviour for the threshold search to agree.
    let count_surviving = |threshold: usize| -> usize {
        let mut sc = std::collections::HashMap::<i64, usize>::new();
        let mut tc = std::collections::HashMap::<i64, usize>::new();
        for i in 0..int_swim.len() {
            *sc.entry(int_swim[i]).or_insert(0) += 1;
            *tc.entry(int_tof[i]).or_insert(0) += 1;
        }
        let mut count = 0usize;
        for i in 0..int_swim.len() {
            let s = sc.get(&int_swim[i]).copied().unwrap_or(0);
            let t = tc.get(&int_tof[i]).copied().unwrap_or(0);
            if s < threshold && t < threshold {
                count += 1;
            }
        }
        count
    };

    // Test 200 thresholds from 5..70
    let n_tests = 200usize;
    let test_min = 5.0f64;
    let test_max = 70.0f64;
    let tested: Vec<f64> = (0..n_tests)
        .map(|i| test_min + (i as f64 / (n_tests - 1) as f64) * (test_max - test_min))
        .collect();
    let remaining: Vec<f64> = tested
        .iter()
        .map(|&t| count_surviving(t as usize) as f64)
        .collect();

    // Gaussian smoothing (sigma=5 in the original, ~15 samples at 200 tests)
    let sigma = 5.0f64;
    let kernel_half = (3.0 * sigma).ceil() as usize;
    let gauss_kernel: Vec<f64> = (0..=2 * kernel_half)
        .map(|i| {
            let x = i as f64 - kernel_half as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let gsum: f64 = gauss_kernel.iter().sum();
    let gauss_kernel: Vec<f64> = gauss_kernel.iter().map(|&v| v / gsum).collect();

    // Gaussian smoothing with reflect boundary (matches scipy default mode='reflect')
    let mut smoothed = vec![0.0f64; n_tests];
    let n = n_tests as isize;
    for i in 0..n_tests {
        let mut val = 0.0;
        for (ki, &gv) in gauss_kernel.iter().enumerate() {
            let mut si = i as isize + ki as isize - kernel_half as isize;
            // Reflect boundary: -k → k-1, n+k → n-1-k (half-sample symmetric)
            if si < 0 {
                si = -si - 1;
            } else if si >= n {
                si = 2 * n - 1 - si;
            }
            si = si.max(0).min(n - 1);
            val += remaining[si as usize] * gv;
        }
        smoothed[i] = val;
    }

    // Gradient
    let mut gradient = vec![0.0f64; n_tests];
    for i in 1..n_tests - 1 {
        gradient[i] = (smoothed[i + 1] - smoothed[i - 1]) / 2.0;
    }
    gradient[0] = smoothed[1] - smoothed[0];
    gradient[n_tests - 1] = smoothed[n_tests - 1] - smoothed[n_tests - 2];

    // Find first peak in gradient (first local maximum).
    // Matches production's: int(tested[gradient_peaks[0]])
    let mut optimal_threshold = tested[n_tests / 2] as usize;
    for i in 1..n_tests - 1 {
        if gradient[i] > gradient[i - 1] && gradient[i] > gradient[i + 1] {
            optimal_threshold = tested[i] as usize;
            break;
        }
    }

    let stripe_threshold = optimal_threshold;
    eprintln!("[rough_binned_fit] optimal stripe threshold: {}", stripe_threshold);
    // Dump curves for comparison with production
    eprintln!("[rough_binned_fit] remaining[0..30]: {:?}", &remaining[..30.min(n_tests)]);
    eprintln!("[rough_binned_fit] gradient[0..30]: {:?}",
        gradient[..30.min(n_tests)].iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    // Dump gradient peak detection
    for i in 1..n_tests.min(25) {
        if gradient[i] > gradient[i.saturating_sub(1)] && gradient[i] > gradient[(i+1).min(n_tests-1)] {
            eprintln!("[rough_binned_fit] gradient local max at i={} value={:.6} tested={:.4}", i, gradient[i], tested[i]);
        }
    }

    // Apply stripe removal with the optimal threshold
    let mut swim_counts = std::collections::HashMap::<i64, usize>::new();
    let mut tof_counts = std::collections::HashMap::<i64, usize>::new();
    for i in 0..filt_tof.len() {
        *swim_counts.entry(int_swim[i]).or_insert(0) += 1;
        *tof_counts.entry(int_tof[i]).or_insert(0) += 1;
    }

    let mut stripe_mask = vec![false; filt_tof.len()];
    for i in 0..filt_tof.len() {
        let sc = swim_counts.get(&int_swim[i]).copied().unwrap_or(0);
        let tc = tof_counts.get(&int_tof[i]).copied().unwrap_or(0);
        stripe_mask[i] = sc < stripe_threshold && tc < stripe_threshold;
    }

    // NOTE: Production bug faithfully ported.
    // Python's filter_dense_lines has keep_tallest=True which modifies
    // combined_mask to re-include the tallest peak on each dense row/col,
    // but then returns points[x_mask & y_mask] (the ORIGINAL mask) instead
    // of points[combined_mask]. So the tallest peaks are never actually kept.
    // We intentionally do NOT keep tallest peaks here to match production.

    let mut stripe_keep = Vec::new();
    for i in 0..filt_tof.len() {
        if stripe_mask[i] { stripe_keep.push(i); }
    }

    let filt_tof: Vec<f64> = stripe_keep.iter().map(|&i| filt_tof[i]).collect();
    let filt_swim: Vec<f64> = stripe_keep.iter().map(|&i| filt_swim[i]).collect();

    eprintln!("[rough_binned_fit] after stripe removal: {} peaks (threshold={})", filt_tof.len(), stripe_threshold);

    // -----------------------------------------------------------------------
    // 7. Linear band filter — for diagnostic/plotting ONLY.
    //    Production passes without_stripes to RANSAC, NOT in_linear.
    //    (rough_autocorrelation_step.py:98 — RANSAC on without_stripes)
    // -----------------------------------------------------------------------
    let band_count = filt_tof.iter().zip(filt_swim.iter())
        .filter(|(&t, &s)| s > 0.8 * t - 20.0 && s < 1.2 * t + 20.0)
        .count();
    eprintln!("[rough_binned_fit] in linear band (diagnostic only): {} peaks", band_count);

    // -----------------------------------------------------------------------
    // 8. RANSAC on stripe-surviving peaks (matching production)
    // -----------------------------------------------------------------------
    if filt_tof.len() < 2 {
        info!("[ransac] rough: too few peaks for RANSAC, falling back to (1.0, 0.0)");
        return Ok((1.0, 0.0));
    }

    // Original uses residual_threshold=10 for both rough and refined RANSAC
    let ransac_residual_threshold = 10.0;
    let (slope, intercept, inlier_count) =
        ransac_fit(&filt_tof, &filt_swim, ransac_residual_threshold, 1000, 42, use_sklearn);

    eprintln!("[rough_binned_fit] RANSAC fit: slope={:.6} intercept={:.4} ({} inliers from {} peaks)",
        slope, intercept, inlier_count, filt_tof.len());
    // Dump first 5 peaks going into RANSAC for comparison
    for i in 0..filt_tof.len().min(5) {
        eprintln!("  peak[{}]: tof={:.4} swim={:.4}", i, filt_tof[i], filt_swim[i]);
    }

    // Dump all RANSAC input peaks to /tmp for diagnostic comparison
    if let Ok(diag) = serde_json::to_string_pretty(&serde_json::json!({
        "peaks": filt_tof.iter().zip(filt_swim.iter()).zip(filt_tof_idx.iter()).zip(filt_swim_idx.iter())
            .map(|(((&t, &s), &ti), &si)| serde_json::json!({"tof": t, "swim": s, "tof_idx": ti, "swim_idx": si}))
            .collect::<Vec<_>>(),
        "slope": slope,
        "intercept": intercept,
        "inlier_count": inlier_count,
        "stripe_threshold": stripe_threshold,
        "subset_shape": [sub_n_tof, sub_n_swim],
    })) {
        let _ = std::fs::write("/tmp/rough_binned_fit_diagnostic.json", diag);
    }

    Ok((slope, intercept))
}

// ---------------------------------------------------------------------------
// Main public RANSAC entry point
// ---------------------------------------------------------------------------

pub fn ransac(
    ds: &DatasetState,
    sh: &mut SherlockState,
    tolerance: f64,
    n_peaks: usize,
) -> Result<RansacResult, String> {
    sh.ransac_slope = None;
    sh.ransac_intercept = None;
    sh.ransac_inlier_mask = None;

    let centroids_tof = sh
        .centroids_tof
        .as_ref()
        .ok_or("No centroids — run centroid first")?;
    let centroids_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;

    let n = centroids_tof.len();
    if n < 2 {
        let slope = 0.0;
        let intercept = if n == 1 { centroids_swim[0] } else { 0.0 };
        sh.ransac_slope = Some(slope);
        sh.ransac_intercept = Some(intercept);
        sh.ransac_inlier_mask = Some(vec![true; n]);
        return Ok(RansacResult {
            slope,
            intercept,
            inlier_count: n,
            outlier_count: 0,
            total: n,
            display_tof: centroids_tof.clone(),
            display_swim: centroids_swim.clone(),
            display_inlier: vec![true; n],
        });
    }

    // -----------------------------------------------------------------------
    // Stage 1: Rough fit on binned raw data (proper autocorrelation pipeline)
    //
    // Bins the TOF axis with quadratic spacing, finds local maxima,
    // filters by range/amplitude/stripes/linear band, then RANSAC fits.
    // This gives a reliable approximate autocorrelation line even on
    // datasets where centroid-based rough RANSAC fails (e.g. sample-006).
    // -----------------------------------------------------------------------
    let use_sklearn = sh.use_sklearn_ransac;
    let (rough_slope, rough_intercept) = rough_binned_fit(ds, tolerance, use_sklearn)?;

    info!(
        "[ransac] using rough fit: swim = {:.4} * tof + {:.4}",
        rough_slope, rough_intercept
    );

    // -----------------------------------------------------------------------
    // Stage 2: Refined fit — filter ALL centroids within tolerance of the
    // rough line, sort by amplitude, take top n_peaks, RANSAC on those.
    // -----------------------------------------------------------------------
    let amplitudes = sh.peak_amplitudes.as_ref();
    let peak_mask = sh.top_n_mask.as_ref().or(sh.stripe_mask.as_ref());

    // Build amplitude for each centroid (surviving peak after top_n/stripes)
    let centroid_amps: Vec<f64> = if let (Some(amps), Some(mask)) = (amplitudes, peak_mask) {
        let surviving: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        surviving.iter().map(|&i| amps[i]).collect()
    } else {
        vec![0.0; n]
    };

    // Filter centroids near the rough line
    let near_line: Vec<usize> = (0..n)
        .filter(|&i| {
            (centroids_swim[i] - (rough_slope * centroids_tof[i] + rough_intercept)).abs()
                < tolerance
        })
        .collect();

    // Sort by amplitude descending, take top n_peaks
    let mut near_sorted = near_line.clone();
    near_sorted.sort_by(|&a, &b| {
        centroid_amps[b]
            .partial_cmp(&centroid_amps[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_n = near_sorted.len().min(n_peaks);
    let sub_idx: Vec<usize> = near_sorted[..top_n].to_vec();

    let sub_tof: Vec<f64> = sub_idx.iter().map(|&i| centroids_tof[i]).collect();
    let sub_swim: Vec<f64> = sub_idx.iter().map(|&i| centroids_swim[i]).collect();

    info!(
        "[ransac] refined: fitting on top {} peaks by amplitude (out of {} near-line, {} total)",
        sub_tof.len(),
        near_line.len(),
        n
    );

    if sub_tof.len() < 2 {
        // Rough fit didn't help — no centroids are near the rough line.
        // Fall back to direct RANSAC on ALL centroids (the rough fit may have
        // returned the (1.0, 0.0) sentinel because the data range didn't cover
        // the hardcoded [400,1000] subset used by rough_binned_fit).
        info!("[ransac] too few near-line peaks ({}) — falling back to direct RANSAC on all {} centroids",
              sub_tof.len(), n);

        let (slope, intercept, _) = ransac_fit(centroids_tof, centroids_swim, 10.0, 1000, 42, use_sklearn);
        info!("[ransac] direct fallback fit: swim = {:.6} * tof + {:.6}", slope, intercept);

        let full_inlier_mask: Vec<bool> = centroids_tof
            .iter()
            .zip(centroids_swim.iter())
            .map(|(&t, &s)| (s - (slope * t + intercept)).abs() < tolerance)
            .collect();
        let inlier_count = full_inlier_mask.iter().filter(|&&m| m).count();
        let outlier_count = n - inlier_count;

        let mut display_rng = rand::thread_rng();
        let display_idx = subsample_indices(n, 5000, &mut display_rng);
        let display_tof: Vec<f64> = display_idx.iter().map(|&i| centroids_tof[i]).collect();
        let display_swim: Vec<f64> = display_idx.iter().map(|&i| centroids_swim[i]).collect();
        let display_inlier: Vec<bool> =
            display_idx.iter().map(|&i| full_inlier_mask[i]).collect();

        // Store the raw rough fit (not the fallback direct fit)
        sh.ransac_slope = Some(rough_slope);
        sh.ransac_intercept = Some(rough_intercept);
        sh.ransac_inlier_mask = Some(full_inlier_mask);

        return Ok(RansacResult {
            slope,
            intercept,
            inlier_count,
            outlier_count,
            total: n,
            display_tof,
            display_swim,
            display_inlier,
        });
    }

    // RANSAC on the refined peak set (residual_threshold=10 matching original)
    let (slope, intercept, _) = ransac_fit(&sub_tof, &sub_swim, 10.0, 1000, 42, use_sklearn);

    info!(
        "[ransac] refined fit: swim = {:.6} * tof + {:.6}",
        slope, intercept
    );

    // Classify ALL centroids against the refined line
    let full_inlier_mask: Vec<bool> = centroids_tof
        .iter()
        .zip(centroids_swim.iter())
        .map(|(&t, &s)| (s - (slope * t + intercept)).abs() < tolerance)
        .collect();

    let inlier_count = full_inlier_mask.iter().filter(|&&m| m).count();
    let outlier_count = n - inlier_count;

    // Display data uses centroid positions for scatter plot
    let mut display_rng = rand::thread_rng();
    let display_idx = subsample_indices(n, 5000, &mut display_rng);

    let display_tof: Vec<f64> = display_idx.iter().map(|&i| centroids_tof[i]).collect();
    let display_swim: Vec<f64> = display_idx.iter().map(|&i| centroids_swim[i]).collect();
    let display_inlier: Vec<bool> = display_idx.iter().map(|&i| full_inlier_mask[i]).collect();

    // Store the RAW rough fit, not the refined fit.
    // precise_autocorrelation uses this to filter full-res peaks,
    // matching production where the precise step sees the actual rough fit.
    sh.ransac_slope = Some(rough_slope);
    sh.ransac_intercept = Some(rough_intercept);
    sh.ransac_inlier_mask = Some(full_inlier_mask);

    Ok(RansacResult {
        slope,
        intercept,
        inlier_count,
        outlier_count,
        total: n,
        display_tof,
        display_swim,
        display_inlier,
    })
}

// ---------------------------------------------------------------------------
// Step 8: Precise autocorrelation
//
// Takes the rough fit from the RANSAC step, computes residuals of ALL full-res
// peaks against it, filters by tolerance (default 1.5 Da), sorts remaining by
// amplitude descending, takes top n_peaks (default 15), computes centroids
// for those peaks, then runs RANSAC on the centroided positions.
// ---------------------------------------------------------------------------

/// Compute the centroid (intensity-weighted center-of-mass) for a single peak.
///
/// `row` and `col` are indices into the dataset grid. The window is
/// `[row - hw_swim .. row + hw_swim] x [col - hw_tof .. col + hw_tof]`.
fn centroid_single_peak(
    ds: &DatasetState,
    row: usize,
    col: usize,
    hw_tof: usize,
    hw_swim: usize,
) -> (f64, f64) {
    let data = ds.data.as_ref().unwrap();
    let tof_coords = ds.tof_coords.as_ref().unwrap();
    let swim_coords = ds.swim_coords.as_ref().unwrap();
    let (n_rows, n_cols) = data.dim();

    let r_lo = row.saturating_sub(hw_swim);
    let r_hi = (row + hw_swim).min(n_rows - 1);
    let c_lo = col.saturating_sub(hw_tof);
    let c_hi = (col + hw_tof).min(n_cols - 1);

    let mut total = 0.0f64;
    let mut weighted_tof = 0.0f64;
    let mut weighted_swim = 0.0f64;

    for r in r_lo..=r_hi {
        for c in c_lo..=c_hi {
            let intensity = data[[r, c]] as f64;
            total += intensity;
            weighted_tof += intensity * tof_coords[c];
            weighted_swim += intensity * swim_coords[r];
        }
    }

    if total > 0.0 {
        (weighted_tof / total, weighted_swim / total)
    } else {
        (tof_coords[col], swim_coords[row])
    }
}

pub fn precise_autocorrelation(
    ds: &DatasetState,
    sh: &mut SherlockState,
    tolerance: f64,
    _n_peaks: usize,
    _half_window_tof: usize,
    _half_window_swim: usize,
) -> Result<PreciseAutocorrelationResult, String> {
    sh.precise_slope = None;
    sh.precise_intercept = None;
    sh.precise_inlier_mask = None;

    let use_sklearn = sh.use_sklearn_ransac;
    let rough_slope = sh.ransac_slope.ok_or("No rough fit — run RANSAC first")?;
    let rough_intercept = sh.ransac_intercept.ok_or("No rough fit")?;

    info!(
        "[precise_autocorrelation] rough fit: swim = {:.6} * tof + {:.6}",
        rough_slope, rough_intercept
    );

    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks — run find_filter first")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peaks")?;
    let peak_amplitudes = sh.peak_amplitudes.as_ref().ok_or("No peak amplitudes")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No TOF coordinates")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;

    let total_peaks = peak_rows.len();
    info!("[precise_autocorrelation] total full-res peaks: {}", total_peaks);

    // Step 1: Filter peaks within tolerance of rough fit line
    let mut near_line_indices: Vec<usize> = Vec::new();
    for i in 0..total_peaks {
        let tof_mass = tof_coords[peak_cols[i]];
        let swim_mass = swim_coords[peak_rows[i]];
        let predicted_swim = rough_slope * tof_mass + rough_intercept;
        let residual = (swim_mass - predicted_swim).abs();
        if residual < tolerance {
            near_line_indices.push(i);
        }
    }

    info!(
        "[precise_autocorrelation] peaks within tolerance {}: {} / {}",
        tolerance, near_line_indices.len(), total_peaks
    );

    if near_line_indices.is_empty() {
        return Err(format!("No peaks within tolerance {} of rough fit line", tolerance));
    }

    // Step 2: Sort by amplitude descending, take top n_peaks
    near_line_indices.sort_by(|&a, &b| {
        peak_amplitudes[b].partial_cmp(&peak_amplitudes[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let peaks_near_line = near_line_indices.len();
    // Production hardcodes n_peaks=15 for precise fit (autocorrelation.py step 8)
    let top_n = near_line_indices.len().min(15);
    let top_indices: Vec<usize> = near_line_indices[..top_n].to_vec();

    info!(
        "[precise_autocorrelation] selected top {} peaks by amplitude (from {} near-line)",
        top_n, peaks_near_line
    );

    // Step 3: Collect raw mass coordinates for these top peaks
    // Production uses use_centroids=False — raw grid coordinates, NOT centroid coordinates
    let mut fit_tof_vals = Vec::with_capacity(top_n);
    let mut fit_swim_vals = Vec::with_capacity(top_n);

    for &idx in &top_indices {
        let row = peak_rows[idx];
        let col = peak_cols[idx];
        fit_tof_vals.push(tof_coords[col]);
        fit_swim_vals.push(swim_coords[row]);
    }

    // Step 4: RANSAC on raw mass positions (residual_threshold=10, 1000 iterations)
    if top_n < 2 {
        let slope = if top_n == 1 { rough_slope } else { 0.0 };
        let intercept = if top_n == 1 {
            fit_swim_vals[0] - rough_slope * fit_tof_vals[0]
        } else { 0.0 };

        sh.precise_slope = Some(slope);
        sh.precise_intercept = Some(intercept);
        sh.precise_inlier_mask = Some(Vec::new());

        return Ok(PreciseAutocorrelationResult {
            slope, intercept, inlier_count: top_n, outlier_count: 0, total: 0,
            peaks_near_line, peaks_used: top_n,
            display_tof: fit_tof_vals, display_swim: fit_swim_vals,
            display_inlier: vec![true; top_n],
        });
    }

    let (slope, intercept, _) = ransac_fit(&fit_tof_vals, &fit_swim_vals, 10.0, 1000, 42, use_sklearn);

    info!(
        "[precise_autocorrelation] RANSAC fit: swim = {:.10} * tof + {:.10}",
        slope, intercept
    );

    // Step 5: Classify ALL centroids (from centroid step) against the precise fit
    let centroids_tof = sh.centroids_tof.as_ref().ok_or("No centroids — run centroid first")?;
    let centroids_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;

    let n_centroids = centroids_tof.len();
    let full_inlier_mask: Vec<bool> = centroids_tof
        .iter()
        .zip(centroids_swim.iter())
        .map(|(&t, &s)| (s - (slope * t + intercept)).abs() < tolerance)
        .collect();

    let inlier_count = full_inlier_mask.iter().filter(|&&m| m).count();
    let outlier_count = n_centroids - inlier_count;

    let mut display_rng = rand::thread_rng();
    let display_idx = subsample_indices(n_centroids, 5000, &mut display_rng);
    let display_tof: Vec<f64> = display_idx.iter().map(|&i| centroids_tof[i]).collect();
    let display_swim: Vec<f64> = display_idx.iter().map(|&i| centroids_swim[i]).collect();
    let display_inlier: Vec<bool> = display_idx.iter().map(|&i| full_inlier_mask[i]).collect();

    sh.precise_slope = Some(slope);
    sh.precise_intercept = Some(intercept);
    sh.precise_inlier_mask = Some(full_inlier_mask);

    Ok(PreciseAutocorrelationResult {
        slope, intercept, inlier_count, outlier_count, total: n_centroids,
        peaks_near_line, peaks_used: top_n,
        display_tof, display_swim, display_inlier,
    })
}

// ---------------------------------------------------------------------------
// Test-only public wrappers for cross-language validation.
// ---------------------------------------------------------------------------

#[cfg(feature = "test-internals")]
pub mod test_internals {
    use crate::dataset::DatasetState;

    /// Public wrapper for ransac_fit.
    /// Returns (slope, intercept, inlier_count).
    pub fn ransac_fit_pub(
        tof: &[f64],
        swim: &[f64],
        tolerance: f64,
        iterations: usize,
        seed: u64,
    ) -> (f64, f64, usize) {
        super::ransac_fit(tof, swim, tolerance, iterations, seed, true)
    }

    /// Public wrapper for rough_binned_fit.
    /// Returns (slope, intercept) from the binned rough autocorrelation pipeline.
    pub fn rough_binned_fit_pub(ds: &DatasetState, tolerance: f64) -> Result<(f64, f64), String> {
        super::rough_binned_fit(ds, tolerance, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{self, DatasetState};
    use std::path::Path;

    #[test]
    fn test_rough_binned_fit_synthetic() {
        // This test requires /tmp/test_binning_synthetic.nc
        // Created by: python3 test_binning.py
        let nc_path = "/tmp/test_binning_synthetic.nc";
        if !Path::new(nc_path).exists() {
            eprintln!("SKIP: {} not found. Run python3 test_binning.py first.", nc_path);
            return;
        }

        let mut ds = DatasetState::default();
        dataset::scan_directory(&mut ds, nc_path).expect("scan failed");
        dataset::load_nc(&mut ds).expect("load failed");

        assert!(ds.loaded);
        assert_eq!(ds.shape, (1025, 50000));

        // Run rough_binned_fit (native RANSAC, no sklearn)
        let result = rough_binned_fit(&ds, 10.0, false);
        assert!(result.is_ok(), "rough_binned_fit failed: {:?}", result.err());

        let (slope, intercept) = result.unwrap();

        // The synthetic data has peaks on the autocorrelation line (swim ≈ tof)
        // so the expected slope is ~1.0 and intercept near 0.
        eprintln!("rough_binned_fit result: slope={:.6}, intercept={:.4}", slope, intercept);

        // Production binning finds:
        // - 1637 TOF bins in range, 122 SWIM rows in range
        // - 22 local maxima, 22 after threshold
        // - 22 unique tof_idx
        //
        // If Rust's binning differs, the slope/intercept will diverge.
        // A slope near 1.0 (±0.1) means the binning found the autocorrelation line.
        // A slope far from 1.0 means the binning produced different peaks.
        assert!(slope > 0.85 && slope < 1.15,
            "Slope {:.4} is too far from 1.0 — binning likely diverged", slope);
        assert!(intercept.abs() < 50.0,
            "Intercept {:.4} is too far from 0 — fit may be on wrong feature", intercept);
    }
}
