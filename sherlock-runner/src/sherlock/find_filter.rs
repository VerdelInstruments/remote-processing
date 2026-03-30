use log::info;
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::VecDeque;

use crate::dataset::DatasetState;
use super::progress::emit_progress;
use super::subsample::{histogram, subsample_indices};
use super::SherlockState;

/// Default analysis range for swim mass (Da). Matches production high_res_range.
pub const DEFAULT_SWIM_RANGE: (f64, f64) = (100.0, 1800.0);
/// Default analysis range for tof mass (Da). Matches production high_res_range.
pub const DEFAULT_TOF_RANGE: (f64, f64) = (100.0, 1800.0);

#[derive(Debug, Serialize)]
pub struct FindFilterResult {
    pub peak_count: usize,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub snr_hist_counts: Vec<u64>,
    pub snr_hist_edges: Vec<f64>,
    pub min_snr_used: f64,
    pub algorithm_used: String,
}

// ---------------------------------------------------------------------------
// Sorted-array running median — O(w) insert/remove, O(1) median.
// For small windows (w <= ~50) this vastly outperforms heap + lazy-deletion
// because the inner loop is a small memcpy with no allocations or hashing.
// ---------------------------------------------------------------------------

struct RunningMedian {
    buf: Vec<f32>,
}

impl RunningMedian {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn push(&mut self, val: f32) {
        let pos = self.buf.partition_point(|&x| x < val);
        self.buf.insert(pos, val);
    }

    fn remove(&mut self, val: f32) {
        let pos = self.buf.partition_point(|&x| x < val);
        // Find exact match (handles duplicates correctly)
        if pos < self.buf.len() && self.buf[pos] == val {
            self.buf.remove(pos);
        } else if pos + 1 < self.buf.len() && self.buf[pos + 1] == val {
            self.buf.remove(pos + 1);
        }
    }

    fn median(&self) -> f32 {
        if self.buf.is_empty() {
            0.0
        } else {
            self.buf[self.buf.len() / 2]
        }
    }
}

// ---------------------------------------------------------------------------
// Column extraction helper — fixes cache thrashing on row-major arrays
// ---------------------------------------------------------------------------

/// Extract a single column from a row-major Array2<f32> into a contiguous Vec.
fn extract_column(data: &Array2<f32>, col: usize) -> Vec<f32> {
    let n_rows = data.nrows();
    let mut out = Vec::with_capacity(n_rows);
    // Use raw slice + stride for best possible performance
    let raw = data.as_slice().unwrap();
    let n_cols = data.ncols();
    let mut idx = col;
    for _ in 0..n_rows {
        out.push(raw[idx]);
        idx += n_cols;
    }
    out
}

// ---------------------------------------------------------------------------
// Rolling median — column-wise, dual-heap, parallelized
// ---------------------------------------------------------------------------

/// Centered rolling median along axis 0 (column-wise / swim direction).
///
/// Production applies median_filter(data, size=(51,1)) on (swim, tof) data,
/// meaning 51 along axis 0 = swim rows.  For each tof column, a 51-element
/// centered median slides down the 1025 swim rows.
///
/// Parallelized across tof columns with rayon.
/// Returns flat row-major Vec<f32> (same layout as data: index = row * n_cols + col).
fn rolling_median_cols(data: &Array2<f32>, window: usize) -> Vec<f32> {
    let (n_rows, n_cols) = data.dim();
    let half = window / 2;
    let data_raw = data.as_slice().unwrap();
    let mut result = vec![0.0f32; n_rows * n_cols];

    // Process each column in parallel.  We collect per-column results, then
    // scatter them into the row-major output.
    let col_results: Vec<Vec<f32>> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            // Extract this column into a contiguous buffer
            let col_data: Vec<f32> = (0..n_rows)
                .map(|r| data_raw[r * n_cols + col])
                .collect();

            let mut med = RunningMedian::new();
            let mut out = vec![0.0f32; n_rows];

            // Initialize centered window for row 0
            let init_right = half.min(n_rows - 1);
            for i in 0..=init_right {
                med.push(col_data[i]);
            }
            out[0] = med.median();

            // Slide the centered window down along the column
            for row in 1..n_rows {
                let new_bottom = row + half;
                if new_bottom < n_rows {
                    med.push(col_data[new_bottom]);
                }
                let old_top = row as isize - half as isize - 1;
                if old_top >= 0 {
                    med.remove(col_data[old_top as usize]);
                }
                out[row] = med.median();
            }
            out
        })
        .collect();

    // Scatter column results into row-major layout
    for (col, col_out) in col_results.iter().enumerate() {
        for (row, &val) in col_out.iter().enumerate() {
            result[row * n_cols + col] = val;
        }
    }

    result
}

/// Column-wise rolling median with reflected boundary padding.
/// Matches scipy.ndimage.median_filter(data, size=(51,1), mode="reflect")
/// on (swim, tof) data: 51-element window slides along axis 0 (swim rows)
/// for each tof column independently.
/// Returns flat row-major Vec<f32> (same layout as data).
/// Public so peaks_above_noise can reuse it.
pub fn rolling_median_precise(data: &Array2<f32>, window: usize) -> Vec<f32> {
    let (n_rows, n_cols) = data.dim();
    let half = window / 2;
    let data_raw = data.as_slice().unwrap();
    let mut result = vec![0.0f32; n_rows * n_cols];

    // Process each tof column in parallel, sliding the median along swim rows.
    let col_results: Vec<Vec<f32>> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            // Extract column into contiguous buffer
            let col_data: Vec<f32> = (0..n_rows)
                .map(|r| data_raw[r * n_cols + col])
                .collect();
            let n = n_rows;
            let mut med = RunningMedian::new();
            let mut out = vec![0.0f32; n];

            // Build initial window for row=0.
            // scipy reflect mode (half-sample symmetric):
            //   left:  pos -k → index k-1  (edge value repeated)
            //   right: pos N+k → index N-1-k
            for i in 0..window.min(n + half) {
                let idx = if i < half {
                    // Reflected left: position -(half-i) → index (half-1-i)
                    (half - 1 - i).min(n - 1)
                } else {
                    (i - half).min(n - 1)
                };
                med.push(col_data[idx]);
            }

            for row in 0..n {
                out[row] = med.median();

                if row + 1 < n {
                    // Remove the topmost element of the old window
                    let remove_idx = if row < half {
                        // Reflected top: position row-half → index half-1-row
                        half - 1 - row
                    } else {
                        row - half
                    };
                    let remove_idx = remove_idx.min(n - 1);
                    med.remove(col_data[remove_idx]);

                    // Add the new bottom element
                    let add_idx = row + 1 + half;
                    let add_idx = if add_idx >= n {
                        // Reflected bottom: position N+k → index N-1-k
                        let reflected = 2 * n - 1 - add_idx;
                        reflected.min(n - 1).max(0)
                    } else {
                        add_idx
                    };
                    med.push(col_data[add_idx]);
                }
            }
            out
        })
        .collect();

    // Scatter column results into row-major layout
    for (col, col_out) in col_results.iter().enumerate() {
        for (row, &val) in col_out.iter().enumerate() {
            result[row * n_cols + col] = val;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Sliding max — single-pass centered monotonic deque (f32 version for noise)
// ---------------------------------------------------------------------------

/// 1D centered sliding maximum using a monotonic deque. O(n). Operates on f32.
/// Kept for potential future use (e.g. noise-based max filtering).
#[allow(dead_code)]
fn sliding_max_1d_f32(input: &[f32], output: &mut [f32], kernel: usize) {
    let n = input.len();
    if n == 0 { return; }
    let half = kernel / 2;
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(kernel);

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


/// 2D maximum filter using separable 1D passes. Matches scipy maximum_filter(size=kernel).
/// All f32 — keeps memory at 2×data instead of 4×data.
/// Returns flat column-major Vec<f32> (index = col * n_rows + row).
fn maximum_filter_2d(data: &Array2<f32>, kernel: usize) -> Vec<f32> {
    let (n_rows, n_cols) = data.dim();

    // Pass 1: row-wise max (contiguous reads, fast)
    let mut intermediate = Array2::<f32>::zeros((n_rows, n_cols));
    {
        let raw_in = data.as_slice().unwrap();
        let raw_out = intermediate.as_slice_mut().unwrap();
        raw_out
            .par_chunks_mut(n_cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let in_row = &raw_in[row * n_cols..(row + 1) * n_cols];
                sliding_max_1d_f32(in_row, out_row, kernel);
            });
    }

    // Pass 2: column-wise max → column-major result
    let mut result = vec![0.0f32; n_rows * n_cols];
    result
        .par_chunks_mut(n_rows)
        .enumerate()
        .for_each(|(col, out)| {
            let input = extract_column(&intermediate, col);
            sliding_max_1d_f32(&input, out, kernel);
        });
    result
}

// ---------------------------------------------------------------------------
// Main find_filter — two-phase memory optimization:
//   Phase 1: noise (f32, 3GB) + data (f32, 3GB) = 6GB. Extract SNR candidates. Drop noise.
//   Phase 2: max filter (f32, 3GB intermediate + 3GB result) + data (f32, 3GB) = 9GB. Check candidates. Drop max filter.
//   Peak memory = max(6, 9) = 9GB.
// ---------------------------------------------------------------------------

pub fn find_filter(
    ds: &DatasetState,
    sh: &mut SherlockState,
    min_snr: f64,
    noise_window: usize,
    peak_filter_size: usize,
    algorithm: &str,
    swim_range: Option<(f64, f64)>,
    tof_range: Option<(f64, f64)>,
) -> Result<FindFilterResult, String> {
    sh.invalidate_from(2);

    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    // Default to production high_res_range [100, 1800] when no range specified
    let swim_range = swim_range.or(Some(DEFAULT_SWIM_RANGE));
    let tof_range = tof_range.or(Some(DEFAULT_TOF_RANGE));
    info!("[find_filter] swim_range: {:?}, tof_range: {:?}", swim_range, tof_range);

    let (n_rows, n_cols) = data.dim();
    let min_snr_f32 = min_snr as f32;

    // -----------------------------------------------------------------------
    // Phase 1: Compute noise floor (f32) and extract SNR candidates
    // -----------------------------------------------------------------------
    emit_progress(0.0, "Computing noise floor...");

    let t1 = std::time::Instant::now();
    let noise_flat = if algorithm == "precise" {
        rolling_median_precise(data, noise_window)
    } else {
        rolling_median_cols(data, noise_window)
    };
    info!("[find_filter] noise floor ({}): {}ms", algorithm, t1.elapsed().as_millis());

    // Diagnostic: print noise values for swim row 0 (= last original freq, our reversed row 0)
    // to compare against Python's transposed noise at swim column 0.
    {
        // Row-major: noise at (row, col) = noise_flat[row * n_cols + col]
        // For row 0 (swim=0): noise_flat[0..n_cols]
        let row0_noise: Vec<f32> = noise_flat[0..10.min(n_cols)].to_vec();
        info!("[find_filter] DIAG row0 noise[:10] (swim=0, tof=0..9): {:?}", row0_noise);

        // For comparison with Python: noise at (swim=last_orig_freq, tof=*)
        // Our row 0 = last original frequency. Python's swim col 0 = first frequency = our row n_rows-1.
        let last_row_noise: Vec<f32> = noise_flat[(n_rows-1)*n_cols..(n_rows-1)*n_cols + 10.min(n_cols)].to_vec();
        info!("[find_filter] DIAG row{} noise[:10] (swim=last=orig_freq_0): {:?}", n_rows-1, last_row_noise);

        // Global noise stats
        let nmin = noise_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let nmax = noise_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let nmean: f32 = noise_flat.iter().sum::<f32>() / noise_flat.len() as f32;
        let nz = noise_flat.iter().filter(|&&v| v > 0.0).count();
        info!("[find_filter] DIAG noise stats: min={:.2} max={:.2} mean={:.2} nonzero={}/{}", nmin, nmax, nmean, nz, noise_flat.len());

        // Remove old column-based diagnostics
        let col0_noise: Vec<f32> = (0..n_rows).map(|r| noise_flat[r * n_cols + 0]).collect();
        info!("[find_filter] DIAG col0 data[:10]: {:?}", &data.as_slice().unwrap()[..10.min(n_cols)].iter().map(|v| *v).collect::<Vec<f32>>());
        // col0 of data: data[row, 0] = data_raw[row * n_cols + 0]
        let col0_data: Vec<f32> = (0..10.min(n_rows)).map(|r| data.as_slice().unwrap()[r * n_cols]).collect();
        info!("[find_filter] DIAG col0 data(col-extracted)[:10]: {:?}", col0_data);
        info!("[find_filter] DIAG col0 noise[:10]: {:?}", &col0_noise[..10.min(n_rows)]);
        info!("[find_filter] DIAG col0 noise[500..510]: {:?}", &col0_noise[500..510.min(n_rows)]);
        let nmin = col0_noise.iter().cloned().fold(f32::INFINITY, f32::min);
        let nmax = col0_noise.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let nmean: f32 = col0_noise.iter().sum::<f32>() / col0_noise.len() as f32;
        info!("[find_filter] DIAG col0 noise stats: min={:.2} max={:.2} mean={:.2}", nmin, nmax, nmean);
    }

    // Extract SNR candidates: cells where data > min_snr * noise.
    // Store (row, col, snr) so we can filter in phase 2 without re-reading noise.
    emit_progress(0.35, "Extracting SNR candidates...");
    let t_cand = std::time::Instant::now();
    let data_raw = data.as_slice().unwrap();

    let per_row_candidates: Vec<Vec<(usize, usize, f64)>> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let data_row = &data_raw[row * n_cols..(row + 1) * n_cols];
            let mut local = Vec::new();
            // Apply range filter: skip peaks outside swim/tof mass ranges
            let swim_ok = match swim_range {
                Some((lo, hi)) => {
                    let s = swim_coords[row];
                    s >= lo && s <= hi
                }
                None => true,
            };
            if !swim_ok { return local; }

            for col in 0..n_cols {
                if let Some((lo, hi)) = tof_range {
                    let t = tof_coords[col];
                    if t < lo || t > hi { continue; }
                }
                let val = data_row[col];
                let idx = row * n_cols + col;  // row-major (same layout as data)
                let nf = noise_flat[idx];
                if val > min_snr_f32 * nf {
                    let snr = if nf > 0.0 { (val / nf) as f64 } else { 0.0 };
                    local.push((row, col, snr));
                }
            }
            local
        })
        .collect();

    // Drop noise — frees ~3GB
    drop(noise_flat);

    // Flatten candidates
    let total_cand: usize = per_row_candidates.iter().map(|v| v.len()).sum();
    let mut candidates: Vec<(usize, usize, f64)> = Vec::with_capacity(total_cand);
    for chunk in per_row_candidates {
        candidates.extend(chunk);
    }
    info!(
        "[find_filter] SNR candidates: {} in {}ms",
        candidates.len(),
        t_cand.elapsed().as_millis()
    );

    // -----------------------------------------------------------------------
    // Phase 2: Compute max filter (f64) and keep only local maxima
    // -----------------------------------------------------------------------
    emit_progress(0.55, "Detecting local maxima...");

    let t2 = std::time::Instant::now();
    let filt_flat = maximum_filter_2d(data, peak_filter_size);
    info!("[find_filter] max filter: {}ms", t2.elapsed().as_millis());

    // Filter candidates: keep where data[row,col] (as f64) == max_filter[col*n_rows+row]
    emit_progress(0.75, "Extracting peaks...");
    let t3 = std::time::Instant::now();

    let mut peak_rows = Vec::new();
    let mut peak_cols = Vec::new();
    let mut amplitudes = Vec::new();
    let mut snr_values = Vec::new();

    for &(row, col, snr) in &candidates {
        let val = data_raw[row * n_cols + col];
        let filt_idx = col * n_rows + row;  // max filter is column-major
        if val == filt_flat[filt_idx] {
            peak_rows.push(row);
            peak_cols.push(col);
            amplitudes.push(val as f64);
            snr_values.push(snr);
        }
    }

    // Drop max filter — frees ~3GB
    drop(filt_flat);
    drop(candidates);

    info!(
        "[find_filter] peak extraction: {}ms ({} peaks)",
        t3.elapsed().as_millis(),
        peak_rows.len()
    );

    let peak_count = peak_rows.len();

    // Step 4: Display data
    emit_progress(0.95, "Preparing display data...");

    let mut rng = rand::thread_rng();
    let indices = subsample_indices(peak_count, 5000, &mut rng);

    let display_tof: Vec<f64> = indices.iter().map(|&i| tof_coords[peak_cols[i]]).collect();
    let display_swim: Vec<f64> = indices.iter().map(|&i| swim_coords[peak_rows[i]]).collect();

    // SNR histogram on log10(snr + 1)
    let log_snr: Vec<f64> = snr_values
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| (s + 1.0).log10())
        .collect();
    let (snr_hist_counts, snr_hist_edges) = histogram(&log_snr, 50);

    // Store state
    sh.peak_row_idx = Some(peak_rows);
    sh.peak_col_idx = Some(peak_cols);
    sh.peak_amplitudes = Some(amplitudes);
    sh.snr_values = Some(snr_values);

    emit_progress(1.0, "Done");

    Ok(FindFilterResult {
        peak_count,
        display_tof,
        display_swim,
        snr_hist_counts,
        snr_hist_edges,
        min_snr_used: min_snr,
        algorithm_used: algorithm.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Raw peak finding — ALL local maxima, no SNR filter.
// Matches production's get_peaks(dataset) which returns every cell where
// data == maximum_filter(data, size=5).  Used as the selection pool for
// top_n_peaks (step 6), matching production's 14M peak pool.
// ---------------------------------------------------------------------------

/// Result from raw peak finding (no SNR filter).
#[derive(Debug, Serialize)]
pub struct FindPeaksRawResult {
    pub peak_count: usize,
}

/// Find ALL local maxima within the given range.  No noise floor, no SNR.
/// Stores peak_row_idx, peak_col_idx, peak_amplitudes in SherlockState.
/// snr_values is set to empty (not computed).
pub fn find_peaks_raw(
    ds: &DatasetState,
    sh: &mut SherlockState,
    peak_filter_size: usize,
    swim_range: Option<(f64, f64)>,
    tof_range: Option<(f64, f64)>,
) -> Result<FindPeaksRawResult, String> {
    sh.invalidate_from(2);

    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    let swim_range = swim_range.or(Some(DEFAULT_SWIM_RANGE));
    let tof_range = tof_range.or(Some(DEFAULT_TOF_RANGE));

    let (n_rows, n_cols) = data.dim();

    emit_progress(0.0, "Computing maximum filter...");

    let t0 = std::time::Instant::now();
    let filt_flat = maximum_filter_2d(data, peak_filter_size);
    info!("[find_peaks_raw] max filter: {}ms", t0.elapsed().as_millis());

    emit_progress(0.5, "Extracting local maxima...");

    let data_raw = data.as_slice().unwrap();

    // Collect all local maxima within range (parallelised per row)
    let per_row: Vec<Vec<(usize, usize, f64)>> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let mut local = Vec::new();
            if let Some((lo, hi)) = swim_range {
                let s = swim_coords[row];
                if s < lo || s > hi { return local; }
            }
            for col in 0..n_cols {
                if let Some((lo, hi)) = tof_range {
                    let t = tof_coords[col];
                    if t < lo || t > hi { continue; }
                }
                let val = data_raw[row * n_cols + col];
                if val > 0.0 && val == filt_flat[col * n_rows + row] {
                    local.push((row, col, val as f64));
                }
            }
            local
        })
        .collect();

    drop(filt_flat);

    let mut peak_rows = Vec::new();
    let mut peak_cols = Vec::new();
    let mut amplitudes = Vec::new();

    for chunk in per_row {
        for (r, c, a) in chunk {
            peak_rows.push(r);
            peak_cols.push(c);
            amplitudes.push(a);
        }
    }

    let peak_count = peak_rows.len();
    info!("[find_peaks_raw] {} local maxima in range ({:.1}s)",
        peak_count, t0.elapsed().as_secs_f64());

    sh.peak_row_idx = Some(peak_rows);
    sh.peak_col_idx = Some(peak_cols);
    sh.peak_amplitudes = Some(amplitudes);
    sh.snr_values = Some(vec![]);

    emit_progress(1.0, "Done");

    Ok(FindPeaksRawResult { peak_count })
}

// ---------------------------------------------------------------------------
// Test-only public wrappers for cross-language validation.
// Enabled with: cargo test --features test-internals
// ---------------------------------------------------------------------------

#[cfg(feature = "test-internals")]
pub mod test_internals {
    use ndarray::Array2;

    /// Public wrapper for rolling_median_precise (reflected boundaries).
    /// Returns flat row-major Vec<f32>, same layout as input.
    pub fn rolling_median_precise_pub(data: &Array2<f32>, window: usize) -> Vec<f32> {
        super::rolling_median_precise(data, window)
    }

    /// Public wrapper for maximum_filter_2d (separable 1D passes).
    /// Returns flat **column-major** Vec<f32> (index = col * n_rows + row).
    pub fn maximum_filter_2d_pub(data: &Array2<f32>, kernel: usize) -> Vec<f32> {
        super::maximum_filter_2d(data, kernel)
    }
}
