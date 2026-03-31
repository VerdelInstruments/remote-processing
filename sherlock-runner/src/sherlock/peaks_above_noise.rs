//! Peaks-above-noise filter (production steps 9-10).
//!
//! Takes ALL full-resolution peaks (from find_filter step) and filters them
//! using a per-column noise floor computed via a 1D median filter along the
//! swim (row) axis with reflected boundary padding.
//!
//! This is a SEPARATE step from find_filter -- it runs on the full peak set
//! AFTER the autocorrelation fit (step 8), not during peak detection.
//! The production pipeline order is:
//!   Step 7: rough autocorrelation (RANSAC on binned peaks)
//!   Step 8: precise autocorrelation (refined two-pass RANSAC on centroids)
//!   Steps 9-10: peaks_above_noise (this step -- noise floor + SNR filter)
//!
//! It applies a column-wise noise floor matching the production
//! `peaks_above_noise_fused.py` implementation:
//!
//!   noise_floor = median_filter(data, size=(window, 1), mode='reflect')
//!   keep = amplitude > min_snr * noise_floor[row, col]
//!
//! The `size=(window, 1)` parameter means the median filter operates along
//! axis 0 (swim/rows) independently for each column (tof), which is exactly
//! what `rolling_median_precise` computes.
//!
//! After filtering, this step replaces the peak arrays in SherlockState with
//! the filtered subset, so downstream steps (precursors, fragments, export)
//! operate only on peaks that survived the noise threshold.

use log::info;
use serde::Serialize;

use crate::dataset::DatasetState;
use super::find_filter::rolling_median_precise;
use super::progress::emit_progress;
use super::subsample::{histogram, subsample_indices};
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct PeaksAboveNoiseResult {
    /// Number of peaks surviving the noise filter.
    pub peak_count: usize,
    /// Number of peaks before filtering (input count).
    pub input_peak_count: usize,
    /// Subsampled TOF coordinates for display (up to 5000).
    pub display_tof: Vec<f64>,
    /// Subsampled SWIM coordinates for display (up to 5000).
    pub display_swim: Vec<f64>,
    /// SNR histogram bin counts (50 bins, log10 scale).
    pub snr_hist_counts: Vec<u64>,
    /// SNR histogram bin edges (51 values).
    pub snr_hist_edges: Vec<f64>,
    /// The min_snr threshold that was applied.
    pub min_snr_used: f64,
    /// The noise_window that was used.
    pub noise_window_used: usize,
}

/// Filter peaks by signal-to-noise ratio using a per-column noise floor.
///
/// This implements the production peaks_above_noise step:
/// 1. Compute noise floor via `median_filter(data, size=(window, 1), mode='reflect')`
///    -- a 1D rolling median along the swim (row) axis for each TOF column.
/// 2. For each peak at (row, col), check if amplitude > min_snr * noise_floor[row, col].
/// 3. Return the filtered peak set.
///
/// # Arguments
/// * `ds` - The loaded dataset containing the full 2D amplitude array.
/// * `sh` - Sherlock state containing peaks from the find_filter step.
///          On success, the filtered peaks replace the existing peak state.
/// * `min_snr` - Minimum signal-to-noise ratio threshold.
/// * `noise_window` - Window size for the 1D median filter along the swim axis.
/// * `app` - Optional Tauri app handle for progress events.
pub fn peaks_above_noise(
    ds: &DatasetState,
    sh: &mut SherlockState,
    min_snr: f64,
    noise_window: usize,
) -> Result<PeaksAboveNoiseResult, String> {
    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    // Use the FULL peak set (stashed before top_n) if available.
    // Production peaks_above_noise operates on the full ~14M peaks, not
    // the 1,200 top_n selection. The stash is set by top_n_peaks.
    let peak_rows = sh.full_peak_row_idx.as_ref()
        .or(sh.peak_row_idx.as_ref())
        .ok_or("No peaks available -- run find_filter or get_peaks_raw first")?;
    let peak_cols = sh.full_peak_col_idx.as_ref()
        .or(sh.peak_col_idx.as_ref())
        .ok_or("No peak columns available")?;
    let peak_amps = sh.full_peak_amplitudes.as_ref()
        .or(sh.peak_amplitudes.as_ref())
        .ok_or("No peak amplitudes available")?;

    let input_peak_count = peak_rows.len();
    info!("[peaks_above_noise] using {} input peaks (full={})",
        input_peak_count, sh.full_peak_row_idx.is_some());
    let min_snr_f32 = min_snr as f32;

    // Step 1: Compute noise floor using column-wise rolling median with reflect mode.
    // This matches scipy's median_filter(data, size=(window, 1), mode='reflect').
    emit_progress(0.0, "Computing per-column noise floor...");

    let noise_flat = rolling_median_precise(data, noise_window);

    emit_progress(0.6, "Filtering peaks by SNR...");

    // Step 2: Filter peaks where amplitude > min_snr * noise_floor[row, col]
    // rolling_median_precise returns flat row-major Vec<f32>
    let n_cols = data.ncols();

    let mut filtered_rows = Vec::new();
    let mut filtered_cols = Vec::new();
    let mut filtered_amps = Vec::new();
    let mut filtered_snr = Vec::new();

    for i in 0..input_peak_count {
        let row = peak_rows[i];
        let col = peak_cols[i];
        let amp = peak_amps[i];

        let noise_val = noise_flat[row * n_cols + col] as f64;
        let snr = if noise_val > 0.0 { amp / noise_val } else { 0.0 };

        if amp > min_snr_f32 as f64 * noise_val {
            filtered_rows.push(row);
            filtered_cols.push(col);
            filtered_amps.push(amp);
            filtered_snr.push(snr);
        }
    }

    let peak_count = filtered_rows.len();

    // Step 3: Prepare display data
    emit_progress(0.9, "Preparing display data...");

    let mut rng = rand::thread_rng();
    let indices = subsample_indices(peak_count, 5000, &mut rng);

    let display_tof: Vec<f64> = indices.iter().map(|&i| tof_coords[filtered_cols[i]]).collect();
    let display_swim: Vec<f64> = indices.iter().map(|&i| swim_coords[filtered_rows[i]]).collect();

    // SNR histogram on log10(snr + 1)
    let log_snr: Vec<f64> = filtered_snr
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| (s + 1.0).log10())
        .collect();
    let (snr_hist_counts, snr_hist_edges) = histogram(&log_snr, 50);

    // Step 4: Update SherlockState with filtered peaks.
    // This replaces the previous peak set with the noise-filtered one.
    // Rebuild all index-dependent state for the new (smaller) peak set.

    sh.peak_row_idx = Some(filtered_rows);
    sh.peak_col_idx = Some(filtered_cols);
    sh.peak_amplitudes = Some(filtered_amps);
    sh.snr_values = Some(filtered_snr);

    // Replace masks with all-true (all filtered peaks are valid)
    sh.stripe_mask = Some(vec![true; peak_count]);
    sh.top_n_mask = Some(vec![true; peak_count]);
    // DO NOT touch centroids — they belong to the pre-noise-filter peak set
    // and are used by compare/export for coordinate matching. The review step
    // handles missing/mismatched centroids gracefully.
    // Free the full peak stash — no longer needed
    sh.full_peak_row_idx = None;
    sh.full_peak_col_idx = None;
    sh.full_peak_amplitudes = None;
    // Clear per-peak downstream state that depends on old indices
    sh.swim_groups = None;
    sh.charges = None;
    sh.isotope_groups = None;
    // Clear per-peak inlier masks (index-dependent), but keep
    // ransac_slope/ransac_intercept/precise_slope/precise_intercept
    // since those are global fit parameters used by precursors.
    sh.ransac_inlier_mask = None;
    sh.precise_inlier_mask = None;

    emit_progress(1.0, "Done");

    Ok(PeaksAboveNoiseResult {
        peak_count,
        input_peak_count,
        display_tof,
        display_swim,
        snr_hist_counts,
        snr_hist_edges,
        min_snr_used: min_snr,
        noise_window_used: noise_window,
    })
}
