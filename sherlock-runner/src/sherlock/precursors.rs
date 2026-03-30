//! Precursor detection (production step 10: get_precursors).
//!
//! Implements the full precursor pipeline:
//! (a) Amplitude threshold: mean + 5*std of peaks_above_noise amplitudes
//! (b) apply_model_to_peaks: calibrate swim coords using precise RANSAC fit
//!     (or rough RANSAC as fallback): swim_calibrated = (swim_raw - intercept) / slope
//! (c) add_centroids: intensity-weighted center-of-mass
//! (d) label_precursors: filter peaks by distance from y=x line (gulley_width_amu=3)
//! (e) detect_all_isotopes: isotope detection on the precursor set
//! (f) filter is_charged_on_autocorrelation: keep only charged peaks near the line

use log::info;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

use crate::dataset::DatasetState;
use super::graph_clique::detect_isotopes_graph_clique;
use super::subsample::subsample_indices;
use super::SherlockState;

/// Result returned from the precursors step.
#[derive(Debug, Serialize)]
pub struct PrecursorsResult {
    /// Number of peaks after amplitude threshold.
    pub peaks_after_threshold: usize,
    /// Number of peaks after calibration.
    pub peaks_calibrated: usize,
    /// Number of precursor groups found by labeling.
    pub precursor_groups: usize,
    /// Total precursor peaks (across all groups).
    pub precursor_peak_count: usize,
    /// Number of isotope groups detected among precursors.
    pub isotope_groups_found: usize,
    /// Number of charged precursors on the autocorrelation line.
    pub charged_on_line: usize,
    /// Charge distribution of final precursors.
    pub charge_distribution: HashMap<i32, usize>,
    /// Calibration slope used (precise or rough).
    pub calibration_slope: f64,
    /// Calibration intercept used (precise or rough).
    pub calibration_intercept: f64,
    /// Whether the precise fit was used (vs rough fallback).
    pub used_precise_fit: bool,
    /// Subsampled display data: calibrated TOF coordinates.
    pub display_tof: Vec<f64>,
    /// Subsampled display data: calibrated SWIM coordinates.
    pub display_swim: Vec<f64>,
    /// Subsampled display data: charge state per point.
    pub display_charge: Vec<i32>,
    /// Subsampled display data: precursor label per point.
    pub display_label: Vec<i32>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute amplitude threshold: mean + multiplier * std of the given amplitudes.
fn amplitude_threshold(amps: &[f64], multiplier: f64) -> f64 {
    let n = amps.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = amps.iter().sum::<f64>() / n;
    let variance = amps.iter().map(|&a| (a - mean) * (a - mean)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    mean + multiplier * std_dev
}

/// Calibrate swim coordinate using the RANSAC model:
///   swim_calibrated = (swim_raw - intercept) / slope
fn calibrate_swim(swim_raw: f64, slope: f64, intercept: f64) -> f64 {
    if slope.abs() < f64::EPSILON {
        swim_raw // avoid division by zero
    } else {
        (swim_raw - intercept) / slope
    }
}

/// Intensity-weighted centroid for a single peak.
/// Returns (centroid_tof, centroid_swim) in physical coordinate space.
pub fn centroid_single(
    data: &ndarray::Array2<f32>,
    tof_coords: &[f64],
    swim_coords: &[f64],
    row: usize,
    col: usize,
    hw_tof: usize,
    hw_swim: usize,
) -> (f64, f64) {
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

/// Label precursors by distance from the autocorrelation line (y=x in
/// calibrated space). A peak is labeled PRECURSOR (label 0) when:
///   |swim_calibrated - centroid_tof| < gulley_width_amu
///
/// This matches the production Python algorithm which checks each peak
/// independently against the y=x diagonal, NOT by grouping consecutive
/// peaks in TOF space.
///
/// Returns a label for each peak (0 = precursor, -1 = not a precursor).
fn label_precursors_by_proximity(
    calibrated_tof: &[f64],
    calibrated_swim: &[f64],
    gulley_width_amu: f64,
) -> Vec<i32> {
    let n = calibrated_tof.len();
    if n == 0 {
        return vec![];
    }

    calibrated_tof
        .iter()
        .zip(calibrated_swim.iter())
        .map(|(&tof, &swim)| {
            if (swim - tof).abs() < gulley_width_amu {
                0 // PRECURSOR
            } else {
                -1 // not a precursor
            }
        })
        .collect()
}

/// Isotope detection on a set of peaks using the graph-clique algorithm.
/// Matches the production Python/NetworkX approach: builds an adjacency
/// graph, finds maximal cliques via Bron-Kerbosch, and assigns charges
/// from median pairwise spacing within each clique.
///
/// Returns (charges, n_groups) where charges[i] is the assigned charge for
/// each peak (0 = uncharged).
pub fn detect_isotopes(
    calibrated_tof: &[f64],
    tolerance: f64,
    max_charge: usize,
) -> (Vec<i32>, usize) {
    detect_isotopes_graph_clique(calibrated_tof, tolerance, max_charge)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the full precursors pipeline (production step 10).
///
/// This step takes the full-resolution peaks (after peaks_above_noise or
/// find_filter + stripes), calibrates them using the RANSAC fit, computes
/// centroids, groups by proximity, detects isotopes, and filters to only
/// charged peaks on the autocorrelation line.
///
/// # Arguments
/// * `ds` - Dataset with loaded 2D amplitude data.
/// * `sh` - SherlockState with peaks and RANSAC results from prior steps.
/// * `amplitude_std_multiplier` - Multiplier for amplitude threshold (default: 5.0).
/// * `gulley_width_amu` - Distance from autocorrelation line (y=x) to label as precursor (default: 3.0).
/// * `isotope_tolerance` - Tolerance for isotope spacing matching (default: 0.3).
/// * `max_charge` - Maximum charge state to detect (default: 10).
/// * `half_window_tof` - Half-window for centroid computation in TOF (default: 3).
/// * `half_window_swim` - Half-window for centroid computation in SWIM (default: 3).
/// * `autocorrelation_tolerance` - Tolerance for "on the autocorrelation line" filter (default: 3.0).
pub fn get_precursors(
    ds: &DatasetState,
    sh: &mut SherlockState,
    amplitude_std_multiplier: f64,
    gulley_width_amu: f64,
    isotope_tolerance: f64,
    max_charge: usize,
    half_window_tof: usize,
    half_window_swim: usize,
    autocorrelation_tolerance: f64,
) -> Result<PrecursorsResult, String> {
    // Invalidate step 9 (precursors) and downstream
    sh.invalidate_from(9);

    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    let peak_rows = sh
        .peak_row_idx
        .as_ref()
        .ok_or("No peaks -- run find_filter first")?;
    let peak_cols = sh
        .peak_col_idx
        .as_ref()
        .ok_or("No peak columns")?;
    let peak_amps = sh
        .peak_amplitudes
        .as_ref()
        .ok_or("No peak amplitudes")?;

    // Determine which calibration to use: precise > rough
    let (cal_slope, cal_intercept, used_precise) =
        if let (Some(s), Some(i)) = (sh.precise_slope, sh.precise_intercept) {
            (s, i, true)
        } else if let (Some(s), Some(i)) = (sh.ransac_slope, sh.ransac_intercept) {
            (s, i, false)
        } else {
            return Err("No RANSAC fit available -- run RANSAC or precise_autocorrelation first".to_string());
        };

    info!(
        "[precursors] using {} fit: swim = {:.6} * tof + {:.6}",
        if used_precise { "precise" } else { "rough" },
        cal_slope,
        cal_intercept
    );

    let total_input = peak_rows.len();

    // -----------------------------------------------------------------------
    // (a) Amplitude threshold: mean + multiplier * std
    // -----------------------------------------------------------------------
    let threshold = amplitude_threshold(peak_amps, amplitude_std_multiplier);

    let mut surviving_indices: Vec<usize> = Vec::new();
    for i in 0..total_input {
        if peak_amps[i] > threshold {
            surviving_indices.push(i);
        }
    }

    let peaks_after_threshold = surviving_indices.len();
    info!(
        "[precursors] amplitude threshold ({:.2} = mean + {}*std): {} -> {} peaks",
        threshold, amplitude_std_multiplier, total_input, peaks_after_threshold
    );

    if peaks_after_threshold == 0 {
        sh.precursor_tof_calibrated = Some(vec![]);
        sh.precursor_swim_calibrated = Some(vec![]);
        sh.precursor_charges = Some(vec![]);
        sh.precursor_labels = Some(vec![]);
        sh.precursor_peak_indices = Some(vec![]);

        return Ok(PrecursorsResult {
            peaks_after_threshold: 0,
            peaks_calibrated: 0,
            precursor_groups: 0,
            precursor_peak_count: 0,
            isotope_groups_found: 0,
            charged_on_line: 0,
            charge_distribution: HashMap::new(),
            calibration_slope: cal_slope,
            calibration_intercept: cal_intercept,
            used_precise_fit: used_precise,
            display_tof: vec![],
            display_swim: vec![],
            display_charge: vec![],
            display_label: vec![],
        });
    }

    // -----------------------------------------------------------------------
    // (b) apply_model_to_peaks: calibrate swim using RANSAC model
    //     swim_calibrated = (swim_raw - intercept) / slope
    //     TOF is already in mass units and doesn't need calibration.
    // -----------------------------------------------------------------------
    let rows: Vec<usize> = surviving_indices.iter().map(|&i| peak_rows[i]).collect();
    let cols: Vec<usize> = surviving_indices.iter().map(|&i| peak_cols[i]).collect();

    let peaks_calibrated = surviving_indices.len();

    info!(
        "[precursors] calibrated {} peaks using model",
        peaks_calibrated
    );

    // -----------------------------------------------------------------------
    // (c) add_centroids: intensity-weighted center-of-mass
    // -----------------------------------------------------------------------
    let centroid_data: Vec<(f64, f64)> = rows
        .par_iter()
        .zip(cols.par_iter())
        .map(|(&row, &col)| {
            centroid_single(
                data,
                tof_coords,
                swim_coords,
                row,
                col,
                half_window_tof,
                half_window_swim,
            )
        })
        .collect();

    // Apply calibration to the centroid swim values
    let centroid_tof: Vec<f64> = centroid_data.iter().map(|&(t, _)| t).collect();
    let centroid_swim_calibrated: Vec<f64> = centroid_data
        .iter()
        .map(|&(_, s)| calibrate_swim(s, cal_slope, cal_intercept))
        .collect();

    info!(
        "[precursors] computed centroids for {} peaks",
        centroid_tof.len()
    );

    // -----------------------------------------------------------------------
    // (d) label_precursors: filter by distance from autocorrelation line
    //     (y=x in calibrated space). Peaks where |swim_cal - tof| <
    //     gulley_width_amu are labeled as PRECURSOR.
    // -----------------------------------------------------------------------
    let labels = label_precursors_by_proximity(&centroid_tof, &centroid_swim_calibrated, gulley_width_amu);

    let precursor_peak_count = labels.iter().filter(|&&l| l >= 0).count();
    let n_groups = if precursor_peak_count > 0 { 1 } else { 0 };

    info!(
        "[precursors] labeled {} precursor peaks ({} non-precursor) by y=x distance < {}",
        precursor_peak_count,
        labels.iter().filter(|&&l| l < 0).count(),
        gulley_width_amu
    );

    // -----------------------------------------------------------------------
    // (e) detect_all_isotopes: isotope detection on the precursor set
    // -----------------------------------------------------------------------
    let (charges, isotope_groups_found) =
        detect_isotopes(&centroid_tof, isotope_tolerance, max_charge);

    info!(
        "[precursors] isotope detection: {} groups found",
        isotope_groups_found
    );

    // -----------------------------------------------------------------------
    // (f) filter is_charged_on_autocorrelation: keep only peaks that are
    //     (1) charged (charge > 0) AND
    //     (2) near the autocorrelation line (|swim_cal - tof| < tolerance)
    //
    //     On the autocorrelation line, calibrated_swim ~= tof (by definition
    //     of the calibration). So we check |swim_calibrated - tof| < tolerance.
    // -----------------------------------------------------------------------
    let mut final_indices: Vec<usize> = Vec::new();
    for i in 0..centroid_tof.len() {
        let is_charged = charges[i] > 0;
        let on_line = (centroid_swim_calibrated[i] - centroid_tof[i]).abs()
            < autocorrelation_tolerance;
        if is_charged && on_line {
            final_indices.push(i);
        }
    }

    let charged_on_line = final_indices.len();

    info!(
        "[precursors] charge + autocorrelation filter: {} -> {} peaks",
        centroid_tof.len(),
        charged_on_line
    );

    // Build charge distribution for final set
    let mut charge_distribution: HashMap<i32, usize> = HashMap::new();
    for &i in &final_indices {
        *charge_distribution.entry(charges[i]).or_insert(0) += 1;
    }

    // -----------------------------------------------------------------------
    // Display data (subsampled)
    // -----------------------------------------------------------------------
    let mut rng = rand::thread_rng();
    let display_count = final_indices.len();
    let display_sample = subsample_indices(display_count, 5000, &mut rng);

    let display_tof: Vec<f64> = display_sample
        .iter()
        .map(|&i| centroid_tof[final_indices[i]])
        .collect();
    let display_swim: Vec<f64> = display_sample
        .iter()
        .map(|&i| centroid_swim_calibrated[final_indices[i]])
        .collect();
    let display_charge: Vec<i32> = display_sample
        .iter()
        .map(|&i| charges[final_indices[i]])
        .collect();
    let display_label: Vec<i32> = display_sample
        .iter()
        .map(|&i| labels[final_indices[i]])
        .collect();

    // -----------------------------------------------------------------------
    // Store results in SherlockState
    // -----------------------------------------------------------------------
    // Store the full (not just final-filtered) calibrated coordinates and
    // charges, so downstream steps can use them. The final_indices mask
    // identifies which ones passed the charged-on-line filter.
    let final_tof: Vec<f64> = final_indices.iter().map(|&i| centroid_tof[i]).collect();
    let final_swim: Vec<f64> = final_indices
        .iter()
        .map(|&i| centroid_swim_calibrated[i])
        .collect();
    let final_charges: Vec<i32> = final_indices.iter().map(|&i| charges[i]).collect();
    let final_labels: Vec<i32> = final_indices.iter().map(|&i| labels[i]).collect();
    let final_peak_indices: Vec<usize> = final_indices
        .iter()
        .map(|&i| surviving_indices[i])
        .collect();

    sh.precursor_tof_calibrated = Some(final_tof);
    sh.precursor_swim_calibrated = Some(final_swim);
    sh.precursor_charges = Some(final_charges);
    sh.precursor_labels = Some(final_labels);
    sh.precursor_peak_indices = Some(final_peak_indices);

    Ok(PrecursorsResult {
        peaks_after_threshold,
        peaks_calibrated,
        precursor_groups: n_groups,
        precursor_peak_count,
        isotope_groups_found,
        charged_on_line,
        charge_distribution,
        calibration_slope: cal_slope,
        calibration_intercept: cal_intercept,
        used_precise_fit: used_precise,
        display_tof,
        display_swim,
        display_charge,
        display_label,
    })
}
