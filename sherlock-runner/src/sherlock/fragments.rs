//! Fragment aggregation (production step 11: get_fragments).
//!
//! For each swim_idx in the tallest peaks (top_n), this step:
//! 1. Collects precursors at that swim_idx
//! 2. Collects tallest peaks at that swim_idx
//! 3. Merges them (union, no duplicates)
//! 4. Adds centroids for peaks that don't already have them
//! 5. Runs isotope detection on the merged fragment set
//! 6. Builds a per-swim-idx fragment dict
//!
//! The final output contains per-swim-idx fragment peaks with their
//! calibrated TOF and SWIM coordinates and charge assignments.

use log::info;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dataset::DatasetState;
use super::precursors::{centroid_single, detect_isotopes};
use super::subsample::subsample_indices;
use super::SherlockState;

/// Result returned from the fragments step.
#[derive(Debug, Serialize)]
pub struct FragmentsResult {
    /// Total number of fragment peaks across all swim indices.
    pub total_fragments: usize,
    /// Number of distinct swim indices in the fragment set.
    pub n_swim_indices: usize,
    /// Charge distribution across all fragment peaks.
    pub charge_distribution: HashMap<i32, usize>,
    /// Number of isotope groups found in the fragment set.
    pub isotope_groups_found: usize,
    /// Min/max fragments per swim index.
    pub min_fragments_per_swim: usize,
    pub max_fragments_per_swim: usize,
    /// Subsampled display data: calibrated TOF coordinates.
    pub display_tof: Vec<f64>,
    /// Subsampled display data: calibrated SWIM coordinates.
    pub display_swim: Vec<f64>,
    /// Subsampled display data: charge state per point.
    pub display_charge: Vec<i32>,
}

/// Calibrate swim coordinate using the RANSAC model:
///   swim_calibrated = (swim_raw - intercept) / slope
fn calibrate_swim(swim_raw: f64, slope: f64, intercept: f64) -> f64 {
    if slope.abs() < f64::EPSILON {
        swim_raw
    } else {
        (swim_raw - intercept) / slope
    }
}

/// Collect the set of swim row indices that have tallest peaks (top_n_mask).
///
/// Returns a sorted list of unique swim_idx values from peaks that pass
/// both the stripe_mask and the top_n_mask.
fn collect_tallest_swim_indices(sh: &SherlockState) -> Result<Vec<usize>, String> {
    let peak_rows = sh
        .peak_row_idx
        .as_ref()
        .ok_or("No peaks -- run find_filter first")?;
    let top_n_mask = sh
        .top_n_mask
        .as_ref()
        .ok_or("No top_n_mask -- run top_n_peaks first")?;

    let mut swim_set: HashSet<usize> = HashSet::new();
    for i in 0..peak_rows.len() {
        if top_n_mask[i] {
            swim_set.insert(peak_rows[i]);
        }
    }

    let mut swim_indices: Vec<usize> = swim_set.into_iter().collect();
    swim_indices.sort_unstable();
    Ok(swim_indices)
}

/// For a given swim_idx, collect all original peak indices (into peak_row_idx etc.)
/// that are tallest peaks at that swim_idx.
fn tallest_peaks_at_swim(sh: &SherlockState, swim_idx: usize) -> Vec<usize> {
    let peak_rows = sh.peak_row_idx.as_ref().unwrap();
    let top_n_mask = sh.top_n_mask.as_ref().unwrap();

    let mut indices = Vec::new();
    for i in 0..peak_rows.len() {
        if top_n_mask[i] && peak_rows[i] == swim_idx {
            indices.push(i);
        }
    }
    indices
}

/// For a given swim_idx, collect all precursor peak indices (into peak_row_idx etc.)
/// at that swim_idx.
///
/// Precursor_peak_indices maps precursor positions back to the original peak arrays.
fn precursor_peaks_at_swim(sh: &SherlockState, swim_idx: usize) -> Vec<usize> {
    let peak_rows = sh.peak_row_idx.as_ref().unwrap();
    let precursor_peak_indices = match sh.precursor_peak_indices.as_ref() {
        Some(v) => v,
        None => return vec![],
    };

    let mut indices = Vec::new();
    for &orig_idx in precursor_peak_indices {
        if orig_idx < peak_rows.len() && peak_rows[orig_idx] == swim_idx {
            indices.push(orig_idx);
        }
    }
    indices
}

/// Run the full fragments pipeline (production step 11).
///
/// For each swim_idx present in the tallest peaks:
/// 1. Collect precursors at that swim_idx
/// 2. Collect tallest peaks at that swim_idx
/// 3. Merge (union, deduplicated by original peak index)
/// 4. Compute centroids for peaks that need them
/// 5. Calibrate SWIM coordinates using the RANSAC model
/// 6. Run isotope detection on the merged set
/// 7. Store per-peak fragment data
///
/// # Arguments
/// * `ds` - Dataset with loaded 2D amplitude data.
/// * `sh` - SherlockState with results from prior steps.
/// * `isotope_tolerance` - Tolerance for isotope spacing matching (default: 0.3).
/// * `max_charge` - Maximum charge state to detect (default: 10).
/// * `half_window_tof` - Half-window for centroid computation in TOF (default: 3).
/// * `half_window_swim` - Half-window for centroid computation in SWIM (default: 3).
pub fn get_fragments(
    ds: &DatasetState,
    sh: &mut SherlockState,
    isotope_tolerance: f64,
    max_charge: usize,
    half_window_tof: usize,
    half_window_swim: usize,
) -> Result<FragmentsResult, String> {
    // Invalidate step 10 (fragments) and downstream
    sh.invalidate_from(10);

    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    // Verify prerequisites
    let _peak_rows = sh
        .peak_row_idx
        .as_ref()
        .ok_or("No peaks -- run find_filter first")?;
    let peak_cols = sh
        .peak_col_idx
        .as_ref()
        .ok_or("No peak columns")?
        .clone();
    let peak_rows = sh
        .peak_row_idx
        .as_ref()
        .ok_or("No peaks")?
        .clone();
    let _top_n_mask = sh
        .top_n_mask
        .as_ref()
        .ok_or("No top_n_mask -- run top_n_peaks first")?;

    // Determine calibration: precise > rough
    let (cal_slope, cal_intercept) =
        if let (Some(s), Some(i)) = (sh.precise_slope, sh.precise_intercept) {
            (s, i)
        } else if let (Some(s), Some(i)) = (sh.ransac_slope, sh.ransac_intercept) {
            (s, i)
        } else {
            return Err(
                "No RANSAC fit available -- run RANSAC or precise_autocorrelation first"
                    .to_string(),
            );
        };

    // Step 1: Collect swim indices that have tallest peaks
    let swim_indices = collect_tallest_swim_indices(sh)?;
    let n_swim_indices = swim_indices.len();

    info!(
        "[fragments] {} swim indices with tallest peaks",
        n_swim_indices
    );

    if n_swim_indices == 0 {
        sh.fragment_tof = Some(vec![]);
        sh.fragment_swim = Some(vec![]);
        sh.fragment_charges = Some(vec![]);
        sh.fragment_swim_idx = Some(vec![]);
        sh.fragment_n_swim_indices = Some(0);

        return Ok(FragmentsResult {
            total_fragments: 0,
            n_swim_indices: 0,
            charge_distribution: HashMap::new(),
            isotope_groups_found: 0,
            min_fragments_per_swim: 0,
            max_fragments_per_swim: 0,
            display_tof: vec![],
            display_swim: vec![],
            display_charge: vec![],
        });
    }

    // Step 2: For each swim_idx, collect and merge peaks
    // Build per-swim fragment groups: swim_idx -> Vec<original_peak_index>
    let mut per_swim_peaks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for &swim_idx in &swim_indices {
        let tallest = tallest_peaks_at_swim(sh, swim_idx);
        let precursors = precursor_peaks_at_swim(sh, swim_idx);

        // Merge (union, deduplicated)
        let mut merged: HashSet<usize> = HashSet::new();
        for idx in tallest {
            merged.insert(idx);
        }
        for idx in precursors {
            merged.insert(idx);
        }

        if !merged.is_empty() {
            let mut sorted: Vec<usize> = merged.into_iter().collect();
            sorted.sort_unstable();
            per_swim_peaks.insert(swim_idx, sorted);
        }
    }

    // Step 3: Compute centroids for all unique peaks across all groups.
    // Collect all unique original peak indices that need centroiding.
    let mut all_peak_indices: HashSet<usize> = HashSet::new();
    for indices in per_swim_peaks.values() {
        for &idx in indices {
            all_peak_indices.insert(idx);
        }
    }

    let unique_indices: Vec<usize> = {
        let mut v: Vec<usize> = all_peak_indices.into_iter().collect();
        v.sort_unstable();
        v
    };

    // Compute centroids in parallel for all unique peaks
    let centroids: Vec<(usize, f64, f64)> = unique_indices
        .par_iter()
        .map(|&orig_idx| {
            let row = peak_rows[orig_idx];
            let col = peak_cols[orig_idx];
            let (ct, cs) = centroid_single(
                data,
                tof_coords,
                swim_coords,
                row,
                col,
                half_window_tof,
                half_window_swim,
            );
            (orig_idx, ct, cs)
        })
        .collect();

    // Build a lookup: original_peak_index -> (centroid_tof, centroid_swim_calibrated)
    let mut centroid_map: HashMap<usize, (f64, f64)> = HashMap::new();
    for &(orig_idx, ct, cs) in &centroids {
        let cs_cal = calibrate_swim(cs, cal_slope, cal_intercept);
        centroid_map.insert(orig_idx, (ct, cs_cal));
    }

    info!(
        "[fragments] computed centroids for {} unique peaks across {} swim groups",
        unique_indices.len(),
        per_swim_peaks.len()
    );

    // Step 4: Per-swim-idx isotope detection and fragment assembly
    let mut all_frag_tof = Vec::new();
    let mut all_frag_swim = Vec::new();
    let mut all_frag_charges = Vec::new();
    let mut all_frag_swim_idx = Vec::new();
    let mut total_isotope_groups = 0usize;
    let mut min_per_swim = usize::MAX;
    let mut max_per_swim = 0usize;

    for (&swim_idx, indices) in &per_swim_peaks {
        let n_in_group = indices.len();
        if n_in_group < min_per_swim {
            min_per_swim = n_in_group;
        }
        if n_in_group > max_per_swim {
            max_per_swim = n_in_group;
        }

        // Gather calibrated TOF values for isotope detection
        let group_tof: Vec<f64> = indices
            .iter()
            .map(|&idx| centroid_map[&idx].0)
            .collect();

        // Run isotope detection on this group's calibrated TOF values
        let (charges, n_groups) = detect_isotopes(&group_tof, isotope_tolerance, max_charge);
        total_isotope_groups += n_groups;

        // Append to global fragment arrays
        for (local_i, &orig_idx) in indices.iter().enumerate() {
            let (ct, cs_cal) = centroid_map[&orig_idx];
            all_frag_tof.push(ct);
            all_frag_swim.push(cs_cal);
            all_frag_charges.push(charges[local_i]);
            all_frag_swim_idx.push(swim_idx);
        }
    }

    if min_per_swim == usize::MAX {
        min_per_swim = 0;
    }

    let total_fragments = all_frag_tof.len();

    info!(
        "[fragments] {} total fragments across {} swim indices ({}-{} per swim), {} isotope groups",
        total_fragments,
        per_swim_peaks.len(),
        min_per_swim,
        max_per_swim,
        total_isotope_groups
    );

    // Build charge distribution
    let mut charge_distribution: HashMap<i32, usize> = HashMap::new();
    for &c in &all_frag_charges {
        *charge_distribution.entry(c).or_insert(0) += 1;
    }

    // Display data (subsampled)
    let mut rng = rand::thread_rng();
    let display_sample = subsample_indices(total_fragments, 5000, &mut rng);

    let display_tof: Vec<f64> = display_sample
        .iter()
        .map(|&i| all_frag_tof[i])
        .collect();
    let display_swim: Vec<f64> = display_sample
        .iter()
        .map(|&i| all_frag_swim[i])
        .collect();
    let display_charge: Vec<i32> = display_sample
        .iter()
        .map(|&i| all_frag_charges[i])
        .collect();

    // Store results in SherlockState
    sh.fragment_tof = Some(all_frag_tof);
    sh.fragment_swim = Some(all_frag_swim);
    sh.fragment_charges = Some(all_frag_charges);
    sh.fragment_swim_idx = Some(all_frag_swim_idx);
    sh.fragment_n_swim_indices = Some(per_swim_peaks.len());

    Ok(FragmentsResult {
        total_fragments,
        n_swim_indices: per_swim_peaks.len(),
        charge_distribution,
        isotope_groups_found: total_isotope_groups,
        min_fragments_per_swim: min_per_swim,
        max_fragments_per_swim: max_per_swim,
        display_tof,
        display_swim,
        display_charge,
    })
}
