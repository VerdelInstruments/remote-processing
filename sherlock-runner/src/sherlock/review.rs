use serde::Serialize;
use std::collections::HashMap;

use crate::dataset::DatasetState;
use super::subsample::subsample_indices;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct ReviewResult {
    pub total_peaks: usize,
    pub inlier_count: usize,
    pub charge_distribution: HashMap<i32, usize>,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub display_amp: Vec<f64>,
    pub display_charge: Vec<i32>,
    pub display_inlier: Vec<bool>,
}

pub fn review(ds: &DatasetState, sh: &SherlockState) -> Result<ReviewResult, String> {
    let peak_amps = sh.peak_amplitudes.as_ref().ok_or("No peaks")?;
    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peak rows")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peak cols")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;

    // Determine the peak set to display.
    // Centroids are the authoritative source (same as what compare uses).
    // peaks_above_noise may have replaced peak_amplitudes with a smaller set
    // but centroids remain from the centroid step.
    let has_centroids = sh.centroids_tof.is_some() && sh.centroids_swim.is_some();
    let centroid_count = sh.centroids_tof.as_ref().map_or(0, |c| c.len());

    // If centroids exist and are larger than the current peak set, use centroids
    // as the primary peak count (matching what compare sees).
    let (n, use_centroids) = if has_centroids && centroid_count > peak_amps.len() {
        (centroid_count, true)
    } else {
        // Use mask-filtered peaks
        let surviving: Vec<usize> = if let Some(mask) = sh.top_n_mask.as_ref().or(sh.stripe_mask.as_ref()) {
            (0..mask.len()).filter(|&i| mask[i]).collect()
        } else {
            (0..peak_amps.len()).collect()
        };
        (surviving.len(), false)
    };

    // Charges (default 0 if isotope wasn't run)
    let default_charges = vec![0i32; n];
    let charges = sh.charges.as_ref().unwrap_or(&default_charges);

    // Inlier mask
    let default_inliers = vec![true; n];
    let inlier_mask = sh
        .precise_inlier_mask
        .as_ref()
        .or(sh.ransac_inlier_mask.as_ref())
        .unwrap_or(&default_inliers);

    let inlier_count = inlier_mask.iter().filter(|&&m| m).count();

    // Build charge distribution
    let mut charge_distribution: HashMap<i32, usize> = HashMap::new();
    for &c in charges.iter().take(n) {
        *charge_distribution.entry(c).or_insert(0) += 1;
    }

    // Subsample for display
    let mut rng = rand::thread_rng();
    let display_idx = subsample_indices(n, 5000, &mut rng);

    let display_tof: Vec<f64>;
    let display_swim: Vec<f64>;
    let display_amp: Vec<f64>;

    if use_centroids {
        let ct = sh.centroids_tof.as_ref().unwrap();
        let cs = sh.centroids_swim.as_ref().unwrap();
        display_tof = display_idx.iter().map(|&i| ct[i]).collect();
        display_swim = display_idx.iter().map(|&i| cs[i]).collect();
        // Amplitudes may not align — use them if same length, else default
        display_amp = display_idx.iter().map(|&i| {
            peak_amps.get(i).copied().unwrap_or(0.0)
        }).collect();
    } else {
        let surviving: Vec<usize> = if let Some(mask) = sh.top_n_mask.as_ref().or(sh.stripe_mask.as_ref()) {
            (0..mask.len()).filter(|&i| mask[i]).collect()
        } else {
            (0..peak_amps.len()).collect()
        };
        display_tof = display_idx.iter().map(|&i| tof_coords[peak_cols[surviving[i]]]).collect();
        display_swim = display_idx.iter().map(|&i| swim_coords[peak_rows[surviving[i]]]).collect();
        display_amp = display_idx.iter().map(|&i| peak_amps[surviving[i]]).collect();
    }

    let display_charge: Vec<i32> = display_idx
        .iter()
        .map(|&i| *charges.get(i).unwrap_or(&0))
        .collect();
    let display_inlier: Vec<bool> = display_idx
        .iter()
        .map(|&i| *inlier_mask.get(i).unwrap_or(&true))
        .collect();

    Ok(ReviewResult {
        total_peaks: n,
        inlier_count,
        charge_distribution,
        display_tof,
        display_swim,
        display_amp,
        display_charge,
        display_inlier,
    })
}
