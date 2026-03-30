use rayon::prelude::*;
use serde::Serialize;

use crate::dataset::DatasetState;
use super::subsample::histogram;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct CentroidResult {
    pub peak_count: usize,
    pub mean_delta_tof: f64,
    pub mean_delta_swim: f64,
    pub tof_hist_counts: Vec<u64>,
    pub tof_hist_edges: Vec<f64>,
    pub swim_hist_counts: Vec<u64>,
    pub swim_hist_edges: Vec<f64>,
}

pub fn centroid(
    ds: &DatasetState,
    sh: &mut SherlockState,
    half_window_tof: usize,
    half_window_swim: usize,
) -> Result<CentroidResult, String> {
    // Only clear centroid results and downstream (swim_groups, isotopes, etc.)
    // Do NOT clear top_n_mask — centroid runs AFTER top_n in the production pipeline.
    sh.centroids_tof = None;
    sh.centroids_swim = None;
    sh.swim_groups = None;
    sh.charges = None;
    sh.isotope_groups = None;

    let data = ds.data.as_ref().ok_or("No data loaded")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;
    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peaks")?;
    // Use top_n_mask (production 13-step) or stripe_mask (old 6-step), whichever is set
    let mask = sh.top_n_mask.as_ref()
        .or(sh.stripe_mask.as_ref())
        .ok_or("No peak mask — run top_n_peaks or remove_stripes first")?;

    let (n_rows, n_cols) = data.dim();

    // Get surviving peak indices
    let surviving: Vec<usize> = (0..peak_rows.len())
        .filter(|&i| mask[i])
        .collect();

    let n_peaks = surviving.len();

    // Compute centroids in parallel
    let centroids: Vec<(f64, f64)> = surviving
        .par_iter()
        .map(|&idx| {
            let row = peak_rows[idx];
            let col = peak_cols[idx];

            let r_lo = row.saturating_sub(half_window_swim);
            let r_hi = (row + half_window_swim).min(n_rows - 1);
            let c_lo = col.saturating_sub(half_window_tof);
            let c_hi = (col + half_window_tof).min(n_cols - 1);

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
                // Fallback to grid position
                (tof_coords[col], swim_coords[row])
            }
        })
        .collect();

    // Compute deltas
    let mut delta_tof = Vec::with_capacity(n_peaks);
    let mut delta_swim = Vec::with_capacity(n_peaks);
    let mut c_tof = Vec::with_capacity(n_peaks);
    let mut c_swim = Vec::with_capacity(n_peaks);

    for (i, &idx) in surviving.iter().enumerate() {
        let (ct, cs) = centroids[i];
        c_tof.push(ct);
        c_swim.push(cs);

        let grid_tof = tof_coords[peak_cols[idx]];
        let grid_swim = swim_coords[peak_rows[idx]];
        delta_tof.push(ct - grid_tof);
        delta_swim.push(cs - grid_swim);
    }

    let mean_delta_tof = if n_peaks > 0 {
        delta_tof.iter().map(|d| d.abs()).sum::<f64>() / n_peaks as f64
    } else {
        0.0
    };
    let mean_delta_swim = if n_peaks > 0 {
        delta_swim.iter().map(|d| d.abs()).sum::<f64>() / n_peaks as f64
    } else {
        0.0
    };

    let (tof_hist_counts, tof_hist_edges) = histogram(&delta_tof, 50);
    let (swim_hist_counts, swim_hist_edges) = histogram(&delta_swim, 50);

    sh.centroids_tof = Some(c_tof);
    sh.centroids_swim = Some(c_swim);

    Ok(CentroidResult {
        peak_count: n_peaks,
        mean_delta_tof,
        mean_delta_swim,
        tof_hist_counts,
        tof_hist_edges,
        swim_hist_counts,
        swim_hist_edges,
    })
}
