use serde::Serialize;
use std::collections::HashMap;

use crate::dataset::DatasetState;
use super::graph_clique::detect_isotopes_graph_clique;
use super::subsample::subsample_indices;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct IsotopeResult {
    pub peak_count: usize,
    pub groups_found: usize,
    pub charge_distribution: HashMap<i32, usize>,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub display_charge: Vec<i32>,
}

pub fn isotope_detect(
    ds: &DatasetState,
    sh: &mut SherlockState,
    tolerance: f64,
    max_charge: usize,
) -> Result<IsotopeResult, String> {
    sh.invalidate_from(5);

    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks")?;
    let mask = sh.top_n_mask.as_ref()
        .or(sh.stripe_mask.as_ref())
        .ok_or("No peak mask — run top_n_peaks or remove_stripes first")?;
    let centroids_tof = sh.centroids_tof.as_ref().ok_or("No centroids — run centroid first")?;
    let centroids_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;

    // Get surviving peak indices (same order as centroids)
    let surviving: Vec<usize> = (0..peak_rows.len())
        .filter(|&i| mask[i])
        .collect();

    let n = surviving.len();
    if n == 0 {
        sh.charges = Some(vec![]);
        sh.isotope_groups = Some(vec![]);
        return Ok(IsotopeResult {
            peak_count: 0,
            groups_found: 0,
            charge_distribution: HashMap::new(),
            display_tof: vec![],
            display_swim: vec![],
            display_charge: vec![],
        });
    }

    // Group peaks by SWIM row (production processes per-trace)
    let mut row_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &orig_idx) in surviving.iter().enumerate() {
        let row = peak_rows[orig_idx];
        row_groups.entry(row).or_default().push(i);
    }

    // Run graph-clique isotope detection per SWIM row
    let mut charges = vec![0i32; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for (_row, group) in &row_groups {
        if group.len() < 2 {
            continue;
        }

        // Extract centroids for this group
        let group_centroids: Vec<f64> = group.iter().map(|&i| centroids_tof[i]).collect();

        let (group_charges, _n_groups) =
            detect_isotopes_graph_clique(&group_centroids, tolerance, max_charge);

        // Map charges back and collect groups
        let mut local_groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (local_idx, &charge) in group_charges.iter().enumerate() {
            let global_idx = group[local_idx];
            charges[global_idx] = charge;
            if charge != 0 {
                local_groups.entry(charge).or_default().push(global_idx);
            }
        }

        for (_charge, members) in local_groups {
            if members.len() >= 2 {
                groups.push(members);
            }
        }
    }

    // Build charge distribution
    let mut charge_distribution: HashMap<i32, usize> = HashMap::new();
    for &c in &charges {
        *charge_distribution.entry(c).or_insert(0) += 1;
    }

    // Display data
    let mut rng = rand::thread_rng();
    let display_idx = subsample_indices(n, 5000, &mut rng);

    let display_tof: Vec<f64> = display_idx.iter().map(|&i| centroids_tof[i]).collect();
    let display_swim: Vec<f64> = display_idx.iter().map(|&i| centroids_swim[i]).collect();
    let display_charge: Vec<i32> = display_idx.iter().map(|&i| charges[i]).collect();

    let _ = swim_coords; // used via peak_rows for grouping

    sh.charges = Some(charges);
    sh.isotope_groups = Some(groups.clone());

    Ok(IsotopeResult {
        peak_count: n,
        groups_found: groups.len(),
        charge_distribution,
        display_tof,
        display_swim,
        display_charge,
    })
}
