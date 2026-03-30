use log::info;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

use crate::dataset::DatasetState;
use super::graph_clique::detect_isotopes_in_group;
use super::subsample::subsample_indices;
use super::SherlockState;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of swim_idx grouping (production step 5).
#[derive(Debug, Serialize)]
pub struct SwimGroupResult {
    /// Number of distinct swim indices with peaks.
    pub n_groups: usize,
    /// Total peaks across all groups.
    pub total_peaks: usize,
    /// Min/max peaks per group.
    pub min_peaks_per_group: usize,
    pub max_peaks_per_group: usize,
}

/// Result of per-group isotope detection (production step 6).
#[derive(Debug, Serialize)]
pub struct SwimGroupIsotopeResult {
    /// Number of peaks processed.
    pub peak_count: usize,
    /// Number of swim groups processed.
    pub n_groups: usize,
    /// Total isotope groups found across all swim groups.
    pub groups_found: usize,
    /// Charge distribution across all peaks.
    pub charge_distribution: HashMap<i32, usize>,
    /// Display data (subsampled).
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
    pub display_charge: Vec<i32>,
}

// ---------------------------------------------------------------------------
// Step 5: Group peaks by swim_idx
// ---------------------------------------------------------------------------

/// Group peaks by their swim row index into a BTreeMap.
///
/// Matches production step 5 (`swim_idx_to_peaks.py`):
/// builds a mapping from swim_idx to the list of peak indices that share
/// that row.  Only stripe-surviving peaks are grouped (using `stripe_mask`).
///
/// Stores `swim_groups: Option<BTreeMap<usize, Vec<usize>>>` in SherlockState,
/// where keys are swim row indices and values are indices into the surviving
/// peaks array (i.e. position within the centroids arrays).
pub fn group_by_swim_idx(
    sh: &mut SherlockState,
) -> Result<SwimGroupResult, String> {
    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks -- run find_filter first")?;
    let mask = sh.top_n_mask.as_ref()
        .or(sh.stripe_mask.as_ref())
        .ok_or("No peak mask -- run top_n_peaks or remove_stripes first")?;

    // Build surviving peak list (indices into the original peak arrays)
    let surviving: Vec<usize> = (0..peak_rows.len())
        .filter(|&i| mask[i])
        .collect();

    // Group by swim row index
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (centroid_idx, &orig_idx) in surviving.iter().enumerate() {
        let row = peak_rows[orig_idx];
        groups.entry(row).or_default().push(centroid_idx);
    }

    let n_groups = groups.len();
    let total_peaks = surviving.len();
    let min_peaks = groups.values().map(|v| v.len()).min().unwrap_or(0);
    let max_peaks = groups.values().map(|v| v.len()).max().unwrap_or(0);

    info!(
        "[swim_group] group_by_swim_idx: {} peaks -> {} groups (min={}, max={} peaks/group)",
        total_peaks, n_groups, min_peaks, max_peaks
    );

    sh.swim_groups = Some(groups);

    Ok(SwimGroupResult {
        n_groups,
        total_peaks,
        min_peaks_per_group: min_peaks,
        max_peaks_per_group: max_peaks,
    })
}

// ---------------------------------------------------------------------------
// Step 5 (full): Group ALL raw peaks by swim_idx (no mask required)
// ---------------------------------------------------------------------------

/// Group ALL raw peaks by their swim row index, without requiring any mask.
///
/// This matches production step 6 (`swim_idx_to_peaks.py`) which groups the
/// FULL ~14M raw peak set by swim_idx BEFORE top_n selection. The resulting
/// grouping can be used so that top_n_peaks can operate per-group or so that
/// downstream steps can iterate over all peaks at a given swim row.
///
/// Unlike `group_by_swim_idx` (which requires stripe_mask or top_n_mask),
/// this function operates on the entire raw peak set from find_filter /
/// find_peaks_raw.
///
/// Stores `swim_groups: Option<BTreeMap<usize, Vec<usize>>>` in SherlockState,
/// where keys are swim row indices and values are indices into the raw peak
/// arrays (peak_row_idx, peak_col_idx, peak_amplitudes).
pub fn group_all_by_swim_idx(
    sh: &mut SherlockState,
) -> Result<SwimGroupResult, String> {
    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks -- run find_filter first")?;
    let total_peaks = peak_rows.len();

    // Group by swim row index -- indices are positions in the raw peak arrays
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, &row) in peak_rows.iter().enumerate() {
        groups.entry(row).or_default().push(idx);
    }

    let n_groups = groups.len();
    let min_peaks = groups.values().map(|v| v.len()).min().unwrap_or(0);
    let max_peaks = groups.values().map(|v| v.len()).max().unwrap_or(0);

    info!(
        "[swim_group] group_all_by_swim_idx: {} peaks -> {} groups (min={}, max={} peaks/group)",
        total_peaks, n_groups, min_peaks, max_peaks
    );

    sh.swim_groups = Some(groups);

    Ok(SwimGroupResult {
        n_groups,
        total_peaks,
        min_peaks_per_group: min_peaks,
        max_peaks_per_group: max_peaks,
    })
}

// ---------------------------------------------------------------------------
// Step 6: Per-group isotope detection with neighbour inclusion
// ---------------------------------------------------------------------------

/// Per-group isotope detection with neighbour inclusion.
///
/// Matches production step 6 (`filter_peaks.py` / `tallest_peaks`):
/// - For each unique swim_idx in the peak set, builds a group that includes
///   peaks from neighbouring swim rows within `row_half_width`.
/// - Runs isotope detection on each expanded group independently using the
///   graph-clique algorithm (matching production NetworkX semantics).
/// - Stores per-peak charges in SherlockState.
///
/// The `row_half_width` parameter controls how many neighbouring swim rows
/// are included in each group.  Production uses row_half_width=4 (so each
/// group includes swim_idx-4 .. swim_idx+4, i.e. 9 rows).
///
/// NOTE: The production code has a known issue where the same peak can be
/// assigned different charges if it appears in multiple overlapping groups.
/// We handle this by using last-write-wins semantics (matching production
/// behaviour).
pub fn per_group_isotope_detect(
    ds: &DatasetState,
    sh: &mut SherlockState,
    row_half_width: usize,
    tolerance: f64,
    max_charge: usize,
) -> Result<SwimGroupIsotopeResult, String> {
    // Only clear isotope/swim_group results, NOT top_n_mask or stripe_mask
    sh.swim_groups = None;
    sh.charges = None;
    sh.isotope_groups = None;

    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks")?;
    let mask = sh.top_n_mask.as_ref()
        .or(sh.stripe_mask.as_ref())
        .ok_or("No peak mask -- run top_n_peaks or remove_stripes first")?;
    let centroids_tof = sh.centroids_tof.as_ref().ok_or("No centroids -- run centroid first")?;
    let centroids_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;
    let _swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;

    // Build surviving peaks and their row-indexed groups
    let surviving: Vec<usize> = (0..peak_rows.len())
        .filter(|&i| mask[i])
        .collect();

    let n = surviving.len();
    if n == 0 {
        sh.charges = Some(vec![]);
        sh.isotope_groups = Some(vec![]);
        sh.swim_groups = Some(BTreeMap::new());
        return Ok(SwimGroupIsotopeResult {
            peak_count: 0,
            n_groups: 0,
            groups_found: 0,
            charge_distribution: HashMap::new(),
            display_tof: vec![],
            display_swim: vec![],
            display_charge: vec![],
        });
    }

    // Group surviving peaks by swim row (centroid_idx -> row)
    let mut row_to_centroid_indices: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (centroid_idx, &orig_idx) in surviving.iter().enumerate() {
        let row = peak_rows[orig_idx];
        row_to_centroid_indices.entry(row).or_default().push(centroid_idx);
    }

    // Store the basic grouping
    sh.swim_groups = Some(row_to_centroid_indices.clone());

    // Collect represented swim indices (unique rows that have peaks)
    let represented_rows: Vec<usize> = row_to_centroid_indices.keys().copied().collect();

    // Per-group isotope detection with neighbour inclusion
    let mut charges = vec![0i32; n];
    let mut total_groups_found = 0usize;

    for &swim_row in &represented_rows {
        // Build expanded group: include peaks from swim_row - hw to swim_row + hw
        let lo = swim_row.saturating_sub(row_half_width);
        let hi = swim_row + row_half_width;

        let mut group_indices: Vec<usize> = Vec::new();
        for (&row, indices) in &row_to_centroid_indices {
            if row >= lo && row <= hi {
                group_indices.extend_from_slice(indices);
            }
        }

        // Deduplicate (indices are already sorted by row, but may overlap across groups)
        group_indices.sort_unstable();
        group_indices.dedup();

        // Run graph-clique isotope detection on this expanded group
        let (group_charges, groups_found) = detect_isotopes_in_group(
            &group_indices,
            centroids_tof,
            tolerance,
            max_charge,
        );

        total_groups_found += groups_found;

        // Write charges back (last-write-wins, matching production behaviour)
        for (centroid_idx, charge) in group_charges {
            if charge != 0 {
                charges[centroid_idx] = charge;
            }
        }
    }

    info!(
        "[swim_group] per_group_isotope_detect: {} peaks, {} swim groups, {} isotope groups found (row_half_width={})",
        n, represented_rows.len(), total_groups_found, row_half_width
    );

    // Build charge distribution
    let mut charge_distribution: HashMap<i32, usize> = HashMap::new();
    for &c in &charges {
        *charge_distribution.entry(c).or_insert(0) += 1;
    }

    // Build isotope_groups (for downstream RANSAC/review compatibility)
    // Collect connected components from the charges for state storage
    let mut isotope_groups: Vec<Vec<usize>> = Vec::new();
    // Group peaks by charge (non-zero = part of an isotope envelope)
    let mut charge_groups: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &c) in charges.iter().enumerate() {
        if c != 0 {
            charge_groups.entry(c).or_default().push(i);
        }
    }
    for (_charge, members) in &charge_groups {
        if members.len() >= 2 {
            isotope_groups.push(members.clone());
        }
    }

    // Display data (subsampled)
    let mut rng = rand::thread_rng();
    let display_idx = subsample_indices(n, 5000, &mut rng);

    let display_tof: Vec<f64> = display_idx.iter().map(|&i| centroids_tof[i]).collect();
    let display_swim: Vec<f64> = display_idx.iter().map(|&i| centroids_swim[i]).collect();
    let display_charge: Vec<i32> = display_idx.iter().map(|&i| charges[i]).collect();

    sh.charges = Some(charges);
    sh.isotope_groups = Some(isotope_groups.clone());

    Ok(SwimGroupIsotopeResult {
        peak_count: n,
        n_groups: represented_rows.len(),
        groups_found: total_groups_found,
        charge_distribution,
        display_tof,
        display_swim,
        display_charge,
    })
}
