use log::info;
use serde::Serialize;
use std::collections::HashMap;

use crate::dataset::DatasetState;
use super::subsample::subsample_indices;
use super::SherlockState;

// ---------------------------------------------------------------------------
// Step 4: Stripe removal (for rough autocorrelation, on binned peaks)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct RemoveStripesResult {
    pub before: usize,
    pub after: usize,
    pub removed: usize,
    pub stripe_rows: Vec<usize>,
    pub stripe_cols: Vec<usize>,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
}

/// Remove instrumental stripe artifacts from peaks.
///
/// Rows or columns with peak count >= `stripe_threshold` are considered
/// stripes.  All peaks on stripe rows/cols are removed.
///
/// NOTE: Production's filter_dense_lines has a keep_tallest bug where the
/// tallest peaks are computed but never actually included in the returned
/// result. This function faithfully reproduces that bug -- tallest peaks
/// on dense lines are NOT kept.
///
/// Stores `stripe_mask: Vec<bool>` (true = keep) in SherlockState.
///
/// This is step 4 in the production pipeline (rough autocorrelation),
/// applied to BINNED peaks.
pub fn remove_stripes(
    ds: &DatasetState,
    sh: &mut SherlockState,
    stripe_threshold: usize,
) -> Result<RemoveStripesResult, String> {
    sh.invalidate_from(3);

    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks -- run find_filter first")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peaks")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    let n_peaks = peak_rows.len();
    let (n_rows, n_cols) = ds.shape;

    // Count peaks per row and per column
    let mut row_counts = vec![0usize; n_rows];
    let mut col_counts = vec![0usize; n_cols];
    for i in 0..n_peaks {
        row_counts[peak_rows[i]] += 1;
        col_counts[peak_cols[i]] += 1;
    }

    // Build mask: keep peaks NOT on dense rows/cols
    let mut mask = vec![false; n_peaks];
    for i in 0..n_peaks {
        mask[i] = row_counts[peak_rows[i]] < stripe_threshold
            && col_counts[peak_cols[i]] < stripe_threshold;
    }

    // Identify stripe rows and cols
    let stripe_rows: Vec<usize> = row_counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= stripe_threshold)
        .map(|(i, _)| i)
        .collect();

    let stripe_cols: Vec<usize> = col_counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c >= stripe_threshold)
        .map(|(i, _)| i)
        .collect();

    // NOTE: Production bug faithfully ported.
    // Python's filter_dense_lines has keep_tallest=True which modifies
    // combined_mask to re-include the tallest peak on each dense row/col,
    // but then returns points[x_mask & y_mask] (the ORIGINAL mask) instead
    // of points[combined_mask]. So the tallest peaks are never actually kept.
    // We intentionally do NOT keep tallest peaks here to match production.

    let before = n_peaks;
    let after = mask.iter().filter(|&&m| m).count();
    let removed = before - after;

    info!(
        "[stripes] remove_stripes: {} peaks -> {} kept, {} removed (threshold={})",
        before, after, removed, stripe_threshold
    );

    // Subsample surviving peaks for display
    let surviving: Vec<usize> = (0..n_peaks).filter(|&i| mask[i]).collect();
    let mut rng = rand::thread_rng();
    let display_idx = subsample_indices(surviving.len(), 5000, &mut rng);

    let display_tof: Vec<f64> = display_idx
        .iter()
        .map(|&i| tof_coords[peak_cols[surviving[i]]])
        .collect();
    let display_swim: Vec<f64> = display_idx
        .iter()
        .map(|&i| swim_coords[peak_rows[surviving[i]]])
        .collect();

    sh.stripe_mask = Some(mask);

    Ok(RemoveStripesResult {
        before,
        after,
        removed,
        stripe_rows,
        stripe_cols,
        display_tof,
        display_swim,
    })
}

// ---------------------------------------------------------------------------
// Step 6: Top N peaks selection (tallest peaks, for refined autocorrelation)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct TopNPeaksResult {
    pub before: usize,
    pub after: usize,
    pub display_tof: Vec<f64>,
    pub display_swim: Vec<f64>,
}

/// Select the top N tallest peaks with per-streak limiting.
///
/// Matches the production pipeline step 6 ("tallest peaks"):
/// - Sort all peaks by amplitude descending
/// - Walk down the sorted list; for each candidate, check if its TOF column
///   (and the neighbourhood within `streak_half_width`) already has
///   `most_peaks_per_streak` selected peaks. If so, skip it.
/// - Stop after collecting `n_peaks` peaks.
///
/// Production parameters:
///   n_peaks = 1200
///   most_peaks_per_streak = 5
///   streak_half_width = 13
///   harmonic_proportional_tolerance = 1
///
/// The `harmonic_proportional_tolerance` parameter controls the streak
/// half-width scaling: the effective half-width for a TOF column is
/// `streak_half_width * harmonic_proportional_tolerance`. With the
/// production value of 1, the effective half-width equals
/// `streak_half_width` (13 columns).
///
/// Reads peaks from SherlockState (using stripe_mask if present to filter
/// to stripe-surviving peaks). Stores `top_n_mask: Vec<bool>` in
/// SherlockState.
pub fn top_n_peaks(
    ds: &DatasetState,
    sh: &mut SherlockState,
    n_peaks: usize,
    most_peaks_per_streak: usize,
    streak_half_width: usize,
    harmonic_proportional_tolerance: f64,
) -> Result<TopNPeaksResult, String> {
    // Invalidate step 6 (top_n) and downstream
    sh.invalidate_top_n();

    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks -- run find_filter first")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peaks")?;
    let amplitudes = sh.peak_amplitudes.as_ref().ok_or("No amplitudes")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;

    let total_peaks = peak_rows.len();
    let (n_rows, n_cols) = ds.shape;

    // Frequency coordinates for harmonic detection (optional — only for freq-domain data)
    let freq_coords = ds.raw_freq_coords.as_ref();

    // Determine which peaks are eligible (respect stripe_mask if present)
    let eligible: Vec<usize> = if let Some(ref smask) = sh.stripe_mask {
        (0..total_peaks).filter(|&i| smask[i]).collect()
    } else {
        (0..total_peaks).collect()
    };

    let n_eligible = eligible.len();

    // Sort eligible indices by amplitude descending
    let mut order = eligible.clone();
    order.sort_by(|&a, &b| amplitudes[b].partial_cmp(&amplitudes[a]).unwrap());

    // Pre-filter: take top n_peaks * 20 candidates (matching production's PeakArrayCollection fast path)
    let candidate_count = n_eligible.min(n_peaks * 20);
    let candidates = &order[..candidate_count];

    let mut keep: Vec<usize> = Vec::with_capacity(n_peaks.min(n_eligible));
    let mut per_tof_count: HashMap<usize, usize> = HashMap::new();

    // For harmonic detection: track (tof_col -> vec of (frequency, amplitude)) of kept peaks
    let mut kept_freqs_by_tof: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    for &idx in candidates {
        let tof = peak_cols[idx];
        let swim_row = peak_rows[idx];
        let amp = amplitudes[idx];

        // Harmonic rejection: check if this peak is a harmonic of an already-kept peak
        // at the same tof_idx. Matches production's top_n_peaks.py lines 56-81.
        if harmonic_proportional_tolerance > 0.0 {
            if let Some(fc) = freq_coords {
                // Get frequency of this peak (reversed index → original freq index)
                let freq_idx = n_rows - 1 - swim_row;
                let candidate_freq = fc[freq_idx];

                // Check against already-kept peaks at same tof_idx
                let mut is_harmonic = false;
                if let Some(parents) = kept_freqs_by_tof.get(&tof) {
                    // Only check parents that are taller (higher amplitude)
                    let parent_freqs: Vec<f64> = parents.iter()
                        .filter(|(_, a)| *a >= amp)
                        .map(|(f, _)| *f)
                        .collect();

                    if !parent_freqs.is_empty() {
                        // Check harmonics 2, 3, 4, 5
                        let harmonic_checks = [2.0_f64, 3.0, 4.0, 5.0];
                        for &n in &harmonic_checks {
                            let harmonic_freq = candidate_freq / n;
                            for &pf in &parent_freqs {
                                let prop_diff = ((pf - harmonic_freq) / harmonic_freq).abs();
                                if prop_diff < harmonic_proportional_tolerance {
                                    is_harmonic = true;
                                    break;
                                }
                            }
                            if is_harmonic { break; }
                        }
                    }
                }

                if is_harmonic {
                    continue;
                }
            }
        }

        // Streak limiting
        if most_peaks_per_streak > 0 {
            let count = *per_tof_count.get(&tof).unwrap_or(&0);
            if count >= most_peaks_per_streak {
                continue;
            }
        }

        keep.push(idx);

        // Track frequency for harmonic detection
        if let Some(fc) = freq_coords {
            let freq_idx = n_rows - 1 - swim_row;
            kept_freqs_by_tof.entry(tof).or_default().push((fc[freq_idx], amp));
        }

        // Increment count for this TOF column and its neighbourhood
        if most_peaks_per_streak > 0 {
            let lo = tof.saturating_sub(streak_half_width);
            let hi = (tof + streak_half_width).min(n_cols - 1);
            for k in lo..=hi {
                *per_tof_count.entry(k).or_insert(0) += 1;
            }
        }

        if keep.len() >= n_peaks {
            break;
        }
    }

    info!(
        "[stripes] top_n_peaks: {} eligible -> {} selected (n_peaks={}, max_per_streak={}, streak_hw={}, harmonic_tol={})",
        n_eligible, keep.len(), n_peaks, most_peaks_per_streak, streak_half_width, harmonic_proportional_tolerance
    );

    // Build full-size mask
    let mut full_mask = vec![false; total_peaks];
    for &idx in &keep {
        full_mask[idx] = true;
    }

    let before = n_eligible;
    let after = keep.len();

    // Subsample for display (all selected peaks, up to 5000)
    let mut rng = rand::thread_rng();
    let display_idx = subsample_indices(keep.len(), 5000, &mut rng);

    let display_tof: Vec<f64> = display_idx
        .iter()
        .map(|&i| tof_coords[peak_cols[keep[i]]])
        .collect();
    let display_swim: Vec<f64> = display_idx
        .iter()
        .map(|&i| swim_coords[peak_rows[keep[i]]])
        .collect();

    sh.top_n_mask = Some(full_mask);

    Ok(TopNPeaksResult {
        before,
        after,
        display_tof,
        display_swim,
    })
}
