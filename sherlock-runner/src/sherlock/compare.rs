use log::info;
use serde::Serialize;

use crate::dataset::DatasetState;
use super::progress::emit_progress;
use super::subsample::subsample_indices;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct NearestPair {
    pub ref_tof: f64,
    pub ref_swim: f64,
    pub our_tof: f64,
    pub our_swim: f64,
    pub tof_diff: f64,
    pub swim_diff: f64,
    pub dist: f64,
}

#[derive(Debug, Serialize)]
pub struct CompareResult {
    pub has_reference: bool,
    pub ref_source: Option<String>,
    pub ref_count: Option<usize>,
    pub our_count: Option<usize>,
    pub matched: Option<usize>,
    pub ref_only: Option<usize>,
    pub our_only: Option<usize>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_match_dist: Option<f64>,
    pub tol_tof: Option<f64>,
    pub tol_swim: Option<f64>,
    pub display_tof: Option<Vec<f64>>,
    pub display_swim: Option<Vec<f64>>,
    pub display_cat: Option<Vec<String>>,
    pub message: Option<String>,
    /// Closest reference↔our pairs (up to 20), sorted by distance.
    /// Helps diagnose coordinate mismatches and tolerance tuning.
    pub nearest_pairs: Option<Vec<NearestPair>>,
}

pub fn compare(
    ds: &DatasetState,
    sh: &SherlockState,
    tol_tof: f64,
    tol_swim: f64,
) -> Result<CompareResult, String> {
    let raw_centroids_tof = sh.centroids_tof.as_ref().ok_or("No centroids")?;
    let raw_centroids_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;

    // Apply calibration to swim centroids.
    // Original Sherlock formula: calibrated = (raw - c) / m
    // where m, c come from the RANSAC autocorrelation fit stored in calibration.pkl.
    // Use RANSAC fit from our pipeline if available, otherwise use raw coordinates.
    let centroids_tof = raw_centroids_tof;
    let calibrated_swim: Vec<f64>;
    let centroids_swim: &Vec<f64> = if let (Some(m), Some(c)) = (sh.ransac_slope, sh.ransac_intercept) {
        if m.abs() > f64::EPSILON {
            info!("[compare] applying RANSAC calibration: m={:.6}, c={:.6}", m, c);
            calibrated_swim = raw_centroids_swim.iter().map(|&s| (s - c) / m).collect();

            // Log sample values for debugging
            if calibrated_swim.len() > 0 {
                info!("[compare] swim sample: raw={:.2} → cal={:.2}", raw_centroids_swim[0], calibrated_swim[0]);
            }

            &calibrated_swim
        } else {
            raw_centroids_swim
        }
    } else {
        info!("[compare] no RANSAC fit available, using raw swim coordinates");
        raw_centroids_swim
    };

    // Log coordinate ranges for diagnostics
    if !centroids_tof.is_empty() {
        let tof_min = centroids_tof.iter().cloned().fold(f64::INFINITY, f64::min);
        let tof_max = centroids_tof.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let swim_min = centroids_swim.iter().cloned().fold(f64::INFINITY, f64::min);
        let swim_max = centroids_swim.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        info!("[compare] our peaks: tof=[{:.2}, {:.2}] swim=[{:.2}, {:.2}] ({})", tof_min, tof_max, swim_min, swim_max, centroids_tof.len());
    }
    if !ds.peaks.is_empty() {
        let ref_tof_min = ds.peaks.iter().map(|p| p.tof_mass).fold(f64::INFINITY, f64::min);
        let ref_tof_max = ds.peaks.iter().map(|p| p.tof_mass).fold(f64::NEG_INFINITY, f64::max);
        let ref_swim_min = ds.peaks.iter().map(|p| p.swim_mass).fold(f64::INFINITY, f64::min);
        let ref_swim_max = ds.peaks.iter().map(|p| p.swim_mass).fold(f64::NEG_INFINITY, f64::max);
        info!("[compare] ref peaks: tof=[{:.2}, {:.2}] swim=[{:.2}, {:.2}] ({})", ref_tof_min, ref_tof_max, ref_swim_min, ref_swim_max, ds.peaks.len());
    }

    if ds.peaks.is_empty() {
        return Ok(CompareResult {
            has_reference: false,
            ref_source: ds.peaks_source.clone(),
            message: Some("No reference peaks available".to_string()),
            ref_count: None,
            our_count: None,
            matched: None,
            ref_only: None,
            our_only: None,
            precision: None,
            recall: None,
            f1: None,
            mean_match_dist: None,
            tol_tof: None,
            tol_swim: None,
            display_tof: None,
            display_swim: None,
            display_cat: None,
            nearest_pairs: None,
        });
    }

    emit_progress(0.0, "Loading reference peaks...");

    let ref_tof: Vec<f64> = ds.peaks.iter().map(|p| p.tof_mass).collect();
    let ref_swim: Vec<f64> = ds.peaks.iter().map(|p| p.swim_mass).collect();
    let n_ref = ref_tof.len();
    let n_our = centroids_tof.len();

    // Build nearest-pairs table: for each reference peak, find its closest
    // our-peak (no tolerance gate) to diagnose coordinate mismatches.
    let nearest_pairs = if n_ref > 0 && n_our > 0 {
        // Sort our peaks by TOF for fast nearest lookup
        let mut our_by_tof: Vec<usize> = (0..n_our).collect();
        our_by_tof.sort_by(|&a, &b| centroids_tof[a].partial_cmp(&centroids_tof[b]).unwrap());
        let sorted_tof: Vec<f64> = our_by_tof.iter().map(|&i| centroids_tof[i]).collect();

        let mut pairs: Vec<NearestPair> = Vec::new();
        for ri in 0..n_ref {
            // Binary search for nearest TOF, then check neighbours
            let pos = sorted_tof.partition_point(|&t| t < ref_tof[ri]);
            let mut best_dist = f64::INFINITY;
            let mut best = None;
            // Check a window around the insertion point
            let lo = pos.saturating_sub(5);
            let hi = (pos + 5).min(n_our);
            for j in lo..hi {
                let oi = our_by_tof[j];
                let dt = ref_tof[ri] - centroids_tof[oi];
                let ds = ref_swim[ri] - centroids_swim[oi];
                let d = (dt * dt + ds * ds).sqrt();
                if d < best_dist {
                    best_dist = d;
                    best = Some(NearestPair {
                        ref_tof: ref_tof[ri],
                        ref_swim: ref_swim[ri],
                        our_tof: centroids_tof[oi],
                        our_swim: centroids_swim[oi],
                        tof_diff: dt.abs(),
                        swim_diff: ds.abs(),
                        dist: d,
                    });
                }
            }
            if let Some(p) = best {
                pairs.push(p);
            }
        }
        pairs.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        pairs.truncate(20);

        if let Some(p) = pairs.first() {
            info!(
                "[compare] closest pair: dist={:.4} tof_diff={:.4} swim_diff={:.4} (tols: tof={}, swim={})",
                p.dist, p.tof_diff, p.swim_diff, tol_tof, tol_swim
            );
        }
        Some(pairs)
    } else {
        None
    };

    // Sort our peaks by TOF for binary search
    let mut our_sort: Vec<usize> = (0..n_our).collect();
    our_sort.sort_by(|&a, &b| {
        centroids_tof[a]
            .partial_cmp(&centroids_tof[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let our_tof_sorted: Vec<f64> = our_sort.iter().map(|&i| centroids_tof[i]).collect();

    emit_progress(0.3, "Matching peaks...");

    let mut matched_ref = vec![false; n_ref];
    let mut matched_our = vec![false; n_our];
    let mut match_distances: Vec<f64> = Vec::new();

    for i in 0..n_ref {
        let lo = our_tof_sorted
            .partition_point(|&t| t < ref_tof[i] - tol_tof);
        let hi = our_tof_sorted
            .partition_point(|&t| t <= ref_tof[i] + tol_tof);

        let mut best_dist = f64::INFINITY;
        let mut best_idx = None;

        for j in lo..hi {
            let orig_idx = our_sort[j];
            let swim_diff = (centroids_swim[orig_idx] - ref_swim[i]).abs();
            if swim_diff >= tol_swim {
                continue;
            }

            let dist = ((centroids_tof[orig_idx] - ref_tof[i]).powi(2)
                + (centroids_swim[orig_idx] - ref_swim[i]).powi(2))
            .sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(orig_idx);
            }
        }

        if let Some(idx) = best_idx {
            matched_ref[i] = true;
            matched_our[idx] = true;
            match_distances.push(best_dist);
        }
    }

    emit_progress(0.7, "Computing statistics...");

    let n_matched = match_distances.len();
    let n_ref_only = n_ref - n_matched;
    let n_our_only = n_our - matched_our.iter().filter(|&&m| m).count();

    let precision = if n_our > 0 {
        n_matched as f64 / n_our as f64
    } else {
        0.0
    };
    let recall = if n_ref > 0 {
        n_matched as f64 / n_ref as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let mean_match_dist = if !match_distances.is_empty() {
        match_distances.iter().sum::<f64>() / match_distances.len() as f64
    } else {
        0.0
    };

    // Display: subsample from each category
    let max_per_cat = 5000 / 3;
    let mut rng = rand::thread_rng();

    let mut display_tof = Vec::new();
    let mut display_swim = Vec::new();
    let mut display_cat = Vec::new();

    // Matched (from our side)
    let matched_indices: Vec<usize> = (0..n_our).filter(|&i| matched_our[i]).collect();
    let sub = subsample_indices(matched_indices.len(), max_per_cat, &mut rng);
    for &i in &sub {
        let idx = matched_indices[i];
        display_tof.push(centroids_tof[idx]);
        display_swim.push(centroids_swim[idx]);
        display_cat.push("matched".to_string());
    }

    // Ref only
    let ref_only_indices: Vec<usize> = (0..n_ref).filter(|&i| !matched_ref[i]).collect();
    let sub = subsample_indices(ref_only_indices.len(), max_per_cat, &mut rng);
    for &i in &sub {
        let idx = ref_only_indices[i];
        display_tof.push(ref_tof[idx]);
        display_swim.push(ref_swim[idx]);
        display_cat.push("ref_only".to_string());
    }

    // Our only
    let our_only_indices: Vec<usize> = (0..n_our).filter(|&i| !matched_our[i]).collect();
    let sub = subsample_indices(our_only_indices.len(), max_per_cat, &mut rng);
    for &i in &sub {
        let idx = our_only_indices[i];
        display_tof.push(centroids_tof[idx]);
        display_swim.push(centroids_swim[idx]);
        display_cat.push("our_only".to_string());
    }

    emit_progress(1.0, "Done");

    Ok(CompareResult {
        has_reference: true,
        ref_source: ds.peaks_source.clone(),
        ref_count: Some(n_ref),
        our_count: Some(n_our),
        matched: Some(n_matched),
        ref_only: Some(n_ref_only),
        our_only: Some(n_our_only),
        precision: Some((precision * 10000.0).round() / 10000.0),
        recall: Some((recall * 10000.0).round() / 10000.0),
        f1: Some((f1 * 10000.0).round() / 10000.0),
        mean_match_dist: Some(mean_match_dist),
        tol_tof: Some(tol_tof),
        tol_swim: Some(tol_swim),
        display_tof: Some(display_tof),
        display_swim: Some(display_swim),
        display_cat: Some(display_cat),
        message: None,
        nearest_pairs,
    })
}
