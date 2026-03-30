use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::run::PeakList;

// ---------------------------------------------------------------------------
// Types (tasks 4.3, 4.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedPair {
    pub a_index: usize,
    pub b_index: usize,
    pub distance: f64,
}

/// Result of comparing two peak lists (task 4.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub run_a_id: String,
    pub run_b_id: String,
    pub run_a_algorithm: String,
    pub run_b_algorithm: String,
    pub matched_pairs: Vec<MatchedPair>,
    pub unmatched_a: Vec<usize>,
    pub unmatched_b: Vec<usize>,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tolerance_used: f64,
    pub peak_comparison: PeakComparisonMetrics,
    pub intermediate_comparisons: Option<HashMap<String, StepComparison>>,
    pub compared_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakComparisonMetrics {
    pub a_count: usize,
    pub b_count: usize,
    pub matched: usize,
    pub mean_match_dist: f64,
}

/// Result of comparing one intermediate step between two runs (task 4.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepComparison {
    pub step_name: String,
    pub peak_count_a: Option<usize>,
    pub peak_count_b: Option<usize>,
    pub peak_count_diff: Option<i64>,
    pub overlap_count: Option<usize>,
    pub mean_coordinate_deviation: Option<f64>,
    pub max_coordinate_deviation: Option<f64>,
    pub fit_param_diffs: Option<FitParamDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitParamDiff {
    pub slope_diff: f64,
    pub intercept_diff: f64,
}

// ---------------------------------------------------------------------------
// Core comparison (task 4.1)
// ---------------------------------------------------------------------------

/// Compare two peak lists using nearest-neighbor matching.
/// Returns matched pairs, unmatched indices, and precision/recall/F1.
pub fn compare_peak_lists(a: &PeakList, b: &PeakList, tolerance: f64) -> ComparisonResult {
    let n_a = a.peaks.len();
    let n_b = b.peaks.len();

    // Sort B peaks by TOF for binary search
    let mut b_sort: Vec<usize> = (0..n_b).collect();
    b_sort.sort_by(|&i, &j| {
        b.peaks[i].tof_mass.partial_cmp(&b.peaks[j].tof_mass)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let b_tof_sorted: Vec<f64> = b_sort.iter().map(|&i| b.peaks[i].tof_mass).collect();

    let mut matched_a = vec![false; n_a];
    let mut matched_b = vec![false; n_b];
    let mut matched_pairs = Vec::new();
    let mut match_distances = Vec::new();

    // For each peak in A, find nearest in B within tolerance
    for ai in 0..n_a {
        let a_tof = a.peaks[ai].tof_mass;
        let a_swim = a.peaks[ai].swim_mass;

        let lo = b_tof_sorted.partition_point(|&t| t < a_tof - tolerance);
        let hi = b_tof_sorted.partition_point(|&t| t <= a_tof + tolerance);

        let mut best_dist = f64::INFINITY;
        let mut best_bi = None;

        for j in lo..hi {
            let bi = b_sort[j];
            let dt = a_tof - b.peaks[bi].tof_mass;
            let ds = a_swim - b.peaks[bi].swim_mass;
            let dist = (dt * dt + ds * ds).sqrt();
            if dist < tolerance && dist < best_dist {
                best_dist = dist;
                best_bi = Some(bi);
            }
        }

        if let Some(bi) = best_bi {
            matched_a[ai] = true;
            matched_b[bi] = true;
            matched_pairs.push(MatchedPair {
                a_index: ai,
                b_index: bi,
                distance: best_dist,
            });
            match_distances.push(best_dist);
        }
    }

    let n_matched = matched_pairs.len();
    let unmatched_a: Vec<usize> = (0..n_a).filter(|&i| !matched_a[i]).collect();
    let unmatched_b: Vec<usize> = (0..n_b).filter(|&i| !matched_b[i]).collect();

    let precision = if n_a > 0 { n_matched as f64 / n_a as f64 } else { 0.0 };
    let recall = if n_b > 0 { n_matched as f64 / n_b as f64 } else { 0.0 };
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

    ComparisonResult {
        run_a_id: String::new(),
        run_b_id: String::new(),
        run_a_algorithm: String::new(),
        run_b_algorithm: String::new(),
        matched_pairs,
        unmatched_a,
        unmatched_b,
        precision: (precision * 10000.0).round() / 10000.0,
        recall: (recall * 10000.0).round() / 10000.0,
        f1: (f1 * 10000.0).round() / 10000.0,
        tolerance_used: tolerance,
        peak_comparison: PeakComparisonMetrics {
            a_count: n_a,
            b_count: n_b,
            matched: n_matched,
            mean_match_dist,
        },
        intermediate_comparisons: None,
        compared_at: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Intermediate comparison (task 4.2)
// ---------------------------------------------------------------------------

/// Compare intermediate step outputs between two runs.
pub fn compare_intermediates(
    a_intermediates: &HashMap<String, serde_json::Value>,
    b_intermediates: &HashMap<String, serde_json::Value>,
    step_name: &str,
) -> StepComparison {
    let a_val = a_intermediates.get(step_name);
    let b_val = b_intermediates.get(step_name);

    let get_peak_count = |v: Option<&serde_json::Value>| -> Option<usize> {
        v.and_then(|v| v.get("summary"))
         .and_then(|s| s.get("filtered_peaks").or_else(|| s.get("peak_count")))
         .and_then(|n| n.as_u64())
         .map(|n| n as usize)
    };

    let peak_count_a = get_peak_count(a_val);
    let peak_count_b = get_peak_count(b_val);
    let peak_count_diff = peak_count_a.zip(peak_count_b)
        .map(|(a, b)| a as i64 - b as i64);

    // Check for fit parameter diffs (RANSAC steps)
    let fit_param_diffs = a_val.zip(b_val).and_then(|(a, b)| {
        let a_slope = a.get("output").and_then(|o| o.get("slope")).and_then(|v| v.as_f64());
        let b_slope = b.get("output").and_then(|o| o.get("slope")).and_then(|v| v.as_f64());
        let a_int = a.get("output").and_then(|o| o.get("intercept")).and_then(|v| v.as_f64());
        let b_int = b.get("output").and_then(|o| o.get("intercept")).and_then(|v| v.as_f64());
        a_slope.zip(b_slope).zip(a_int.zip(b_int)).map(|((as_, bs), (ai, bi))| {
            FitParamDiff {
                slope_diff: as_ - bs,
                intercept_diff: ai - bi,
            }
        })
    });

    StepComparison {
        step_name: step_name.to_string(),
        peak_count_a,
        peak_count_b,
        peak_count_diff,
        overlap_count: None,
        mean_coordinate_deviation: None,
        max_coordinate_deviation: None,
        fit_param_diffs,
    }
}

// ---------------------------------------------------------------------------
// Tests (task 4.6)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sherlock::run::Peak;

    fn make_peaks(coords: &[(f64, f64)]) -> PeakList {
        PeakList {
            peaks: coords.iter().map(|&(t, s)| Peak {
                tof_mass: t,
                swim_mass: s,
                amplitude: 100.0,
                charge: 1,
                inlier: true,
            }).collect(),
            ransac: None,
            rebased: false,
            peak_count: coords.len(),
        }
    }

    #[test]
    fn test_identical_runs_f1_one() {
        let peaks = make_peaks(&[(100.0, 200.0), (150.0, 250.0), (200.0, 300.0)]);
        let result = compare_peak_lists(&peaks, &peaks, 1.0);
        assert_eq!(result.f1, 1.0);
        assert_eq!(result.precision, 1.0);
        assert_eq!(result.recall, 1.0);
        assert!(result.unmatched_a.is_empty());
        assert!(result.unmatched_b.is_empty());
        assert_eq!(result.matched_pairs.len(), 3);
        for mp in &result.matched_pairs {
            assert_eq!(mp.distance, 0.0);
        }
    }

    #[test]
    fn test_partial_overlap() {
        let a = make_peaks(&[
            (100.0, 200.0), (150.0, 250.0), (200.0, 300.0), (250.0, 350.0),
        ]);
        let b = make_peaks(&[
            (100.0, 200.0), (150.0, 250.0), // match
            (500.0, 600.0), (550.0, 650.0), // no match
        ]);
        let result = compare_peak_lists(&a, &b, 1.0);
        assert_eq!(result.matched_pairs.len(), 2);
        assert_eq!(result.unmatched_a.len(), 2);
        assert_eq!(result.unmatched_b.len(), 2);
        assert_eq!(result.precision, 0.5);
        assert_eq!(result.recall, 0.5);
    }

    #[test]
    fn test_no_overlap() {
        let a = make_peaks(&[(100.0, 200.0)]);
        let b = make_peaks(&[(500.0, 600.0)]);
        let result = compare_peak_lists(&a, &b, 1.0);
        assert_eq!(result.f1, 0.0);
        assert_eq!(result.matched_pairs.len(), 0);
    }

    #[test]
    fn test_empty_lists() {
        let empty = make_peaks(&[]);
        let non_empty = make_peaks(&[(100.0, 200.0)]);
        let result = compare_peak_lists(&empty, &non_empty, 1.0);
        assert_eq!(result.precision, 0.0);
        assert_eq!(result.recall, 0.0);
        assert_eq!(result.f1, 0.0);
    }

    #[test]
    fn test_compare_intermediates_with_fit_params() {
        let mut a_int = HashMap::new();
        a_int.insert("ransac-rough".to_string(), serde_json::json!({
            "output": {"slope": 1.0023, "intercept": -0.5},
            "summary": {"peak_count": 500}
        }));
        let mut b_int = HashMap::new();
        b_int.insert("ransac-rough".to_string(), serde_json::json!({
            "output": {"slope": 1.0025, "intercept": -0.48},
            "summary": {"peak_count": 510}
        }));

        let result = compare_intermediates(&a_int, &b_int, "ransac-rough");
        assert_eq!(result.peak_count_a, Some(500));
        assert_eq!(result.peak_count_b, Some(510));
        assert_eq!(result.peak_count_diff, Some(-10));
        let fit = result.fit_param_diffs.unwrap();
        assert!((fit.slope_diff - (-0.0002)).abs() < 0.0001);
    }
}
