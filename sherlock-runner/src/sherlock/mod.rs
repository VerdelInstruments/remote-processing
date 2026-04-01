pub mod calibrate;
pub mod fragments;
pub mod centroid;
pub mod graph_clique;
pub mod compare;
pub mod export;
pub mod find_filter;
pub mod isotope;
pub mod peaks_above_noise;
pub mod precursors;
pub mod progress;
pub mod ransac;
pub mod review;
pub mod run;
pub mod run_compare;
pub mod stripes;
pub mod subsample;
pub mod swim_group;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// Re-export result types for lib.rs
pub use calibrate::CalibrateResult;
pub use centroid::CentroidResult;
pub use compare::CompareResult;
pub use export::ExportResult;
pub use fragments::FragmentsResult;
pub use find_filter::FindFilterResult;
pub use find_filter::FindPeaksRawResult;
pub use isotope::IsotopeResult;
pub use peaks_above_noise::PeaksAboveNoiseResult;
pub use precursors::PrecursorsResult;
pub use ransac::{PreciseAutocorrelationResult, RansacResult};
pub use review::ReviewResult;
pub use stripes::RemoveStripesResult;
pub use stripes::TopNPeaksResult;
pub use swim_group::{SwimGroupResult, SwimGroupIsotopeResult};

/// Calibration parameters for frequency → mass conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParams {
    pub scale: f64,
    pub offset: f64,
}

/// Persistent state holding intermediate results between Sherlock wizard steps.
/// Each step stores its output here; downstream steps read from here.
#[derive(Debug, Default)]
pub struct SherlockState {
    // Per-step intermediate capture for result set serialisation (task 2.1)
    pub step_intermediates: Vec<run::StepIntermediate>,

    // Algorithm variant: controls RANSAC implementation choice
    // "rust-native" = pure Rust RANSAC, "rust-sklearn-bridge" = Python sklearn via subprocess
    pub use_sklearn_ransac: bool,

    // Step 1: Calibrate
    pub calibration: Option<CalibrationParams>,

    // Step 2: Find & Filter
    pub peak_row_idx: Option<Vec<usize>>,
    pub peak_col_idx: Option<Vec<usize>>,
    pub peak_amplitudes: Option<Vec<f64>>,
    pub snr_values: Option<Vec<f64>>,

    // Step 3: Remove Stripes
    pub stripe_mask: Option<Vec<bool>>,

    // Step 4: Centroid
    pub centroids_tof: Option<Vec<f64>>,
    pub centroids_swim: Option<Vec<f64>>,

    // Step 5: Swim group (group peaks by swim row index)
    /// Mapping from swim row index to centroid indices in that row.
    pub swim_groups: Option<BTreeMap<usize, Vec<usize>>>,

    // Step 5b: Isotope (per-group or flat)
    pub charges: Option<Vec<i32>>,
    pub isotope_groups: Option<Vec<Vec<usize>>>,

    // Full peak set (saved before top_n overwrites peak arrays).
    // peaks_above_noise needs the original 14M peaks, not the 1,200 top_n.
    pub full_peak_row_idx: Option<Vec<usize>>,
    pub full_peak_col_idx: Option<Vec<usize>>,
    pub full_peak_amplitudes: Option<Vec<f64>>,

    // Step 6: Top N peaks (tallest peak selection)
    pub top_n_mask: Option<Vec<bool>>,

    // Step 7: RANSAC (rough fit)
    pub ransac_slope: Option<f64>,
    pub ransac_intercept: Option<f64>,
    pub ransac_inlier_mask: Option<Vec<bool>>,

    // Step 8: Precise Autocorrelation (refined fit)
    pub precise_slope: Option<f64>,
    pub precise_intercept: Option<f64>,
    pub precise_inlier_mask: Option<Vec<bool>>,

    // Step 9: Precursors (calibrated, labeled, isotope-detected, charge-filtered)
    pub precursor_tof_calibrated: Option<Vec<f64>>,
    pub precursor_swim_calibrated: Option<Vec<f64>>,
    pub precursor_charges: Option<Vec<i32>>,
    pub precursor_labels: Option<Vec<i32>>,
    /// Indices into the original peak arrays (peak_row_idx etc.) for each precursor.
    pub precursor_peak_indices: Option<Vec<usize>>,

    // Step 10: Fragments (per-swim aggregation of precursors + tallest peaks)
    pub fragment_tof: Option<Vec<f64>>,
    pub fragment_swim: Option<Vec<f64>>,
    pub fragment_charges: Option<Vec<i32>>,
    pub fragment_swim_idx: Option<Vec<usize>>,
    pub fragment_n_swim_indices: Option<usize>,
}

impl SherlockState {
    /// Clear the given step and all downstream results.
    /// Steps: 1=calibrate, 2=find_filter, 3=stripes, 4=centroid,
    ///        5=isotope, 6=top_n_peaks, 7=ransac, 8=precise_autocorrelation,
    ///        9=precursors, 10=fragments
    pub fn invalidate_from(&mut self, step: u8) {
        // Remove intermediates for invalidated steps
        self.step_intermediates.retain(|i| (i.step_number as u8) < step);

        if step <= 1 {
            self.calibration = None;
        }
        if step <= 2 {
            self.peak_row_idx = None;
            self.peak_col_idx = None;
            self.peak_amplitudes = None;
            self.snr_values = None;
        }
        if step <= 3 {
            self.stripe_mask = None;
        }
        if step <= 4 {
            self.centroids_tof = None;
            self.centroids_swim = None;
        }
        if step <= 5 {
            self.swim_groups = None;
            self.charges = None;
            self.isotope_groups = None;
        }
        if step <= 6 {
            self.top_n_mask = None;
        }
        if step <= 7 {
            self.ransac_slope = None;
            self.ransac_intercept = None;
            self.ransac_inlier_mask = None;
        }
        if step <= 8 {
            self.precise_slope = None;
            self.precise_intercept = None;
            self.precise_inlier_mask = None;
        }
        if step <= 9 {
            self.precursor_tof_calibrated = None;
            self.precursor_swim_calibrated = None;
            self.precursor_charges = None;
            self.precursor_labels = None;
            self.precursor_peak_indices = None;
        }
        if step <= 10 {
            self.fragment_tof = None;
            self.fragment_swim = None;
            self.fragment_charges = None;
            self.fragment_swim_idx = None;
            self.fragment_n_swim_indices = None;
        }
    }

    /// Clear top_n_peaks selection and all downstream results.
    pub fn invalidate_top_n(&mut self) {
        self.top_n_mask = None;
        self.ransac_slope = None;
        self.ransac_intercept = None;
        self.ransac_inlier_mask = None;
        self.precise_slope = None;
        self.precise_intercept = None;
        self.precise_inlier_mask = None;
        self.precursor_tof_calibrated = None;
        self.precursor_swim_calibrated = None;
        self.precursor_charges = None;
        self.precursor_labels = None;
        self.precursor_peak_indices = None;
    }

    /// Reset all state (e.g. when loading a new dataset).
    pub fn reset(&mut self) {
        self.invalidate_from(1);
    }
}
