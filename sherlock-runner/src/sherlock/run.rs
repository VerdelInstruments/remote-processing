use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Run ID generation (task 1.3)
// ---------------------------------------------------------------------------

/// Generate a unique 8-char alphanumeric run ID from timestamp + random bytes.
pub fn generate_run_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let ts_bytes = now.as_millis().to_le_bytes();
    let mut rng = rand::thread_rng();
    let rand_bytes: [u8; 4] = rng.gen();

    // Mix timestamp and random bytes, encode as base36
    let mut combined: u64 = 0;
    for &b in ts_bytes.iter().take(8) {
        combined = combined.wrapping_mul(31).wrapping_add(b as u64);
    }
    for &b in &rand_bytes {
        combined = combined.wrapping_mul(31).wrapping_add(b as u64);
    }

    // Encode as 8-char alphanumeric (base 36: 0-9, a-z)
    const CHARS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut id = String::with_capacity(8);
    let mut val = combined;
    for _ in 0..8 {
        id.push(CHARS[(val % 36) as usize] as char);
        val /= 36;
    }
    id
}

// ---------------------------------------------------------------------------
// Result set types (tasks 1.1, 1.2)
// ---------------------------------------------------------------------------

/// Complete snapshot of all parameter values used in a pipeline run, keyed by step name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    #[serde(default)]
    pub calibrate: Option<CalibrateConfig>,
    #[serde(default)]
    pub find_filter: Option<FindFilterConfig>,
    #[serde(default)]
    pub remove_stripes: Option<RemoveStripesConfig>,
    #[serde(default)]
    pub top_n: Option<TopNConfig>,
    #[serde(default)]
    pub centroid: Option<CentroidConfig>,
    #[serde(default)]
    pub isotope: Option<IsotopeConfig>,
    #[serde(default)]
    pub ransac: Option<RansacConfig>,
    #[serde(default)]
    pub precise_autocorrelation: Option<PreciseAutocorrelationConfig>,
    #[serde(default)]
    pub precursors: Option<PrecursorsConfig>,
    #[serde(default)]
    pub fragments: Option<FragmentsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrateConfig {
    pub scale: f64,
    pub offset: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindFilterConfig {
    pub min_snr: f64,
    pub noise_window: usize,
    pub peak_filter_size: usize,
    pub algorithm: String,
    pub swim_range: Option<(f64, f64)>,
    pub tof_range: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveStripesConfig {
    pub stripe_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopNConfig {
    pub n_peaks: usize,
    pub most_peaks_per_streak: usize,
    pub streak_half_width: usize,
    pub harmonic_proportional_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidConfig {
    pub half_window_tof: usize,
    pub half_window_swim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotopeConfig {
    pub tolerance: f64,
    pub max_charge: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacConfig {
    pub tolerance: f64,
    pub n_peaks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreciseAutocorrelationConfig {
    pub tolerance: f64,
    pub n_peaks: usize,
    pub half_window_tof: usize,
    pub half_window_swim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecursorsConfig {
    pub amplitude_std_multiplier: f64,
    pub gulley_width_amu: f64,
    pub isotope_tolerance: f64,
    pub max_charge: usize,
    pub half_window_tof: usize,
    pub half_window_swim: usize,
    pub autocorrelation_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentsConfig {
    pub isotope_tolerance: f64,
    pub max_charge: usize,
    pub half_window_tof: usize,
    pub half_window_swim: usize,
}

/// Summary metrics in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub raw_peaks: Option<usize>,
    pub final_peaks: usize,
    pub precursors: Option<usize>,
    pub ransac_slope: Option<f64>,
    pub ransac_intercept: Option<f64>,
}

/// Run manifest — metadata for a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub dataset: String,
    pub algorithm: String,
    pub config: RunConfig,
    pub origin: RunOrigin,
    pub status: RunStatus,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub error: Option<String>,
    pub summary: Option<RunSummary>,
    pub store_intermediates: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RunOrigin {
    Local,
    Cloud,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// A single peak in the result set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    pub tof_mass: f64,
    pub swim_mass: f64,
    pub amplitude: f64,
    pub charge: i32,
    pub inlier: bool,
}

/// RANSAC fit parameters stored in the peak list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RansacFit {
    pub slope: f64,
    pub intercept: f64,
}

/// Peak list — the final output of a pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakList {
    pub peaks: Vec<Peak>,
    pub ransac: Option<RansacFit>,
    pub rebased: bool,
    pub peak_count: usize,
}

/// Per-step intermediate output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepIntermediate {
    pub step_number: usize,
    pub step_name: String,
    pub duration_ms: u64,
    pub output: serde_json::Value,
    pub summary: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Disk persistence (tasks 2.2-2.6)
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};

/// Get the runs directory for a dataset: `{dataset_dir}/.sherlock/runs/`
fn runs_dir(dataset_dir: &Path) -> PathBuf {
    dataset_dir.join(".sherlock").join("runs")
}

/// Get the directory for a specific run: `{dataset_dir}/.sherlock/runs/run-{id}/`
fn run_dir(dataset_dir: &Path, run_id: &str) -> PathBuf {
    runs_dir(dataset_dir).join(format!("run-{}", run_id))
}

/// Write a complete run to disk (task 2.2).
pub fn write_run_to_disk(
    dataset_dir: &Path,
    manifest: &RunManifest,
    peaks: &PeakList,
    intermediates: Option<&[StepIntermediate]>,
) -> Result<PathBuf, String> {
    let dir = run_dir(dataset_dir, &manifest.run_id);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create run dir {}: {}", dir.display(), e))?;

    // Write manifest.json
    let manifest_path = dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| format!("Failed to write manifest: {}", e))?;

    // Write peaks.json
    let peaks_path = dir.join("peaks.json");
    let peaks_json = serde_json::to_string_pretty(peaks).map_err(|e| e.to_string())?;
    std::fs::write(&peaks_path, peaks_json)
        .map_err(|e| format!("Failed to write peaks: {}", e))?;

    // Write intermediates if provided
    if let Some(intermediates) = intermediates {
        let int_dir = dir.join("intermediates");
        std::fs::create_dir_all(&int_dir)
            .map_err(|e| format!("Failed to create intermediates dir: {}", e))?;
        for step in intermediates {
            let filename = format!("{:02}-{}.json", step.step_number, step.step_name);
            let step_json = serde_json::to_string_pretty(step).map_err(|e| e.to_string())?;
            std::fs::write(int_dir.join(&filename), step_json)
                .map_err(|e| format!("Failed to write intermediate {}: {}", filename, e))?;
        }
    }

    Ok(dir)
}

/// List all runs for a dataset, sorted by started_at descending (task 2.3).
pub fn list_runs(dataset_dir: &Path) -> Result<Vec<RunManifest>, String> {
    let dir = runs_dir(dataset_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut manifests = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| format!("Failed to read runs dir: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest_path = path.join("manifest.json");
        if !manifest_path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read {}: {}", manifest_path.display(), e))?;
        match serde_json::from_str::<RunManifest>(&content) {
            Ok(m) => manifests.push(m),
            Err(e) => {
                log::warn!("Skipping invalid manifest {}: {}", manifest_path.display(), e);
            }
        }
    }

    // Sort by started_at descending (most recent first)
    manifests.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    Ok(manifests)
}

/// Load a run's peaks.json (task 2.4).
pub fn load_run_peaks(dataset_dir: &Path, run_id: &str) -> Result<PeakList, String> {
    let path = run_dir(dataset_dir, run_id).join("peaks.json");
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read peaks for run {}: {}", run_id, e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse peaks for run {}: {}", run_id, e))
}

/// Load a run's intermediate files (task 2.5).
pub fn load_run_intermediates(
    dataset_dir: &Path,
    run_id: &str,
) -> Result<HashMap<String, serde_json::Value>, String> {
    let int_dir = run_dir(dataset_dir, run_id).join("intermediates");
    if !int_dir.exists() {
        return Ok(HashMap::new());
    }

    let mut result = HashMap::new();
    let entries = std::fs::read_dir(&int_dir)
        .map_err(|e| format!("Failed to read intermediates dir: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        // Key by step name extracted from the JSON or filename
        let step_name = value.get("step_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                path.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
        result.insert(step_name, value);
    }

    Ok(result)
}

/// Delete a run directory from disk (task 2.6).
pub fn delete_run(dataset_dir: &Path, run_id: &str) -> Result<(), String> {
    let dir = run_dir(dataset_dir, run_id);
    if !dir.exists() {
        return Err(format!("Run {} not found", run_id));
    }
    std::fs::remove_dir_all(&dir)
        .map_err(|e| format!("Failed to delete run {}: {}", run_id, e))
}

// ---------------------------------------------------------------------------
// Tests (task 1.4)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_id_generation_uniqueness() {
        let id1 = generate_run_id();
        let id2 = generate_run_id();
        assert_eq!(id1.len(), 8);
        assert_eq!(id2.len(), 8);
        assert_ne!(id1, id2);
        // All chars should be alphanumeric
        assert!(id1.chars().all(|c| c.is_ascii_alphanumeric()));
        assert!(id2.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_manifest_serialisation_roundtrip() {
        let manifest = RunManifest {
            run_id: "a1b2c3d4".to_string(),
            dataset: "/data/sample-010".to_string(),
            algorithm: "local-rust-v1".to_string(),
            config: RunConfig {
                calibrate: Some(CalibrateConfig { scale: 1.0, offset: 0.0 }),
                find_filter: Some(FindFilterConfig {
                    min_snr: 70.0,
                    noise_window: 51,
                    peak_filter_size: 5,
                    algorithm: "local_maxima".to_string(),
                    swim_range: None,
                    tof_range: None,
                }),
                remove_stripes: None,
                top_n: Some(TopNConfig {
                    n_peaks: 1200,
                    most_peaks_per_streak: 5,
                    streak_half_width: 13,
                    harmonic_proportional_tolerance: 1.0,
                }),
                centroid: None,
                isotope: None,
                ransac: None,
                precise_autocorrelation: None,
                precursors: None,
                fragments: None,
            },
            origin: RunOrigin::Local,
            status: RunStatus::Completed,
            started_at: "2026-03-30T10:00:00Z".to_string(),
            completed_at: Some("2026-03-30T10:05:00Z".to_string()),
            error: None,
            summary: Some(RunSummary {
                raw_peaks: Some(14_000_000),
                final_peaks: 847,
                precursors: Some(200),
                ransac_slope: Some(1.0023),
                ransac_intercept: Some(-0.5),
            }),
            store_intermediates: false,
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let roundtrip: RunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.run_id, "a1b2c3d4");
        assert_eq!(roundtrip.status, RunStatus::Completed);
        assert_eq!(roundtrip.origin, RunOrigin::Local);
        assert_eq!(roundtrip.summary.as_ref().unwrap().final_peaks, 847);
    }

    #[test]
    fn test_peak_list_serialisation_roundtrip() {
        let peaks = PeakList {
            peaks: vec![
                Peak {
                    tof_mass: 100.5,
                    swim_mass: 200.3,
                    amplitude: 1500.0,
                    charge: 2,
                    inlier: true,
                },
                Peak {
                    tof_mass: 150.1,
                    swim_mass: 250.7,
                    amplitude: 800.0,
                    charge: 1,
                    inlier: false,
                },
            ],
            ransac: Some(RansacFit {
                slope: 1.0023,
                intercept: -0.5,
            }),
            rebased: true,
            peak_count: 2,
        };

        let json = serde_json::to_string_pretty(&peaks).unwrap();
        let roundtrip: PeakList = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.peak_count, 2);
        assert_eq!(roundtrip.peaks.len(), 2);
        assert!(roundtrip.rebased);
        assert_eq!(roundtrip.peaks[0].tof_mass, 100.5);
        assert_eq!(roundtrip.peaks[1].charge, 1);
        assert!(!roundtrip.peaks[1].inlier);
    }

    #[test]
    fn test_step_intermediate_serialisation() {
        let step = StepIntermediate {
            step_number: 2,
            step_name: "find-filter".to_string(),
            duration_ms: 1234,
            output: serde_json::json!({"filtered_peaks": 50000}),
            summary: serde_json::json!({"raw_peaks": 14200000, "filtered_peaks": 50000}),
        };

        let json = serde_json::to_string_pretty(&step).unwrap();
        let roundtrip: StepIntermediate = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.step_number, 2);
        assert_eq!(roundtrip.step_name, "find-filter");
        assert_eq!(roundtrip.duration_ms, 1234);
    }

    #[test]
    fn test_write_and_read_run() {
        let tmp = std::env::temp_dir().join(format!("sherlock_test_{}", generate_run_id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let manifest = RunManifest {
            run_id: "test1234".to_string(),
            dataset: tmp.to_string_lossy().to_string(),
            algorithm: "local-rust-v1".to_string(),
            config: RunConfig {
                calibrate: None, find_filter: None, remove_stripes: None,
                top_n: None, centroid: None, isotope: None, ransac: None,
                precise_autocorrelation: None, precursors: None, fragments: None,
            },
            origin: RunOrigin::Local,
            status: RunStatus::Completed,
            started_at: "2026-03-30T10:00:00Z".to_string(),
            completed_at: Some("2026-03-30T10:05:00Z".to_string()),
            error: None,
            summary: Some(RunSummary {
                raw_peaks: None, final_peaks: 2, precursors: None,
                ransac_slope: None, ransac_intercept: None,
            }),
            store_intermediates: true,
        };

        let peaks = PeakList {
            peaks: vec![Peak {
                tof_mass: 100.0, swim_mass: 200.0, amplitude: 500.0,
                charge: 1, inlier: true,
            }],
            ransac: None,
            rebased: false,
            peak_count: 1,
        };

        let intermediates = vec![StepIntermediate {
            step_number: 1,
            step_name: "calibrate".to_string(),
            duration_ms: 10,
            output: serde_json::json!({}),
            summary: serde_json::json!({"scale": 1.0}),
        }];

        // Write
        let run_path = write_run_to_disk(&tmp, &manifest, &peaks, Some(&intermediates)).unwrap();
        assert!(run_path.join("manifest.json").exists());
        assert!(run_path.join("peaks.json").exists());
        assert!(run_path.join("intermediates/01-calibrate.json").exists());

        // List
        let runs = list_runs(&tmp).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, "test1234");

        // Load peaks
        let loaded_peaks = load_run_peaks(&tmp, "test1234").unwrap();
        assert_eq!(loaded_peaks.peak_count, 1);

        // Load intermediates
        let loaded_int = load_run_intermediates(&tmp, "test1234").unwrap();
        assert!(loaded_int.contains_key("calibrate"));

        // Delete
        delete_run(&tmp, "test1234").unwrap();
        let runs = list_runs(&tmp).unwrap();
        assert_eq!(runs.len(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
