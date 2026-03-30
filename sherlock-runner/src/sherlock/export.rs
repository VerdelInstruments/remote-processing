use serde::Serialize;
use std::io::Write;

use crate::dataset::DatasetState;
use super::run::PeakList;
use super::SherlockState;

#[derive(Debug, Serialize)]
pub struct ExportResult {
    pub path: String,
    pub peak_count: usize,
    pub format: String,
    /// Whether swim coordinates were rebased using the RANSAC/precise calibration.
    pub rebased: bool,
}

#[derive(Serialize)]
struct ExportPeak {
    tof_mass: f64,
    swim_mass: f64,
    amplitude: f64,
    charge: i32,
    inlier: bool,
}

#[derive(Serialize)]
struct ExportJson {
    peaks: Vec<ExportPeak>,
    ransac: Option<RansacInfo>,
    rebased: bool,
}

#[derive(Serialize)]
struct RansacInfo {
    slope: f64,
    intercept: f64,
}

/// Rebase a raw swim mass coordinate using the RANSAC/precise autocorrelation fit.
///
/// The calibration line is: swim_raw = slope * tof_mass + intercept
/// Inverting: swim_calibrated = (swim_raw - intercept) / slope
///
/// This maps from the raw frequency-derived swim mass into the calibrated
/// coordinate system where swim_mass ~ tof_mass along the autocorrelation line.
///
/// ## Equivalence to production rebase
///
/// Production reloads the dataset with updated calibration
/// (`cal.update_with_fit(slope, intercept)`) then maps each peak's swim_idx
/// to the new coordinate grid: `swim_mass = swim_axis[swim_idx]`.
///
/// The updated calibration's `f_to_m` computes:
///   `y_prime = scale / (freq + offset)`   (initial raw mass)
///   `edited = (y_prime - new_c) / new_m`
///
/// Where `update_with_fit(slope, intercept)` sets `new_m = 1.0 * slope`,
/// `new_c = 0.0 + intercept` (from initial m=1, c=0). So:
///   `edited = (raw_mass - intercept) / slope`
///
/// This is exactly `rebase_swim(raw_mass, slope, intercept)`. The linear
/// transform is therefore mathematically equivalent to the production rebase
/// when the initial calibration has m=1, c=0 (which is always the case for
/// fresh datasets). No non-linear rebase is needed.
fn rebase_swim(raw_swim: f64, slope: f64, intercept: f64) -> f64 {
    (raw_swim - intercept) / slope
}

pub fn export(
    ds: &DatasetState,
    sh: &SherlockState,
    format: &str,
    rebase: bool,
) -> Result<ExportResult, String> {
    let stripe_mask = sh.top_n_mask.as_ref()
        .or(sh.stripe_mask.as_ref())
        .ok_or("No peak mask -- run top_n_peaks or remove_stripes first")?;
    let peak_amps = sh.peak_amplitudes.as_ref().ok_or("No amplitudes")?;
    let peak_rows = sh.peak_row_idx.as_ref().ok_or("No peaks")?;
    let peak_cols = sh.peak_col_idx.as_ref().ok_or("No peaks")?;
    let tof_coords = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;
    let swim_coords = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;

    let surviving: Vec<usize> = (0..stripe_mask.len())
        .filter(|&i| stripe_mask[i])
        .collect();

    let n = surviving.len();

    // Use centroids if available, otherwise grid positions
    let default_tof: Vec<f64> = surviving.iter().map(|&i| tof_coords[peak_cols[i]]).collect();
    let default_swim: Vec<f64> = surviving.iter().map(|&i| swim_coords[peak_rows[i]]).collect();
    let c_tof = sh.centroids_tof.as_ref().unwrap_or(&default_tof);
    let c_swim = sh.centroids_swim.as_ref().unwrap_or(&default_swim);

    let default_charges = vec![0i32; n];
    let charges = sh.charges.as_ref().unwrap_or(&default_charges);

    // Prefer precise autocorrelation (step 8), fall back to rough RANSAC (step 7)
    let default_inliers = vec![true; n];
    let inliers = sh
        .precise_inlier_mask
        .as_ref()
        .or(sh.ransac_inlier_mask.as_ref())
        .unwrap_or(&default_inliers);

    // Determine rebase parameters: prefer precise fit, fall back to rough RANSAC.
    // Only rebase if requested AND a fit is available.
    let rebase_params: Option<(f64, f64)> = if rebase {
        sh.precise_slope
            .zip(sh.precise_intercept)
            .or_else(|| sh.ransac_slope.zip(sh.ransac_intercept))
            .filter(|(slope, _)| slope.abs() > f64::EPSILON)
    } else {
        None
    };
    let rebased = rebase_params.is_some();

    let peaks: Vec<ExportPeak> = (0..n)
        .map(|i| {
            let raw_swim = c_swim[i];
            let swim_mass = match rebase_params {
                Some((slope, intercept)) => rebase_swim(raw_swim, slope, intercept),
                None => raw_swim,
            };
            ExportPeak {
                tof_mass: c_tof[i],
                swim_mass,
                amplitude: peak_amps[surviving[i]],
                charge: *charges.get(i).unwrap_or(&0),
                inlier: *inliers.get(i).unwrap_or(&true),
            }
        })
        .collect();

    let out_dir = ds
        .directory
        .as_ref()
        .ok_or("No dataset directory")?;

    // Prefer precise fit for the reported RANSAC info, fall back to rough RANSAC
    let ransac_info = sh
        .precise_slope
        .zip(sh.precise_intercept)
        .or_else(|| sh.ransac_slope.zip(sh.ransac_intercept))
        .map(|(s, i)| RansacInfo {
            slope: s,
            intercept: i,
        });

    let (filename, content) = match format {
        "json" => {
            let export = ExportJson {
                peaks,
                ransac: ransac_info,
                rebased,
            };
            let json = serde_json::to_string_pretty(&export).map_err(|e| e.to_string())?;
            ("sherlock_peaks.json".to_string(), json)
        }
        _ => {
            // CSV
            let mut buf = String::from("tof_mass,swim_mass,amplitude,charge,inlier\n");
            for p in &peaks {
                buf.push_str(&format!(
                    "{},{},{},{},{}\n",
                    p.tof_mass, p.swim_mass, p.amplitude, p.charge, p.inlier
                ));
            }
            ("sherlock_peaks.csv".to_string(), buf)
        }
    };

    let out_path = out_dir.join(&filename);
    let mut file = std::fs::File::create(&out_path)
        .map_err(|e| format!("Failed to create {}: {}", out_path.display(), e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    Ok(ExportResult {
        path: out_path.to_string_lossy().to_string(),
        peak_count: n,
        format: format.to_string(),
        rebased,
    })
}

/// Export fragments (production step 13).
///
/// This exports the fragments built by `get_fragments` (production step 12).
/// Fragment coordinates are already calibrated (swim coordinates were rebased
/// during the fragments step using the RANSAC/precise fit), so no additional
/// rebase is applied.
///
/// Production step 13 flattens the per-swim-idx fragment dict, rebases peaks
/// to the calibrated dataset, and writes CSV. Since the fragments step already
/// applies the equivalent rebase (`calibrate_swim`), this function simply
/// writes the stored fragment data to disk.
pub fn export_fragments(
    ds: &DatasetState,
    sh: &SherlockState,
    format: &str,
) -> Result<ExportResult, String> {
    let frag_tof = sh.fragment_tof.as_ref()
        .ok_or("No fragment data -- run get_fragments first")?;
    let frag_swim = sh.fragment_swim.as_ref()
        .ok_or("No fragment swim data")?;
    let frag_charges = sh.fragment_charges.as_ref()
        .ok_or("No fragment charges")?;
    let frag_swim_idx = sh.fragment_swim_idx.as_ref()
        .ok_or("No fragment swim indices")?;

    let n = frag_tof.len();

    // Prefer precise fit for reported RANSAC info, fall back to rough RANSAC
    let ransac_info = sh
        .precise_slope
        .zip(sh.precise_intercept)
        .or_else(|| sh.ransac_slope.zip(sh.ransac_intercept))
        .map(|(s, i)| RansacInfo {
            slope: s,
            intercept: i,
        });

    let peaks: Vec<ExportPeak> = (0..n)
        .map(|i| ExportPeak {
            tof_mass: frag_tof[i],
            swim_mass: frag_swim[i],
            amplitude: 0.0, // fragments don't store amplitude separately
            charge: frag_charges[i],
            inlier: true,
        })
        .collect();

    let out_dir = ds
        .directory
        .as_ref()
        .ok_or("No dataset directory")?;

    let (filename, content) = match format {
        "json" => {
            let export = ExportJson {
                peaks,
                ransac: ransac_info,
                rebased: true, // fragments are already calibrated
            };
            let json = serde_json::to_string_pretty(&export).map_err(|e| e.to_string())?;
            ("sherlock_fragments.json".to_string(), json)
        }
        _ => {
            // CSV with swim_idx column for fragment grouping
            let mut buf = String::from("swim_idx,tof_mass,swim_mass,charge\n");
            for i in 0..n {
                buf.push_str(&format!(
                    "{},{},{},{}\n",
                    frag_swim_idx[i], frag_tof[i], frag_swim[i], frag_charges[i],
                ));
            }
            ("sherlock_fragments.csv".to_string(), buf)
        }
    };

    let out_path = out_dir.join(&filename);
    let mut file = std::fs::File::create(&out_path)
        .map_err(|e| format!("Failed to create {}: {}", out_path.display(), e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    Ok(ExportResult {
        path: out_path.to_string_lossy().to_string(),
        peak_count: n,
        format: format.to_string(),
        rebased: true,
    })
}

/// Export CSV from a saved run's PeakList (task 3.3).
/// Generates a CSV in the existing format from a run's peaks.json.
pub fn export_csv_from_run(
    peak_list: &PeakList,
    output_dir: &std::path::Path,
) -> Result<ExportResult, String> {
    let mut buf = String::from("tof_mass,swim_mass,amplitude,charge,inlier\n");
    for p in &peak_list.peaks {
        buf.push_str(&format!(
            "{},{},{},{},{}\n",
            p.tof_mass, p.swim_mass, p.amplitude, p.charge, p.inlier
        ));
    }

    let out_path = output_dir.join("sherlock_peaks.csv");
    let mut file = std::fs::File::create(&out_path)
        .map_err(|e| format!("Failed to create {}: {}", out_path.display(), e))?;
    file.write_all(buf.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;

    Ok(ExportResult {
        path: out_path.to_string_lossy().to_string(),
        peak_count: peak_list.peak_count,
        format: "csv".to_string(),
        rebased: peak_list.rebased,
    })
}
