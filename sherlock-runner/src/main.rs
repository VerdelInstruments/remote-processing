//! Sherlock Lambda handler — runs the Rust mass spectrometry pipeline on AWS Lambda.
//!
//! Input: JSON with S3 path to .nc file + pipeline config
//! Output: JSON with run manifest + S3 path to results

use aws_sdk_s3::Client as S3Client;
use lambda_runtime::{service_fn, Error, LambdaEvent};
use log::info;
use serde::{Deserialize, Serialize};
use std::time::Instant;

mod dataset;
mod sherlock;

use dataset::DatasetState;
use sherlock::SherlockState;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SherlockRequest {
    /// S3 URI: s3://bucket/key/to/file.nc
    s3_input_path: String,
    /// S3 prefix for results: s3://bucket/runs/run-abc123/
    s3_output_prefix: String,
    /// Pipeline configuration
    config: PipelineConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct PipelineConfig {
    // Algorithm variant: "rust-native" or "rust-sklearn-bridge"
    #[serde(default = "default_variant")]
    algorithm_variant: String,
    // Calibration
    #[serde(default = "default_scale")]
    scale: f64,
    #[serde(default = "default_offset")]
    offset: f64,
    // Find peaks
    #[serde(default = "default_peak_filter_size")]
    peak_filter_size: usize,
    #[serde(default = "default_swim_min")]
    swim_min: f64,
    #[serde(default = "default_swim_max")]
    swim_max: f64,
    #[serde(default = "default_tof_min")]
    tof_min: f64,
    #[serde(default = "default_tof_max")]
    tof_max: f64,
    // Top N
    #[serde(default = "default_n_peaks")]
    n_peaks: usize,
    #[serde(default = "default_most_peaks_per_streak")]
    most_peaks_per_streak: usize,
    #[serde(default = "default_streak_half_width")]
    streak_half_width: usize,
    #[serde(default = "default_harmonic_tolerance")]
    harmonic_proportional_tolerance: f64,
    // Centroid + Isotope
    #[serde(default = "default_half_window_tof")]
    half_window_tof: usize,
    #[serde(default = "default_half_window_swim")]
    half_window_swim: usize,
    #[serde(default = "default_isotope_tolerance")]
    isotope_tolerance: f64,
    #[serde(default = "default_max_charge")]
    max_charge: usize,
    #[serde(default = "default_row_half_width")]
    row_half_width: usize,
    // RANSAC
    #[serde(default = "default_ransac_tolerance")]
    ransac_tolerance: f64,
    #[serde(default = "default_ransac_n_peaks")]
    ransac_n_peaks: usize,
    // Noise
    #[serde(default = "default_min_snr")]
    min_snr: f64,
    #[serde(default = "default_noise_window")]
    noise_window: usize,
    // Precursors
    #[serde(default = "default_amplitude_std_multiplier")]
    amplitude_std_multiplier: f64,
    #[serde(default = "default_gulley_width")]
    gulley_width_amu: f64,
    #[serde(default = "default_autocorrelation_tolerance")]
    autocorrelation_tolerance: f64,
}

fn default_variant() -> String { "rust-sklearn-bridge".to_string() }
fn default_scale() -> f64 { 39.0 }
fn default_offset() -> f64 { 0.01 }
fn default_peak_filter_size() -> usize { 5 }
fn default_swim_min() -> f64 { 100.0 }
fn default_swim_max() -> f64 { 1800.0 }
fn default_tof_min() -> f64 { 100.0 }
fn default_tof_max() -> f64 { 1800.0 }
fn default_n_peaks() -> usize { 1200 }
fn default_most_peaks_per_streak() -> usize { 5 }
fn default_streak_half_width() -> usize { 13 }
fn default_harmonic_tolerance() -> f64 { 1.0 }
fn default_half_window_tof() -> usize { 4 }
fn default_half_window_swim() -> usize { 0 }
fn default_isotope_tolerance() -> f64 { 0.1 }
fn default_max_charge() -> usize { 6 }
fn default_row_half_width() -> usize { 4 }
fn default_ransac_tolerance() -> f64 { 1.5 }
fn default_ransac_n_peaks() -> usize { 15 }
fn default_min_snr() -> f64 { 70.0 }
fn default_noise_window() -> usize { 51 }
fn default_amplitude_std_multiplier() -> f64 { 5.0 }
fn default_gulley_width() -> f64 { 3.0 }
fn default_autocorrelation_tolerance() -> f64 { 3.0 }

#[derive(Debug, Serialize)]
struct SherlockResponse {
    run_id: String,
    status: String,
    algorithm_variant: String,
    peak_count: usize,
    duration_ms: u64,
    results_prefix: String,
    steps: Vec<StepTiming>,
}

#[derive(Debug, Serialize)]
struct StepTiming {
    name: String,
    duration_ms: u64,
}

// ---------------------------------------------------------------------------
// S3 helpers
// ---------------------------------------------------------------------------

/// Parse s3://bucket/key into (bucket, key)
fn parse_s3_uri(uri: &str) -> Result<(String, String), String> {
    let stripped = uri.strip_prefix("s3://").ok_or("Invalid S3 URI: must start with s3://")?;
    let slash = stripped.find('/').ok_or("Invalid S3 URI: no key after bucket")?;
    Ok((stripped[..slash].to_string(), stripped[slash + 1..].to_string()))
}

async fn download_from_s3(client: &S3Client, s3_uri: &str, local_path: &str) -> Result<(), String> {
    use tokio::io::AsyncWriteExt;

    let (bucket, key) = parse_s3_uri(s3_uri)?;
    info!("[s3] downloading s3://{}/{} → {}", bucket, key, local_path);
    let t0 = Instant::now();

    let resp = client
        .get_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
        .map_err(|e| format!("S3 download error: {}", e))?;

    // Stream to disk — never hold the full file in RAM
    let mut file = tokio::fs::File::create(local_path)
        .await
        .map_err(|e| format!("Failed to create {}: {}", local_path, e))?;

    let mut stream = resp.body;
    let mut written: u64 = 0;
    while let Some(chunk) = stream
        .try_next()
        .await
        .map_err(|e| format!("S3 stream error: {}", e))?
    {
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        written += chunk.len() as u64;
    }
    file.flush().await.map_err(|e| format!("Flush error: {}", e))?;

    info!("[s3] streamed {:.0} MB to disk in {}ms", written as f64 / 1e6, t0.elapsed().as_millis());
    Ok(())
}

async fn upload_to_s3(client: &S3Client, s3_uri: &str, data: &[u8]) -> Result<(), String> {
    let (bucket, key) = parse_s3_uri(s3_uri)?;
    info!("[s3] uploading {} bytes → s3://{}/{}", data.len(), bucket, key);

    client
        .put_object()
        .bucket(&bucket)
        .key(&key)
        .body(data.to_vec().into())
        .content_type("application/json")
        .send()
        .await
        .map_err(|e| format!("S3 upload error: {}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Pipeline execution
// ---------------------------------------------------------------------------

fn run_pipeline(ds: &mut DatasetState, sh: &mut SherlockState, config: &PipelineConfig) -> Result<Vec<StepTiming>, String> {
    let mut timings = Vec::new();

    macro_rules! step {
        ($name:expr, $body:expr) => {{
            let t = Instant::now();
            let result = $body;
            timings.push(StepTiming {
                name: $name.to_string(),
                duration_ms: t.elapsed().as_millis() as u64,
            });
            info!("[pipeline] {} completed in {}ms", $name, t.elapsed().as_millis());
            result?
        }};
    }

    // Step 1: Calibrate
    step!("calibrate", sherlock::calibrate::calibrate(
        ds, sh, config.scale, config.offset
    ));

    // Step 2: Find all peaks
    step!("find_peaks", sherlock::find_filter::find_peaks_raw(
        ds, sh, config.peak_filter_size,
        Some((config.swim_min, config.swim_max)),
        Some((config.tof_min, config.tof_max)),
    ));

    // Step 3: Top N peaks
    step!("top_n_peaks", sherlock::stripes::top_n_peaks(
        ds, sh, config.n_peaks, config.most_peaks_per_streak,
        config.streak_half_width, config.harmonic_proportional_tolerance,
    ));

    // Step 4: Centroid + swim group + isotope
    step!("centroid_isotope", {
        sherlock::swim_group::group_by_swim_idx(sh)?;
        sherlock::centroid::centroid(ds, sh, config.half_window_tof, config.half_window_swim)?;
        sherlock::swim_group::per_group_isotope_detect(
            ds, sh, config.row_half_width, config.isotope_tolerance, config.max_charge,
        )
    });

    // Step 5: Rough RANSAC
    step!("ransac_rough", sherlock::ransac::ransac(
        ds, sh, config.ransac_tolerance, config.ransac_n_peaks,
    ));

    // Step 6: Precise autocorrelation
    step!("ransac_precise", sherlock::ransac::precise_autocorrelation(
        ds, sh, config.ransac_tolerance, config.ransac_n_peaks,
        config.half_window_tof, config.half_window_swim,
    ));

    // Step 7: Peaks above noise
    step!("peaks_above_noise", sherlock::peaks_above_noise::peaks_above_noise(
        ds, sh, config.min_snr, config.noise_window,
    ));

    // Step 8: Precursors
    step!("precursors", sherlock::precursors::get_precursors(
        ds, sh,
        config.amplitude_std_multiplier, config.gulley_width_amu,
        config.isotope_tolerance, config.max_charge,
        config.half_window_tof, config.half_window_swim,
        config.autocorrelation_tolerance,
    ));

    // Step 9: Fragments
    step!("fragments", sherlock::fragments::get_fragments(
        ds, sh, config.isotope_tolerance, config.max_charge,
        config.half_window_tof, config.half_window_swim,
    ));

    Ok(timings)
}

// ---------------------------------------------------------------------------
// Lambda handler
// ---------------------------------------------------------------------------

async fn handler(event: LambdaEvent<SherlockRequest>) -> Result<SherlockResponse, Error> {
    let req = event.payload;
    let t0 = Instant::now();

    info!("[handler] input={}, output={}", req.s3_input_path, req.s3_output_prefix);

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let s3 = S3Client::new(&config);

    // Download .nc to /tmp
    let local_nc = "/tmp/input.nc";
    download_from_s3(&s3, &req.s3_input_path, local_nc).await?;

    // Load dataset into memory
    let mut ds = DatasetState::default();
    dataset::scan_directory(&mut ds, local_nc)?;
    dataset::load_nc(&mut ds)?;
    // Keep the .nc file on disk for optional Python diagnostic bridge.
    // Deleted after diagnostics complete (end of handler).

    info!("[handler] dataset loaded: {:?}", ds.shape);

    // Run pipeline
    let mut sh = SherlockState::default();
    sh.use_sklearn_ransac = req.config.algorithm_variant == "rust-sklearn-bridge";
    info!("[handler] algorithm_variant={}, use_sklearn={}", req.config.algorithm_variant, sh.use_sklearn_ransac);
    let timings = run_pipeline(&mut ds, &mut sh, &req.config)?;

    // Build result set
    let run_id = sherlock::run::generate_run_id();
    let peaks = build_peak_list(&ds, &sh)?;
    let peak_count = peaks.peak_count;

    // Upload results to S3
    let manifest_json = serde_json::to_string_pretty(&serde_json::json!({
        "run_id": run_id,
        "algorithm": "cloud-rust-v1",
        "origin": "cloud",
        "status": "completed",
        "peak_count": peak_count,
        "duration_ms": t0.elapsed().as_millis() as u64,
        "steps": timings,
    }))?;
    let peaks_json = serde_json::to_string_pretty(&peaks)?;

    let prefix = req.s3_output_prefix.trim_end_matches('/');
    upload_to_s3(&s3, &format!("{}/manifest.json", prefix), manifest_json.as_bytes()).await?;
    upload_to_s3(&s3, &format!("{}/peaks.json", prefix), peaks_json.as_bytes()).await?;

    // Upload diagnostic dump if it exists (from rough_binned_fit)
    if let Ok(diag) = std::fs::read("/tmp/rough_binned_fit_diagnostic.json") {
        upload_to_s3(&s3, &format!("{}/diagnostic_rust_binning.json", prefix), &diag).await?;
        info!("[handler] uploaded Rust binning diagnostic");
    }

    // Run Python binning bridge for comparison (if available)
    {
        use std::process::{Command, Stdio};
        use std::io::Write as IoWrite;

        // Use the original .nc file still on disk
        {
            let bridge_req = serde_json::json!({
                "nc_path": local_nc,
                "n_bins": 5000,
                "swim_range": [400, 1000],
                "tof_range": [400, 1000],
            });

            let py_result = Command::new("python3")
                .arg("/opt/binning_bridge.py")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .and_then(|mut child| {
                    if let Some(ref mut stdin) = child.stdin {
                        let _ = stdin.write_all(bridge_req.to_string().as_bytes());
                    }
                    child.wait_with_output()
                });

            match py_result {
                Ok(output) if output.status.success() => {
                    upload_to_s3(&s3, &format!("{}/diagnostic_python_binning.json", prefix), &output.stdout).await?;
                    info!("[handler] uploaded Python binning diagnostic");
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    info!("[handler] Python binning bridge failed: {}", stderr);
                }
                Err(e) => info!("[handler] Python binning bridge not available: {}", e),
            }
        }
    }

    // Clean up /tmp
    let _ = std::fs::remove_file(local_nc);
    let _ = std::fs::remove_file("/tmp/rough_binned_fit_diagnostic.json");

    let duration_ms = t0.elapsed().as_millis() as u64;
    info!("[handler] complete: {} peaks, {}ms", peak_count, duration_ms);

    Ok(SherlockResponse {
        run_id,
        status: "completed".to_string(),
        algorithm_variant: req.config.algorithm_variant.clone(),
        peak_count,
        duration_ms,
        results_prefix: prefix.to_string(),
        steps: timings,
    })
}

fn build_peak_list(
    _ds: &DatasetState,
    sh: &SherlockState,
) -> Result<sherlock::run::PeakList, String> {
    // Use centroids as the authoritative peak set — these are the refined
    // positions from the centroid step (matching what compare uses).
    // peaks_above_noise may have replaced peak_amplitudes with a different
    // set, but centroids are untouched.
    let c_tof = sh.centroids_tof.as_ref().ok_or("No centroids — run centroid step first")?;
    let c_swim = sh.centroids_swim.as_ref().ok_or("No centroids")?;
    let n = c_tof.len();

    let default_charges = vec![0i32; n];
    let charges = sh.charges.as_ref().unwrap_or(&default_charges);

    let default_inliers = vec![true; n];
    let inliers = sh.precise_inlier_mask.as_ref()
        .or(sh.ransac_inlier_mask.as_ref())
        .unwrap_or(&default_inliers);

    // Rebase swim using precise/rough RANSAC fit
    let rebase_params = sh.precise_slope.zip(sh.precise_intercept)
        .or_else(|| sh.ransac_slope.zip(sh.ransac_intercept))
        .filter(|(slope, _)| slope.abs() > f64::EPSILON);
    let rebased = rebase_params.is_some();

    let ransac_fit = sh.precise_slope.zip(sh.precise_intercept)
        .or_else(|| sh.ransac_slope.zip(sh.ransac_intercept))
        .map(|(s, i)| sherlock::run::RansacFit { slope: s, intercept: i });

    // Amplitude: use peak_amplitudes if it matches centroid count, else 0
    let peak_amps = sh.peak_amplitudes.as_ref();
    let amps_aligned = peak_amps.map_or(false, |a| a.len() == n);

    let peaks: Vec<sherlock::run::Peak> = (0..n)
        .map(|i| {
            let raw_swim = c_swim[i];
            let swim_mass = match rebase_params {
                Some((slope, intercept)) => (raw_swim - intercept) / slope,
                None => raw_swim,
            };
            sherlock::run::Peak {
                tof_mass: c_tof[i],
                swim_mass,
                amplitude: if amps_aligned { peak_amps.unwrap()[i] } else { 0.0 },
                charge: *charges.get(i).unwrap_or(&0),
                inlier: *inliers.get(i).unwrap_or(&true),
            }
        })
        .collect();

    Ok(sherlock::run::PeakList {
        peak_count: peaks.len(),
        peaks,
        ransac: ransac_fit,
        rebased,
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();
    lambda_runtime::run(service_fn(handler)).await
}
