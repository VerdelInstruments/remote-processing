//! Dataset loading for Lambda — loads NetCDF from local /tmp path.
//!
//! Stripped-down version of 2discover's dataset.rs:
//! - No Tauri managed state
//! - No rusqlite/csv peak loading (not needed for pipeline execution)
//! - No overview/slice/bin (UI-only functions)

use log::info;
use ndarray::Array2;
use netcdf::Extents;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Peak type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    pub swim_mass: f64,
    pub tof_mass: f64,
    pub amplitude: f64,
    pub centroid: Option<f64>,
}

// ---------------------------------------------------------------------------
// Dataset state
// ---------------------------------------------------------------------------

pub struct DatasetState {
    pub data: Option<Array2<f32>>,
    pub swim_coords: Option<Vec<f64>>,
    pub tof_coords: Option<Vec<f64>>,
    pub raw_freq_coords: Option<Vec<f64>>,
    pub is_frequency: bool,
    pub y_dim: String,
    pub x_dim: String,
    pub name: String,
    pub shape: (usize, usize),
    pub nc_path: Option<PathBuf>,
    pub directory: Option<PathBuf>,
    pub loaded: bool,
    pub peaks: Vec<Peak>,
    pub peaks_source: Option<String>,
}

impl Default for DatasetState {
    fn default() -> Self {
        Self {
            data: None,
            swim_coords: None,
            tof_coords: None,
            raw_freq_coords: None,
            is_frequency: false,
            y_dim: String::new(),
            x_dim: String::new(),
            name: String::new(),
            shape: (0, 0),
            nc_path: None,
            directory: None,
            loaded: false,
            peaks: Vec::new(),
            peaks_source: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scan — find .nc file in a directory
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ScanResult {
    pub name: String,
    pub shape: (usize, usize),
    pub has_nc: bool,
    pub nc_size: u64,
}

pub fn scan_directory(state: &mut DatasetState, path: &str) -> Result<ScanResult, String> {
    let t0 = std::time::Instant::now();

    let input = PathBuf::from(path);
    let (directory, nc_path) = if input.is_file() && input.extension().map_or(false, |e| e == "nc") {
        // Direct .nc path
        let dir = input.parent().unwrap_or(&input).to_path_buf();
        (dir, Some(input.clone()))
    } else {
        // Directory — find .nc inside
        let nc = find_nc_file(&input);
        (input.clone(), nc)
    };

    state.directory = Some(directory.clone());
    state.name = directory
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let has_nc = nc_path.is_some();
    let nc_size = nc_path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    state.nc_path = nc_path;

    info!(
        "[dataset] scan: nc={} ({:.0} MB), {}ms",
        has_nc,
        nc_size as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_millis()
    );

    Ok(ScanResult {
        name: state.name.clone(),
        shape: state.shape,
        has_nc,
        nc_size,
    })
}

// ---------------------------------------------------------------------------
// Load — read .nc into RAM
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct LoadResult {
    pub shape: (usize, usize),
    pub load_time_ms: u64,
}

pub fn load_nc(state: &mut DatasetState) -> Result<LoadResult, String> {
    if state.nc_path.is_none() {
        return Err("No .nc path — call scan first".into());
    }
    if state.loaded {
        return Ok(LoadResult {
            shape: state.shape,
            load_time_ms: 0,
        });
    }

    let nc_path = state.nc_path.as_ref().unwrap().clone();
    let t0 = std::time::Instant::now();

    let file = netcdf::open(&nc_path).map_err(|e| format!("NetCDF open error: {}", e))?;
    info!("[dataset] netcdf::open: {}ms", t0.elapsed().as_millis());

    let estimated_mem: u64 = file
        .variables()
        .find(|v| v.dimensions().len() == 2)
        .map(|v| v.dimensions().iter().map(|d| d.len() as u64).product())
        .unwrap_or(0)
        * 4;

    info!(
        "[dataset] loading {} (est {:.0} MB in RAM)",
        nc_path.display(),
        estimated_mem as f64 / (1024.0 * 1024.0)
    );

    let var_name = file
        .variables()
        .find(|v| v.dimensions().len() == 2)
        .map(|v| v.name().to_string())
        .ok_or("No 2D variable found in NetCDF file")?;

    let var = file
        .variable(&var_name)
        .ok_or(format!("Variable '{}' not found", var_name))?;

    let dims: Vec<_> = var.dimensions().iter().map(|d| d.name().to_string()).collect();
    if dims.len() != 2 {
        return Err(format!("Expected 2D variable, got {}D", dims.len()));
    }

    let y_dim = &dims[0];
    let x_dim = &dims[1];
    let y_len = var.dimensions()[0].len();
    let x_len = var.dimensions()[1].len();

    info!(
        "[dataset] variable '{}': {}x{} dims={}/{}",
        var_name, y_len, x_len, y_dim, x_dim
    );

    let t_read = std::time::Instant::now();
    let values: Vec<f32> = var
        .get::<f32, _>(Extents::All)
        .map_err(|e| format!("Failed to read data: {}", e))?
        .into_raw_vec();
    info!("[dataset] read data ({}x{}): {}ms", y_len, x_len, t_read.elapsed().as_millis());

    let mut data = Array2::from_shape_vec((y_len, x_len), values)
        .map_err(|e| format!("Array shape error: {}", e))?;

    let y_coords: Vec<f64> = file
        .variable(y_dim)
        .ok_or(format!("Coordinate '{}' not found", y_dim))?
        .get::<f64, _>(Extents::All)
        .map_err(|e| format!("Failed to read {} coords: {}", y_dim, e))?
        .into_raw_vec();

    let x_coords: Vec<f64> = file
        .variable(x_dim)
        .ok_or(format!("Coordinate '{}' not found", x_dim))?
        .get::<f64, _>(Extents::All)
        .map_err(|e| format!("Failed to read {} coords: {}", x_dim, e))?
        .into_raw_vec();

    let t_freq = std::time::Instant::now();
    let is_frequency = y_dim == "frequency";
    let (swim_coords, raw_freq, data) = if is_frequency {
        let freq_coords = y_coords.clone();
        let mut swim: Vec<f64> = y_coords.iter().map(|f| 39.0 / (f + 0.01)).collect();
        swim.reverse();
        let nrows = data.nrows();
        let ncols = data.ncols();
        let raw = data.as_slice_mut().expect("array not contiguous");
        for i in 0..nrows / 2 {
            let j = nrows - 1 - i;
            let (lo, hi) = raw.split_at_mut(j * ncols);
            lo[i * ncols..(i + 1) * ncols].swap_with_slice(&mut hi[..ncols]);
        }
        (swim, Some(freq_coords), data)
    } else {
        (y_coords, None, data)
    };

    if is_frequency {
        info!("[dataset] frequency→mass conversion: {}ms", t_freq.elapsed().as_millis());
    }

    state.y_dim = y_dim.clone();
    state.x_dim = x_dim.clone();
    state.is_frequency = is_frequency;
    state.swim_coords = Some(swim_coords);
    state.tof_coords = Some(x_coords);
    state.raw_freq_coords = raw_freq;
    state.shape = (data.nrows(), data.ncols());
    state.data = Some(data);
    state.loaded = true;

    let elapsed = t0.elapsed();
    info!(
        "[dataset] load complete: shape={:?}, dims={}/{}, freq={}, {}ms",
        state.shape, state.y_dim, state.x_dim, state.is_frequency, elapsed.as_millis()
    );

    Ok(LoadResult {
        shape: state.shape,
        load_time_ms: elapsed.as_millis() as u64,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_nc_file(dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut nc_files: Vec<PathBuf> = Vec::new();
        let mut fourier_files: Vec<PathBuf> = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with("_fourierdomain.nc") {
                    fourier_files.push(path);
                } else if name.ends_with(".nc") && name != "overview.nc" {
                    nc_files.push(path);
                }
            }
        }

        if let Some(f) = fourier_files.into_iter().next() {
            return Some(f);
        }
        if let Some(f) = nc_files.into_iter().next() {
            return Some(f);
        }
    }

    let overview = dir.join("overview.nc");
    if overview.exists() {
        return Some(overview);
    }

    None
}
