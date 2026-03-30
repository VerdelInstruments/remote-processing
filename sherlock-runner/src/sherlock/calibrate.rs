use log::info;
use serde::Serialize;

use crate::dataset::DatasetState;
use super::{CalibrationParams, SherlockState};

#[derive(Debug, Serialize)]
pub struct CalibrateResult {
    pub freq: Vec<f64>,
    pub mass: Vec<f64>,
    pub swim_range: (f64, f64),
    pub tof_range: (f64, f64),
    pub shape: (usize, usize),
    pub note: Option<String>,
}

pub fn calibrate(
    ds: &mut DatasetState,
    sh: &mut SherlockState,
    scale: f64,
    offset: f64,
) -> Result<CalibrateResult, String> {
    sh.calibration = Some(CalibrationParams { scale, offset });
    sh.invalidate_from(2);

    let tof = ds.tof_coords.as_ref().ok_or("No tof coordinates")?;
    let tof_range = (*tof.first().unwrap_or(&0.0), *tof.last().unwrap_or(&0.0));

    if ds.is_frequency {
        let raw_freq = ds
            .raw_freq_coords
            .as_ref()
            .ok_or("No raw frequency coordinates")?;

        // Recompute swim coordinates using user's calibration: mass = scale / (freq + offset)
        let mut new_swim: Vec<f64> = raw_freq.iter().map(|f| scale / (f + offset)).collect();
        new_swim.reverse(); // frequency is descending mass — reverse to ascending

        info!(
            "[calibrate] recalculated swim coords: scale={}, offset={}, range=[{:.2}, {:.2}]",
            scale, offset,
            new_swim.first().unwrap_or(&0.0),
            new_swim.last().unwrap_or(&0.0),
        );

        let swim_range = (
            *new_swim.first().unwrap_or(&0.0),
            *new_swim.last().unwrap_or(&0.0),
        );

        // Preview curve (subsampled)
        let step = (raw_freq.len() / 200).max(1);
        let freq_sub: Vec<f64> = raw_freq.iter().step_by(step).copied().collect();
        let mass_sub: Vec<f64> = freq_sub.iter().map(|f| scale / (f + offset)).collect();

        // Update the dataset's swim coordinates so all downstream steps use them
        ds.swim_coords = Some(new_swim);

        Ok(CalibrateResult {
            freq: freq_sub,
            mass: mass_sub,
            swim_range,
            tof_range,
            shape: ds.shape,
            note: None,
        })
    } else {
        let swim = ds.swim_coords.as_ref().ok_or("No swim coordinates")?;
        let swim_range = (*swim.first().unwrap_or(&0.0), *swim.last().unwrap_or(&0.0));

        Ok(CalibrateResult {
            freq: vec![],
            mass: vec![],
            swim_range,
            tof_range,
            shape: ds.shape,
            note: Some("Data already in mass coordinates — no conversion needed".to_string()),
        })
    }
}
