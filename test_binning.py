#!/usr/bin/env python3
"""
Unit test: compare Rust binning vs production (xarray) binning on synthetic data.

Creates a synthetic NetCDF with known peaks on the autocorrelation line,
runs both the Python (production) binning and dumps the expected output
for comparison with the Rust implementation.
"""
import json
import tempfile
import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter

def create_synthetic_nc(path, n_swim=1025, n_tof=50000, n_peaks=20, seed=42):
    """Create a synthetic NetCDF file with peaks on the autocorrelation line.
    n_swim=1025 matches production datasets (1025 frequency channels)."""
    rng = np.random.RandomState(seed)

    # Frequency axis: ascending (as stored in NetCDF).
    # Loading code reverses and converts: mass = 39/(freq+0.01), then [::-1]
    # Result: ascending mass from ~78 to ~1950
    freq = np.linspace(0.01, 0.5, n_swim)  # ascending freq in NC
    # Pre-compute the mass coordinates for peak placement
    # (what they'll be after freq→mass conversion + reversal)
    mass_from_freq_raw = 39.0 / (freq + 0.01)  # descending mass
    mass_from_freq = mass_from_freq_raw[::-1]  # ascending mass (after load reversal)

    # TOF axis (ascending mass)
    tof = np.linspace(100.0, 1600.0, n_tof)

    # Create empty data array
    data = np.zeros((n_swim, n_tof), dtype=np.float32)

    # Place peaks along the autocorrelation line: swim_mass ≈ tof_mass
    # We index into the NC data array (freq ascending, before reversal).
    # mass_from_freq_raw[i] is the mass at freq index i (descending mass).
    # After load reversal, freq index i maps to swim index (n_swim-1-i).
    peak_tof_indices = rng.choice(np.arange(n_tof // 4, 3 * n_tof // 4), size=n_peaks, replace=False)
    peak_tof_indices.sort()

    placed_peaks = []
    for tof_idx in peak_tof_indices:
        target_swim_mass = tof[tof_idx]  # On the autocorrelation line
        # Find closest swim index in the POST-REVERSAL mass array
        swim_idx_post = np.argmin(np.abs(mass_from_freq - target_swim_mass))
        # Convert to PRE-REVERSAL index for the NC data array
        swim_idx_pre = n_swim - 1 - swim_idx_post
        amplitude = rng.uniform(1e6, 1e10)
        data[swim_idx_pre, tof_idx] = amplitude
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                si = swim_idx_pre + di
                tj = tof_idx + dj
                if 0 <= si < n_swim and 0 <= tj < n_tof:
                    data[si, tj] = max(data[si, tj], amplitude * rng.uniform(0.3, 0.9))
        placed_peaks.append({
            "tof_idx": int(tof_idx),
            "swim_idx_post_reversal": int(swim_idx_post),
            "tof_mass": float(tof[tof_idx]),
            "swim_mass": float(mass_from_freq[swim_idx_post]),
            "amplitude": float(amplitude),
        })

    # Save as NetCDF with frequency dimension (matching production format)
    ds = xr.Dataset(
        {"amplitude": (["frequency", "mass_charge"], data)},
        coords={
            "frequency": freq,
            "mass_charge": tof,
        },
    )
    ds.to_netcdf(path)

    return {
        "n_swim": n_swim,
        "n_tof": n_tof,
        "n_peaks": n_peaks,
        "swim_range": [float(mass_from_freq.min()), float(mass_from_freq.max())],
        "tof_range": [float(tof[0]), float(tof[-1])],
        "placed_peaks": placed_peaks,
    }


def production_binning(nc_path, n_bins=5000, swim_range=(400, 1000), tof_range=(400, 1000)):
    """Run the exact production binning pipeline."""
    ds = xr.open_dataset(nc_path)

    data = ds["amplitude"]
    y_dim = data.dims[0]  # "frequency"
    x_dim = data.dims[1]  # "mass_charge"

    # Frequency → mass conversion
    freq = ds[y_dim].values
    mass = 39.0 / (freq + 0.01)
    mass_reversed = mass[::-1]
    data = data.isel({y_dim: slice(None, None, -1)})
    data = data.assign_coords({y_dim: mass_reversed})
    data = data.rename({y_dim: "swim_mass", x_dim: "tof_mass"})

    tof_mass = data["tof_mass"]
    tof_idx_coord = xr.DataArray(np.arange(len(tof_mass)), dims=["tof_mass"])
    data = data.assign_coords(tof_idx=("tof_mass", np.arange(len(tof_mass))))

    # Bin edges
    min_tof = float(tof_mass.min())
    max_tof = float(tof_mass.max())
    bin_edges = min_tof + (np.linspace(0, 1, n_bins + 1) ** 2) * (max_tof - min_tof)
    bin_edges[-1] = max_tof + 1e-9

    # Bin the data
    binned = data.groupby_bins("tof_mass", bins=bin_edges, right=True).sum().fillna(0)
    binned_idx = data["tof_idx"].groupby_bins("tof_mass", bins=bin_edges, right=True).mean()
    # Handle NaN in binned_idx (bins with no data)
    binned_idx = binned_idx.fillna(-1).astype(int)
    binned = binned.rename({"tof_mass_bins": "tof_mass"})
    binned = binned.assign_coords(tof_idx=("tof_mass", binned_idx.values))

    # Select range
    subset = binned.sel(swim_mass=slice(*swim_range), tof_mass=slice(*tof_range))

    n_tof_sel = subset.sizes["tof_mass"]
    n_swim_sel = subset.sizes["swim_mass"]

    # Find local maxima
    values = subset.values  # (swim, tof)
    values_t = values.T  # (tof, swim) — matches production
    max_filt = maximum_filter(values_t, size=5)
    peaks_mask = (values_t > 0) & (values_t == max_filt)

    # Extract peaks
    tof_coords = subset["tof_mass"].values
    swim_coords = subset["swim_mass"].values
    tof_idx_vals = subset["tof_idx"].values

    # Get midpoints of intervals
    tof_mid = np.array([float(v.mid) if hasattr(v, "mid") else float(v) for v in tof_coords])
    swim_vals = np.array([float(v) for v in swim_coords])

    ti_indices, si_indices = np.where(peaks_mask)
    peaks = []
    for k in range(len(ti_indices)):
        ti = ti_indices[k]
        si = si_indices[k]
        peaks.append({
            "tof": float(tof_mid[ti]),
            "swim": float(swim_vals[si]),
            "amplitude": float(values_t[ti, si]),
            "tof_idx": int(tof_idx_vals[ti]),
        })

    # Amplitude threshold
    mean_val = float(np.mean(values_t))
    std_val = float(np.std(values_t))
    threshold = mean_val + 0.5 * std_val
    threshold_peaks = [p for p in peaks if p["amplitude"] > threshold]

    unique_tof = len(set(p["tof_idx"] for p in threshold_peaks))

    return {
        "n_tof_bins_selected": n_tof_sel,
        "n_swim_selected": n_swim_sel,
        "local_maxima": len(peaks),
        "threshold_peaks": len(threshold_peaks),
        "unique_tof_idx": unique_tof,
        "threshold": threshold,
        "mean": mean_val,
        "std": std_val,
        "peaks": threshold_peaks,
        "bin_edges_first_10": bin_edges[:10].tolist(),
        "bin_edges_last_5": bin_edges[-5:].tolist(),
    }


def main():
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        nc_path = f.name

    print("=== Creating synthetic dataset ===")
    info = create_synthetic_nc(nc_path, n_swim=1025, n_tof=50000, n_peaks=30)
    print(f"  Shape: {info['n_swim']} x {info['n_tof']}")
    print(f"  TOF range: {info['tof_range']}")
    print(f"  SWIM range: {info['swim_range']}")
    print(f"  Placed {info['n_peaks']} peaks on autocorrelation line")
    print()

    print("=== Running production binning ===")
    result = production_binning(nc_path)
    print(f"  Bins selected: {result['n_tof_bins_selected']} TOF x {result['n_swim_selected']} SWIM")
    print(f"  Local maxima: {result['local_maxima']}")
    print(f"  After threshold ({result['threshold']:.2f}): {result['threshold_peaks']}")
    print(f"  Unique tof_idx: {result['unique_tof_idx']}")
    print(f"  Bin edges [0:10]: {[f'{e:.2f}' for e in result['bin_edges_first_10']]}")
    print(f"  Bin edges [-5:]: {[f'{e:.2f}' for e in result['bin_edges_last_5']]}")
    print()

    # Save expected output for Rust comparison
    expected = {
        "nc_path": nc_path,
        "n_tof_bins_selected": result["n_tof_bins_selected"],
        "n_swim_selected": result["n_swim_selected"],
        "local_maxima": result["local_maxima"],
        "threshold_peaks_count": result["threshold_peaks"],
        "unique_tof_idx": result["unique_tof_idx"],
        "threshold": result["threshold"],
        "peaks": result["peaks"],
    }

    with open("/tmp/binning_expected.json", "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Expected output saved to /tmp/binning_expected.json")
    print(f"Synthetic NC at: {nc_path}")
    print()

    # Print peaks for comparison
    print("=== Threshold peaks (production) ===")
    for i, p in enumerate(result["peaks"][:20]):
        print(f"  [{i}] tof={p['tof']:.2f} swim={p['swim']:.2f} amp={p['amplitude']:.0f} tof_idx={p['tof_idx']}")

    # Keep the NC file for Rust comparison
    print(f"\nNC file kept at: {nc_path}")
    print(f"To compare with Rust, run the Lambda with this file.")
    print()
    print(f"=== Key values for Rust comparison ===")
    print(f"n_tof_bins_in_range: {result['n_tof_bins_selected']}")
    print(f"n_swim_in_range: {result['n_swim_selected']}")
    print(f"local_maxima: {result['local_maxima']}")
    print(f"threshold_peaks: {result['threshold_peaks']}")
    print(f"unique_tof_idx: {result['unique_tof_idx']}")
    print(f"threshold: {result['threshold']:.2f}")
    print(f"mean: {result['mean']:.2f}")
    print(f"std: {result['std']:.2f}")


if __name__ == "__main__":
    main()
