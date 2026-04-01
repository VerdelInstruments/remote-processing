#!/usr/bin/env python3
"""
Binning bridge: runs the exact production binning pipeline on a dataset,
returning the binned peaks that feed into RANSAC.

Reads JSON on stdin with:
  - nc_path: path to the NetCDF file
  - n_bins: number of bins (default 5000)
  - swim_range: [lo, hi] (default [400, 1000])
  - tof_range: [lo, hi] (default [400, 1000])

Writes JSON to stdout with:
  - peaks: [{tof, swim, amplitude, tof_idx, swim_idx}, ...]
  - n_bins_selected: number of TOF bins in the subset
  - n_swim_selected: number of swim rows in the subset
  - threshold_peaks: peaks after amplitude thresholding
"""
import json
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter

req = json.load(sys.stdin)
nc_path = req["nc_path"]
n_bins = req.get("n_bins", 5000)
swim_range = tuple(req.get("swim_range", [400, 1000]))
tof_range = tuple(req.get("tof_range", [400, 1000]))

# Load dataset
ds = xr.open_dataset(nc_path)
# Find the 2D data variable
data_var = None
for name in ds.data_vars:
    if len(ds[name].dims) == 2:
        data_var = name
        break

if data_var is None:
    json.dump({"error": "No 2D variable found"}, sys.stdout)
    sys.exit(0)

data = ds[data_var]

# Check if y-dimension is frequency and convert to mass
y_dim = data.dims[0]
x_dim = data.dims[1]

if y_dim == "frequency":
    freq = ds[y_dim].values
    mass = 39.0 / (freq + 0.01)
    # Reverse to ascending mass
    mass = mass[::-1]
    data = data.isel({y_dim: slice(None, None, -1)})
    data = data.assign_coords({y_dim: mass})
    data = data.rename({y_dim: "swim_mass"})
    y_dim = "swim_mass"

# Rename x dimension if needed
if x_dim != "tof_mass":
    tof_vals = ds[x_dim].values
    data = data.rename({x_dim: "tof_mass"})
else:
    tof_vals = ds["tof_mass"].values

tof_mass = data["tof_mass"]

# Add tof_idx coordinate
tof_idx = xr.DataArray(np.arange(len(tof_vals)), dims=["tof_mass"])
data = data.assign_coords(tof_idx=("tof_mass", np.arange(len(tof_vals))))

# Step 1: Compute bin edges (quadratic spacing)
min_tof = float(tof_mass.min())
max_tof = float(tof_mass.max())
bin_edges = min_tof + (np.linspace(0, 1, n_bins + 1) ** 2) * (max_tof - min_tof)
bin_edges[-1] = max_tof + 1e-9

# Step 2: Bin the data
binned = data.groupby_bins("tof_mass", bins=bin_edges, right=True).sum().fillna(0)

# Bin the indices
binned_idx = data["tof_idx"].groupby_bins("tof_mass", bins=bin_edges, right=True).mean().astype(int)
binned = binned.rename({"tof_mass_bins": "tof_mass"})
binned = binned.assign_coords(tof_idx=("tof_mass", binned_idx.values))

# Step 3: Select range
selection = {}
if swim_range:
    selection["swim_mass"] = slice(*swim_range)
if tof_range:
    selection["tof_mass"] = slice(*tof_range)

subset = binned.sel(**selection)

n_tof_sel = subset.sizes["tof_mass"]
n_swim_sel = subset.sizes["swim_mass"]

# Step 4: Find local maxima (kernel=5)
values = subset.values  # (swim, tof)
# Production transposes to (tof, swim) before local_maximum
values_t = values.T  # (tof, swim)
max_filt = maximum_filter(values_t, size=5)
peaks_mask = (values_t > 0) & (values_t == max_filt)

tof_coords_sel = subset["tof_mass"].values
swim_coords_sel = subset["swim_mass"].values if hasattr(subset["swim_mass"], "values") else np.array([float(x) for x in subset["swim_mass"].values])
tof_idx_sel = subset["tof_idx"].values

# Handle Interval coordinates for tof
tof_mid = []
for v in tof_coords_sel:
    if hasattr(v, "mid"):
        tof_mid.append(float(v.mid))
    else:
        tof_mid.append(float(v))
tof_mid = np.array(tof_mid)

# Handle swim coordinates
swim_vals = []
for v in swim_coords_sel:
    swim_vals.append(float(v))
swim_vals = np.array(swim_vals)

peaks = []
ti_indices, si_indices = np.where(peaks_mask)
for k in range(len(ti_indices)):
    ti = ti_indices[k]
    si = si_indices[k]
    peaks.append({
        "tof": float(tof_mid[ti]),
        "swim": float(swim_vals[si]),
        "amplitude": float(values_t[ti, si]),
        "tof_idx": int(tof_idx_sel[ti]),
        "swim_idx": int(si + np.searchsorted(data["swim_mass"].values, swim_range[0])),
    })

# Step 5: Amplitude threshold (mean + 0.5 * std of subset)
subset_vals = values_t.flatten()
mean_val = float(np.mean(subset_vals))
std_val = float(np.std(subset_vals))
threshold = mean_val + 0.5 * std_val

threshold_peaks = [p for p in peaks if p["amplitude"] > threshold]

# Diagnostics
unique_tof_idx = len(set(p["tof_idx"] for p in threshold_peaks))
unique_swim_idx = len(set(p["swim_idx"] for p in threshold_peaks))

json.dump({
    "n_bins_selected": n_tof_sel,
    "n_swim_selected": n_swim_sel,
    "local_maxima_count": len(peaks),
    "threshold": threshold,
    "mean": mean_val,
    "std": std_val,
    "threshold_peaks_count": len(threshold_peaks),
    "unique_tof_idx": unique_tof_idx,
    "unique_swim_idx": unique_swim_idx,
    "threshold_peaks": threshold_peaks[:500],  # Cap to avoid huge output
}, sys.stdout)
