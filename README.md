# Sherlock Remote Processing

Rust mass spectrometry pipeline running on AWS Lambda (ARM64, container image).

## Architecture

```
S3 (source .nc) → Lambda (sherlock-runner) → S3 (results)
                      │
                      ├── Downloads .nc to /tmp (5GB ephemeral)
                      ├── Runs 9-step pipeline in Rust
                      └── Uploads manifest.json + peaks.json
```

- **Runtime**: provided.al2023, ARM64
- **Memory**: 10,240 MB (10 GB)
- **Timeout**: 900s (15 min)
- **Ephemeral storage**: 5,120 MB (5 GB)
- **Region**: eu-west-2

## AWS Resources

| Resource | Name/ARN |
|----------|----------|
| Lambda function | `sherlock-runner` |
| ECR repository | `987686461587.dkr.ecr.eu-west-2.amazonaws.com/sherlock-runner` |
| CodeBuild project | `sherlock-runner-build` |
| Source data bucket | `two-two-one-b-files-987686461587` (read-only) |
| Results bucket | `verdel-sherlock-results` |
| Lambda role | `lambda-sherlock-role` |
| CodeBuild role | `codebuild-sherlock-role` |

## Invoking the Lambda

```bash
aws lambda invoke \
  --function-name sherlock-runner \
  --cli-binary-format raw-in-base64-out \
  --payload '{
    "s3_input_path": "s3://two-two-one-b-files-987686461587/path/to/file_fourierdomain.nc",
    "s3_output_prefix": "s3://verdel-sherlock-results/runs/my-run-id",
    "config": {}
  }' \
  --profile verdel --region eu-west-2 \
  /tmp/response.json
```

### Config parameters (all optional, defaults shown)

```json
{
  "scale": 39.0,
  "offset": 0.01,
  "peak_filter_size": 5,
  "swim_min": 100.0,
  "swim_max": 1800.0,
  "tof_min": 100.0,
  "tof_max": 1800.0,
  "n_peaks": 1200,
  "most_peaks_per_streak": 5,
  "streak_half_width": 13,
  "harmonic_proportional_tolerance": 1.0,
  "half_window_tof": 4,
  "half_window_swim": 0,
  "isotope_tolerance": 0.1,
  "max_charge": 6,
  "row_half_width": 4,
  "ransac_tolerance": 1.5,
  "ransac_n_peaks": 15,
  "min_snr": 70.0,
  "noise_window": 51,
  "amplitude_std_multiplier": 5.0,
  "gulley_width_amu": 3.0,
  "autocorrelation_tolerance": 3.0
}
```

### Response format

```json
{
  "run_id": "y21zwsca",
  "status": "completed",
  "peak_count": 1200,
  "duration_ms": 113000,
  "results_prefix": "s3://verdel-sherlock-results/runs/test-001",
  "steps": [
    { "name": "calibrate", "duration_ms": 0 },
    { "name": "find_peaks", "duration_ms": 8000 },
    { "name": "top_n_peaks", "duration_ms": 4000 },
    { "name": "centroid_isotope", "duration_ms": 100 },
    { "name": "ransac_rough", "duration_ms": 3000 },
    { "name": "ransac_precise", "duration_ms": 130 },
    { "name": "peaks_above_noise", "duration_ms": 17000 },
    { "name": "precursors", "duration_ms": 0 },
    { "name": "fragments", "duration_ms": 0 }
  ]
}
```

### Output files in S3

- `{output_prefix}/manifest.json` — run metadata, timing, peak count
- `{output_prefix}/peaks.json` — full peak list with coordinates, charges, inlier flags

## Building and Deploying

### Prerequisites

- AWS CLI with `verdel` profile configured (SSO)
- GitHub repo: `VerdelInstruments/remote-processing` (public, for CodeBuild)

### Build (via CodeBuild — no local Docker needed)

```bash
# Authenticate
aws sso login --profile verdel

# Trigger build
aws codebuild start-build \
  --project-name sherlock-runner-build \
  --profile verdel --region eu-west-2

# Check status
aws codebuild batch-get-builds \
  --ids "sherlock-runner-build:<build-id>" \
  --profile verdel --region eu-west-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

The build:
1. Pulls source from GitHub
2. Builds HDF5 1.12.3 + NetCDF-C 4.9.2 from source (AL2023 ARM64)
3. Installs Rust toolchain
4. Compiles the Lambda binary (`cargo build --release`)
5. Packages into a container image
6. Pushes to ECR

Takes ~12-15 minutes.

### Deploy (update Lambda to use new image)

After a successful build:

```bash
aws lambda update-function-code \
  --function-name sherlock-runner \
  --image-uri 987686461587.dkr.ecr.eu-west-2.amazonaws.com/sherlock-runner:latest \
  --profile verdel --region eu-west-2
```

### Full rebuild from scratch

If the Lambda or ECR repo don't exist:

```bash
# 1. Create ECR repo
aws ecr create-repository --repository-name sherlock-runner \
  --profile verdel --region eu-west-2

# 2. Build via CodeBuild (see above)

# 3. Create Lambda
aws lambda create-function \
  --function-name sherlock-runner \
  --package-type Image \
  --code ImageUri=987686461587.dkr.ecr.eu-west-2.amazonaws.com/sherlock-runner:latest \
  --role arn:aws:iam::987686461587:role/lambda-sherlock-role \
  --architectures arm64 \
  --memory-size 10240 \
  --timeout 900 \
  --ephemeral-storage Size=5120 \
  --environment 'Variables={RUST_LOG=info}' \
  --profile verdel --region eu-west-2
```

## Updating the Pipeline Code

1. Make changes in `sherlock-runner/src/`
2. Commit and push to GitHub
3. Trigger CodeBuild: `aws codebuild start-build --project-name sherlock-runner-build --profile verdel --region eu-west-2`
4. Wait for build to complete (~15 min)
5. Update Lambda: `aws lambda update-function-code --function-name sherlock-runner --image-uri 987686461587.dkr.ecr.eu-west-2.amazonaws.com/sherlock-runner:latest --profile verdel --region eu-west-2`
6. Test with an invoke

## Monitoring

```bash
# Recent invocation logs
aws logs tail /aws/lambda/sherlock-runner \
  --since 1h --profile verdel --region eu-west-2

# CloudWatch metrics (duration, memory, errors)
# Console: https://eu-west-2.console.aws.amazon.com/lambda/home?region=eu-west-2#/functions/sherlock-runner?tab=monitoring
```

## Performance (observed)

| File size | Total duration | Peak memory | S3 download | Pipeline |
|-----------|---------------|-------------|-------------|----------|
| 1.5 GB | ~35s | ~4 GB | ~20s | ~15s |
| 3.2 GB | ~114s | ~9.6 GB | ~45s | ~65s |

- Cold start: ~1s (Rust, ARM64)
- Binary size: 54 MB (container image)
- Memory limit: 10 GB. Files up to ~3.2 GB process successfully. Larger files would need ECS Fargate.

## Test Results (2026-04-02, latest)

10 non-DSTL samples from `two-two-one-b-files-987686461587`, all 3.2 GB fourierdomain.nc files. Each compared against the production reference `peaks.db` using 1.5 Da nearest-neighbour tolerance. Variant: `rust-sklearn-bridge`.

| Sample | Peaks | Ref | Matched | F1 | Top50 | Top100 | Duration | Notes |
|--------|-------|-----|---------|-----|-------|--------|----------|-------|
| jrlipids-000005 | 1200 | 1200 | 855 | 71.2% | 50.0% | 49.0% | 217s | See known issues |
| jrlipids-000009 | 1200 | 1200 | 1170 | 97.5% | 98.0% | 98.0% | 196s | |
| jrlipids-000006 | 1200 | 1200 | 1191 | 99.2% | 100% | 99.0% | 198s | |
| jrlipids-000011 | 1200 | 1200 | 1174 | 97.8% | 100% | 98.0% | 198s | |
| jrlipids-000002 | 1200 | 1200 | 1195 | 99.6% | 100% | 100% | 198s | |
| jr-1 | 1200 | 1200 | 401 | 33.4% | 60.0% | 55.0% | 201s | RANSAC divergence |
| jrlipids-000003 | 1200 | 1200 | 1175 | 97.9% | 98.0% | 96.0% | 197s | |
| jrlipids-000007 | 1200 | 1200 | 1157 | 96.4% | 94.0% | 92.0% | 197s | |
| jrlipids-000001 | 1200 | 1200 | 401 | 33.4% | 60.0% | 55.0% | 199s | Same as jr-1 |
| jrlipids-000004 | 1200 | 1200 | 1196 | 99.7% | 100% | 100% | 200s | |

**Average F1: 82.6%** (10 samples). **7 of 10 samples above 96%.**

### Interpretation

- **7 samples at 96-99.7%**: Pipeline matches production closely. Top-50 recall 94-100%.
- **1 sample at 71.2%** (jrlipids-000005): Moderate gap, top_n selection differences.
- **2 samples at 33.4%** (jr-1, jrlipids-000001): RANSAC rough fit divergence causes ~9 Da systematic swim coordinate shift. See investigation notes below.

### Investigation History

The recall improved through a series of fixes, each validated by the test harness:

| Fix | Avg F1 | Commit |
|-----|--------|--------|
| Initial (peaks_above_noise on 1,200 top_n) | 8.4% | — |
| Output centroids (1,200) instead of noise-filtered | 69.5% | `9dc6af0` |
| Store raw rough fit (not refined) for precise step | **82.6%** | `1954656` |

**Root cause of remaining 33% samples**: Diagnostic comparison (Rust vs Python binning bridge, same Lambda invocation) confirmed the binning produces **identical** peaks (same 61 peaks, coordinates matching to 4 decimal places). The RANSAC fit (via sklearn bridge) is also identical. The divergence occurs because the rough fit (slope=0.804) feeds into the precise step which selects different full-res peaks than production. Tolerance sweep confirms the peaks ARE found — at 20 Da tolerance, recall reaches 99.9% on these samples. The coordinate offset is systematic (~9 Da), not random.

### Known Issues

1. **RANSAC rough fit divergence on 2 of 10 samples**: The rough fit on "jr-1" and "jrlipids-000001" produces slope=0.804 while production likely gets a different slope. Even with the sklearn bridge (same algorithm, same parameters, same input peaks), the fit matches our Rust binning output — so the divergence must come from a subtle difference in how production bins or thresholds differently for these specific datasets. The precise autocorrelation then amplifies the difference.

2. **top_n_peaks selection overlap ~96-98%**: Harmonic filter matches production closely but not perfectly. 17-40 of 1200 peaks differ per sample. Accounts for the 71% sample.

3. **Peak output path**: The Lambda outputs the 1,200 centroided top_n peaks with RANSAC-calibrated swim coordinates. The peaks_above_noise step runs on the full 14M peaks (confirmed in logs) but its output feeds precursors/fragments, not the final peak list.

## Code Provenance

The pipeline code in `sherlock-runner/src/sherlock/` was copied from `VerdelInstruments/2Discover` (`src-tauri/src/sherlock/`) with these modifications:

| File | Change |
|------|--------|
| `progress.rs` | Replaced Tauri `emit_progress(app, ...)` with log-only no-op |
| `find_filter.rs` | Removed `app` param; memory-optimised `maximum_filter_2d` (single-buffer, validated by 5 unit tests) |
| `compare.rs` | Removed `app` param |
| `peaks_above_noise.rs` | Removed `app` param; reads from `full_peak_*` stash (14M peaks) instead of current peak arrays |
| `mod.rs` | Added `full_peak_*` stash fields, `use_sklearn_ransac` flag; removed `NativeSherlock` Tauri wrapper |
| `stripes.rs` | Stashes full peaks before top_n filtering |
| `ransac.rs` | Added `use_sklearn` param threading; stores raw rough fit (not refined); dumps diagnostic JSON; updated bridge paths for Lambda |
| `review.rs` | Takes `DatasetState` param; handles centroid/peak count mismatch |
| `dataset.rs` | Stripped to scan + load_nc only (no UI functions, no rusqlite/csv) |
| `main.rs` | Lambda handler with S3 streaming download, algorithm_variant routing, diagnostic dumps |

All other sherlock modules (`calibrate.rs`, `centroid.rs`, `precursors.rs`, `fragments.rs`, `isotope.rs`, `swim_group.rs`, `graph_clique.rs`, `subsample.rs`, `run.rs`, `run_compare.rs`, `export.rs`) are unmodified copies from 2discover.

### What's in the Lambda vs 2discover

The Lambda code includes fixes NOT yet in 2discover:
- **Full peak stash** (`full_peak_*` fields in SherlockState, stash in `stripes.rs`, read in `peaks_above_noise.rs`) — ensures peaks_above_noise operates on 14M peaks
- **Memory-optimised `maximum_filter_2d`** — single-buffer approach reducing peak memory from 9.6GB to 6.4GB
- **Raw rough fit storage** — `ransac()` stores rough_slope/rough_intercept (not refined) so precise_autocorrelation sees the correct starting point
- **Algorithm variants** — `algorithm_variant` field routes RANSAC through native Rust or Python sklearn bridge
- **Diagnostic dumps** — rough_binned_fit outputs diagnostic JSON to S3, Python binning bridge comparison

The Lambda container includes:
- Python 3 + numpy + scikit-learn + scipy + xarray + netCDF4
- `/opt/ransac_bridge.py` — sklearn RANSAC bridge
- `/opt/binning_bridge.py` — production-equivalent binning for diagnostics

The Lambda code does NOT include:
- Any Tauri UI integration
- rusqlite/csv peak loading (reference peaks not needed for processing)

## IAM Policies

### lambda-sherlock-role
- `AWSLambdaBasicExecutionRole` (CloudWatch Logs)
- `s3:GetObject` on `arn:aws:s3:::two-two-one-b-files-987686461587/*` (read-only source data)
- `s3:PutObject` on `arn:aws:s3:::verdel-sherlock-results/*` (write results only)

**No write access to source data bucket.**

### codebuild-sherlock-role
- CloudWatch Logs (create/write)
- ECR (push images to `sherlock-runner` repository only)
