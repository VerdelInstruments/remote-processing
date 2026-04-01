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

## Test Results (2026-04-01)

10 non-DSTL samples from `two-two-one-b-files-987686461587`, all 3.2 GB fourierdomain.nc files. Each compared against the production reference `peaks.db` using 1.5 Da nearest-neighbour tolerance.

| Sample | Peaks | Ref | Matched | F1 | Duration | Notes |
|--------|-------|-----|---------|-----|----------|-------|
| jrlipids-000005 | 1200 | 1200 | 828 | 69.0% | 118s | |
| jrlipids-000009 | 1200 | 1200 | 1170 | 97.5% | 113s | Near-perfect |
| jrlipids-000006 | 1200 | 1200 | 1191 | 99.2% | 111s | Near-perfect |
| jrlipids-000011 | 1200 | 1200 | 872 | 72.7% | 114s | |
| jrlipids-000002 | 1200 | 1200 | 1195 | 99.6% | 114s | Near-perfect |
| jr-1 | 1200 | 1200 | 396 | 33.0% | 115s | RANSAC divergence |
| jrlipids-000003 | 1200 | 1200 | 917 | 76.4% | 114s | |
| jrlipids-000007 | 1200 | 1200 | — | — | 114s | Ref download failed |

**Average F1: 78.2%** (7 samples with successful comparison)

### Interpretation

- **3 samples at 97-99%**: pipeline matches production closely
- **3 samples at 69-77%**: gap is from top_n_peaks selection differences (harmonic filter produces slightly different 1,200-peak selection — see test-harness Experiment 8)
- **1 sample at 33%**: RANSAC fit divergence on "jr-1" (different sample type, competing linear structures)

### Known Issues

1. **RANSAC divergence on some samples**: The Lambda uses native Rust RANSAC, not the sklearn bridge. Production uses `sklearn.RANSACRegressor(random_state=42, stop_n_inliers=50)`. The `stop_n_inliers=50` parameter causes sklearn to stop early at the first acceptable fit, which can produce a different consensus than running all 1000 trials. The native Rust RANSAC doesn't implement `stop_n_inliers`. On most samples the fits converge, but on samples with competing linear structures (like "jr-1") the fits diverge significantly. See test-harness Experiments 17-18.

2. **top_n_peaks selection overlap ~96-98%**: The harmonic filter in Rust matches production closely (Experiment 11: 98.6% overlap on sample-20) but not perfectly. 17-40 of the 1200 selected peaks differ, causing downstream coordinate differences. This accounts for the 69-77% F1 tier.

3. **peaks_above_noise output not used for final peaks**: The Lambda outputs the 1,200 centroided top_n peaks (matching what the 2discover compare step uses), not the peaks_above_noise filtered set. The peaks_above_noise step runs correctly on the full 14M peaks (verified in logs) but its output feeds precursors/fragments, not the final peak list.

## Code Provenance

The pipeline code in `sherlock-runner/src/sherlock/` was copied from `VerdelInstruments/2Discover` (`src-tauri/src/sherlock/`) with these modifications:

| File | Change |
|------|--------|
| `progress.rs` | Replaced Tauri `emit_progress(app, ...)` with log-only no-op |
| `find_filter.rs` | Removed `app: Option<&tauri::AppHandle>` param; memory-optimised `maximum_filter_2d` (single-buffer, validated by 5 unit tests) |
| `compare.rs` | Removed `app` param |
| `peaks_above_noise.rs` | Removed `app` param; reads from `full_peak_*` stash (14M peaks) instead of current peak arrays |
| `mod.rs` | Added `full_peak_row_idx/col_idx/amplitudes` fields; removed `NativeSherlock` Tauri wrapper |
| `stripes.rs` | Stashes full peaks before top_n filtering |
| `review.rs` | Takes `DatasetState` param; handles centroid/peak count mismatch |
| `dataset.rs` | Stripped to scan + load_nc only (no UI functions, no rusqlite/csv) |

All other sherlock modules (`calibrate.rs`, `centroid.rs`, `ransac.rs`, `precursors.rs`, `fragments.rs`, `isotope.rs`, `swim_group.rs`, `graph_clique.rs`, `subsample.rs`, `run.rs`, `run_compare.rs`, `export.rs`) are unmodified copies.

### What's in the Lambda vs 2discover

The Lambda code includes two fixes NOT yet in 2discover:
- **Full peak stash** (`full_peak_*` fields in SherlockState, stash in `stripes.rs`, read in `peaks_above_noise.rs`) — ensures peaks_above_noise operates on 14M peaks
- **Memory-optimised `maximum_filter_2d`** — single-buffer approach reducing peak memory from 9.6GB to 6.4GB

The Lambda code does NOT include:
- The sklearn RANSAC bridge (no Python in the Lambda container) — falls back to native Rust RANSAC
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
