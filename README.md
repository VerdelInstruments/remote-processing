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
  "peak_count": 138,
  "duration_ms": 34776,
  "results_prefix": "s3://verdel-sherlock-results/runs/test-001",
  "steps": [
    { "name": "calibrate", "duration_ms": 0 },
    { "name": "find_peaks", "duration_ms": 3318 },
    ...
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

| Metric | Value |
|--------|-------|
| Cold start | ~2s |
| S3 download (1.5 GB) | ~20s |
| Pipeline execution | ~15s |
| Total (warm) | ~35s |
| Memory used | ~4 GB peak |
| Binary size | 54 MB |
