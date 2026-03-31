#!/bin/bash
# Run 10 non-DSTL samples through the Lambda and compare against reference peaks.
# Usage: ./test-samples.sh
set -euo pipefail

PROFILE="verdel"
REGION="eu-west-2"
BUCKET="two-two-one-b-files-987686461587"
RESULTS_BUCKET="verdel-sherlock-results"
OUTPUT_DIR="/tmp/sherlock-test-results"

mkdir -p "$OUTPUT_DIR"

echo "=== Finding samples with reference peaks (excluding DSTL) ==="

# Get packages that have both a fourierdomain.nc AND sherlock/peaks.db, skip DSTL
PACKAGES=$(aws s3 ls "s3://$BUCKET/" --recursive --profile "$PROFILE" --region "$REGION" \
  | grep "sherlock/peaks.db$" \
  | grep -v "dstl" \
  | awk '{print $4}' \
  | sed 's|/sherlock/peaks.db||' \
  | head -10)

echo "Found $(echo "$PACKAGES" | wc -l | tr -d ' ') packages"
echo ""

# For each package, find the fourierdomain.nc and run
IDX=0
for PKG_PATH in $PACKAGES; do
  IDX=$((IDX + 1))

  # Find the fourierdomain.nc in this package
  NC_KEY=$(aws s3 ls "s3://$BUCKET/$PKG_PATH/converted/" --profile "$PROFILE" --region "$REGION" \
    | grep "_fourierdomain.nc$" \
    | awk '{print $4}' \
    | head -1)

  if [ -z "$NC_KEY" ]; then
    echo "[$IDX] SKIP — no fourierdomain.nc in $PKG_PATH"
    continue
  fi

  NC_PATH="$PKG_PATH/converted/$NC_KEY"
  SAMPLE_NAME=$(echo "$NC_KEY" | sed 's/_fourierdomain.nc//')
  RUN_ID="test-$(printf '%03d' $IDX)"

  echo "[$IDX] $SAMPLE_NAME"
  echo "    NC: s3://$BUCKET/$NC_PATH"
  echo "    Ref: s3://$BUCKET/$PKG_PATH/sherlock/peaks.db"

  # Invoke Lambda
  RESPONSE_FILE="$OUTPUT_DIR/$RUN_ID-response.json"
  aws lambda invoke \
    --function-name sherlock-runner \
    --cli-binary-format raw-in-base64-out \
    --payload "{
      \"s3_input_path\": \"s3://$BUCKET/$NC_PATH\",
      \"s3_output_prefix\": \"s3://$RESULTS_BUCKET/test-batch/$RUN_ID\",
      \"config\": {}
    }" \
    --profile "$PROFILE" --region "$REGION" \
    "$RESPONSE_FILE" > /dev/null 2>&1

  # Check for Lambda error
  if grep -q "errorMessage" "$RESPONSE_FILE" 2>/dev/null; then
    ERROR=$(python3 -c "import json; print(json.load(open('$RESPONSE_FILE')).get('errorMessage','unknown'))" 2>/dev/null)
    echo "    FAILED: $ERROR"
    echo ""
    continue
  fi

  # Parse response
  PEAK_COUNT=$(python3 -c "import json; print(json.load(open('$RESPONSE_FILE')).get('peak_count', 0))")
  DURATION=$(python3 -c "import json; print(json.load(open('$RESPONSE_FILE')).get('duration_ms', 0))")
  STATUS=$(python3 -c "import json; print(json.load(open('$RESPONSE_FILE')).get('status', 'unknown'))")

  echo "    Status: $STATUS | Peaks: $PEAK_COUNT | Duration: ${DURATION}ms"

  # Download reference peaks.db and our peaks.json for comparison
  aws s3 cp "s3://$BUCKET/$PKG_PATH/sherlock/peaks.db" "$OUTPUT_DIR/$RUN_ID-ref-peaks.db" \
    --profile "$PROFILE" --region "$REGION" > /dev/null 2>&1

  aws s3 cp "s3://$RESULTS_BUCKET/test-batch/$RUN_ID/peaks.json" "$OUTPUT_DIR/$RUN_ID-peaks.json" \
    --profile "$PROFILE" --region "$REGION" > /dev/null 2>&1

  echo "    Downloaded reference + results for comparison"
  echo ""
done

echo "=== All invocations complete ==="
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Run the comparison script next:"
echo "  python3 compare-results.py"
