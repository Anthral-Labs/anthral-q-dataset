#!/bin/bash
# Master script: Download + Reconstruct GDELT NGrams
# All output goes to both terminal AND /home/ubuntu/gdelt/pipeline.log
#
# Run: bash run_all.sh
# Monitor from another terminal: tail -f /home/ubuntu/gdelt/pipeline.log

set -e
cd /home/ubuntu/gdelt

LOGFILE="/home/ubuntu/gdelt/pipeline.log"

# Tee all output to logfile
exec > >(tee -a "$LOGFILE") 2>&1

echo "================================================"
echo "  GDELT NGrams Pipeline"
echo "  $(date)"
echo "  Log: $LOGFILE"
echo "================================================"
echo ""

# Step 1: Download
echo "=== STEP 1/3: Download NGrams (Aug 2025 – Mar 2026) ==="
echo "  8 files/hour × 24 hours × 243 days = ~46K files"
echo "  16 parallel threads, no rate limits"
echo ""
python3 download_ngrams.py --workers 16

echo ""

# Step 2: Reconstruct
echo "=== STEP 2/3: Reconstruct articles ==="
echo ""
python3 reconstruct.py

echo ""

# Step 3: Sync to S3
echo "=== STEP 3/3: Sync to S3 ==="
echo ""
if [ -f s3_bucket.txt ]; then
    BUCKET=$(cat s3_bucket.txt)
    echo "Syncing to s3://$BUCKET/gdelt-forecasting/"
    aws s3 sync /home/ubuntu/gdelt/reconstructed/ "s3://$BUCKET/gdelt-forecasting/reconstructed/" --storage-class STANDARD_IA
    echo "Sync complete."
else
    echo "No s3_bucket.txt found. Skipping S3 sync."
    echo "To sync later: echo 'forecast-research' > s3_bucket.txt"
fi

echo ""
echo "================================================"
echo "  ALL DONE — $(date)"
echo "  Log: $LOGFILE"
echo "================================================"
