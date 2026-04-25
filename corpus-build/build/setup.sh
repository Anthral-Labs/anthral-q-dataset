#!/bin/bash
# Setup script for Azure A100 instance
# Installs all dependencies for the retrieval pipeline

set -e

echo "=== Retrieval Pipeline Setup ==="

# System deps
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip awscli

# Python deps
pip3 install --user torch transformers sentence-transformers faiss-gpu numpy boto3 accelerate

# Download data from S3
echo ""
echo "=== Downloading data from S3 ==="
mkdir -p /data/{reconstructed,cleaned,embeddings,retrieval,questions,training}

# Reconstructed articles (33GB)
echo "Downloading reconstructed articles..."
aws s3 sync s3://forecast-research/gdelt-forecasting/reconstructed/ /data/reconstructed/

# Questions + search queries
echo "Downloading questions..."
aws s3 sync s3://forecast-research/questions/ /data/questions/

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the pipeline:"
echo "  python3 step1_clean.py --input /data/reconstructed --output /data/cleaned"
echo "  python3 step2_embed.py --input /data/cleaned --output /data/embeddings"
echo "  python3 step3_retrieve.py --index /data/embeddings --questions /data/questions/filtered/polymarket_final.json --queries /data/questions/polymarket_search_queries.json --output /data/retrieval/retrieved_context_raw.json"
echo "  python3 step4_leakage_check.py submit --input /data/retrieval/retrieved_context_raw.json"
echo "  python3 step4_leakage_check.py check"
echo "  python3 step4_leakage_check.py apply --input /data/retrieval/retrieved_context_raw.json --output /data/retrieval/retrieved_context_clean.json"
echo "  python3 step5_assemble.py --context /data/retrieval/retrieved_context_clean.json --questions /data/questions/filtered/polymarket_final.json --output /data/training"
