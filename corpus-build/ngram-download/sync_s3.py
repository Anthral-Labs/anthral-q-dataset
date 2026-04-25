"""
Step 4: Sync results to S3 for persistent storage.

Syncs both the raw NGrams files and reconstructed articles to S3.

Usage: python3 sync_s3.py --bucket your-bucket-name
"""

import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sync(local_dir: str, s3_path: str):
    """Sync a local directory to S3."""
    cmd = f"aws s3 sync {local_dir} {s3_path} --storage-class STANDARD_IA"
    logger.info("Running: %s", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Synced %s → %s", local_dir, s3_path)
    else:
        logger.error("Sync failed: %s", result.stderr[:500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    args = parser.parse_args()

    base = f"s3://{args.bucket}/gdelt-forecasting"

    # Sync reconstructed articles (most important — skip raw to save storage)
    sync("/home/ubuntu/gdelt/reconstructed", f"{base}/reconstructed/")

    # Sync filtered/search results
    sync("/home/ubuntu/gdelt/filtered", f"{base}/filtered/")

    # Optionally sync raw NGrams (600GB — expensive to store, skip unless needed)
    # sync("/home/ubuntu/gdelt/raw", f"{base}/raw/")

    logger.info("Done. Data persisted in s3://%s/gdelt-forecasting/", args.bucket)


if __name__ == "__main__":
    main()
