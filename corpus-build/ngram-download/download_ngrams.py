"""
Download GDELT Web NGrams 3.0 files.

Strategy: For each 15-minute window, try every minute (0-14, 15-29, 30-34, 45-59).
GDELT publishes up to 5 files per window as a burst. We grab everything available.
404s are instant and cost nothing. No rate limits on these static file downloads.

Resumable: skips files already on disk.

Usage:
  python3 download_ngrams.py
  python3 download_ngrams.py --start 20250801 --end 20260401 --workers 16
"""

import os
import sys
import time
import logging
import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = Path("/home/ubuntu/gdelt/raw")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR.parent / "download.log")),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "http://data.gdeltproject.org/gdeltv3/webngrams/"
stats = {"downloaded": 0, "skipped": 0, "missing": 0, "errors": 0, "bytes": 0}


def generate_timestamps(start_date: str, end_date: str):
    """
    Generate a timestamp for every minute of every hour in the date range.
    GDELT publishes up to 5 files per 15-min window, burst timing varies.
    We try all 60 minutes per hour — 404s are instant, no overhead.
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    timestamps = []

    current_day = start
    while current_day < end:
        for hour in range(24):
            for minute in range(60):
                ts = current_day.replace(hour=hour, minute=minute, second=0)
                timestamps.append(ts.strftime("%Y%m%d%H%M%S"))
        current_day += timedelta(days=1)

    return timestamps


def download_one(timestamp: str) -> dict:
    """Download a single NGrams file. Returns status dict."""
    url = f"{BASE_URL}{timestamp}.webngrams.json.gz"
    local_path = OUTPUT_DIR / f"{timestamp}.webngrams.json.gz"

    if local_path.exists() and local_path.stat().st_size > 100:
        return {"status": "skipped", "size": local_path.stat().st_size}

    try:
        resp = requests.get(url, timeout=60, stream=True)

        if resp.status_code == 200:
            size = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=131072):
                    f.write(chunk)
                    size += len(chunk)
            return {"status": "downloaded", "size": size}
        elif resp.status_code == 404:
            return {"status": "missing", "size": 0}
        else:
            return {"status": "error", "size": 0, "code": resp.status_code}

    except requests.exceptions.Timeout:
        return {"status": "error", "size": 0, "code": "timeout"}
    except Exception as e:
        return {"status": "error", "size": 0, "code": str(e)[:50]}


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def main():
    parser = argparse.ArgumentParser(description="Download GDELT NGrams 3.0 files")
    parser.add_argument("--start", default="20250801", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20260401", help="End date YYYYMMDD")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download threads")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamps = generate_timestamps(args.start, args.end)
    total = len(timestamps)

    # Count already downloaded
    already_files = list(OUTPUT_DIR.glob("*.webngrams.json.gz"))
    already_count = len(already_files)
    already_bytes = sum(f.stat().st_size for f in already_files)

    logger.info("=" * 70)
    logger.info("GDELT NGrams Download")
    logger.info("=" * 70)
    logger.info("Date range: %s to %s", args.start, args.end)
    logger.info("Strategy: try every minute, grab all files, skip 404s")
    logger.info("Timestamps to try: %d", total)
    logger.info("Workers: %d", args.workers)
    logger.info("Already on disk: %d files (%s)", already_count, format_size(already_bytes))
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    remaining = [
        ts for ts in timestamps
        if not (OUTPUT_DIR / f"{ts}.webngrams.json.gz").exists()
        or (OUTPUT_DIR / f"{ts}.webngrams.json.gz").stat().st_size <= 100
    ]

    if not remaining:
        logger.info("All files already downloaded!")
        return

    logger.info("Remaining: %d timestamps to try", len(remaining))
    logger.info("Starting download...")
    logger.info("")

    start_time = time.time()
    completed = 0
    total_bytes = already_bytes

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, ts): ts for ts in remaining}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["status"] == "downloaded":
                stats["downloaded"] += 1
                stats["bytes"] += result["size"]
                total_bytes += result["size"]
            elif result["status"] == "skipped":
                stats["skipped"] += 1
            elif result["status"] == "missing":
                stats["missing"] += 1
            else:
                stats["errors"] += 1

            if completed % 1000 == 0 or completed == len(remaining):
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (len(remaining) - completed) / rate if rate > 0 else 0
                pct = completed / len(remaining) * 100
                total_files = stats["downloaded"] + stats["skipped"] + already_count

                logger.info(
                    "[%5.1f%%] %d/%d tried | %d files on disk (%s) | "
                    "%.0f/sec | ETA %s | new:%d miss:%d err:%d",
                    pct, completed, len(remaining),
                    total_files, format_size(total_bytes),
                    rate, format_time(eta),
                    stats["downloaded"], stats["missing"], stats["errors"],
                )

    elapsed = time.time() - start_time
    total_files = stats["downloaded"] + stats["skipped"] + already_count
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info("Time: %s", format_time(elapsed))
    logger.info("New files: %d (%s)", stats["downloaded"], format_size(stats["bytes"]))
    logger.info("Already had: %d", already_count + stats["skipped"])
    logger.info("Not found (404): %d", stats["missing"])
    logger.info("Errors: %d", stats["errors"])
    logger.info("Total files on disk: %d (%s)", total_files, format_size(total_bytes))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
