"""
Step 2: Reconstruct full articles from GDELT NGrams files.

Processes each day's NGrams files:
  1. Parse all .webngrams.json.gz files for that day
  2. Group n-grams by article URL
  3. Merge overlapping n-grams into full text
  4. Save reconstructed articles as one JSON per day

English-only. Skips articles shorter than 200 chars.

Usage:
  python3 reconstruct.py
  python3 reconstruct.py --workers 4
"""

import os
import json
import gzip
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/ubuntu/gdelt/reconstruct.log"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("/home/ubuntu/gdelt/raw")
OUTPUT_DIR = Path("/home/ubuntu/gdelt/reconstructed")
MIN_ARTICLE_CHARS = 200


def parse_ngrams_file(filepath: Path) -> dict:
    """Parse a single .webngrams.json.gz file. Returns {url: [(pos, snippet)]}."""
    articles = defaultdict(list)

    try:
        with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    url = record.get("url", "")
                    lang = record.get("lang", "")

                    if lang and lang != "en":
                        continue

                    # Each record has pre (before), ngram (target word), post (after)
                    # Full context snippet = pre + ngram + post
                    pre = record.get("pre", "")
                    ngram = record.get("ngram", "")
                    post = record.get("post", "")
                    pos = record.get("pos", 0)

                    if url and (pre or ngram or post):
                        snippet = f"{pre} {ngram} {post}".strip()
                        articles[url].append((pos, snippet))
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as e:
        logger.debug("Error parsing %s: %s", filepath.name, e)

    return dict(articles)


def merge_ngrams(fragments: list) -> str:
    """Reconstruct article text by merging overlapping context snippets."""
    if not fragments:
        return ""

    # Sort by position
    fragments.sort(key=lambda x: x[0])

    # Deduplicate by position (keep longest snippet at each position)
    by_pos = {}
    for pos, snippet in fragments:
        if pos not in by_pos or len(snippet) > len(by_pos[pos]):
            by_pos[pos] = snippet

    sorted_frags = sorted(by_pos.items())
    if not sorted_frags:
        return ""

    # Merge overlapping snippets
    text = sorted_frags[0][1]
    for _, snippet in sorted_frags[1:]:
        if snippet in text:
            continue
        # Find best overlap between end of text and start of snippet
        best_overlap = 0
        for overlap_len in range(min(len(text), len(snippet)), 0, -1):
            if text.endswith(snippet[:overlap_len]):
                best_overlap = overlap_len
                break
        text += snippet[best_overlap:]

    return text


def process_day(args_tuple):
    """Process all NGrams files for a single day. Run in subprocess."""
    day, filepaths = args_tuple
    output_file = OUTPUT_DIR / f"{day}_articles.json"

    # Skip if already done
    if output_file.exists():
        try:
            with open(output_file) as f:
                existing = json.load(f)
            return day, len(existing), True
        except (json.JSONDecodeError, IOError):
            pass  # Corrupted, redo

    # Parse all files for this day
    all_ngrams = defaultdict(list)
    for fp in filepaths:
        file_ngrams = parse_ngrams_file(fp)
        for url, ngrams in file_ngrams.items():
            all_ngrams[url].extend(ngrams)

    # Reconstruct each article
    articles = []
    for url, ngrams in all_ngrams.items():
        text = merge_ngrams(ngrams)
        if len(text) >= MIN_ARTICLE_CHARS:
            articles.append({
                "url": url,
                "text": text,
                "date": day,
                "char_count": len(text),
            })

    # Save
    with open(output_file, "w") as f:
        json.dump(articles, f)

    return day, len(articles), False


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Reconstruct articles from GDELT NGrams")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help=f"Parallel workers (default: {max(1, cpu_count() - 1)})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group files by day
    files = sorted(RAW_DIR.glob("*.webngrams.json.gz"))
    if not files:
        logger.error("No NGrams files found in %s. Run download_ngrams.py first.", RAW_DIR)
        return

    logger.info("=" * 70)
    logger.info("ARTICLE RECONSTRUCTION")
    logger.info("=" * 70)
    logger.info("NGrams files: %d", len(files))
    logger.info("Workers: %d", args.workers)

    # Group by day (first 8 chars of filename = YYYYMMDD)
    def day_key(f):
        return f.name[:8]

    days = []
    for day, day_files in groupby(files, key=day_key):
        days.append((day, list(day_files)))

    total_days = len(days)
    logger.info("Days to process: %d", total_days)

    # Check already done
    already_done = 0
    already_articles = 0
    for day, _ in days:
        output_file = OUTPUT_DIR / f"{day}_articles.json"
        if output_file.exists():
            already_done += 1
            try:
                with open(output_file) as f:
                    already_articles += len(json.load(f))
            except:
                pass

    if already_done > 0:
        logger.info("Already reconstructed: %d/%d days (%d articles)", already_done, total_days, already_articles)

    logger.info("Starting reconstruction...")
    logger.info("")

    start_time = time.time()
    completed = 0
    total_articles = already_articles
    skipped = 0

    with Pool(processes=args.workers) as pool:
        for day, num_articles, was_cached in pool.imap_unordered(process_day, days):
            completed += 1

            if was_cached:
                skipped += 1
                total_articles += num_articles  # Already counted but track
            else:
                total_articles += num_articles

            if True:  # log every day
                elapsed = time.time() - start_time
                days_done = completed - skipped  # Exclude cached from rate calc
                rate = days_done / elapsed if elapsed > 0 and days_done > 0 else 0
                eta = (total_days - completed) / rate if rate > 0 else 0
                pct = completed / total_days * 100

                # Disk usage
                disk_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*_articles.json"))

                logger.info(
                    "[%5.1f%%] Day %s | %d/%d days | %d articles total | "
                    "%s on disk | %.2f days/sec | ETA %s",
                    pct, day, completed, total_days, total_articles,
                    format_size(disk_bytes), rate, format_time(eta),
                )

    elapsed = time.time() - start_time

    # Final stats
    output_files = list(OUTPUT_DIR.glob("*_articles.json"))
    total_size = sum(f.stat().st_size for f in output_files)

    logger.info("")
    logger.info("=" * 70)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 70)
    logger.info("Time: %s", format_time(elapsed))
    logger.info("Days processed: %d (skipped %d cached)", total_days, skipped)
    logger.info("Total articles: %d", total_articles)
    logger.info("Output files: %d", len(output_files))
    logger.info("Output size: %s", format_size(total_size))
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
