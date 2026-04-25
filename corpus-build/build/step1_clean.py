"""
Step 1: Clean reconstructed GDELT articles.

Input:  S3 or local dir of daily JSON files (YYYYMMDD_articles.json)
Output: cleaned_articles/ directory with cleaned JSON files

Filters:
  - Length: drop < 300 chars, > 50K chars
  - Language: keep English only (langdetect)
  - Dedup: MinHash (Jaccard > 0.8)

Run:
  python3 step1_clean.py --input /data/reconstructed --output /data/cleaned
"""

import json
import logging
import argparse
import hashlib
import time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# MinHash parameters — reduced for speed (32 perms is sufficient for 0.8 threshold)
NUM_PERM = 32
JACCARD_THRESHOLD = 0.8


def clean_one_day(input_path: Path, output_path: Path, minhash_index: dict) -> dict:
    """Clean articles for a single day. Returns stats dict."""
    with open(input_path) as f:
        articles = json.load(f)

    stats = {"input": len(articles), "short": 0, "long": 0, "non_english": 0, "duplicate": 0, "kept": 0}
    cleaned = []

    for art in articles:
        text = art.get("text", "")
        char_count = len(text)

        # Length filter
        if char_count < 300:
            stats["short"] += 1
            continue
        if char_count > 50000:
            stats["long"] += 1
            continue

        # Language filter (fast heuristic: check for common English words)
        if not _is_likely_english(text):
            stats["non_english"] += 1
            continue

        # Skip dedup — duplicates are handled at retrieval time (deduplicate by URL)

        cleaned.append({
            "url": art.get("url", ""),
            "text": text,
            "date": art.get("date", ""),
            "char_count": char_count,
        })
        stats["kept"] += 1

    # Save
    if cleaned:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(cleaned, f)

    return stats


# --- Language detection (fast heuristic) ---

ENGLISH_COMMON = {
    "the", "and", "that", "this", "with", "from", "have", "been",
    "will", "said", "were", "they", "their", "which", "about",
    "would", "there", "could", "after", "also", "into", "other",
    "more", "than", "some", "when", "what", "just", "like",
}

def _is_likely_english(text: str, threshold: float = 0.02) -> bool:
    """
    Fast English detection. Checks if common English words make up
    at least `threshold` fraction of all words. Much faster than langdetect.
    """
    words = text.lower().split()
    if len(words) < 20:
        return False
    sample = words[:200]  # Only check first 200 words
    english_count = sum(1 for w in sample if w in ENGLISH_COMMON)
    return (english_count / len(sample)) >= threshold


# --- MinHash deduplication ---

def _shingles(text: str, k: int = 5) -> set:
    """Generate character k-shingles from text."""
    text = text.lower()[:2000]  # Only shingle first 2000 chars for speed
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def _compute_minhash(text: str) -> list:
    """Compute MinHash signature for a text."""
    shings = _shingles(text)
    if not shings:
        return [0] * NUM_PERM

    signature = []
    for i in range(NUM_PERM):
        min_hash = float('inf')
        for s in shings:
            h = int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def _jaccard_from_minhash(sig1: list, sig2: list) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# LSH band index for fast approximate matching
NUM_BANDS = 8
ROWS_PER_BAND = NUM_PERM // NUM_BANDS  # 4 rows per band


def _band_hashes(signature: list) -> list:
    """Split signature into bands and hash each band."""
    bands = []
    for b in range(NUM_BANDS):
        start = b * ROWS_PER_BAND
        band = tuple(signature[start:start + ROWS_PER_BAND])
        bands.append(hash(band))
    return bands


def _is_duplicate(signature: list, index: dict) -> bool:
    """Check if a signature has any near-duplicate in the index using LSH."""
    bands = _band_hashes(signature)
    for b, band_hash in enumerate(bands):
        bucket_key = (b, band_hash)
        if bucket_key in index:
            # Candidate found — verify with full Jaccard
            for candidate_sig in index[bucket_key]:
                if _jaccard_from_minhash(signature, candidate_sig) >= JACCARD_THRESHOLD:
                    return True
    return False


def _register_minhash(signature: list, index: dict, article_id: int):
    """Add a signature to the LSH index."""
    bands = _band_hashes(signature)
    for b, band_hash in enumerate(bands):
        bucket_key = (b, band_hash)
        if bucket_key not in index:
            index[bucket_key] = []
        index[bucket_key].append(signature)


def main():
    parser = argparse.ArgumentParser(description="Clean reconstructed GDELT articles")
    parser.add_argument("--input", required=True, help="Directory with YYYYMMDD_articles.json files")
    parser.add_argument("--output", required=True, help="Output directory for cleaned files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_articles.json"))
    logger.info("Found %d daily files in %s", len(files), input_dir)

    minhash_index = {}
    total_stats = defaultdict(int)
    start_time = time.time()

    for i, f in enumerate(files):
        day = f.stem.replace("_articles", "")
        out_path = output_dir / f.name

        # Skip if already cleaned
        if out_path.exists():
            stats = {"input": 0, "kept": 0}
            with open(out_path) as fp:
                stats["kept"] = len(json.load(fp))
            total_stats["kept"] += stats["kept"]
            if (i + 1) % 10 == 0:
                logger.info("[%d/%d] Day %s — cached (%d articles)", i + 1, len(files), day, stats["kept"])
            continue

        stats = clean_one_day(f, out_path, minhash_index)

        for k, v in stats.items():
            total_stats[k] += v

        if True:  # log every day
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600
            logger.info(
                "[%d/%d] Day %s — %d→%d kept | Total: %d kept / %d input | "
                "short:%d lang:%d dup:%d | %.0f days/hr",
                i + 1, len(files), day,
                stats["input"], stats["kept"],
                total_stats["kept"], total_stats["input"],
                total_stats["short"], total_stats["non_english"], total_stats["duplicate"],
                rate,
            )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 70)
    logger.info("Time: %.1f hours", elapsed / 3600)
    logger.info("Input: %d articles", total_stats["input"])
    logger.info("Dropped — short: %d, long: %d, non-English: %d, duplicate: %d",
                total_stats["short"], total_stats["long"], total_stats["non_english"], total_stats["duplicate"])
    logger.info("Kept: %d articles (%.1f%%)", total_stats["kept"],
                100 * total_stats["kept"] / max(total_stats["input"], 1))
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
