"""
Parallel download of all 644 OpenAI embedding batch outputs.

The sequential check phase is too slow (~30s per batch × 644 = 5h).
This script uses a thread pool to download 20 batches concurrently.

State tracked in /mnt/data2/openai_embed/state.json — same format as
step2_openai_embed.py so subsequent assemble/index phases work unchanged.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/mnt/data2/openai_embed")
BATCH_OUTPUT_DIR = WORK_DIR / "batch_outputs"
STATE_PATH = WORK_DIR / "state.json"
MAX_WORKERS = 20
MAX_RETRIES = 3


def download_one(client: OpenAI, key: str, info: dict):
    out_path = BATCH_OUTPUT_DIR / f"{key}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return key, "already", info

    for attempt in range(MAX_RETRIES):
        try:
            batch = client.batches.retrieve(info["batch_id"])
            if batch.status != "completed":
                return key, f"not_complete:{batch.status}", info
            if not batch.output_file_id:
                return key, "no_output_file", info
            content = client.files.content(batch.output_file_id).read()
            if len(content) < 1000:
                raise RuntimeError(f"output suspiciously small: {len(content)} bytes")
            with open(out_path, "wb") as f:
                f.write(content)
            info["downloaded_at"] = time.time()
            info["output_file_id"] = batch.output_file_id
            info["status"] = batch.status
            return key, "downloaded", info
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return key, f"error:{str(e)[:200]}", info


def main():
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = json.load(open(STATE_PATH))
    batches = state.get("batches", {})
    logger.info("Total batches in state: %d", len(batches))

    pending = {k: v for k, v in batches.items()
               if not (BATCH_OUTPUT_DIR / f"{k}.jsonl").exists()
               or (BATCH_OUTPUT_DIR / f"{k}.jsonl").stat().st_size < 1000}
    logger.info("Pending downloads: %d", len(pending))

    if not pending:
        logger.info("Everything already downloaded.")
        return

    client = OpenAI()
    counts = {"downloaded": 0, "already": 0, "error": 0, "not_complete": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, client, k, v): k for k, v in pending.items()}
        done = 0
        for fut in as_completed(futures):
            key, status, info = fut.result()
            done += 1
            if status == "downloaded":
                counts["downloaded"] += 1
                state["batches"][key] = info
            elif status == "already":
                counts["already"] += 1
            elif status.startswith("not_complete"):
                counts["not_complete"] += 1
            else:
                counts["error"] += 1
                logger.warning("%s: %s", key, status)
            if done % 50 == 0 or done == len(pending):
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(pending) - done) / rate if rate > 0 else 0
                logger.info("[%d/%d] downloaded=%d errors=%d, rate=%.1f/s, eta=%.0fs",
                            done, len(pending), counts["downloaded"], counts["error"],
                            rate, eta)
                # Persist state mid-run in case of crash
                with open(STATE_PATH, "w") as f:
                    json.dump(state, f, indent=2)

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    for k, v in counts.items():
        logger.info("  %s: %d", k, v)


if __name__ == "__main__":
    main()
