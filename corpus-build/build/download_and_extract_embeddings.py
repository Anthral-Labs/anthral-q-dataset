"""
Download OpenAI embedding batch outputs and write directly to compact .npy shards.

Replaces the naive JSONL download (which would need ~900 GB for 644 batches).

For each batch:
  1. GET file content from OpenAI
  2. Stream-parse the JSONL response
  3. Extract embeddings (1536 floats each) into numpy array
  4. Save to /mnt/data2/openai_embed/embeddings_shards/{key}.npy
     plus a sidecar /mnt/data2/openai_embed/embeddings_shards/{key}.idx.json
     with chunk_id ranges so we know where each shard maps

Memory per batch: 49K rows × 1536 cols × 4 bytes = ~287 MB
Disk per batch:   same — 287 MB
Total for 644 batches: ~184 GB (fits on /mnt/data2's 1 TB)

Resumable: skips shards that already exist.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/mnt/data2/openai_embed")
SHARDS_DIR = WORK_DIR / "embeddings_shards"
STATE_PATH = WORK_DIR / "state.json"
EMBED_DIM = 1536
MAX_WORKERS = 8     # smaller — each download is ~1.4 GB raw, peaks ~12 GB total
MAX_RETRIES = 3


def parse_one_batch(content: bytes):
    """Parse a batch output JSONL bytes blob and return (embeddings, chunk_ids).

    Each line is one /v1/embeddings response. The custom_id encodes the start
    chunk_id: 'b{batch_num}-c{start_chunk_id}'. The response.body.data list
    contains the embeddings in input order.
    """
    embeddings = []
    chunk_ids = []
    for line in content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id", "")
        try:
            start_chunk_id = int(cid.split("-c")[1])
        except (IndexError, ValueError):
            logger.warning("bad custom_id: %s", cid)
            continue
        try:
            data = obj["response"]["body"]["data"]
        except (KeyError, TypeError):
            continue
        for i, item in enumerate(data):
            emb = item.get("embedding")
            if emb is None or len(emb) != EMBED_DIM:
                continue
            embeddings.append(emb)
            chunk_ids.append(start_chunk_id + i)
    if not embeddings:
        return None, None
    arr = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(chunk_ids, dtype=np.int64)
    return arr, ids


def process_one(client: OpenAI, key: str, info: dict):
    npy_path = SHARDS_DIR / f"{key}.npy"
    ids_path = SHARDS_DIR / f"{key}.ids.npy"
    if npy_path.exists() and ids_path.exists():
        return key, "already", None

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            batch = client.batches.retrieve(info["batch_id"])
            if batch.status != "completed" or not batch.output_file_id:
                return key, f"not_complete:{batch.status}", None
            content = client.files.content(batch.output_file_id).read()
            arr, ids = parse_one_batch(content)
            del content
            if arr is None:
                return key, "empty", None
            np.save(npy_path, arr)
            np.save(ids_path, ids)
            return key, "ok", arr.shape
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(2 ** attempt)
    return key, f"error:{last_err}", None


def main():
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    state = json.load(open(STATE_PATH))
    batches = state.get("batches", {})
    logger.info("Total batches: %d", len(batches))

    pending = {k: v for k, v in batches.items()
               if not (SHARDS_DIR / f"{k}.npy").exists()}
    logger.info("Pending: %d", len(pending))
    if not pending:
        logger.info("All shards exist.")
        return

    client = OpenAI()
    counts = {"ok": 0, "already": 0, "error": 0, "not_complete": 0, "empty": 0}
    total_rows = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, client, k, v): k for k, v in pending.items()}
        done = 0
        for fut in as_completed(futures):
            key, status, shape = fut.result()
            done += 1
            if status == "ok":
                counts["ok"] += 1
                if shape is not None:
                    total_rows += shape[0]
            elif status == "already":
                counts["already"] += 1
            elif status.startswith("not_complete"):
                counts["not_complete"] += 1
            elif status == "empty":
                counts["empty"] += 1
            else:
                counts["error"] += 1
                logger.warning("%s: %s", key, status)
            if done % 20 == 0 or done == len(pending):
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(pending) - done) / rate if rate > 0 else 0
                logger.info("[%d/%d] ok=%d errors=%d rows=%d rate=%.2f/s eta=%.0fs",
                            done, len(pending), counts["ok"], counts["error"],
                            total_rows, rate, eta)

    logger.info("=" * 60)
    logger.info("DOWNLOAD + EXTRACT COMPLETE")
    logger.info("=" * 60)
    for k, v in counts.items():
        logger.info("  %s: %d", k, v)
    logger.info("Total rows extracted: %d", total_rows)


if __name__ == "__main__":
    main()
