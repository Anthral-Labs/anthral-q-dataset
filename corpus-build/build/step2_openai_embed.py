"""
Step 2 (OpenAI version): Embed the full corpus via text-embedding-3-small.

Uses OpenAI Batch API for 50% discount. The full 31.5M chunks cost ~$75.

Pipeline phases:

  build     — read /data/cleaned/, chunk articles, write metadata.jsonl + batch input files
  submit    — upload all batch input files, create batches, save state
  check     — poll all batches, download outputs when ready
  assemble  — parse outputs into embeddings.npy (chunk_id order)
  index     — build FAISS IVFFlat index from embeddings.npy

Run each phase as:
  python3 step2_openai_embed.py build
  python3 step2_openai_embed.py submit
  python3 step2_openai_embed.py check
  python3 step2_openai_embed.py assemble
  python3 step2_openai_embed.py index

State is persisted in /mnt/data2/openai_embed/state.json so you can run
check/submit multiple times safely.
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
CLEANED_DIR = Path("/data/cleaned")
WORK_DIR = Path("/mnt/data2/openai_embed")
BATCH_INPUT_DIR = WORK_DIR / "batch_inputs"
BATCH_OUTPUT_DIR = WORK_DIR / "batch_outputs"
METADATA_PATH = WORK_DIR / "metadata.jsonl"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npy"
INDEX_PATH = WORK_DIR / "faiss.index"
STATE_PATH = WORK_DIR / "state.json"

# Embedding config
MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNKS_PER_REQUEST = 1000      # each batch request embeds 1000 texts in one API call
REQUESTS_PER_FILE = 49         # 49 × 1000 = 49K inputs per file (OpenAI limit: 50K per batch)
CHUNKS_PER_FILE = CHUNKS_PER_REQUEST * REQUESTS_PER_FILE  # 49K chunks per file
MAX_TEXT_CHARS = 8000          # cap input text (embedding-3 max ~8191 tokens ~32K chars, plenty)

# Step2 compatibility — same chunking
def chunk_article(text: str, max_chars: int = 2000) -> list:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        space_idx = text.rfind(" ", start, end)
        if space_idx > start:
            end = space_idx
        chunks.append(text[start:end])
        start = end + 1
    return chunks


# ---------- BUILD ----------

def phase_build():
    """Read cleaned articles, write metadata.jsonl + batch input JSONL files."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CLEANED_DIR.glob("*_articles.json"))
    logger.info("Reading %d day files from %s", len(files), CLEANED_DIR)

    chunk_id = 0
    batch_file_idx = 0
    current_batch_file = None
    current_requests_in_file = 0
    current_batch_texts = []
    current_batch_start_chunk_id = 0

    metadata_f = open(METADATA_PATH, "w")

    def flush_current_request():
        """Write one /v1/embeddings request with the accumulated texts."""
        nonlocal current_batch_texts, current_batch_start_chunk_id
        nonlocal current_requests_in_file, current_batch_file, batch_file_idx

        if not current_batch_texts:
            return

        req = {
            "custom_id": f"b{batch_file_idx:04d}-c{current_batch_start_chunk_id}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": MODEL,
                "input": current_batch_texts,
            },
        }
        if current_batch_file is None:
            path = BATCH_INPUT_DIR / f"batch_{batch_file_idx:04d}.jsonl"
            current_batch_file = open(path, "w")
            logger.info("Opened batch file %s", path)

        current_batch_file.write(json.dumps(req) + "\n")
        current_requests_in_file += 1
        current_batch_texts = []

        if current_requests_in_file >= REQUESTS_PER_FILE:
            current_batch_file.close()
            logger.info("Closed batch file %d (%d requests)",
                        batch_file_idx, current_requests_in_file)
            current_batch_file = None
            current_requests_in_file = 0
            batch_file_idx += 1

    start_time = time.time()
    for fi, f in enumerate(files):
        day = f.stem.replace("_articles", "")
        with open(f) as fp:
            articles = json.load(fp)

        for art in articles:
            text = art.get("text", "")
            if not text or len(text) < 50:
                continue
            chunks = chunk_article(text)
            for ci, chunk_text in enumerate(chunks):
                # Write metadata
                metadata_f.write(json.dumps({
                    "chunk_id": chunk_id,
                    "article_url": art.get("url", ""),
                    "article_date": art.get("date", day),
                    "chunk_index": ci,
                }) + "\n")

                # Start new batch request at every CHUNKS_PER_REQUEST boundary
                if len(current_batch_texts) == 0:
                    current_batch_start_chunk_id = chunk_id

                current_batch_texts.append(chunk_text[:MAX_TEXT_CHARS])
                chunk_id += 1

                if len(current_batch_texts) >= CHUNKS_PER_REQUEST:
                    flush_current_request()

        if (fi + 1) % 20 == 0 or (fi + 1) == len(files):
            elapsed = time.time() - start_time
            logger.info("[%d/%d] %s — total chunks=%d batch_files=%d elapsed=%.0fs",
                        fi + 1, len(files), day, chunk_id, batch_file_idx, elapsed)

    # Flush remaining partial request + close last file
    flush_current_request()
    if current_batch_file is not None:
        current_batch_file.close()
        logger.info("Closed final batch file %d (%d requests)",
                    batch_file_idx, current_requests_in_file)

    metadata_f.close()

    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("Total chunks: %d", chunk_id)
    logger.info("Total batch files: %d", batch_file_idx + 1)
    logger.info("Metadata: %s", METADATA_PATH)
    logger.info("Batch inputs: %s", BATCH_INPUT_DIR)

    # Save state
    state = {
        "phase": "build_done",
        "total_chunks": chunk_id,
        "num_batch_files": batch_file_idx + 1,
        "model": MODEL,
        "embed_dim": EMBED_DIM,
        "batches": {},  # populated by submit
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------- SUBMIT ----------

def phase_submit():
    from openai import OpenAI
    state = json.load(open(STATE_PATH))
    client = OpenAI()
    batch_files = sorted(BATCH_INPUT_DIR.glob("batch_*.jsonl"))
    logger.info("Submitting %d batch files", len(batch_files))

    state.setdefault("batches", {})
    n_submitted = 0
    n_skipped = 0
    for bf in batch_files:
        key = bf.stem  # batch_0000
        if key in state["batches"] and state["batches"][key].get("batch_id"):
            n_skipped += 1
            continue
        logger.info("Uploading %s...", bf.name)
        file_obj = client.files.create(file=open(bf, "rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"task": "openai_embed_corpus", "file": bf.name},
        )
        state["batches"][key] = {
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "status": batch.status,
            "submitted_at": time.time(),
        }
        logger.info("  %s → %s status=%s", bf.name, batch.id, batch.status)
        n_submitted += 1
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)

    logger.info("Submitted: %d, skipped (already submitted): %d",
                n_submitted, n_skipped)


# ---------- CHECK ----------

def phase_check():
    from openai import OpenAI
    state = json.load(open(STATE_PATH))
    client = OpenAI()
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_completed = 0
    n_in_progress = 0
    n_failed = 0
    n_downloaded = 0
    n_already_downloaded = 0

    for key, info in state["batches"].items():
        out_path = BATCH_OUTPUT_DIR / f"{key}.jsonl"
        if out_path.exists() and info.get("downloaded_at"):
            n_already_downloaded += 1
            continue
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        rc = batch.request_counts
        if batch.status == "completed":
            n_completed += 1
            if batch.output_file_id:
                content = client.files.content(batch.output_file_id).read()
                with open(out_path, "wb") as f:
                    f.write(content)
                info["downloaded_at"] = time.time()
                info["output_file_id"] = batch.output_file_id
                n_downloaded += 1
                logger.info("downloaded %s (%d completed, %d failed)",
                            key, rc.completed if rc else "?", rc.failed if rc else "?")
        elif batch.status in ("failed", "expired", "cancelled"):
            n_failed += 1
            info["error_at"] = time.time()
            logger.warning("%s: %s", key, batch.status)
        else:
            n_in_progress += 1
            if rc:
                logger.info("%s: %s (%d/%d)",
                            key, batch.status, rc.completed, rc.total)

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    logger.info("=" * 60)
    logger.info("CHECK SUMMARY")
    logger.info("=" * 60)
    logger.info("Total batches: %d", len(state["batches"]))
    logger.info("  in progress:       %d", n_in_progress)
    logger.info("  newly completed:   %d", n_downloaded)
    logger.info("  already downloaded: %d", n_already_downloaded)
    logger.info("  failed:            %d", n_failed)


# ---------- ASSEMBLE ----------

def phase_assemble():
    import numpy as np
    state = json.load(open(STATE_PATH))
    n_chunks = state["total_chunks"]
    logger.info("Assembling %d chunks × %d dims into %s",
                n_chunks, EMBED_DIM, EMBEDDINGS_PATH)

    # Memory-mapped output array (saves RAM)
    # 31.5M × 1536 × 4 bytes = 193 GB — needs disk
    out = np.memmap(str(EMBEDDINGS_PATH), dtype=np.float32, mode="w+",
                    shape=(n_chunks, EMBED_DIM))

    filled = np.zeros(n_chunks, dtype=bool)

    batch_files = sorted(BATCH_OUTPUT_DIR.glob("batch_*.jsonl"))
    logger.info("Parsing %d batch output files", len(batch_files))

    for bi, bf in enumerate(batch_files):
        with open(bf) as fp:
            for line in fp:
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                # custom_id format: "b0000-c{start_chunk_id}"
                try:
                    start_chunk_id = int(cid.split("-c")[1])
                except (IndexError, ValueError):
                    logger.warning("bad custom_id: %s", cid)
                    continue
                try:
                    data = obj["response"]["body"]["data"]
                except (KeyError, TypeError):
                    logger.warning("missing response data for %s", cid)
                    continue
                for i, item in enumerate(data):
                    emb = item.get("embedding")
                    if emb is None or len(emb) != EMBED_DIM:
                        continue
                    cid_i = start_chunk_id + i
                    if cid_i >= n_chunks:
                        logger.warning("chunk_id %d out of range", cid_i)
                        continue
                    out[cid_i] = np.asarray(emb, dtype=np.float32)
                    filled[cid_i] = True
        if (bi + 1) % 10 == 0:
            logger.info("  [%d/%d] filled=%d/%d",
                        bi + 1, len(batch_files), int(filled.sum()), n_chunks)

    out.flush()
    n_missing = int((~filled).sum())
    logger.info("Assemble complete. Missing: %d / %d", n_missing, n_chunks)
    if n_missing:
        logger.warning("Some embeddings are missing — the FAISS index will"
                       " include zero vectors at those positions.")


# ---------- INDEX ----------

def phase_index():
    import numpy as np
    import faiss
    state = json.load(open(STATE_PATH))
    n_chunks = state["total_chunks"]

    logger.info("Loading embeddings (memmap) %s", EMBEDDINGS_PATH)
    emb = np.memmap(str(EMBEDDINGS_PATH), dtype=np.float32, mode="r",
                    shape=(n_chunks, EMBED_DIM))

    nlist = min(int(np.sqrt(n_chunks)), 4096)
    logger.info("Building IVFFlat index: %d vectors × %d dims, nlist=%d",
                n_chunks, EMBED_DIM, nlist)

    quantizer = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist,
                               faiss.METRIC_INNER_PRODUCT)

    # Train on a sample (500K random rows, matching step2 behavior)
    logger.info("Training index on 500K sample...")
    np.random.seed(0)
    sample_idx = np.random.choice(n_chunks, size=min(500_000, n_chunks), replace=False)
    sample = np.ascontiguousarray(emb[sample_idx])
    index.train(sample)
    del sample
    logger.info("Training complete.")

    # Add vectors in chunks of 1M
    logger.info("Adding vectors in chunks of 1M...")
    chunk_size = 1_000_000
    for start in range(0, n_chunks, chunk_size):
        end = min(start + chunk_size, n_chunks)
        block = np.ascontiguousarray(emb[start:end])
        index.add(block)
        del block
        logger.info("  Added %d/%d", end, n_chunks)

    index.nprobe = 32
    faiss.write_index(index, str(INDEX_PATH))
    size_gb = INDEX_PATH.stat().st_size / 1e9
    logger.info("FAISS index saved: %s (%.1f GB)", INDEX_PATH, size_gb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "submit", "check", "assemble", "index"])
    args = parser.parse_args()
    {
        "build": phase_build,
        "submit": phase_submit,
        "check": phase_check,
        "assemble": phase_assemble,
        "index": phase_index,
    }[args.phase]()


if __name__ == "__main__":
    main()
