"""
Assemble the 644 OpenAI embedding shards into a single memmap + build a new
FAISS IVFFlat index.

Input:
  /mnt/data2/openai_embed/embeddings_shards/batch_XXXX.npy       (49K × 1536 f32)
  /mnt/data2/openai_embed/embeddings_shards/batch_XXXX.ids.npy   (49K int64, chunk_ids)
  /mnt/data2/openai_embed/metadata.jsonl                          (31.5M chunk records)

Output:
  /mnt/data2/openai_embed/embeddings.npy    memmap, shape (31536589, 1536) f32
  /mnt/data2/openai_embed/faiss.index       IVFFlat with IP metric
  /mnt/data2/openai_embed/filled_mask.npy   bool array — which chunk_ids were filled

Phases:
  assemble   read all shards, write embeddings into the big memmap at chunk_id
             positions. track which chunk_ids are filled.
  index      build FAISS IVFFlat IP index from the filled embeddings.

Run:
  python3 assemble_openai_embed_index.py assemble
  python3 assemble_openai_embed_index.py index
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/mnt/data2/openai_embed")
SHARDS_DIR = WORK_DIR / "embeddings_shards"
STATE_PATH = WORK_DIR / "state.json"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npy"
FILLED_MASK_PATH = WORK_DIR / "filled_mask.npy"
INDEX_PATH = WORK_DIR / "faiss.index"

EMBED_DIM = 1536


def phase_assemble():
    state = json.load(open(STATE_PATH))
    n_chunks = state["total_chunks"]
    logger.info("Target: %d chunks × %d dims = %.1f GB",
                n_chunks, EMBED_DIM, n_chunks * EMBED_DIM * 4 / 1e9)

    shards = sorted(SHARDS_DIR.glob("batch_*.npy"))
    shards = [s for s in shards if not s.name.endswith(".ids.npy")]
    logger.info("Found %d embedding shards", len(shards))

    EMBEDDINGS_PATH.unlink(missing_ok=True)
    out = np.memmap(
        str(EMBEDDINGS_PATH),
        dtype=np.float32,
        mode="w+",
        shape=(n_chunks, EMBED_DIM),
    )
    filled = np.zeros(n_chunks, dtype=bool)

    total_rows = 0
    start = time.time()
    for i, shard_path in enumerate(shards):
        ids_path = shard_path.with_suffix(".ids.npy")
        if not ids_path.exists():
            logger.warning("missing ids for %s", shard_path.name)
            continue
        emb = np.load(shard_path, mmap_mode="r")
        ids = np.load(ids_path)
        if emb.shape[0] != ids.shape[0]:
            logger.warning("%s: embedding/ids length mismatch %d vs %d",
                           shard_path.name, emb.shape[0], ids.shape[0])
            continue
        valid = (ids >= 0) & (ids < n_chunks)
        if not valid.all():
            logger.warning("%s: %d out-of-range chunk_ids dropped",
                           shard_path.name, (~valid).sum())
            emb = emb[valid]
            ids = ids[valid]
        out[ids] = emb
        filled[ids] = True
        total_rows += emb.shape[0]
        if (i + 1) % 50 == 0 or (i + 1) == len(shards):
            elapsed = time.time() - start
            logger.info("[%d/%d] shards, %d rows filled, %.0fs elapsed",
                        i + 1, len(shards), total_rows, elapsed)

    out.flush()
    np.save(FILLED_MASK_PATH, filled)

    n_filled = int(filled.sum())
    logger.info("=" * 60)
    logger.info("ASSEMBLE COMPLETE")
    logger.info("=" * 60)
    logger.info("Filled: %d / %d chunks (%.1f%%)",
                n_filled, n_chunks, 100 * n_filled / max(n_chunks, 1))
    logger.info("Embeddings: %s", EMBEDDINGS_PATH)
    logger.info("Filled mask: %s", FILLED_MASK_PATH)


def phase_index():
    import faiss
    state = json.load(open(STATE_PATH))
    n_chunks = state["total_chunks"]

    logger.info("Loading embeddings memmap from %s", EMBEDDINGS_PATH)
    emb = np.memmap(
        str(EMBEDDINGS_PATH),
        dtype=np.float32,
        mode="r",
        shape=(n_chunks, EMBED_DIM),
    )
    filled = np.load(FILLED_MASK_PATH)
    filled_idxs = np.where(filled)[0]
    n_filled = len(filled_idxs)
    logger.info("Filled: %d / %d (%.1f%%)", n_filled, n_chunks, 100 * n_filled / n_chunks)

    if n_filled < n_chunks:
        logger.warning("Index will contain zero vectors at %d unfilled positions",
                       n_chunks - n_filled)

    nlist = min(int(np.sqrt(n_chunks)), 4096)
    logger.info("Building IndexIVFFlat IP, nlist=%d", nlist)
    quantizer = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train on a 500K random sample from filled positions
    logger.info("Training on 500K random sample...")
    np.random.seed(0)
    sample_size = min(500_000, n_filled)
    sample_idxs = np.random.choice(filled_idxs, size=sample_size, replace=False)
    sample = np.ascontiguousarray(emb[sample_idxs])
    index.train(sample)
    del sample
    logger.info("Training complete.")

    # Add all vectors in 1M chunks (including zero vectors for unfilled — they're
    # just not going to match anything meaningful)
    logger.info("Adding all %d vectors in 1M chunks...", n_chunks)
    chunk_size = 1_000_000
    for start_i in range(0, n_chunks, chunk_size):
        end_i = min(start_i + chunk_size, n_chunks)
        block = np.ascontiguousarray(emb[start_i:end_i])
        index.add(block)
        del block
        logger.info("  added %d/%d", end_i, n_chunks)

    index.nprobe = 32
    faiss.write_index(index, str(INDEX_PATH))
    gb = INDEX_PATH.stat().st_size / 1e9
    logger.info("Saved %s (%.1f GB)", INDEX_PATH, gb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["assemble", "index"])
    args = parser.parse_args()
    {"assemble": phase_assemble, "index": phase_index}[args.phase]()


if __name__ == "__main__":
    main()
