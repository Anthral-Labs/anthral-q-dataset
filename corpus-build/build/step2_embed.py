"""
Step 2: Chunk articles and embed with Qwen3-Embedding-8B.

Uses sentence-transformers with flash_attention_2 for fast GPU embedding.
Processes one day at a time, streams embeddings to disk.

Run:
  python3 step2_embed.py --input /data/cleaned --output /data/embeddings --batch-size 256
"""

import json
import logging
import argparse
import time
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(model_name: str = "Qwen/Qwen3-Embedding-8B"):
    """Load embedding model via sentence-transformers with flash attention."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading %s...", model_name)

    import torch
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    logger.info("Model loaded.")
    return model


def chunk_article(text: str, max_chars: int = 2000) -> list:
    """Split long articles. Most are <2000 chars and pass through as-is."""
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


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Embed articles with Qwen3-Embedding-8B")
    parser.add_argument("--input", required=True, help="Directory with cleaned article JSONs")
    parser.add_argument("--output", required=True, help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model name (0.6B is 10x faster, still top-tier)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    files = sorted(input_dir.glob("*_articles.json"))
    logger.info("Found %d cleaned files", len(files))

    # Process day by day, save embeddings per day to avoid memory issues
    metadata_file = open(output_dir / "metadata.jsonl", "w")
    chunk_id = 0
    total_articles = 0
    total_chunks = 0
    start_time = time.time()

    for file_idx, f in enumerate(files):
        day = f.stem.replace("_articles", "")
        day_emb_path = output_dir / f"{day}_embeddings.npy"

        # Skip if already done
        if day_emb_path.exists():
            # Count existing chunks from metadata
            total_articles += 1  # approximate
            if (file_idx + 1) % 10 == 0:
                logger.info("[%d/%d] Day %s — cached, skipping", file_idx + 1, len(files), day)
            continue

        # Load articles
        with open(f) as fp:
            articles = json.load(fp)

        # Build chunks
        day_texts = []
        day_meta = []
        for art in articles:
            text = art.get("text", "")
            if not text or len(text) < 50:
                continue

            chunks = chunk_article(text)
            for ci, chunk_text in enumerate(chunks):
                day_texts.append(chunk_text)
                day_meta.append({
                    "chunk_id": chunk_id,
                    "article_url": art.get("url", ""),
                    "article_date": art.get("date", day),
                    "chunk_index": ci,
                })
                chunk_id += 1

        total_articles += len(articles)
        total_chunks += len(day_texts)

        if not day_texts:
            continue

        # Embed entire day — sentence-transformers handles batching internally
        logger.info("  Embedding %d chunks for day %s...", len(day_texts), day)
        day_embeddings = model.encode(
            day_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Save embeddings for this day
        np.save(day_emb_path, day_embeddings.astype(np.float32))

        # Write metadata
        for meta in day_meta:
            metadata_file.write(json.dumps(meta) + "\n")

        elapsed = time.time() - start_time
        rate = total_chunks / elapsed if elapsed > 0 else 0
        eta = (len(files) - file_idx - 1) / ((file_idx + 1) / elapsed) if elapsed > 0 else 0

        logger.info(
            "[%d/%d] Day %s | %d articles, %d chunks | %.0f chunks/sec | ETA %s | disk: %s",
            file_idx + 1, len(files), day,
            len(articles), len(day_texts),
            rate, format_time(eta),
            _disk_usage(output_dir),
        )

    metadata_file.close()

    # Build combined FAISS index from all daily embedding files
    logger.info("")
    logger.info("Building FAISS index from daily embeddings...")
    _build_combined_index(output_dir)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 70)
    logger.info("Time: %s", format_time(elapsed))
    logger.info("Articles: %d", total_articles)
    logger.info("Chunks: %d", total_chunks)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)


def _build_combined_index(output_dir: Path):
    """Build full FAISS IVFFlat index from daily embeddings. Adds vectors day-by-day to manage memory."""
    import faiss

    emb_files = sorted(output_dir.glob("*_embeddings.npy"))
    logger.info("Loading %d daily embedding files...", len(emb_files))

    # First pass: get dimensions and total count
    total_n = 0
    dim = None
    for ef in emb_files:
        emb = np.load(ef, mmap_mode='r')
        total_n += emb.shape[0]
        if dim is None:
            dim = emb.shape[1]
    logger.info("Total vectors: %d, dim: %d (%.1f GB uncompressed)", total_n, dim, total_n * dim * 4 / 1e9)

    nlist = min(int(np.sqrt(total_n)), 4096)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train on a sample
    logger.info("Training IVFFlat index (nlist=%d)...", nlist)
    train_chunks = []
    train_n = 0
    for ef in emb_files:
        emb = np.load(ef)
        train_chunks.append(emb)
        train_n += emb.shape[0]
        if train_n >= 500000:
            break
    train_data = np.vstack(train_chunks)[:500000]
    index.train(train_data)
    del train_data, train_chunks
    logger.info("Training complete.")

    # Add vectors day by day to avoid loading everything into memory at once
    logger.info("Adding vectors day by day...")
    added = 0
    for i, ef in enumerate(emb_files):
        emb = np.load(ef)
        index.add(emb)
        added += emb.shape[0]
        del emb
        if (i + 1) % 20 == 0:
            logger.info("  Added %d/%d vectors (%d/%d files)", added, total_n, i + 1, len(emb_files))

    index.nprobe = 32
    faiss.write_index(index, str(output_dir / "faiss.index"))
    index_size = Path(output_dir / "faiss.index").stat().st_size / 1e9
    logger.info("FAISS IVFFlat index saved: %d vectors, %d dims, %.1f GB on disk", total_n, dim, index_size)


def _disk_usage(path: Path) -> str:
    """Get human-readable disk usage of a directory."""
    import subprocess
    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    return result.stdout.split()[0] if result.stdout else "?"


if __name__ == "__main__":
    main()
