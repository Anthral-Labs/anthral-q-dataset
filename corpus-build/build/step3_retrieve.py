"""
Step 3: Retrieve context for each Polymarket question.

Input:
  - FAISS index + metadata from step2 (on /mnt/data2)
  - Reconstructed articles (on /data/reconstructed) for text lookup
  - polymarket_final.json + search queries

Output:
  - retrieved_context_raw.json: top-5 chunks per question (before leakage check)

Applies Gate 1 (date buffer: article must be ≥30 days before resolution).
Gate 2 (LLM leakage check) runs separately in step4.

Run:
  python3 step3_retrieve.py
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
INDEX_DIR = Path("/mnt/data2")
ARTICLES_DIR = Path("/data/reconstructed")
QUESTIONS_PATH = Path("/data/questions/filtered/polymarket_final.json")
QUERIES_PATH = Path("/data/questions/polymarket_search_queries.json")
OUTPUT_PATH = Path("/data/retrieval/retrieved_context_raw.json")
DATE_BUFFER_DAYS = 30


def load_faiss_index():
    """Load FAISS index and metadata."""
    import faiss

    logger.info("Loading FAISS index from %s...", INDEX_DIR / "faiss.index")
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    logger.info("Index loaded: %d vectors", index.ntotal)

    logger.info("Loading metadata...")
    metadata = []
    with open(INDEX_DIR / "metadata.jsonl") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    logger.info("Metadata: %d entries", len(metadata))

    # Fix: IVFFlat default nprobe=1 only searches 1 of 4096 clusters → garbage recall.
    # nprobe=32 searches 32 clusters (~250K articles) per query. Much better recall.
    import faiss as _faiss
    _faiss.downcast_index(index).nprobe = 32
    logger.info("Set nprobe=32")

    return index, metadata


def load_article_text_index():
    """
    Build a lookup: (date, url) → article text from reconstructed files.
    Only loads article URLs and texts into a dict for fast lookup.
    """
    logger.info("Building article text index from %s...", ARTICLES_DIR)
    text_lookup = {}
    files = sorted(ARTICLES_DIR.glob("*_articles.json"))

    for i, f in enumerate(files):
        day = f.stem.replace("_articles", "")
        with open(f) as fp:
            articles = json.load(fp)
        for art in articles:
            url = art.get("url", "")
            text = art.get("text", "")
            if url and text:
                text_lookup[(day, url)] = text

        if (i + 1) % 20 == 0:
            logger.info("  Loaded %d/%d days (%d articles indexed)", i + 1, len(files), len(text_lookup))

    logger.info("Article text index: %d articles", len(text_lookup))
    return text_lookup


def load_embedding_model():
    """Load the same embedding model used for articles."""
    from sentence_transformers import SentenceTransformer
    import torch

    logger.info("Loading embedding model...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    logger.info("Model loaded.")
    return model


def embed_queries(queries: list, model) -> np.ndarray:
    """Embed search queries."""
    embeddings = model.encode(
        queries,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def passes_date_filter(article_date: str, resolution_date: str) -> bool:
    """Gate 1: article must be at least 30 days before resolution."""
    try:
        art = article_date.replace("-", "")[:8]
        res = resolution_date.replace("-", "")[:8]
        art_dt = datetime.strptime(art, "%Y%m%d")
        res_dt = datetime.strptime(res, "%Y%m%d")
        return (res_dt - art_dt).days >= DATE_BUFFER_DAYS
    except (ValueError, TypeError):
        return False


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load everything
    index, metadata = load_faiss_index()
    text_lookup = load_article_text_index()
    model = load_embedding_model()

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    logger.info("Questions: %d", len(questions))

    with open(QUERIES_PATH) as f:
        queries_map = json.load(f)
    logger.info("Search queries: %d questions covered", len(queries_map))

    # Embed all queries at once (batch)
    logger.info("Embedding all search queries...")
    all_query_texts = []
    query_to_question = []  # (question_idx, query_idx_within_question)

    for i in range(len(questions)):
        q_queries = queries_map.get(str(i), [questions[i].get("title", "")[:60]])
        for j, sq in enumerate(q_queries):
            all_query_texts.append(sq)
            query_to_question.append((i, j))

    logger.info("Total queries to embed: %d", len(all_query_texts))
    query_embeddings = embed_queries(all_query_texts, model)
    logger.info("Queries embedded: %s", query_embeddings.shape)

    # Search FAISS — top 10 per query
    logger.info("Searching FAISS index...")
    top_k = 10
    scores, indices = index.search(query_embeddings, top_k)
    logger.info("Search complete.")

    # Group results by question, deduplicate, apply date filter, get text
    logger.info("Processing results per question...")
    results = []
    with_context = 0
    start_time = time.time()

    # Pre-group search results by question index
    question_hits = {}  # question_idx → [(score, chunk_idx)]
    for q_idx, (q_question_idx, _) in enumerate(query_to_question):
        if q_question_idx not in question_hits:
            question_hits[q_question_idx] = []
        for rank in range(top_k):
            chunk_idx = int(indices[q_idx][rank])
            score = float(scores[q_idx][rank])
            if chunk_idx >= 0:
                question_hits[q_question_idx].append((score, chunk_idx))

    for i, q in enumerate(questions):
        resolution_date = q.get("actual_resolve_time", "2026-01-01")
        hits = question_hits.get(i, [])

        # Deduplicate by URL, keep highest score
        seen_urls = {}
        for score, chunk_idx in hits:
            if chunk_idx >= len(metadata):
                continue
            meta = metadata[chunk_idx]
            url = meta.get("article_url", "")
            if url not in seen_urls or score > seen_urls[url][0]:
                seen_urls[url] = (score, chunk_idx, meta)

        # Apply Gate 1, get text, rank by score
        candidates = []
        for url, (score, chunk_idx, meta) in seen_urls.items():
            article_date = meta.get("article_date", "")

            if not passes_date_filter(article_date, resolution_date):
                continue

            # Look up article text
            text = text_lookup.get((article_date, url), "")
            if not text:
                continue

            candidates.append({
                "chunk_id": meta.get("chunk_id"),
                "url": url,
                "date": article_date,
                "text": text[:5000],  # Cap at 5K chars
                "score": round(score, 4),
            })

        # Sort by score, keep top 5
        candidates.sort(key=lambda x: -x["score"])
        top_chunks = candidates[:5]

        if top_chunks:
            with_context += 1

        results.append({
            "question_idx": i,
            "question_id": q.get("id"),
            "title": q.get("title", ""),
            "resolution": str(q.get("resolution", "")),
            "resolution_date": q.get("actual_resolve_time", ""),
            "num_chunks": len(top_chunks),
            "chunks": top_chunks,
        })

        if (i + 1) % 500 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - start_time
            logger.info(
                "[%d/%d] %d with context (%.1f%%) | %.1f q/sec",
                i + 1, len(questions), with_context,
                100 * with_context / (i + 1),
                (i + 1) / elapsed,
            )

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    total_chunks = sum(r["num_chunks"] for r in results)
    logger.info("")
    logger.info("=" * 70)
    logger.info("RETRIEVAL COMPLETE")
    logger.info("=" * 70)
    logger.info("Questions: %d", len(results))
    logger.info("With context: %d (%.1f%%)", with_context, 100 * with_context / len(results))
    logger.info("Total chunks: %d", total_chunks)
    logger.info("Avg chunks/question: %.1f", total_chunks / max(len(results), 1))
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("")
    logger.info("Next: python3 step4_leakage_check.py submit --input %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
