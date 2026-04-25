"""
Step 3: Search reconstructed articles for our Polymarket questions.

For each question + its 6 search queries, find matching articles
from the reconstructed corpus. Simple keyword matching + ranking
by relevance (term frequency).

Usage: python3 search_articles.py --questions /path/to/polymarket_final.json --queries /path/to/search_queries.json
"""

import json
import re
import logging
import argparse
import time
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTICLES_DIR = Path("/home/ubuntu/gdelt/reconstructed")
OUTPUT_PATH = Path("/home/ubuntu/gdelt/filtered/question_context.json")


def load_all_articles() -> list:
    """Load all reconstructed articles into memory."""
    articles = []
    for f in sorted(ARTICLES_DIR.glob("*_articles.json")):
        with open(f) as fp:
            day_articles = json.load(fp)
            articles.extend(day_articles)
    return articles


def build_index(articles: list) -> dict:
    """Build a simple inverted index: word → list of article indices."""
    index = {}
    for i, art in enumerate(articles):
        text = art.get("text", "").lower()
        words = set(re.findall(r'\b\w{3,}\b', text))
        for w in words:
            if w not in index:
                index[w] = []
            index[w].append(i)
    return index


def search(query: str, index: dict, articles: list, max_results: int = 20) -> list:
    """Search articles by keyword query. Returns ranked article indices."""
    query_words = re.findall(r'\b\w{3,}\b', query.lower())
    if not query_words:
        return []

    # Count how many query words each article matches
    scores = Counter()
    for word in query_words:
        for idx in index.get(word, []):
            scores[idx] += 1

    # Return top results sorted by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [idx for idx, score in ranked[:max_results] if score >= 2]  # At least 2 word matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Path to polymarket_final.json")
    parser.add_argument("--queries", required=True, help="Path to polymarket_search_queries.json")
    parser.add_argument("--max-articles", type=int, default=10, help="Max articles per question")
    args = parser.parse_args()

    # Load questions and search queries
    with open(args.questions) as f:
        questions = json.load(f)
    with open(args.queries) as f:
        queries_map = json.load(f)

    logger.info("Loaded %d questions", len(questions))

    # Load all articles
    logger.info("Loading reconstructed articles...")
    articles = load_all_articles()
    logger.info("Loaded %d articles", len(articles))

    if not articles:
        logger.error("No articles found. Run reconstruct.py first.")
        return

    # Build index
    logger.info("Building search index...")
    index = build_index(articles)
    logger.info("Index built: %d unique terms", len(index))

    # Search for each question
    results = []
    start_time = time.time()

    for i, q in enumerate(questions):
        search_queries = queries_map.get(str(i), [q.get("title", "")[:60]])
        question_date = q.get("actual_resolve_time", "2026-01-01")[:10].replace("-", "")

        # Search across all queries, deduplicate results
        seen_urls = set()
        matched_articles = []

        for sq in search_queries:
            hits = search(sq, index, articles, max_results=10)
            for idx in hits:
                art = articles[idx]
                url = art.get("url", "")
                art_date = art.get("date", "")

                # Skip if already seen or too close to resolution
                if url in seen_urls:
                    continue
                # Date buffer: skip articles within 30 days of resolution
                if art_date and question_date and art_date > question_date[:6]:
                    continue

                seen_urls.add(url)
                matched_articles.append({
                    "url": url,
                    "date": art_date,
                    "text": art["text"][:5000],  # Cap at 5K chars
                    "char_count": art.get("char_count", len(art.get("text", ""))),
                })

                if len(matched_articles) >= args.max_articles:
                    break

            if len(matched_articles) >= args.max_articles:
                break

        results.append({
            "question_idx": i,
            "question_id": q.get("id"),
            "title": q.get("title", ""),
            "num_articles": len(matched_articles),
            "articles": matched_articles,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            with_ctx = sum(1 for r in results if r["num_articles"] > 0)
            logger.info(
                "Searched %d/%d questions (%.0f/sec) — %d with context",
                i + 1, len(questions), (i + 1) / elapsed, with_ctx,
            )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, default=str)

    with_context = sum(1 for r in results if r["num_articles"] > 0)
    total_articles = sum(r["num_articles"] for r in results)
    logger.info("=" * 60)
    logger.info("Search complete")
    logger.info("  Questions with context: %d/%d (%.1f%%)",
                with_context, len(results), 100 * with_context / len(results))
    logger.info("  Total matched articles: %d", total_articles)
    logger.info("  Avg per question: %.1f", total_articles / max(len(results), 1))
    logger.info("  Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
