"""Rebuild metadata.jsonl to match the embedding order exactly."""
import json, logging, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLEANED_DIR = Path("/data/reconstructed")  # same source step2 used
OUTPUT = Path("/mnt/data2/metadata.jsonl")

def chunk_article(text, max_chars=2000):
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

files = sorted(CLEANED_DIR.glob("*_articles.json"))
logger.info("Rebuilding metadata from %d files", len(files))

chunk_id = 0
start = time.time()

with open(OUTPUT, "w") as out:
    for i, f in enumerate(files):
        day = f.stem.replace("_articles", "")
        with open(f) as fp:
            articles = json.load(fp)

        for art in articles:
            text = art.get("text", "")
            if not text or len(text) < 50:
                continue
            chunks = chunk_article(text)
            for ci in range(len(chunks)):
                out.write(json.dumps({
                    "chunk_id": chunk_id,
                    "article_url": art.get("url", ""),
                    "article_date": art.get("date", day),
                    "chunk_index": ci,
                }) + "\n")
                chunk_id += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            logger.info("[%d/%d] %d chunks so far", i + 1, len(files), chunk_id)

elapsed = time.time() - start
logger.info("Done. %d chunks in %.0fs. Saved to %s", chunk_id, elapsed, OUTPUT)
