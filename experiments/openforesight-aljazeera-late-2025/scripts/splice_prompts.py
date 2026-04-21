"""
Splice our GDELT-retrieved chunks into the OpenForesight prompt template.
Input: aljzLate2025.jsonl (their data) + retrieval_{loose,strict}.json (our chunks)
Output: aljz_with_our_retrieval_{loose,strict}.jsonl — a dataset shaped like the
        HF aljazeeraLate2025 split but with `prompt` field rewritten to use our articles.
"""
import json
import sys

EVAL_SRC = "/data/eval/aljz/aljzLate2025.jsonl"

def splice_one(their_prompt: str, our_chunks: list[dict]) -> str:
    art_marker = "Relevant passages from retrieved news articles:"
    tail_marker = "Think step by step"
    art_start = their_prompt.find(art_marker)
    tail_start = their_prompt.find(tail_marker)
    if art_start < 0 or tail_start < 0:
        # Fall back: keep their prompt as-is
        return their_prompt
    header = their_prompt[: art_start + len(art_marker)]
    tail = their_prompt[tail_start:]
    parts = []
    for i, c in enumerate(our_chunks[:5], start=1):
        title = c.get("title") or c.get("url", "")[:80]
        source = c.get("domain") or ""
        date = c.get("date") or ""
        passage = (c.get("text") or "")[:2500]
        parts.append(
            f"\nArticle {i}:\nTitle: {title}\nSource: {source}\nArticle Publish Date: {date}\nRelevant Passage: {passage}"
        )
    return header + "".join(parts) + "\n\n" + tail


def main(variant: str):
    ret_path = f"/data/eval/aljz/retrieval_{variant}.json"
    out_path = f"/data/eval/aljz/aljz_with_our_retrieval_{variant}.jsonl"
    print(f"Loading retrieval from {ret_path}")
    retrieval = {r.get("question_id") or r.get("id"): r.get("chunks", []) for r in json.load(open(ret_path))}
    print(f"  loaded {len(retrieval)} retrieval entries")

    with open(EVAL_SRC) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} original rows")

    with open(out_path, "w") as f:
        spliced = 0
        for r in rows:
            qid = r["qid"]
            chunks = retrieval.get(qid, [])
            if not chunks:
                # No retrieval for this qid — write row with original prompt unchanged
                f.write(json.dumps(r) + "\n")
                continue
            r_out = dict(r)
            r_out["prompt"] = splice_one(r["prompt"], chunks)
            f.write(json.dumps(r_out) + "\n")
            spliced += 1
    print(f"Wrote {out_path}: {spliced}/{len(rows)} spliced")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "loose")
