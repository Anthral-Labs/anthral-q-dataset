"""Build OpenAI batch job for semantic leakage check on dayminus1 retrieval.
Rates each question's retrieved chunks as EXPLICIT / IMPLIED / NO leakage of the ground-truth answer.
"""
import json
from pathlib import Path

EVAL_SRC = "/data/eval/aljz/aljzLate2025.jsonl"
RETRIEVAL = "/data/eval/aljz/retrieval_dayminus1.json"
BATCH_OUT = "/data/eval/aljz/leakage_batch_dayminus1.jsonl"

# Load the 491 aljz questions
with open(EVAL_SRC) as f:
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open(EVAL_SRC)}

# Load retrieval
retrieval = {r["question_id"]: r for r in json.load(open(RETRIEVAL))}

PROMPT_TEMPLATE = """You are checking whether retrieved news articles contain the answer to a specific forecasting question.

Question: {question}
Ground truth answer: "{answer}"
Resolution criteria: {criteria}

Retrieved articles:

{articles}

Task: decide whether the combined text of these articles reveals the answer "{answer}" to the question. Use exactly one of these labels:

EXPLICIT - the articles literally state the answer, or a clear paraphrase of it. A model reading this would just copy the answer.
IMPLIED - the articles don't state the answer verbatim but make it unambiguous from context (e.g., the event is reported with enough specificity that the answer is inescapable).
NO - the answer is NOT present in the articles. To get the answer right, a model would need to reason, combine clues across articles, or use prior knowledge outside the provided text.

Respond on the FIRST LINE with exactly one word: EXPLICIT, IMPLIED, or NO. Then one brief sentence of justification."""

batch_records = []
missing = 0
for qid, row in rows.items():
    r = retrieval.get(qid)
    if not r or not r.get("chunks"):
        missing += 1
        continue
    articles = []
    for i, c in enumerate(r["chunks"][:5], 1):
        title = c.get("title") or c.get("url","")[:80]
        source = c.get("domain","")
        date = c.get("date","")
        text = (c.get("text") or "")[:2000]
        articles.append(f"Article {i} (source: {source}, date: {date}):\nTitle: {title}\n{text}")
    articles_text = "\n\n---\n\n".join(articles)

    prompt = PROMPT_TEMPLATE.format(
        question=row["question_title"],
        answer=row["answer"],
        criteria=row.get("resolution_criteria","")[:600],
        articles=articles_text,
    )
    batch_records.append({
        "custom_id": qid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
        }
    })

with open(BATCH_OUT,"w") as f:
    for r in batch_records:
        f.write(json.dumps(r)+"\n")
print(f"Wrote {len(batch_records)} batch requests to {BATCH_OUT}")
print(f"Skipped (no retrieval chunks): {missing}")
