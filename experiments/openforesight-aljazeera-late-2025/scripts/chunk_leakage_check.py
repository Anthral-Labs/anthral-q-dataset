"""Per-chunk leakage check on the 45 EXPLICIT questions.
For each chunk of each EXPLICIT question, ask GPT-4o: does this chunk contain the answer?
"""
import json, re

LABEL_RE = re.compile(r"^\s*(EXPLICIT|IMPLIED|NO)\b", re.IGNORECASE | re.MULTILINE)

# Find the 45 EXPLICIT questions
leakage = {}
for line in open("/data/eval/aljz/leakage_results_dayminus1.jsonl"):
    r = json.loads(line)
    qid = r["custom_id"]
    try:
        text = r["response"]["body"]["choices"][0]["message"]["content"]
        m = LABEL_RE.search(text)
        leakage[qid] = m.group(1).upper() if m else None
    except:
        leakage[qid] = None

explicit_qids = {qid for qid, l in leakage.items() if l == "EXPLICIT"}
print(f"EXPLICIT questions: {len(explicit_qids)}")

# Load rows + retrieval
aljz = {json.loads(l)["qid"]: json.loads(l) for l in open("/data/eval/aljz/aljzLate2025.jsonl")}
retrieval = {r["question_id"]: r for r in json.load(open("/data/eval/aljz/retrieval_dayminus1.json"))}

PROMPT = """You are checking whether a single news article chunk contains (or directly reveals) the answer to a specific question.

Question: {question}
Ground truth answer: "{answer}"

Article chunk (Source: {source}, Date: {date}):
{text}

Does this chunk contain the answer "{answer}" to the question, or directly reveal it in a way that makes it obvious?

Respond on the FIRST LINE with exactly one word: YES or NO. Then one brief sentence of justification."""

batch_records = []
for qid in sorted(explicit_qids):
    row = aljz[qid]
    r = retrieval.get(qid)
    if not r or not r.get("chunks"):
        continue
    for chunk_idx, c in enumerate(r["chunks"][:5]):
        text = (c.get("text") or "")[:2000]
        prompt = PROMPT.format(
            question=row["question_title"],
            answer=row["answer"],
            source=c.get("domain",""),
            date=c.get("date",""),
            text=text,
        )
        batch_records.append({
            "custom_id": f"{qid}:{chunk_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role":"user","content":prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            }
        })

with open("/data/eval/aljz/chunk_leakage_batch.jsonl","w") as f:
    for r in batch_records:
        f.write(json.dumps(r)+"\n")
print(f"Wrote {len(batch_records)} per-chunk requests to chunk_leakage_batch.jsonl")
