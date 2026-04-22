"""Download chunk-leakage results, identify leaky chunks, rebuild prompts stripping them."""
import json, re
from openai import OpenAI
from collections import defaultdict

client = OpenAI()
batch_id = open("/data/eval/aljz/chunk_batch_id.txt").read().strip()
out_id = client.batches.retrieve(batch_id).output_file_id
raw = client.files.content(out_id).text
open("/data/eval/aljz/chunk_leakage_results.jsonl","w").write(raw)

YES_RE = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)

# Parse per-chunk labels
chunk_leak = defaultdict(dict)  # qid -> {chunk_idx: YES/NO}
for line in raw.splitlines():
    r = json.loads(line)
    cid = r["custom_id"]  # "qid:chunk_idx"
    qid, idx = cid.rsplit(":", 1)
    idx = int(idx)
    try:
        text = r["response"]["body"]["choices"][0]["message"]["content"]
        m = YES_RE.search(text)
        chunk_leak[qid][idx] = m.group(1).upper() if m else "NO"
    except:
        chunk_leak[qid][idx] = "NO"

# Stats
total_chunks = sum(len(v) for v in chunk_leak.values())
leaky_chunks = sum(sum(1 for l in v.values() if l == "YES") for v in chunk_leak.values())
print(f"Parsed {total_chunks} chunk labels across {len(chunk_leak)} EXPLICIT questions")
print(f"Chunks flagged as leaking the answer: {leaky_chunks} ({leaky_chunks/total_chunks*100:.1f}%)")

# Distribution of how many chunks per question are leaky
from collections import Counter
per_q_leak = Counter(sum(1 for l in v.values() if l == "YES") for v in chunk_leak.values())
print(f"Leaky-chunks-per-question distribution: {dict(per_q_leak)}")

# Build new shim: for EXPLICIT questions, rebuild prompt with leaky chunks stripped.
aljz = {json.loads(l)["qid"]: json.loads(l) for l in open("/data/eval/aljz/aljzLate2025.jsonl")}
retrieval = {r["question_id"]: r for r in json.load(open("/data/eval/aljz/retrieval_dayminus1.json"))}

def build_prompt(row, chunks):
    """Splice our chunks into OpenForecaster's template. Same as splice_prompts.py but inline."""
    their = json.loads(json.dumps(row))  # copy
    # Use the row's existing 'prompt' field as a template — find article block, replace
    p = row["prompt"]  # loose variant
    art_marker = "Relevant passages from retrieved news articles:"
    tail_marker = "Think step by step"
    ai = p.find(art_marker); ti = p.find(tail_marker)
    if ai < 0 or ti < 0:
        return None
    header = p[:ai+len(art_marker)]
    tail = p[ti:]
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"\nArticle {i}:\nTitle: {c.get('title','') or c.get('url','')[:80]}\nSource: {c.get('domain','')}\nArticle Publish Date: {c.get('date','')}\nRelevant Passage: {(c.get('text') or '')[:2500]}")
    return header + "".join(parts) + "\n\n" + tail

# For each EXPLICIT question, build a row with only non-leaky chunks
rebuilt = []
kept_chunk_counts = []
for qid, leak_map in chunk_leak.items():
    row = aljz[qid]
    chunks = retrieval[qid].get("chunks", [])
    kept = [c for i, c in enumerate(chunks[:5]) if leak_map.get(i, "NO") != "YES"]
    kept_chunk_counts.append(len(kept))
    # Build deep-copied row with new 'prompt' field
    new_row = dict(row)
    if kept:
        new_prompt = build_prompt(row, kept)
    else:
        # Fall back to prompt_without_retrieval if all chunks were leaky
        new_prompt = row["prompt_without_retrieval"]
    if new_prompt:
        new_row["prompt"] = new_prompt
        rebuilt.append(new_row)

print(f"Rebuilt {len(rebuilt)} question rows with leaky chunks stripped")
print(f"Remaining-chunks-per-question: {dict(Counter(kept_chunk_counts))}")

# Write as JSONL (same format eval_openforesight.py --local_jsonl expects)
with open("/data/eval/aljz/aljz_dayminus1_deleaked.jsonl","w") as f:
    for r in rebuilt:
        f.write(json.dumps(r)+"\n")
print("Wrote /data/eval/aljz/aljz_dayminus1_deleaked.jsonl")
