"""Convert aljzLate2025.jsonl into the format retrieve_union.py expects.
Produces two shim files differing only in which date field maps to `created_time`.
"""
import json
import sys

SRC = "/data/eval/aljz/aljzLate2025.jsonl"

with open(SRC) as f:
    rows = [json.loads(line) for line in f]

for variant, date_field in [("loose", "resolution_date"), ("strict", "question_start_date")]:
    out = []
    for r in rows:
        title = r.get("question_title", "") + " " + (r.get("background") or "")
        out.append({
            "id": r["qid"],
            "title": title.strip(),
            "description": r.get("resolution_criteria", ""),
            "created_time": r[date_field] + "T23:59:59Z",
            "actual_resolve_time": r["resolution_date"],
            "outcomes": ["yes", "no"],
            "resolution": "yes",
            "lossfunk_category": "GEOPOLITICS",
            "source": "openforesight",
        })
    path = f"/data/eval/aljz/aljz_shim_{variant}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {path}: {len(out)} questions, date_field={date_field}")
