"""Shim for cutoff = resolution_date - 1 day per question."""
import json
from datetime import date, timedelta
SRC = "/data/eval/aljz/aljzLate2025.jsonl"
with open(SRC) as f:
    rows = [json.loads(l) for l in f]
out = []
for r in rows:
    cutoff = date.fromisoformat(r["resolution_date"]) - timedelta(days=1)
    out.append({
        "id": r["qid"],
        "title": (r.get("question_title","") + " " + (r.get("background") or "")).strip(),
        "description": r.get("resolution_criteria",""),
        "created_time": cutoff.isoformat() + "T23:59:59Z",
        "actual_resolve_time": r["resolution_date"],
        "outcomes": ["yes","no"],
        "resolution": "yes",
        "lossfunk_category": "GEOPOLITICS",
        "source": "openforesight",
    })
with open("/data/eval/aljz/aljz_shim_dayminus1.json","w") as f:
    json.dump(out, f)
print(f"Wrote {len(out)} rows with cutoff=resolution-1day")
