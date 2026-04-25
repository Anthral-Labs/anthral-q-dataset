# GDELT corpus construction

Builds the 31.5M-article retrieval substrate used by every experiment in this repo.

## Pipeline

```
┌─────────────────────────┐
│  ngram-download/        │  Download raw GDELT 2.0 n-grams from S3,
│  (run first)            │  reconstruct article text, dump daily JSONLs
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  step1_clean.py         │  HTML strip, dedupe, boilerplate removal
│                         │  → /data/cleaned/{yyyymmdd}_articles.json
└────────────┬────────────┘
             ▼
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐     ┌──────────────────┐
│ TF-IDF  │     │ OpenAI embed-3-  │
│ matrix  │     │ small via batch  │
│         │     │ API              │
│ step2_  │     │ step2_openai_    │
│ embed   │     │ embed            │
└────┬────┘     └─────────┬────────┘
     │                    ▼
     │          ┌──────────────────┐
     │          │ parallel_download│
     │          │ _embed_batches   │
     │          │ (OAI batch pull) │
     │          └─────────┬────────┘
     │                    ▼
     │          ┌──────────────────┐
     │          │ assemble_openai_ │
     │          │ embed_index      │
     │          │ (IVF FAISS)      │
     │          └─────────┬────────┘
     ▼                    ▼
/data/tfidf_index/      /mnt/data2/openai_embed/
  ├─ vectorizer.pkl       ├─ faiss.index          (181 GB)
  ├─ tfidf_matrix.npz     ├─ embeddings.npy       (181 GB)
  └─ docs_meta.json       ├─ metadata.jsonl       (5.6 GB)
                          └─ state.json
```

## Runtime / cost notes

- **ngram download**: ~12 hrs over LTE/gigabit, S3 egress charges apply
- **step1_clean**: ~2 hrs single-machine (CPU-bound)
- **step2_embed (TF-IDF)**: ~30 min on a beefy CPU machine
- **step2_openai_embed**: ~24 hrs elapsed time with OpenAI batch API (50% off; ~$300 for the full 31.5M chunks)
- **assemble_openai_embed_index**: ~1 hr (IVF training + add)

## Where the built artifacts live (on VM #1)

- `/data/cleaned/{yyyymmdd}_articles.json` — 243 daily files, ~40 GB total
- `/data/tfidf_index/` — 10 GB
- `/mnt/data2/openai_embed/` — 367 GB total
  - `faiss.index` + `embeddings.npy` + `metadata.jsonl` + shards dir

**These artifacts do NOT ship with this repo** — they're too large for git. To run experiments, either:
1. Rebuild via this pipeline (multi-day + ~$300)
2. Use the VM snapshot of `qwen-server` which has them pre-built

See `infra/data-layout.md` for the VM-level file paths each experiment expects.

## Relationship to experiment leakage audits

`step4_leakage_check.py` is an *older* corpus-level leakage scanner (used during corpus construction to flag questionable article sources). The **current** leakage audits used in experiments 010 and 011 are question-level and chunk-level, implemented separately — see those experiment cards.
