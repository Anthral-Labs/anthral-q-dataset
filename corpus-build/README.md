# GDELT corpus construction

Code to build the news-article retrieval substrate underlying the [`gdelt-forecast-binary`](https://huggingface.co/datasets/rajatagarwal457/gdelt-forecast-binary) and [`gdelt-forecast-freeform`](https://huggingface.co/datasets/rajatagarwal457/gdelt-forecast-freeform) Hugging Face datasets.

The pipeline turns ~12 months of [GDELT 2.0](https://www.gdeltproject.org/) global news n-grams into:

- ~31.5 M cleaned article-text records, sharded as daily JSONL files
- A 100K-term TF-IDF index over the entire corpus
- An OpenAI `text-embedding-3-small` FAISS IVF index over the same corpus (~181 GB on disk)

Together these support the union-retrieval pipeline (TF-IDF ∪ OpenAI FAISS → OpenAI rerank → top-k) used to build evidence sets for forecasting questions in a strict-cutoff posture.

## Pipeline

```
┌───────────────────────────────┐
│  ngram-download/              │  Download raw GDELT 2.0 n-grams from S3,
│  (run first; ~12 hrs)         │  reconstruct article text per day.
│  → /data/reconstructed/       │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│  build/step1_clean.py         │  HTML strip, dedupe, boilerplate removal.
│  (~2 hrs CPU)                 │  → /data/cleaned/{yyyymmdd}_articles.json
└──────────────┬────────────────┘
               ▼
       ┌───────┴────────────────┐
       ▼                        ▼
┌─────────────┐         ┌──────────────────────┐
│  TF-IDF     │         │  OpenAI text-        │
│  index      │         │  embedding-3-small   │
│             │         │  via batch API       │
│ build/      │         │                      │
│ step2_      │         │ build/step2_openai_  │
│ embed.py    │         │ embed.py             │
│ (~30 min)   │         │ (~24 hrs, ~$300)     │
└──────┬──────┘         └──────────┬───────────┘
       │                           ▼
       │              ┌──────────────────────┐
       │              │  build/parallel_     │
       │              │  download_embed_     │
       │              │  batches.py          │
       │              │  (pull batch outputs)│
       │              └──────────┬───────────┘
       │                         ▼
       │              ┌──────────────────────┐
       │              │  build/assemble_     │
       │              │  openai_embed_       │
       │              │  index.py            │
       │              │  (IVF FAISS, ~1 hr)  │
       │              └──────────┬───────────┘
       ▼                         ▼
/data/tfidf_index/       /mnt/data2/openai_embed/
  ├─ vectorizer.pkl        ├─ faiss.index          (181 GB)
  ├─ tfidf_matrix.npz      ├─ embeddings.npy       (181 GB)
  └─ docs_meta.json        ├─ metadata.jsonl       (5.6 GB)
                           └─ state.json
```

## Subdirectories

- [`ngram-download/`](ngram-download/) — Phase 0: pull raw GDELT 2.0 n-grams from S3 and reconstruct article text per day. See subdirectory's own scripts; `run_all.sh` chains them end-to-end.
- [`build/`](build/) — Phases 1–5: clean, embed (TF-IDF and OpenAI), assemble. See [`build/README.md`](build/README.md) for per-script details and runtime expectations.

## What you need before running

| | |
|---|---|
| **GDELT n-grams S3 access** | Public; AWS CLI or boto3 with no creds works for the `gdelt-open-data` bucket. Egress charges apply if not running on AWS. |
| **OpenAI API key** | For `step2_openai_embed.py` (~$300 of `text-embedding-3-small` calls via batch API at 50% off). |
| **Disk** | ~600 GB total for cleaned articles + TF-IDF index + OpenAI embed index + intermediate batch outputs. |
| **CPU/RAM** | Mid-tier server (32 GB RAM, ~12 cores) is enough; nothing is GPU-bound. |
| **Wall-clock** | ~2–3 days end-to-end if everything goes smoothly. The OpenAI embed step is the long pole (waiting on batch API). |

## Adapting paths

Most scripts hardcode our VM's paths (`/data/cleaned/`, `/mnt/data2/openai_embed/`) for convenience. They're either CLI-overridable via `--input` / `--output` flags, or live as `Path(...)` constants near the top of each file. Search for `WORK_DIR`, `CLEANED_DIR`, `OUTPUT` and adjust before running.

## Date range

The corpus the published Hugging Face datasets were built from spans roughly **August 2025 → April 2026**. The pipeline is not date-bound — point `ngram-download/sync_s3.py` at any GDELT date range to reproduce or extend.

## Cost summary

| Phase | Wall-clock | $$$ |
|---|---|---|
| ngram-download | ~12 hrs | S3 egress only (~$5 if outside AWS) |
| step1_clean | ~2 hrs | — |
| step2_embed (TF-IDF) | ~30 min | — |
| step2_openai_embed | ~24 hrs (batch) | **~$300** |
| download + assemble FAISS | ~2 hrs | — |
| **Total** | **~2 days** | **~$305** |

## License

This code is released under MIT (matching the parent [`anthral-q-dataset`](https://github.com/Anthral-Labs/anthral-q-dataset) repo). The corpus *outputs* (cleaned articles) inherit GDELT 2.0's CC-BY 4.0 license — attribution to GDELT required for any redistribution.

## Citing

If you use this pipeline (or the datasets it produces), please cite:

```bibtex
@misc{anthral2026gdeltforecast,
  title  = {GDELT-Forecast: Strict-cutoff forecasting benchmarks from GDELT news clusters},
  author = {Rajat Agarwal and Anthral Labs},
  year   = {2026},
  url    = {https://github.com/Anthral-Labs/anthral-q-dataset},
}
```
