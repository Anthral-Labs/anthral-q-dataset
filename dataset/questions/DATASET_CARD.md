# Anthral Q-Dataset: Leakage-Proof Prediction-Market Evaluation Set

**Version 1.0 — April 2026**

A curated evaluation dataset of resolved prediction-market questions from
Polymarket, Kalshi, and Metaculus, constructed specifically to support
leakage-controlled evaluation of LLM forecasting. 10,206 markets grouped
into **3,086 distinct real-world events**; a clean **1,347-question binary
evaluation subset** is provided for direct use.

---

## 1. Summary

| | Markets | Events |
|---|---:|---:|
| Full set (`eval_set.json`) | 10,206 | 3,086 |
| Binary subset (`eval_set_binary.json`) | 1,347 | 1,347 |
| Multi-option subset (`eval_set_multi.json`) | 8,859 | 1,739 |

Every binary question in `eval_set_binary.json` is its own event (1:1). The
multi-option subset contains events with multiple outcomes (e.g., an
election with separate markets for each candidate) that have been decomposed
into one binary market per outcome — the median multi event has ~5 markets,
the largest has 93 (a single March Madness bracket).

**Temporal coverage.** Questions created between 2016-05-09 and 2026-03-30;
all resolved between 2025-08-01 and 2026-04-01. Resolution balance on the
binary subset is 47% YES / 53% NO.

**Sources.**

| Source | Markets | Events | Markets/event |
|---|---:|---:|---:|
| Kalshi | 6,019 | 1,909 | 3.2 |
| Polymarket | 3,552 | 542 | 6.6 |
| Metaculus | 635 | 635 | 1.0 |

---

## 2. Files

### Data

```
dataset/questions/
├── eval_set.json          # 10,206 markets (full set, all 6 categories)
├── eval_set_binary.json   # 1,347 binary markets (our primary eval subset)
├── eval_set_multi.json    # 8,859 multi-option markets (decomposed to binary)
└── DATASET_CARD.md        # this file
```

### Predictions (paired results)

```
experiments/retrieval_politics_geo/
├── experiment.json                                          # run manifest
└── qwen35_27b__faiss_creation_cutoff__politics_geo__320.jsonl
```

The predictions file contains Qwen3.5-27B outputs with-context (TF-IDF
retrieval over a frozen 31.7M-article GDELT corpus with a strict
creation-date filter) on the 320 politics+geopolitics questions drawn from
the binary subset. This is the headline result reported in `docs/results-brief.md`.

Additional experiment runs (no-context baseline, union retrieval, prompt
variants) also live under the same directory as comparison artifacts; see
`docs/results-brief.md` Appendix F for methodology.

---

## 3. Schema

Every record in `eval_set*.json` is a JSON object with these fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique market ID (`mc_*` = Metaculus, `kx_*` = Kalshi, `pm_*` = Polymarket) |
| `title` | string | Market question (the query presented to forecasters) |
| `description` | string | Resolution criteria and source context |
| `outcomes` | list[string] | Always `["Yes", "No"]` after binarization |
| `outcome_prices` | string | JSON-encoded pair of final settlement prices |
| `resolution` | string | `"yes"` or `"no"` — the ground-truth outcome |
| `actual_resolve_time` | string | ISO date the market resolved |
| `created_time` | string | ISO timestamp the market was created (used for leakage filter) |
| `close_time` | string | ISO timestamp the market closed to trading |
| `volume` | float | Trading volume (USD for Polymarket/Kalshi; 0 for Metaculus) |
| `url` | string | Canonical URL for the market on its source platform |
| `source` | string | `"polymarket"`, `"kalshi"`, or `"metaculus"` |
| `event_title` | string | Human-readable event name (may equal `title` if 1:1) |
| `event_id` | string | 12-char hex event ID (group key — 3,086 distinct values) |
| `event_size` | int | Number of markets in the event |
| `event_yes_count` | int | Number of YES-resolved markets in the event (for multi-option) |
| `lossfunk_category` | string | One of: `POLITICS`, `GEOPOLITICS`, `FINANCE`, `SPORTS`, `ENTERTAINMENT`, `TECHNOLOGY` |
| `lossfunk_reason` | string | One-line GPT-4o-mini classifier rationale |
| `scoring_weight` | float | Per-event down-weight to prevent large brackets (sports) from dominating aggregate metrics |
| `twap` | float | Time-weighted average price over the market's lifetime |
| `twap_source` | string | `"market"` or `"imputed"` |
| `difficulty` | float | `abs(twap − resolution_indicator)` — larger = harder |
| `n_price_days` | int | Distinct days with price data |
| `baseline_answer` | string | `"YES"` / `"NO"` — zero-context LLM baseline |
| `baseline_confidence` | float | Baseline predicted probability |
| `baseline_correct` | bool | Whether baseline matched resolution |
| `dedup_flag` | string | Non-empty if the market was flagged as a near-duplicate during curation |
| `dedup_near_match` | string | ID of the near-match market (if flagged) |

### Predictions file schema

Each line of `qwen35_27b__faiss_creation_cutoff__politics_geo__320.jsonl` is
a JSON object:

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Matches `id` in the eval file |
| `title`, `category`, `resolution_date`, `outcomes` | — | Copied from eval for convenience |
| `known_answer` | int | `1` if resolution was YES, `0` if NO |
| `market_price` | float \| null | Market settlement price at close |
| `num_chunks_available` | int | Retrieved article chunks passed to the model |
| `num_chunks_used` | int | Chunks actually included after dedup/length cap |
| `prompt` | string | Full prompt sent to vLLM (system + user, chat template applied) |
| `raw_output` | string | Full model output including reasoning trace |
| `extracted_probability` | float | Parsed probability in [0, 1] — the forecast |
| `extraction_method` | string | `"json"` (parsed from JSON block) or fallback |
| `model` | string | `Qwen/Qwen3.5-27B` |
| `mode` | string | `with_context` or `no_context` |
| `thinking_mode` | bool | `true` — reasoning traces included |
| `temperature` | float | `0.0` |
| `prompt_tokens`, `completion_tokens` | int | vLLM-reported token counts |
| `run_id` | string | Run identifier |
| `timestamp` | string | UTC timestamp of inference |

---

## 4. Category distribution

### Full `eval_set.json`

| Category | Markets | Events |
|---|---:|---:|
| FINANCE | 4,192 | 718 |
| SPORTS | 3,730 | 1,600 |
| POLITICS | 917 | 350 |
| ENTERTAINMENT | 753 | 205 |
| GEOPOLITICS | 425 | 150 |
| TECHNOLOGY | 189 | 78 |

### Binary `eval_set_binary.json`

| Category | Binary questions |
|---|---:|
| SPORTS | 474 |
| FINANCE | 390 |
| POLITICS | 207 |
| GEOPOLITICS | 113 |
| ENTERTAINMENT | 106 |
| TECHNOLOGY | 57 |

### Politics + Geopolitics test subset (320 questions)

The 320-question subset on which we report our primary results is the
intersection `lossfunk_category ∈ {POLITICS, GEOPOLITICS}` within the binary
set (207 politics + 113 geopolitics = 320). 13 questions are excluded from
evaluation per `analysis/excluded_questions.json`; the remaining 252 form
the treatment set for our Brier-delta statistics.

---

## 5. Construction pipeline

High-level: **410,000 raw → 16,574 post-clean → 10,895 post-classify →
10,206 final → 1,347 binary subset**.

1. **Normalization.** Polymarket, Kalshi, and Metaculus all mapped to a
   common schema (see §3). Source-specific quirks (Polymarket's multi-token
   outcome encoding, Kalshi's event-series hierarchy, Metaculus's binary vs
   numeric resolution) are handled in `pipeline/`.
2. **Hard filters.** Sportsbook-style markets, banned categories (celebrity,
   reality TV), duplicates, low-volume crypto markets removed.
3. **Classification.** Six-category taxonomy assigned via GPT-4o-mini Batch
   API (`lossfunk_category` / `lossfunk_reason`). Off-topic questions
   (classified `IRRELEVANT`) dropped.
4. **Curation.** Multi-option events decomposed into binary markets (one
   market per outcome). TWAP and difficulty computed from price history.
   Markets with < 7 days of price data dropped.
5. **Dedup.** Semantic deduplication across sources using sentence embeddings.
   Near-duplicates within the same event kept but flagged via `dedup_flag`.
6. **Splits.** Binary (`eval_set_binary.json`) vs multi-option
   (`eval_set_multi.json`). Binary is the primary evaluation set.

The full pipeline is in `pipeline/`. Reproduction requires access to the
three source APIs (see §7 for upstream source terms).

---

## 6. Leakage defense (what makes this dataset useful)

The core methodological contribution is a **strict creation-date filter
applied before retrieval**, not after. For each market, only articles with
publication date strictly earlier than `created_time` are eligible as
retrieved context. This is stricter than any published benchmark we are
aware of:

- Standard date-filtered web search (`before:YYYY-MM-DD` in Google/Bing)
  leaks ~71% of the time per [Paleka et al. (2025)](https://arxiv.org/abs/2506.00723)
  — pages update silently, resolution-date buffers admit post-creation
  articles, and URL slugs can encode post-resolution information.
- Our retrieval operates over a **static 31.7M-article GDELT corpus**
  frozen at ingest, with a per-chunk creation-date check that runs before
  the search (not as a post-filter).
- Verification: zero leakage violations across the 320-question test set
  (every article publication date < every question creation date, checked
  per-chunk).

Full methodology and the per-chunk verification procedure are in
`docs/results-brief.md` Appendix A.

---

## 7. How to use

### Evaluating a new model

```python
import json

# Load the 320-question test subset
eval_set = json.load(open("dataset/questions/eval_set_binary.json"))
test = [r for r in eval_set
        if r["lossfunk_category"] in ("POLITICS", "GEOPOLITICS")]

# (excluded questions per analysis/excluded_questions.json)

for q in test:
    title = q["title"]
    resolve_date = q["actual_resolve_time"]
    ground_truth = 1 if q["resolution"] == "yes" else 0
    # ... run your model, record predicted_probability ...
    # brier = (predicted_probability - ground_truth) ** 2
```

### Comparing against our Qwen3.5-27B baseline

Predictions in
`experiments/retrieval_politics_geo/qwen35_27b__faiss_creation_cutoff__politics_geo__320.jsonl`
contain `question_id` matching `id` in the eval file and
`extracted_probability` as the forecast. Paired-bootstrap significance
testing is described in `docs/results-brief.md` Appendix B.4.

### Building retrieval context

The retrieval pipeline used to produce our results is in
`scripts/retrieve_hybrid.py` (TF-IDF + embedding rerank) and
`scripts/retrieve_union.py` (TF-IDF ∪ OpenAI embedding, reranked). Both
enforce the creation-date filter before search.

---

## 8. License and attribution

### Our curation

The curation, normalization, classification, and annotations in this
dataset are released under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**. If you use this dataset, please cite as:

> Anthral Labs (2026). *Anthral Q-Dataset: Leakage-Proof Prediction-Market
> Evaluation Set.* https://github.com/Anthral-Labs/questions-dataset

### Upstream source terms

The underlying market data is derived from three public APIs. Each has its
own terms; downstream users are responsible for compliance when
redistributing derivative works that incorporate the underlying data.

- **Polymarket** — public market data, accessed via Polymarket's public
  API. See [Polymarket Terms of Service](https://polymarket.com/tos).
- **Kalshi** — public event-market data, accessed via Kalshi's public API.
  See [Kalshi Terms of Service](https://kalshi.com/docs/tos).
- **Metaculus** — questions and resolutions accessed via Metaculus's public
  API. Metaculus content is generally licensed CC BY-SA 4.0; see the
  [Metaculus Terms of Use](https://www.metaculus.com/terms/).

The code in this repository (under `scripts/`, `pipeline/`, `docs/`) is
released under the MIT License — see the top-level `LICENSE` file.

---

## 9. Known limitations

- **Category imbalance.** Sports and finance markets dominate the full set
  (77% of markets by count). Deliberately, the binary subset down-weights
  both via `scoring_weight` to keep politics and geopolitics — the
  categories where news retrieval is most plausibly useful — non-trivially
  represented.
- **Temporal range is narrow.** All resolutions are within Aug 2025 – Apr
  2026. Long-horizon forecasting (> 1 year) is under-represented.
- **Metaculus is 1:1 by construction.** Metaculus events map to single
  questions, which means the 635 Metaculus rows contribute 635 events —
  inflating the Metaculus events/market ratio vs. the Kalshi/Polymarket
  equivalent. This is a property of how the source platforms differ, not a
  curation artifact.
- **Leakage defense is corpus-level, not web-level.** Users who retrieve
  from live web sources (Google, Exa, Perplexity, etc.) instead of our
  frozen corpus are not protected by the creation-date filter and should
  expect contamination at the Paleka-et-al. baseline rate (~71%).

---

## 10. Citation

```bibtex
@dataset{anthral_q_dataset_2026,
  author = {{Anthral Labs}},
  title  = {Anthral Q-Dataset: Leakage-Proof Prediction-Market Evaluation Set},
  year   = {2026},
  url    = {https://github.com/Anthral-Labs/questions-dataset},
  note   = {Version 1.0, 10,206 markets across 3,086 events, 1,347 binary eval subset}
}
```

## 11. Contact

Issues and pull requests: https://github.com/Anthral-Labs/questions-dataset/issues
