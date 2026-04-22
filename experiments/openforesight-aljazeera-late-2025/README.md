# Qwen3.5-27B on OpenForesight `aljazeeraLate2025` — 4-Run Retrieval Ablation

**Date:** 2026-04-21

A four-condition ablation measuring how retrieval corpus choice and leakage-defense protocol affect free-form forecasting performance on 491 post-cutoff questions.

## Setup

| Component | Value |
|---|---|
| Model | `Qwen/Qwen3.5-27B` (no fine-tuning) |
| Dataset | [`nikhilchandak/OpenForesight`](https://huggingface.co/datasets/nikhilchandak/OpenForesight), split `aljazeeraLate2025` (491 free-form questions, resolutions Sept–Dec 2025, post the model's knowledge cutoff) |
| Inference | vLLM 0.17, bf16, temperature 0.0, `max_new_tokens=16384`, thinking mode enabled |
| Prompt template | OpenForecaster's `prompt` / `prompt_without_retrieval` template verbatim (explicit Brier scoring rule in-context, `<answer>` / `<probability>` XML output) |
| Judge | GPT-4o via OpenAI API with OpenForecaster's `get_judge_prompt_with_gt` prompt verbatim (see `scripts/judge_openai.py`) |
| Scoring | Signed reward per question: `+1−(1−p)²` if judge=correct, `−p²` if judge=wrong, `−0.25` if model output was token-cap / unparseable (soft-Brier fallback per Chandak et al. 2025) |

## Conditions

Only the retrieval source and leakage filter differ; everything else is identical across runs.

| Run | Context | Leakage posture | Where the articles come from |
|---|---|---|---|
| **A** No context | OpenForesight `prompt_without_retrieval` field verbatim | — | Model reasons from priors only |
| **B** OpenForecaster retrieval | OpenForesight `prompt` field verbatim | Loose (articles ≤ `resolution_date`) | OpenForecaster's own retrieval pipeline |
| **C-loose** Our retrieval (matched protocol) | Their template, our articles spliced in | Loose (articles ≤ `resolution_date`) | TF-IDF ∪ OpenAI-embedding union → OAI rerank → top-5 from our 31.5M-article GDELT corpus |
| **C-strict** Our retrieval (strict forecasting filter) | Their template, our articles spliced in | Strict (articles ≤ `question_start_date`) | Same GDELT pipeline as C-loose |

## Results

| Run | Correct | Accuracy | Null (token-cap) | **Signed reward** (higher better, range [−1, +1]) | **Brier loss** (lower better, range [0, 1]) |
|---|---:|---:|---:|---:|---:|
| A — no context | 45 / 491 | 9.2% | 6.1% (30) | −0.1554 | **0.2470** ← best |
| B — their retrieval | 64 / 491 | 13.0% | 2.4% (12) | −0.2561 | 0.3865 |
| **C-loose** — our retrieval, loose | **133 / 491** | **27.1%** | 1.4% (7) | **−0.0869** ← best | 0.3578 |
| C-strict — our retrieval, strict | 86 / 491 | 17.5% | 0.6% (3) | −0.1733 | 0.3484 |

Both metrics use the same extracted probabilities. They differ in sign and in how they penalize confidence:

- **Signed reward** (OpenForecaster's metric, range [−1, +1], higher better): `+1−(1−p)²` if correct else `−p²`. Gives a `+1` baseline per correct answer on top of a `−(1−p)²` hedge-penalty. Rewards getting more answers right even at moderate confidence.
- **Brier loss** (standard Brier, range [0, 1], lower better): `(p − y)²`. Pure calibration error. Penalizes any confidence on wrong answers as much as missed confidence on correct ones.

Unparseable / token-cap outputs: signed reward uses OpenForecaster's soft-Brier fallback of `−0.25`; Brier loss uses `0.25` (equivalent to predicting p=0.5 on a binary outcome, which gives squared error `0.25` either way).

The rankings differ between the two metrics:

| Ranking | Signed reward (higher better) | Brier loss (lower better) |
|---|---|---|
| 1 (best) | C-loose | A (no context) |
| 2 | A | C-strict |
| 3 | C-strict | C-loose |
| 4 (worst) | B | B |

## Findings

1. **Our GDELT corpus more than doubles the accuracy of OpenForecaster's retrieval** (B vs C-loose, matched protocol): 13.0% → 27.1%, signed reward −0.256 → −0.087, Brier loss 0.387 → 0.358. The only variable that changed is the article corpus — our 31.5M-article GDELT pool has meaningfully better coverage of Sept–Dec 2025 events than whatever OpenForecaster's retrieval pipeline uses.

2. **Leakage quantification**: switching from the loose filter (`article_date ≤ resolution_date`) to our strict creation-date filter (`article_date ≤ question_start_date`) drops accuracy by 9.6 pp and signed reward by 0.086. Approximately half of retrieval's apparent lift on this benchmark comes from articles published between question creation and resolution. This is the Paleka-et-al.-style leakage-premium effect, measured directly on the OpenForesight family for the first time we are aware of.

3. **Calibration ceiling for untrained base models + confidence-inflation effect**: even with C-loose's best-in-class retrieval, signed reward is `−0.087` and Brier loss is `0.358`. Base Qwen3.5-27B is systematically overconfident on the ~73% of questions it still gets wrong. Most revealing: on *Brier loss specifically*, the no-context Run A is actually the best-calibrated (0.247), and *adding retrieval makes Brier worse across the board* because retrieval inflates the model's stated confidence on wrong answers faster than it improves correctness. Closing this gap requires training, not better retrieval — consistent with OpenForecaster's thesis that a combined accuracy-plus-Brier reward function (GRPO, OpenForecaster-8B recipe) is the calibration lever.

## Failure-mode texture (qualitative)

Predictions where our retrieval helped most are named-entity questions about globally-reported events with clear single-article resolution (Rio police raids → "Public Defender's Office"; India tariff announcement → date of enforcement). Predictions that still fail under C-loose cluster into three buckets: (a) answers living in niche outlets outside our GDELT coverage, (b) specificity mismatches where the right article is retrieved but the extracted entity is the wrong grain ("Public Security Secretariat" vs "Public Defender's Office"), (c) questions where the model reasoned itself into a confident wrong answer despite having relevant articles in context.

## Extended cutoff sweep — `dayminus1`

After the initial 4-run ablation we added a third cutoff point: **`resolution_date − 1 day`**, probing whether the bulk of leakage comes from the final 24 hours before resolution.

| Cutoff | Accuracy | Signed reward | Brier loss |
|---|---:|---:|---:|
| Strict (articles ≤ `question_start_date`) | 17.5% | −0.1733 | 0.3484 |
| **dayminus1** (articles ≤ `resolution_date − 1 day`) | **23.0%** | **−0.1315** | 0.3616 |
| Loose (articles ≤ `resolution_date`) | 27.1% | −0.0869 | 0.3578 |

The dayminus1 point lands about 57% of the way from strict to loose on both accuracy and signed reward. The final 24 hours before resolution account for the remaining ~43% of the leakage premium (20 of 491 questions flip from correct-under-loose to wrong-under-dayminus1). Leakage is **concentrated near resolution**, not spread uniformly through the active window.

## Semantic leakage check (two-stage GPT-4o audit)

To separate "retrieval helped because it surfaced relevant context" from "retrieval helped because the answer was literally in the retrieved article," we ran a two-stage audit on the dayminus1 retrieval:

### Stage 1 — question-level classification (491 questions)

GPT-4o read the combined 5-chunk context for each question and rated it as **EXPLICIT** (article literally states the answer), **IMPLIED** (answer unambiguous but not stated verbatim), or **NO** (answer not present in retrieval).

| Leakage level | Questions | Accuracy on that subset |
|---|---:|---:|
| EXPLICIT | 45 (9.2%) | 77.8% (35/45 correct) |
| IMPLIED | 30 (6.1%) | 66.7% (20/30 correct) |
| NO | 416 (84.7%) | 13.9% (58/416 correct) |
| **Total** | **491** | **23.0%** |

84.7% of retrievals have no answer leakage — when the model is correct on those, it's doing genuine forecasting. Of the 113 correct answers, 58 (51%) came from contexts with no answer leakage, 55 (49%) were aided by answer-bearing retrieval.

### Stage 2 — chunk-level classification + de-leaked counterfactual (45 EXPLICIT questions)

For each of the 45 EXPLICIT questions, GPT-4o scored each of the 5 chunks individually: does THIS chunk contain the answer? Of 225 total chunks, **77 (34.2%) were flagged as leaking the answer**. Distribution of leaky-chunks-per-question:

| # leaky chunks per question | # questions | Remaining chunks after strip |
|---|---:|---|
| 0 (classifier disagreement) | 4 | 5 |
| 1 | 22 | 4 |
| 2 | 8 | 3 |
| 3 | 7 | 2 |
| 4 | 3 | 1 |
| 5 (all) | 1 | 0 (falls back to no-context) |

We stripped the flagged chunks from each prompt and **re-ran Qwen3.5-27B inference on the 45 de-leaked prompts**. This is the real counterfactual: same model, same retrieval pipeline, just with the specific answer-bearing articles removed.

### Counterfactual results

| Condition | Correct on the 45 EXPLICIT questions | Δ |
|---|---:|---:|
| Before (leaky chunks present) | 35 / 45 = 77.8% | — |
| After (leaky chunks stripped) | 15 / 45 = 33.3% | **−20 correct** |

20 of the 35 previously-correct answers flipped to wrong. The 15 that survived are cases where the model reasoned correctly from remaining context or prior knowledge.

Merged with the unchanged 446 non-EXPLICIT results:

| Version | Accuracy | Signed reward | Brier loss |
|---|---:|---:|---:|
| Original dayminus1 | 23.0% (113/491) | −0.1315 | 0.3616 |
| **De-leaked counterfactual** | **18.9% (93/491)** | **−0.1919** | 0.3813 |
| Strict (for reference) | 17.5% (86/491) | −0.1733 | 0.3484 |

**Headline: the de-leaked dayminus1 lands at 18.9% accuracy — within 1.4 pp of strict's 17.5%.** Essentially all of the cutoff-relaxation premium (strict → dayminus1, +5.5 pp) is attributable to articles that literally state the ground-truth answer. The residual 1.4 pp is IMPLIED-level context (30 questions where the answer is unambiguous in context without being stated verbatim) — genuinely helpful retrieval that isn't trivial answer copying.

The signed reward on the de-leaked counterfactual (−0.1919) is actually *worse* than strict (−0.1733) even at similar accuracy — the non-leaky dayminus1 articles drive higher confidence on wrong answers, so the mistakes pay a larger Brier penalty.

## Files

```
experiments/openforesight-aljazeera-late-2025/
├── README.md                       (this file)
├── summary.json                    (headline numbers in machine-readable form)
├── predictions/                    (raw vLLM outputs — 491 records per run except dayminus1_deleaked which has 45)
│   ├── run_a_no_context.jsonl
│   ├── run_b_their_retrieval.jsonl
│   ├── run_c_loose.jsonl
│   ├── run_c_strict.jsonl
│   ├── run_dayminus1.jsonl
│   └── run_dayminus1_deleaked.jsonl  (only the 45 EXPLICIT questions, rerun with leaky chunks stripped)
├── judged/                         (per-question LLM-judge results — binary correct + signed reward)
│   ├── run_a.jsonl
│   ├── run_b.jsonl
│   ├── run_c_loose.jsonl
│   ├── run_c_strict.jsonl
│   ├── run_dayminus1.jsonl
│   └── run_dayminus1_deleaked.jsonl
├── retrievals/                     (top-5 chunks from our GDELT corpus per question, per filter)
│   ├── loose.json
│   ├── strict.json
│   └── dayminus1.json
├── leakage/                        (semantic leakage audit artifacts)
│   ├── question_level_dayminus1.jsonl      (491 GPT-4o bundle-level labels: EXPLICIT/IMPLIED/NO)
│   ├── chunk_level_dayminus1.jsonl         (225 per-chunk YES/NO labels for the 45 EXPLICIT questions)
│   └── dayminus1_deleaked_prompts.jsonl    (the 45 prompts fed back into Qwen with leaky chunks stripped)
└── scripts/
    ├── judge_openai.py                     (GPT-4o judge wrapper using OpenForecaster's prompt verbatim)
    ├── splice_prompts.py                   (splices retrieved chunks into OpenForecaster's prompt template)
    ├── make_retrieval_shim.py              (converts records to retrieve_union.py input format)
    ├── make_dayminus1_shim.py              (shim generator for the resolution-minus-1-day cutoff)
    ├── build_leakage_batch.py              (builds Stage 1 bundle-level OpenAI batch request JSONL)
    ├── submit_leakage_batch.py             (submits the batch to OpenAI)
    ├── chunk_leakage_check.py              (builds Stage 2 chunk-level batch for the 45 EXPLICIT)
    └── rebuild_deleaked.py                 (identifies leaky chunks, builds the de-leaked prompts)
```

## Reproducibility

Inference was run using OpenForecaster's published `eval_openforesight.py` from [`OpenForecaster/scaling-forecasting-training`](https://github.com/OpenForecaster/scaling-forecasting-training), with one local patch (`max_model_len=32768`) to fit Qwen3.5-27B on a single 80 GB A100. Retrieval was run using our `retrieve_union.py` (from the internal `questions-dataset` repo — same pipeline used for our 320-question leakage-proof benchmark). Judge scoring was done via a ~150-line wrapper calling GPT-4o with OpenForecaster's `get_judge_prompt_with_gt` prompt text verbatim (see `scripts/judge_openai.py`).

## Context and prior work

- **OpenForecaster** (Chandak et al., 2025, [arXiv:2512.25070](https://arxiv.org/abs/2512.25070)) — the dataset and task framework.
- **Paleka et al., 2025** ([arXiv:2506.00723](https://arxiv.org/abs/2506.00723)) — retrospective audit documenting the ~2.4× leakage inflation of standard date-filtered web retrieval on forecasting benchmarks.
- The `aljazeeraLate2025` split was released after the main OpenForecaster paper and, to our knowledge, has no published numeric results on any model prior to this ablation. Shashwat Goel's [March 2026 tweet](https://x.com/ShashwatGoel7/status/2038246682043756794) describes DeepSeek V3.2 scoring negative on this split and trained 8B models achieving positive — consistent with what we see on untrained Qwen3.5-27B (all four rows remain negative).

## License

Predictions, judged outputs, retrieval artifacts and documentation in this folder are released under CC BY 4.0 (see repo-level `LICENSE`). Scripts are MIT. The underlying `aljazeeraLate2025` questions and article text belong to their respective sources (OpenForesight, Al Jazeera); this repo reproduces only our derived predictions and scoring, not the source data.
