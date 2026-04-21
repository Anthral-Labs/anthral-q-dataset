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

| Run | Correct | Accuracy | Null (token-cap) | **Mean signed reward** |
|---|---:|---:|---:|---:|
| A — no context | 45 / 491 | 9.2% | 6.1% (30) | **−0.1554** |
| B — their retrieval | 64 / 491 | 13.0% | 2.4% (12) | **−0.2561** |
| **C-loose** — our retrieval, loose | **133 / 491** | **27.1%** | 1.4% (7) | **−0.0869** |
| C-strict — our retrieval, strict | 86 / 491 | 17.5% | 0.6% (3) | **−0.1733** |

Range of mean signed reward is `[−1, +1]`. Zero is neutral; negative means systematically-wrong-with-confidence dominates correctly-reasoned-with-confidence.

## Findings

1. **Our GDELT corpus more than doubles the accuracy of OpenForecaster's retrieval** (B vs C-loose, matched protocol): 13.0% → 27.1%, mean reward −0.256 → −0.087. The only variable that changed is the article corpus — our 31.5M-article GDELT pool has meaningfully better coverage of Sept–Dec 2025 events than whatever OpenForecaster's retrieval pipeline uses.

2. **Leakage quantification**: switching from the loose filter (`article_date ≤ resolution_date`) to our strict creation-date filter (`article_date ≤ question_start_date`) drops accuracy by 9.6 pp and mean reward by 0.086. Approximately half of retrieval's apparent lift on this benchmark comes from articles published between question creation and resolution. This is the Paleka-et-al.-style leakage-premium effect, measured directly on the OpenForesight family for the first time we are aware of.

3. **Calibration ceiling for untrained base models**: even with C-loose's best-in-class retrieval, mean signed reward is `−0.087`. Base Qwen3.5-27B is systematically overconfident on the ~73% of questions it still gets wrong. Closing this gap requires training, not better retrieval — consistent with OpenForecaster's thesis that a combined accuracy-plus-Brier reward function (GRPO, OpenForecaster-8B recipe) is the calibration lever.

## Failure-mode texture (qualitative)

Predictions where our retrieval helped most are named-entity questions about globally-reported events with clear single-article resolution (Rio police raids → "Public Defender's Office"; India tariff announcement → date of enforcement). Predictions that still fail under C-loose cluster into three buckets: (a) answers living in niche outlets outside our GDELT coverage, (b) specificity mismatches where the right article is retrieved but the extracted entity is the wrong grain ("Public Security Secretariat" vs "Public Defender's Office"), (c) questions where the model reasoned itself into a confident wrong answer despite having relevant articles in context.

## Files

```
experiments/openforesight-aljazeera-late-2025/
├── README.md                       (this file)
├── summary.json                    (headline numbers in machine-readable form)
├── predictions/                    (raw vLLM outputs per run — one JSONL per run, 491 records each)
│   ├── run_a_no_context.jsonl
│   ├── run_b_their_retrieval.jsonl
│   ├── run_c_loose.jsonl
│   └── run_c_strict.jsonl
├── judged/                         (per-question LLM-judge results — binary correct + signed reward)
│   ├── run_a.jsonl
│   ├── run_b.jsonl
│   ├── run_c_loose.jsonl
│   └── run_c_strict.jsonl
├── retrievals/                     (top-5 chunks from our GDELT corpus per question, both filters)
│   ├── loose.json
│   └── strict.json
└── scripts/
    ├── judge_openai.py             (GPT-4o judge wrapper using OpenForecaster's prompt verbatim)
    ├── splice_prompts.py           (splices our retrieved chunks into OpenForecaster's prompt template)
    └── make_retrieval_shim.py      (converts aljazeeraLate2025 records to the format our retrieve_union.py expects)
```

## Reproducibility

Inference was run using OpenForecaster's published `eval_openforesight.py` from [`OpenForecaster/scaling-forecasting-training`](https://github.com/OpenForecaster/scaling-forecasting-training), with one local patch (`max_model_len=32768`) to fit Qwen3.5-27B on a single 80 GB A100. Retrieval was run using our `retrieve_union.py` (from the internal `questions-dataset` repo — same pipeline used for our 320-question leakage-proof benchmark). Judge scoring was done via a ~150-line wrapper calling GPT-4o with OpenForecaster's `get_judge_prompt_with_gt` prompt text verbatim (see `scripts/judge_openai.py`).

## Context and prior work

- **OpenForecaster** (Chandak et al., 2025, [arXiv:2512.25070](https://arxiv.org/abs/2512.25070)) — the dataset and task framework.
- **Paleka et al., 2025** ([arXiv:2506.00723](https://arxiv.org/abs/2506.00723)) — retrospective audit documenting the ~2.4× leakage inflation of standard date-filtered web retrieval on forecasting benchmarks.
- The `aljazeeraLate2025` split was released after the main OpenForecaster paper and, to our knowledge, has no published numeric results on any model prior to this ablation. Shashwat Goel's [March 2026 tweet](https://x.com/ShashwatGoel7/status/2038246682043756794) describes DeepSeek V3.2 scoring negative on this split and trained 8B models achieving positive — consistent with what we see on untrained Qwen3.5-27B (all four rows remain negative).

## License

Predictions, judged outputs, retrieval artifacts and documentation in this folder are released under CC BY 4.0 (see repo-level `LICENSE`). Scripts are MIT. The underlying `aljazeeraLate2025` questions and article text belong to their respective sources (OpenForesight, Al Jazeera); this repo reproduces only our derived predictions and scoring, not the source data.
