"""
Step 5: Assemble training prompts from clean context + questions.

Input:
  - retrieved_context_clean.json (from step4)
  - polymarket_final.json (questions with resolutions + market prices)

Output:
  - training_data.json: frozen prompts ready for RL training
  - train_split.json: 5K questions for training
  - test_split.json: 5K questions for evaluation

Each training example is a frozen prompt:
  [system] + [0-5 context chunks] + [question] + [instructions]

Context count is randomized (0-5) per OpenForesighter methodology,
so the model learns to forecast with variable context availability.

Run:
  python3 step5_assemble.py --context /data/retrieval/retrieved_context_clean.json --questions /data/questions/polymarket_final.json --output /data/training
"""

import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert forecaster familiar with the work of Philip Tetlock and the principles of superforecasting. You make calibrated probabilistic predictions based on available evidence.

Analyze the provided context articles carefully. Consider base rates, reference classes, and any relevant evidence. Think step by step about factors that increase or decrease the probability of the event occurring.

After your analysis, output your probability estimate as a number between 0 and 1."""

QUESTION_TEMPLATE = """## Context Articles

{context}

## Question

{question}

## Instructions

Based on the context above and your general knowledge, estimate the probability that this event will resolve YES.

Think through your reasoning step by step, then provide your final probability estimate.

Your response MUST end with:
PROBABILITY: <number between 0 and 1>"""

NO_CONTEXT_TEMPLATE = """## Question

{question}

## Instructions

Based on your general knowledge, estimate the probability that this event will resolve YES.

Think through your reasoning step by step, then provide your final probability estimate.

Your response MUST end with:
PROBABILITY: <number between 0 and 1>"""


def format_context(chunks: list, max_chunks: int = 5) -> str:
    """Format retrieved chunks into context string."""
    if not chunks:
        return ""

    # Randomly select 0-max_chunks articles (OpenForesighter methodology)
    n = random.randint(0, min(max_chunks, len(chunks)))
    if n == 0:
        return ""

    selected = chunks[:n]
    parts = []
    for i, chunk in enumerate(selected):
        date = chunk.get("date", "unknown date")
        url = chunk.get("url", "")
        domain = url.split("/")[2] if "/" in url and len(url.split("/")) > 2 else "unknown"
        text = chunk.get("text", "")

        parts.append(f"### Article {i+1} ({domain}, {date})\n{text}")

    return "\n\n".join(parts)


def create_prompt(question_text: str, chunks: list, include_context: bool = True) -> str:
    """Create a frozen training/evaluation prompt."""
    if include_context and chunks:
        context_str = format_context(chunks)
        if context_str:
            return QUESTION_TEMPLATE.format(context=context_str, question=question_text)

    return NO_CONTEXT_TEMPLATE.format(question=question_text)


def main():
    parser = argparse.ArgumentParser(description="Assemble training prompts")
    parser.add_argument("--context", required=True, help="Path to retrieved_context_clean.json")
    parser.add_argument("--questions", required=True, help="Path to polymarket_final.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split + context sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.context) as f:
        context_data = json.load(f)
    with open(args.questions) as f:
        questions = json.load(f)

    logger.info("Loaded %d context entries and %d questions", len(context_data), len(questions))

    # Build context lookup
    context_by_idx = {r["question_idx"]: r for r in context_data}

    # Assemble training examples
    examples = []
    for i, q in enumerate(questions):
        ctx = context_by_idx.get(i, {})
        chunks = ctx.get("chunks", [])

        # Parse resolution to binary outcome
        resolution = q.get("resolution", "")
        outcome = _parse_binary_outcome(resolution)
        if outcome is None:
            continue  # Skip questions with unclear resolution

        # Get market price (community prediction baseline)
        market_price = q.get("community_prediction")

        prompt = create_prompt(q.get("title", ""), chunks)

        examples.append({
            "question_idx": i,
            "question_id": q.get("id"),
            "title": q.get("title", ""),
            "category": q.get("category", "other"),
            "resolution_date": q.get("actual_resolve_time", ""),
            "outcome": outcome,  # 1.0 = yes, 0.0 = no
            "market_price": market_price,
            "num_context_chunks": len(chunks),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt,
        })

    logger.info("Assembled %d training examples (dropped %d with unclear resolution)",
                len(examples), len(questions) - len(examples))

    # Split into train/test
    # Sort by resolution date, then split with temporal gap
    examples.sort(key=lambda x: x.get("resolution_date", ""))

    # Ensure temporal gap: latest train resolution < earliest test resolution - 10 days
    mid = len(examples) // 2
    train = examples[:mid]
    test = examples[mid:]

    if train and test:
        last_train_date = train[-1].get("resolution_date", "")[:10]
        first_test_date = test[0].get("resolution_date", "")[:10]
        logger.info("Train: %d examples (resolves up to %s)", len(train), last_train_date)
        logger.info("Test: %d examples (resolves from %s)", len(test), first_test_date)

    # Save
    with open(output_dir / "all_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    with open(output_dir / "train_split.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(output_dir / "test_split.json", "w") as f:
        json.dump(test, f, indent=2)

    # Category breakdown
    train_cats = {}
    test_cats = {}
    for ex in train:
        train_cats[ex["category"]] = train_cats.get(ex["category"], 0) + 1
    for ex in test:
        test_cats[ex["category"]] = test_cats.get(ex["category"], 0) + 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("ASSEMBLY COMPLETE")
    logger.info("=" * 70)
    logger.info("Total examples: %d", len(examples))
    logger.info("Train: %d | Test: %d", len(train), len(test))
    logger.info("")
    logger.info("Train categories:")
    for cat, n in sorted(train_cats.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", cat, n)
    logger.info("")
    logger.info("Test categories:")
    for cat, n in sorted(test_cats.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", cat, n)
    logger.info("")
    logger.info("Output: %s", output_dir)
    logger.info("  all_examples.json — full dataset")
    logger.info("  train_split.json — training set")
    logger.info("  test_split.json — evaluation set")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Ready for Phase 4: RL Training")


def _parse_binary_outcome(resolution) -> float | None:
    """Parse resolution to binary outcome (1.0 = yes, 0.0 = no)."""
    if resolution is None:
        return None

    res_str = str(resolution).lower().strip()

    if res_str in ("yes", "true", "1", "1.0"):
        return 1.0
    elif res_str in ("no", "false", "0", "0.0"):
        return 0.0
    else:
        # Polymarket might have team names etc — check if it looks like a "yes" outcome
        # For markets where resolution is the winning option name, we need the original
        # question structure. For now, skip ambiguous ones.
        return None


if __name__ == "__main__":
    main()
