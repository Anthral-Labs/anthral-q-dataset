"""
Step 4: Gate 2 — LLM leakage check on retrieved chunks.

Input:  retrieved_context_raw.json (from step3)
Output: retrieved_context_clean.json (leakage-free chunks only)

Batches all (question, chunk) pairs through gpt-4o-mini via OpenAI Batch API
at 50% discount. Each pair is checked: "Does this article reveal the answer?"

Run:
  python3 step4_leakage_check.py submit --input /data/retrieval/retrieved_context_raw.json
  python3 step4_leakage_check.py check
  python3 step4_leakage_check.py apply --input /data/retrieval/retrieved_context_raw.json --output /data/retrieval/retrieved_context_clean.json
"""

import json
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/data/retrieval")
BATCH_INPUT = DATA_DIR / "batch_leakage_input.jsonl"
BATCH_OUTPUT = DATA_DIR / "batch_leakage_output.jsonl"
BATCH_ID_FILE = DATA_DIR / "batch_leakage_id.txt"

LEAKAGE_PROMPT = """You are a temporal leakage detector for a forecasting research project.

FORECASTING QUESTION:
{question}

The known correct answer to this question is: "{answer}"

Check specifically whether the article:
1. Directly states this answer or outcome
2. Contains information that makes this answer obvious
3. Reports on the resolution of this event

ARTICLE TEXT:
{article}

TASK: Determine whether this article contains information that reveals or strongly
hints at the outcome of the forecasting question. We want articles that provide
BACKGROUND CONTEXT for making a prediction, NOT articles that report the OUTCOME.

An article FAILS (contains leakage) if it:
- Directly reports the outcome/resolution of the event
- Contains results, scores, or final decisions that answer the question
- Was clearly written AFTER the event resolved and describes what happened

An article PASSES (is clean) if it:
- Provides background context, analysis, or predictions
- Discusses the event in a forward-looking way
- Contains information that would help someone PREDICT the outcome

Respond with exactly one of:
PASSES - [brief reason]
FAILS - [brief reason]
"""


def cmd_submit(args):
    """Create and submit batch for leakage checking."""
    import requests
    import os

    with open(args.input) as f:
        results = json.load(f)

    # Build batch requests
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    request_count = 0

    with open(BATCH_INPUT, "w") as f:
        for r in results:
            question = r["title"]
            answer = r.get("resolution", "")

            for j, chunk in enumerate(r["chunks"]):
                text = chunk.get("text", "")[:4000]

                prompt = LEAKAGE_PROMPT.format(
                    question=question,
                    answer=answer,
                    article=text,
                )

                request = {
                    "custom_id": f"q{r['question_idx']}-c{j}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a temporal leakage detector. Respond with exactly: PASSES - [reason] or FAILS - [reason]. Nothing else."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": 80,
                    },
                }
                f.write(json.dumps(request) + "\n")
                request_count += 1

    logger.info("Created %d leakage check requests", request_count)
    logger.info("Estimated cost: ~$%.2f (with 50%% batch discount)", request_count * 0.0002)

    # Submit batch(es)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "OPENAI_API_KEY" in line:
                        api_key = line.strip().split("=", 1)[1]

    if not api_key:
        logger.error("No OPENAI_API_KEY found. Set it in environment or .env file.")
        return

    # Split if > 50K requests
    lines = BATCH_INPUT.read_text().strip().split("\n")
    batch_ids = []

    for part_idx in range(0, len(lines), 50000):
        part_lines = lines[part_idx:part_idx + 50000]
        part_path = DATA_DIR / f"batch_leakage_input_{part_idx // 50000}.jsonl"
        part_path.write_text("\n".join(part_lines) + "\n")

        # Upload
        logger.info("Uploading part %d (%d requests)...", part_idx // 50000, len(part_lines))
        upload_resp = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (part_path.name, open(part_path, "rb"), "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
        file_id = upload_resp.json()["id"]

        # Create batch
        batch_resp = requests.post(
            "https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"description": f"leakage_check_part_{part_idx // 50000}"},
            },
            timeout=30,
        )
        batch_id = batch_resp.json()["id"]
        batch_ids.append(batch_id)
        logger.info("Submitted batch: %s", batch_id)

    with open(BATCH_ID_FILE, "w") as f:
        f.write("\n".join(batch_ids) + "\n")

    logger.info("All batches submitted. Run 'step4_leakage_check.py check' to poll status.")


def cmd_check(args):
    """Check batch status and download results."""
    import requests
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "OPENAI_API_KEY" in line:
                        api_key = line.strip().split("=", 1)[1]

    with open(BATCH_ID_FILE) as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    all_complete = True
    all_results = []

    for batch_id in batch_ids:
        resp = requests.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        status = resp.json()
        state = status.get("status")
        counts = status.get("request_counts", {})
        logger.info("Batch %s: %s (%d/%d completed, %d failed)",
                     batch_id, state,
                     counts.get("completed", 0), counts.get("total", 0),
                     counts.get("failed", 0))

        if state != "completed":
            all_complete = False
            continue

        # Download results
        output_file_id = status.get("output_file_id")
        resp2 = requests.get(
            f"https://api.openai.com/v1/files/{output_file_id}/content",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        for line in resp2.text.strip().split("\n"):
            if line.strip():
                all_results.append(json.loads(line))

    if not all_complete:
        logger.info("Not all batches complete. Run again later.")
        return

    # Save combined results
    with open(BATCH_OUTPUT, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    logger.info("Downloaded %d leakage check results to %s", len(all_results), BATCH_OUTPUT)
    logger.info("Run 'step4_leakage_check.py apply' to filter results.")


def cmd_apply(args):
    """Apply leakage verdicts to retrieved context."""
    with open(args.input) as f:
        results = json.load(f)

    # Parse verdicts
    verdicts = {}  # "q{idx}-c{j}" → True (passes) / False (fails)
    with open(BATCH_OUTPUT) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row["custom_id"]
            content = row["response"]["body"]["choices"][0]["message"]["content"].strip()
            verdicts[custom_id] = content.upper().startswith("PASSES")

    # Apply
    total_kept = 0
    total_rejected = 0
    clean_results = []

    for r in results:
        clean_chunks = []
        for j, chunk in enumerate(r["chunks"]):
            key = f"q{r['question_idx']}-c{j}"
            if verdicts.get(key, True):  # Default to keep if verdict missing
                clean_chunks.append(chunk)
                total_kept += 1
            else:
                total_rejected += 1

        clean_results.append({
            "question_idx": r["question_idx"],
            "question_id": r["question_id"],
            "title": r["title"],
            "resolution": r.get("resolution", ""),
            "resolution_date": r.get("resolution_date", ""),
            "num_chunks": len(clean_chunks),
            "chunks": clean_chunks,
        })

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    with_context = sum(1 for r in clean_results if r["num_chunks"] > 0)
    logger.info("")
    logger.info("=" * 70)
    logger.info("LEAKAGE FILTERING COMPLETE")
    logger.info("=" * 70)
    logger.info("Chunks kept: %d", total_kept)
    logger.info("Chunks rejected (leaky): %d", total_rejected)
    logger.info("Rejection rate: %.1f%%", 100 * total_rejected / max(total_kept + total_rejected, 1))
    logger.info("Questions with context: %d/%d", with_context, len(clean_results))
    logger.info("Output: %s", output_path)
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next: run step5_assemble.py to create training prompts")


def main():
    parser = argparse.ArgumentParser(description="Gate 2: LLM leakage check")
    sub = parser.add_subparsers(dest="cmd")

    p_submit = sub.add_parser("submit", help="Submit batch for leakage checking")
    p_submit.add_argument("--input", required=True, help="retrieved_context_raw.json")

    p_check = sub.add_parser("check", help="Check batch status and download results")

    p_apply = sub.add_parser("apply", help="Apply verdicts to filter chunks")
    p_apply.add_argument("--input", required=True, help="retrieved_context_raw.json")
    p_apply.add_argument("--output", required=True, help="retrieved_context_clean.json")

    args = parser.parse_args()

    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "apply":
        cmd_apply(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
