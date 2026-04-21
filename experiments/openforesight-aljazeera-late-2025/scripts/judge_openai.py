"""
Judge OpenForecaster-format predictions via GPT-4o.
Uses the EXACT judge prompt from local_judge/llm_judge.py:get_judge_prompt_with_gt.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Their judge prompt, verbatim
def get_judge_prompt_with_gt(question, target, response, cot=True):
    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have the same information as in the ground-truth (not less nor unnecessary extra). 
The response can be more specific than the ground-truth (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be <= 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
"""
    prompt += f"""Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    return prompt

ANSWER_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)
client = OpenAI()

def extract_answer_from_pred(pred):
    ea = pred.get("extracted_answer")
    if not ea or not isinstance(ea, list) or not ea[0]:
        return None, None
    d = ea[0]
    # key = answer_str, value = prob
    ans_str = list(d.keys())[0]
    prob = list(d.values())[0]
    if ans_str == "null":
        return None, None
    return ans_str, prob

def judge_one(question, target, response):
    prompt = get_judge_prompt_with_gt(question, target, response)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )
            text = r.choices[0].message.content
            m = ANSWER_RE.search(text)
            if m:
                return int(m.group(1)), text
            # One more try with explicit reminder
            break
        except Exception as e:
            time.sleep(2 ** attempt)
    return None, None  # judge failure

def signed_reward(correct, prob):
    """Their scoring: correct -> 1-(1-p)^2, wrong -> -p^2, null -> -0.25 (soft Brier)."""
    if correct is None or prob is None:
        return -0.25
    if correct == 1:
        return 1.0 - (1.0 - prob) ** 2
    else:
        return -(prob ** 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--concurrency", type=int, default=20)
    args = p.parse_args()

    with open(args.input) as f:
        preds = [json.loads(l) for l in f]
    print(f"Loaded {len(preds)} predictions from {args.input}")

    def process(i_pred):
        i, pred = i_pred
        response, prob = extract_answer_from_pred(pred)
        question = pred.get("question_title", "")
        target = pred.get("answer", "")
        if response is None:
            # Token-cap / null — automatic miss but soft-brier penalty
            return {"idx": i, "judge": None, "response_given": None,
                    "correct": 0, "prob": None, "reward": -0.25,
                    "answer_gt": target, "question_title": question}
        corr, judge_text = judge_one(question, target, response)
        if corr is None:
            # Judge failed — treat as missing
            return {"idx": i, "judge": "JUDGE_FAILURE", "response_given": response,
                    "correct": 0, "prob": prob, "reward": -0.25,
                    "answer_gt": target, "question_title": question}
        return {"idx": i, "judge": judge_text, "response_given": response,
                "correct": corr, "prob": prob, "reward": signed_reward(corr, prob),
                "answer_gt": target, "question_title": question}

    results = [None] * len(preds)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(process, (i, p)): i for i, p in enumerate(preds)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  {done}/{len(preds)} judged | {rate:.1f}/s")

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    correct = sum(1 for r in results if r["correct"] == 1)
    null_count = sum(1 for r in results if r["response_given"] is None)
    rewards = [r["reward"] for r in results]
    mean_reward = sum(rewards) / len(rewards)
    print()
    print(f"=== RESULTS for {args.input} ===")
    print(f"Total: {len(results)}")
    print(f"Null (token-cap/unparsed): {null_count} ({null_count/len(results)*100:.1f}%)")
    print(f"Correct (judge says match): {correct} ({correct/len(results)*100:.1f}%)")
    print(f"Mean signed reward: {mean_reward:.4f}  (range: -1 worst wrong to +1 perfect)")
    print(f"Brier-style loss (−reward): {-mean_reward:.4f}")

if __name__ == "__main__":
    main()
