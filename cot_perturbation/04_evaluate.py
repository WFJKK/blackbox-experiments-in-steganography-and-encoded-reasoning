#!/usr/bin/env python3
"""Step 4: Evaluate task accuracy on baseline + conditioned answers.

Uses Claude Haiku as a judge to check whether the model's answer matches the
MATH-500 ground truth. Writes one JSONL record per (trial_id, condition) pair
where condition is either 'baseline' or one of the 8 perturbation types.

Resume: skips (trial_id, condition) already evaluated.

Usage:
    python 04_evaluate.py --model r1-distill-8b
    python 04_evaluate.py --model qwen3-8b
"""

import argparse
import sys
import time
from datetime import datetime, timezone

from configs import MODELS, baseline_path, conditioned_path, evaluation_path
from utils.io import load_jsonl, append_jsonl
from utils.haiku_client import get_haiku_client, call_haiku


JUDGE_SYSTEM = (
    "You are a math answer checker. You will be given a math problem, the "
    "correct answer, and a model's response. Determine whether the model's "
    "final answer matches the correct answer (allowing for different but "
    "equivalent forms such as 1/2 vs 0.5, or different orderings of a set).\n\n"
    "Respond with ONLY 'correct' or 'incorrect'. Nothing else."
)


def judge_user_prompt(problem: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"Correct answer: {ground_truth}\n\n"
        f"Model response: {model_answer}\n\n"
        "Did the model arrive at the correct answer?"
    )


def is_correct(verdict: str) -> bool:
    v = verdict.lower().strip()
    return "correct" in v and "incorrect" not in v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    baselines = load_jsonl(baseline_path(args.model))
    conditioned = load_jsonl(conditioned_path(args.model))
    if not baselines:
        print(f"ERROR: no baselines at {baseline_path(args.model)}.")
        sys.exit(1)

    output = evaluation_path(args.model)
    existing_keys = {
        (r["trial_id"], r["condition"])
        for r in load_jsonl(output)
    }

    haiku = get_haiku_client()

    # Build the list of trials to judge: baselines + each perturbation
    judgments_needed = []
    for b in baselines:
        judgments_needed.append({
            "trial_id": b["trial_id"],
            "condition": "baseline",
            "problem": b["problem"],
            "ground_truth": b["ground_truth"],
            "model_answer": b["answer"],
        })
    for c in conditioned:
        judgments_needed.append({
            "trial_id": c["trial_id"],
            "condition": c["perturbation_type"],
            "problem": baselines_by_id_lookup(baselines, c["trial_id"])["problem"],
            "ground_truth": c["ground_truth"],
            "model_answer": c["conditioned_answer"],
        })

    total = len(judgments_needed)
    n_done = len(existing_keys)
    print(f"Total judgments: {total}, already done: {n_done}")

    for j in judgments_needed:
        key = (j["trial_id"], j["condition"])
        if key in existing_keys:
            continue

        if j["model_answer"].startswith("SKIPPED:") or j["model_answer"].startswith("ERROR:"):
            verdict = "incorrect"
            correct = False
        else:
            t0 = time.time()
            verdict = call_haiku(
                haiku, JUDGE_SYSTEM,
                judge_user_prompt(j["problem"], j["ground_truth"], j["model_answer"]),
                max_tokens=16,
            )
            correct = is_correct(verdict)

        record = {
            "trial_id": j["trial_id"],
            "model": args.model,
            "condition": j["condition"],
            "model_answer": j["model_answer"],
            "ground_truth": j["ground_truth"],
            "verdict_raw": verdict,
            "correct": correct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(output, record)
        n_done += 1
        if n_done % 20 == 0 or n_done <= 5:
            print(f"[{n_done}/{total}] trial {j['trial_id']} / {j['condition']}: "
                  f"{'ok' if correct else 'WRONG'}")

    print(f"\nDone. {n_done} judgments saved to {output}")


def baselines_by_id_lookup(baselines, trial_id):
    for b in baselines:
        if b["trial_id"] == trial_id:
            return b
    return None


if __name__ == "__main__":
    main()
