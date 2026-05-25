#!/usr/bin/env python3
"""Step 1: Generate baseline responses for each MATH-500 problem.

Sends each problem to the model and records the full response, parsed into
the CoT and the final answer. Writes one JSONL record per trial.

Usage:
    python 01_generate_baselines.py --model r1-distill-8b --n 50
    python 01_generate_baselines.py --model qwen3-8b --n 50 --vllm-port 8000
    python 01_generate_baselines.py --model r1-distill-8b --resume

Resume: skips any trial_id already in the output file.
"""

import argparse
import sys
import time
from datetime import datetime, timezone

from configs import MODELS, N_BASELINES, baseline_path
from utils.io import sample_problems, load_jsonl, append_jsonl
from utils.vllm_client import (
    get_vllm_client,
    get_served_model,
    generate_baseline,
    parse_cot_and_answer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument(
        "--n", type=int, default=N_BASELINES, help="Number of baselines"
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_config = MODELS[args.model]
    output = baseline_path(args.model)

    # Verify vLLM server is serving the right model
    client = get_vllm_client(args.vllm_port)
    try:
        served = get_served_model(client)
    except Exception as e:
        print(f"ERROR: cannot reach vLLM server at port {args.vllm_port}: {e}")
        print(
            f"Start it with: vllm serve {model_config['hf_id']} --port {args.vllm_port}"
        )
        sys.exit(1)

    if model_config["hf_id"] not in served:
        print(
            f"WARNING: vLLM is serving '{served}', expected '{model_config['hf_id']}'"
        )
        print("Continuing anyway in case of name aliasing...")

    # Load sampled problems and existing results
    problems = sample_problems(args.n, seed=args.seed)
    existing = {r["trial_id"] for r in load_jsonl(output)}
    print(f"Model: {model_config['name']} (vLLM: {served})")
    print(f"Total problems: {len(problems)}, already done: {len(existing)}")

    n_done = len(existing)
    for problem in problems:
        if problem["trial_id"] in existing:
            continue

        t0 = time.time()
        print(
            f"\n[{n_done + 1}/{len(problems)}] trial {problem['trial_id']}: {problem['problem'][:80]}..."
        )

        response_text = generate_baseline(
            client,
            served,
            problem["problem"],
            max_tokens=model_config["max_tokens"],
            temperature=model_config["temperature"],
        )

        cot, answer = parse_cot_and_answer(response_text)
        elapsed = time.time() - t0

        record = {
            "trial_id": problem["trial_id"],
            "model": args.model,
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "subject": problem.get("subject", ""),
            "level": problem.get("level", ""),
            "problem_uid": problem.get("id", ""),
            "full_response": response_text,
            "cot": cot,
            "answer": answer,
            "cot_word_count": len(cot.split()),
            "elapsed_sec": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(output, record)
        n_done += 1

        print(
            f"  CoT: {len(cot.split())} words | answer: {answer[:80]!r} | {elapsed:.1f}s"
        )

    print(f"\nDone. {n_done} baselines saved to {output}")


if __name__ == "__main__":
    main()
