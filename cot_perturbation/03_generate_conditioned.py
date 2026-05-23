#!/usr/bin/env python3
"""Step 3: For each (trial, perturbation), pre-fill the perturbed CoT into the
model and let it generate a final answer conditioned on that perturbed reasoning.

This is the core experiment: testing whether the model's answer depends on
the structure or surface of the CoT, vs only its semantic content.

Reads from results/{model}_perturbed_cots.jsonl, writes to
results/{model}_conditioned.jsonl. Each output record contains the
conditioned answer (what the model generated AFTER the perturbed think block).

Resume support: skips (trial_id, perturbation_type) pairs already done.

Usage:
    python 03_generate_conditioned.py --model r1-distill-8b
    python 03_generate_conditioned.py --model qwen3-8b --vllm-port 8000
"""

import argparse
import sys
import time
from datetime import datetime, timezone

from configs import MODELS, baseline_path, perturbed_cot_path, conditioned_path
from utils.io import load_jsonl, append_jsonl
from utils.vllm_client import get_vllm_client, get_served_model, generate_conditioned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--vllm-port", type=int, default=8000)
    args = parser.parse_args()

    model_config = MODELS[args.model]

    # Load problems (from baselines), perturbations, existing results
    baselines = {r["trial_id"]: r for r in load_jsonl(baseline_path(args.model))}
    if not baselines:
        print(f"ERROR: no baselines at {baseline_path(args.model)}. Run step 1 first.")
        sys.exit(1)

    perturbations = load_jsonl(perturbed_cot_path(args.model))
    if not perturbations:
        print(f"ERROR: no perturbed CoTs at {perturbed_cot_path(args.model)}. Run step 2 first.")
        sys.exit(1)

    output = conditioned_path(args.model)
    existing_keys = {
        (r["trial_id"], r["perturbation_type"])
        for r in load_jsonl(output)
    }

    # vLLM
    client = get_vllm_client(args.vllm_port)
    try:
        served = get_served_model(client)
    except Exception as e:
        print(f"ERROR: cannot reach vLLM at port {args.vllm_port}: {e}")
        sys.exit(1)

    total = len(perturbations)
    n_done = len(existing_keys)
    print(f"Total conditioned generations: {total}, already done: {n_done}")

    for record in perturbations:
        trial_id = record["trial_id"]
        pert_type = record["perturbation_type"]
        key = (trial_id, pert_type)
        if key in existing_keys:
            continue

        baseline = baselines.get(trial_id)
        if baseline is None:
            print(f"WARNING: no baseline for trial {trial_id}, skipping")
            continue

        if record["perturbed_cot"].startswith("ERROR:"):
            # Perturbation failed; record an error and move on
            append_jsonl(output, {
                "trial_id": trial_id,
                "model": args.model,
                "perturbation_type": pert_type,
                "category": record["category"],
                "conditioned_answer": f"SKIPPED: {record['perturbed_cot']}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            n_done += 1
            continue

        t0 = time.time()
        conditioned_answer = generate_conditioned(
            client, served,
            problem=baseline["problem"],
            perturbed_cot=record["perturbed_cot"],
            max_tokens=1024,  # answer-only, no need for CoT length
            temperature=model_config["temperature"],
        )
        elapsed = time.time() - t0

        out_record = {
            "trial_id": trial_id,
            "model": args.model,
            "perturbation_type": pert_type,
            "category": record["category"],
            "ground_truth": baseline["ground_truth"],
            "conditioned_answer": conditioned_answer,
            "elapsed_sec": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(output, out_record)
        n_done += 1

        print(f"[{n_done}/{total}] trial {trial_id} / {pert_type}: "
              f"{conditioned_answer[:80]!r} ({elapsed:.1f}s)")

    print(f"\nDone. {n_done} conditioned answers saved to {output}")


if __name__ == "__main__":
    main()
