#!/usr/bin/env python3
"""Step 2: For each baseline CoT, generate all 8 perturbed versions.

Reads baselines from results/{model}_baselines.jsonl and writes one JSONL
record per (trial_id, perturbation_type) pair to results/{model}_perturbed_cots.jsonl.

Three of the perturbations (style, content_perturb, content_replace) call Haiku,
so the ANTHROPIC_API_KEY env var must be set.

Resume: skips (trial_id, perturbation_type) pairs already done.

Usage:
    python 02_generate_perturbed_cots.py --model r1-distill-8b
    python 02_generate_perturbed_cots.py --model qwen3-8b
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone

from configs import MODELS, PERTURBATIONS, baseline_path, perturbed_cot_path
from utils.io import load_jsonl, append_jsonl
from utils.haiku_client import get_haiku_client
from perturbations import apply_perturbation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baselines = load_jsonl(baseline_path(args.model))
    if not baselines:
        print(f"ERROR: no baselines at {baseline_path(args.model)}. Run step 1 first.")
        sys.exit(1)

    output = perturbed_cot_path(args.model)
    existing_keys = {
        (r["trial_id"], r["perturbation_type"])
        for r in load_jsonl(output)
    }

    haiku_client = get_haiku_client()
    rng = random.Random(args.seed)

    total_needed = len(baselines) * len(PERTURBATIONS)
    n_done = len(existing_keys)
    print(f"Total perturbations: {total_needed}, already done: {n_done}")

    for baseline in baselines:
        trial_id = baseline["trial_id"]
        cot = baseline["cot"]

        for pert_type, category, _description in PERTURBATIONS:
            key = (trial_id, pert_type)
            if key in existing_keys:
                continue

            t0 = time.time()
            try:
                perturbed = apply_perturbation(pert_type, cot, rng, haiku_client)
            except Exception as e:
                perturbed = f"ERROR: {e}"

            elapsed = time.time() - t0

            record = {
                "trial_id": trial_id,
                "model": args.model,
                "perturbation_type": pert_type,
                "category": category,
                "original_word_count": len(cot.split()),
                "perturbed_word_count": len(perturbed.split()),
                "perturbed_cot": perturbed,
                "elapsed_sec": round(elapsed, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(output, record)
            n_done += 1
            print(f"[{n_done}/{total_needed}] trial {trial_id} / {pert_type}: "
                  f"{len(cot.split())} -> {len(perturbed.split())} words "
                  f"({elapsed:.1f}s)")

    print(f"\nDone. {n_done} perturbations saved to {output}")


if __name__ == "__main__":
    main()
