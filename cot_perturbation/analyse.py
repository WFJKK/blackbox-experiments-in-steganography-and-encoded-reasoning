#!/usr/bin/env python3
"""Aggregate all results from steps 1-7 into a final report.

Produces a summary table with accuracy per perturbation, accuracy changes
relative to baseline, and 95% confidence intervals using Wilson score interval
(appropriate for proportions with small N).

Usage:
    python analyse.py --model r1-distill-8b
    python analyse.py --model qwen3-8b
    python analyse.py --all   # both models side-by-side
"""

import argparse
import math
from collections import defaultdict
from configs import (
    MODELS, PERTURBATIONS, evaluation_path, transfer_path,
    response_prediction_path, entanglement_path, baseline_path,
)
from utils.io import load_jsonl


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (point, low, high)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - half), min(1, center + half)


def fmt_pct_ci(k: int, n: int) -> str:
    p, lo, hi = wilson_ci(k, n)
    return f"{p:.0%} [{lo:.0%}, {hi:.0%}]"


def analyse_model(model_key: str):
    print(f"\n{'='*78}")
    print(f"  Model: {MODELS[model_key]['name']}")
    print(f"{'='*78}")

    # --- Step 4: task accuracy per condition ---
    evaluations = load_jsonl(evaluation_path(model_key))
    by_condition = defaultdict(list)
    for e in evaluations:
        by_condition[e["condition"]].append(e["correct"])

    if "baseline" not in by_condition:
        print("No baseline evaluations found. Has step 4 run?")
        return

    baseline_correct = sum(by_condition["baseline"])
    baseline_n = len(by_condition["baseline"])
    baseline_acc = baseline_correct / baseline_n if baseline_n else 0

    print(f"\n--- Task accuracy (E_1 perturbation effects) ---")
    print(f"{'Condition':<22} {'Category':<18} {'N':>4} {'Accuracy':>22} {'Δ from baseline':>16}")
    print("-" * 86)
    print(f"{'baseline':<22} {'-':<18} {baseline_n:>4} "
          f"{fmt_pct_ci(baseline_correct, baseline_n):>22} {'-':>16}")
    for pert_type, category, _ in PERTURBATIONS:
        results = by_condition.get(pert_type, [])
        if not results:
            print(f"{pert_type:<22} {category:<18} {'-':>4} (no data)")
            continue
        k, n = sum(results), len(results)
        acc = k / n
        delta = acc - baseline_acc
        print(f"{pert_type:<22} {category:<18} {n:>4} "
              f"{fmt_pct_ci(k, n):>22} {delta:>+15.0%}")

    # --- Step 5: E_2 cross-model transfer ---
    transfer_data = load_jsonl(transfer_path(model_key))
    if transfer_data:
        # We need to score Haiku's answers ourselves; for now report response counts
        # Real scoring requires comparing haiku_response to ground_truth
        # We'll do a simple "contains ground truth answer" check
        by_cond = defaultdict(lambda: {"n": 0, "correct": 0})
        for t in transfer_data:
            cond = t["condition"]
            gt = t["ground_truth"].strip().lower()
            resp = t["haiku_response"].strip().lower()
            # Try to find "Answer:" line and compare
            answer_line = ""
            for line in resp.split("\n"):
                if "answer:" in line.lower():
                    answer_line = line.split(":", 1)[-1].strip()
                    break
            if not answer_line:
                answer_line = resp[-200:]  # fallback: last bit of response
            match = gt in answer_line or answer_line in gt
            by_cond[cond]["n"] += 1
            by_cond[cond]["correct"] += int(match)

        print(f"\n--- Cross-model transfer (E_2): Haiku solves with model's CoT ---")
        print(f"{'Condition':<22} {'N':>4} {'Haiku accuracy':>22}")
        print("-" * 50)
        ordered = ["baseline"] + [p[0] for p in PERTURBATIONS]
        for cond in ordered:
            if cond not in by_cond:
                continue
            d = by_cond[cond]
            print(f"{cond:<22} {d['n']:>4} {fmt_pct_ci(d['correct'], d['n']):>22}")

    # --- Step 6: E_3 response prediction ---
    rp = load_jsonl(response_prediction_path(model_key))
    if rp:
        # Aggregate by run
        by_run = defaultdict(list)
        for r in rp:
            by_run[r["run_id"]].append(r["match"])
        print(f"\n--- Response prediction (E_3.1): does CoT determine answer? ---")
        print(f"{'Run':>6} {'N':>4} {'Match rate':>22}")
        print("-" * 36)
        all_matches = []
        for run_id, matches in sorted(by_run.items()):
            k, n = sum(matches), len(matches)
            print(f"{run_id:>6} {n:>4} {fmt_pct_ci(k, n):>22}")
            all_matches.extend(matches)
        if all_matches:
            k, n = sum(all_matches), len(all_matches)
            print(f"{'all':>6} {n:>4} {fmt_pct_ci(k, n):>22}")

    # --- Step 7: E_4 entanglement ---
    ent = load_jsonl(entanglement_path(model_key))
    if ent:
        valid = [e for e in ent if e.get("parse_ok") and e["score"] > 0]
        n = len(valid)
        if n > 0:
            mean_score = sum(e["score"] for e in valid) / n
            contradiction_rate = sum(e["contradiction"] for e in valid) / n
            ignored_rate = sum(e["ignored_conclusion"] for e in valid) / n
            print(f"\n--- Entanglement (E_4.1.2): rubric scoring of CoT-answer pairs ---")
            print(f"N = {n}, mean score = {mean_score:.2f}/5")
            print(f"Contradictions: {fmt_pct_ci(sum(e['contradiction'] for e in valid), n)}")
            print(f"Ignored conclusions: {fmt_pct_ci(sum(e['ignored_conclusion'] for e in valid), n)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Run for all models in configs")
    args = parser.parse_args()

    if args.all:
        for m in MODELS:
            analyse_model(m)
    elif args.model:
        analyse_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
