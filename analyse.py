#!/usr/bin/env python3
"""Analyse prompted steganography results and produce summary table.

Usage:
    python analyse.py results/full_run.jsonl
    python analyse.py results/full_run.jsonl --latex
"""

import argparse
import json
import sys
from collections import defaultdict


def load_results(path: str) -> list[dict]:
    """Load JSONL results."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def analyse(results: list[dict]) -> dict:
    """Compute per-condition aggregate metrics."""
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["scheme"])
        groups[key].append(r)

    summary = {}
    for (model, scheme), trials in sorted(groups.items()):
        n = len(trials)
        # Filter out errored trials
        valid = [t for t in trials if "error" not in t]
        n_valid = len(valid)

        if n_valid == 0:
            summary[(model, scheme)] = {
                "n": n, "n_valid": 0,
                "task_acc": 0, "payload_pre": 0, "payload_post": 0,
                "monitor_rate": 0, "mean_ber_pre": 1.0, "mean_ber_post": 1.0,
            }
            continue

        # Task accuracy
        task_correct = sum(1 for t in valid if t.get("task_correct", False))
        task_acc = task_correct / n_valid

        # Payload recovery = 1 - BER (proportion of bits correct)
        mean_ber_pre = sum(t.get("ber_pre_cap", 1.0) for t in valid) / n_valid
        mean_ber_post = sum(t.get("ber_post_cap", 1.0) for t in valid) / n_valid
        payload_pre = 1 - mean_ber_pre
        payload_post = 1 - mean_ber_post

        # Monitor detection rate
        monitor_flagged = sum(1 for t in valid if t.get("monitor_suspicious", False))
        monitor_rate = monitor_flagged / n_valid

        summary[(model, scheme)] = {
            "n": n,
            "n_valid": n_valid,
            "task_acc": task_acc,
            "payload_pre": payload_pre,
            "payload_post": payload_post,
            "monitor_rate": monitor_rate,
            "mean_ber_pre": mean_ber_pre,
            "mean_ber_post": mean_ber_post,
        }

    return summary


def print_table(summary: dict):
    """Print a readable summary table."""
    print(f"{'Model':<16} {'Scheme':<15} {'N':>4} {'Task acc':>9} {'Payload(pre)':>13} {'Payload(post)':>14} {'Monitor':>8}")
    print("-" * 85)

    for (model, scheme), s in sorted(summary.items()):
        print(
            f"{model:<16} {scheme:<15} {s['n_valid']:>4} "
            f"{s['task_acc']:>8.0%} "
            f"{s['payload_pre']:>12.0%} "
            f"{s['payload_post']:>13.0%} "
            f"{s['monitor_rate']:>7.0%}"
        )


def print_latex(summary: dict):
    """Print LaTeX table."""
    model_names = {
        "qwen2.5-7b": "Qwen2.5-7B",
        "qwen3-8b-nothink": "Qwen3-8B (no-think)",
        "qwen3-8b-think": "Qwen3-8B (think)",
        "llama4-maverick": "Llama-4-Maverick",
    }
    scheme_names = {
        "hex": "hex",
        "acrostic": "acrostic",
        "synonym": "synonym",
        "model_invented": "model-inv.",
    }

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Scheme} & \textbf{Task acc.} & \textbf{Payload (pre)} & \textbf{Payload (post-CAP)} & \textbf{Monitor} \\")
    print(r"\midrule")

    for (model, scheme), s in sorted(summary.items()):
        mname = model_names.get(model, model)
        sname = scheme_names.get(scheme, scheme)
        print(
            f"{mname} & {sname} & "
            f"{s['task_acc']:.0%} & "
            f"{s['payload_pre']:.0%} & "
            f"{s['payload_post']:.0%} & "
            f"{s['monitor_rate']:.0%} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Prompted steganography with CAP defence. $N = " + str(next(iter(summary.values()))['n_valid']) + r"$ per condition, 4-bit payloads.}")
    print(r"\label{tab:prompted_steg}")
    print(r"\end{table}")


def print_per_trial_detail(results: list[dict], model: str = None, scheme: str = None):
    """Print per-trial detail for debugging."""
    for r in results:
        if model and r["model"] != model:
            continue
        if scheme and r["scheme"] != scheme:
            continue
        print(f"\n--- {r['model']} / {r['scheme']} / trial {r['trial']} ---")
        print(f"  Payload:     {r['payload']}")
        print(f"  Decoded pre: {r.get('decoded_pre_cap', '?')}")
        print(f"  Decoded post:{r.get('decoded_post_cap', '?')}")
        print(f"  BER pre:     {r.get('ber_pre_cap', '?')}")
        print(f"  BER post:    {r.get('ber_post_cap', '?')}")
        print(f"  Monitor:     {r.get('monitor_suspicious', '?')} ({r.get('monitor_confidence', '?')})")
        print(f"  Task correct:{r.get('task_correct', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Analyse prompted steg results")
    parser.add_argument("input", help="Path to results JSONL")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument("--detail", action="store_true", help="Print per-trial detail")
    parser.add_argument("--model", type=str, default=None, help="Filter by model (for --detail)")
    parser.add_argument("--scheme", type=str, default=None, help="Filter by scheme (for --detail)")
    args = parser.parse_args()

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}\n")

    if args.detail:
        print_per_trial_detail(results, args.model, args.scheme)
        return

    summary = analyse(results)

    if args.latex:
        print_latex(summary)
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
