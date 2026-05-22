#!/usr/bin/env python3
"""Download the full MATH-500 dataset from HuggingFace.

Run this once before the full experiment:
    python download_math500.py
"""

import json
import os

def main():
    cache_file = "data/math500.json"

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            existing = json.load(f)
        if len(existing) >= 500:
            print(f"Already have {len(existing)} problems in {cache_file}. Nothing to do.")
            return
        print(f"Only {len(existing)} problems cached. Downloading full dataset...")

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    problems = []
    for row in ds:
        problems.append({
            "problem": row["problem"],
            "solution": row.get("solution", ""),
            "answer": row["answer"],
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
            "id": row.get("unique_id", ""),
        })

    os.makedirs("data", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Done: {len(problems)} problems saved to {cache_file}")


if __name__ == "__main__":
    main()
