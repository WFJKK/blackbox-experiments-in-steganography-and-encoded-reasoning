"""Common helpers: JSONL I/O, trial assignment, MATH-500 loading."""

import json
import os
import random
import sys


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file. Returns an empty list if the file doesn't exist."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: str, record: dict):
    """Append one JSON record as a line to the file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_math500() -> list[dict]:
    """Load MATH-500. Looks in ../data/ first, then current dir."""
    candidates = [
        "../data/math500.json",
        "data/math500.json",
        os.path.join(os.path.dirname(__file__), "..", "data", "math500.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    print("ERROR: MATH-500 not found. Run download_math500.py in the parent directory.", file=sys.stderr)
    sys.exit(1)


def sample_problems(n: int, seed: int = 42) -> list[dict]:
    """Sample n MATH-500 problems with a fixed seed for reproducibility across all scripts."""
    problems = load_math500()
    rng = random.Random(seed)
    sampled = rng.sample(problems, min(n, len(problems)))
    # Tag each with a stable trial_id matching its index in the sampled list
    for i, p in enumerate(sampled):
        p["trial_id"] = i
    return sampled
