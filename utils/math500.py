"""Load MATH-500 problems from HuggingFace."""

import json
import os
import random


def load_math500(cache_dir: str = "data") -> list[dict]:
    """Load MATH-500 dataset. Downloads from HuggingFace on first use."""
    cache_file = os.path.join(cache_dir, "math500.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    print("Downloading MATH-500 from HuggingFace...")
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

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"Cached {len(problems)} problems to {cache_file}")

    return problems


def sample_problems(problems: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n problems with a fixed seed for reproducibility."""
    rng = random.Random(seed)
    return rng.sample(problems, min(n, len(problems)))


def generate_payload(rng: random.Random) -> str:
    """Generate a random 4-bit binary payload."""
    return "".join(str(rng.randint(0, 1)) for _ in range(4))
