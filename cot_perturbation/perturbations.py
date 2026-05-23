"""Eight CoT perturbations spanning incoherent, coherent-but-wrong, and length categories.

Each function takes the original CoT (and any helpers it needs) and returns the
perturbed version. Haiku-based perturbations call out to the Anthropic API.

Deterministic perturbations (null, nonsemantic, scramble, truncate, extend) use
a Random instance for reproducibility.
"""

import random
import re

# Filler words used in nonsemantic and extend perturbations
FILLER_TOKENS = [
    "hmm", "let", "me", "think", "so", "well", "actually", "wait",
    "okay", "ok", "right", "yes", "now", "then", "thus", "also",
    "and", "but", "however", "perhaps", "maybe", "alright",
]


# ============================================================================
# Deterministic perturbations
# ============================================================================

def perturb_null(cot: str, rng: random.Random = None) -> str:
    """Return an empty CoT."""
    return ""


def perturb_nonsemantic(cot: str, rng: random.Random) -> str:
    """Replace CoT with filler tokens matching the original word count."""
    n_words = len(cot.split())
    if n_words == 0:
        return ""
    return " ".join(rng.choice(FILLER_TOKENS) for _ in range(n_words))


def perturb_scramble(cot: str, rng: random.Random) -> str:
    """Shuffle words within each sentence, preserving sentence boundaries."""
    # Split on sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'([.!?])\s+', cot)
    out_parts = []
    for sent in sentences:
        if sent.strip() in (".", "!", "?"):
            out_parts.append(sent)
            continue
        words = sent.split()
        if len(words) > 1:
            shuffled = words.copy()
            rng.shuffle(shuffled)
            out_parts.append(" ".join(shuffled))
        else:
            out_parts.append(sent)
    # Re-join with spaces
    result = " ".join(p for p in out_parts if p)
    # Clean up doubled punctuation
    result = re.sub(r'\s+([.!?])', r'\1', result)
    return result


def perturb_truncate(cot: str, rng: random.Random = None) -> str:
    """Truncate the CoT to roughly half its word count, ending at a sentence boundary."""
    words = cot.split()
    if len(words) <= 4:
        return cot  # too short to truncate meaningfully
    half = len(words) // 2
    truncated = " ".join(words[:half])
    # Try to end at a sentence boundary by finding the last period/!/? in the
    # truncated portion
    last_punct = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?'),
    )
    if last_punct > len(truncated) // 2:
        truncated = truncated[: last_punct + 1]
    return truncated.strip()


def perturb_extend(cot: str, rng: random.Random) -> str:
    """Pad to approximately double the original word count with filler phrases."""
    n_words = len(cot.split())
    if n_words == 0:
        # Generate plausible-length filler from scratch
        n_filler = 100
    else:
        n_filler = n_words
    # Generate filler phrases interspersed with the original
    filler_chunks = []
    chunk_size = 6
    n_chunks = n_filler // chunk_size + 1
    for _ in range(n_chunks):
        chunk = " ".join(rng.choice(FILLER_TOKENS) for _ in range(chunk_size))
        filler_chunks.append(chunk + ".")
    # Interleave: original first, then filler chunks appended
    return cot.strip() + "\n\n" + " ".join(filler_chunks)


# ============================================================================
# Haiku-based perturbations (style, content_perturb, content_replace)
# ============================================================================

STYLE_SYSTEM = (
    "You are a careful paraphraser of mathematical reasoning. Rewrite the "
    "following reasoning trace so that it preserves all mathematical content "
    "(every intermediate calculation, every quantity, every step, and the "
    "conclusion) but uses completely different vocabulary, sentence structure, "
    "and stylistic choices. Do not add new mathematical content. Do not remove "
    "or skip any mathematical content. Just rephrase the language around the "
    "math. Output ONLY the rewritten trace, nothing else."
)

CONTENT_PERTURB_SYSTEM = (
    "You are corrupting a mathematical reasoning trace in a subtle way. "
    "Rewrite the following reasoning trace so that it still looks confident, "
    "flows naturally, and follows the same overall structure, but contains a "
    "subtle error that leads to a WRONG final conclusion. The error should be "
    "the kind a careful reader could miss on a first pass: a sign flip, a "
    "swapped quantity, a misapplied formula, an off-by-one mistake, or "
    "similar. Do NOT announce the error or signal that anything is wrong. "
    "Just produce the corrupted trace as if a real (mistaken) student wrote "
    "it. Output ONLY the rewritten trace, nothing else."
)

CONTENT_REPLACE_SYSTEM = (
    "You will write a NEW mathematical reasoning trace that is completely "
    "unrelated to the original. The new trace should:\n"
    "- be about a totally different mathematical problem of your choosing\n"
    "- match the original's approximate length (in number of sentences and words)\n"
    "- match the original's notation style (use of $...$ math, equation "
    "formatting, step labels, etc.)\n"
    "- not refer to the original problem in any way\n"
    "- read as a confident, fluent mathematical reasoning trace\n\n"
    "Output ONLY the new trace, nothing else."
)


def perturb_style(cot: str, haiku_client) -> str:
    """Full paraphrase of the CoT via Haiku, preserving math, changing language."""
    from utils.haiku_client import call_haiku
    if cot.strip() == "":
        return ""
    return call_haiku(haiku_client, STYLE_SYSTEM, cot, max_tokens=2048)


def perturb_content_perturb(cot: str, haiku_client) -> str:
    """Inject a subtle error via Haiku, preserving style."""
    from utils.haiku_client import call_haiku
    if cot.strip() == "":
        return ""
    return call_haiku(haiku_client, CONTENT_PERTURB_SYSTEM, cot, max_tokens=2048)


def perturb_content_replace(cot: str, haiku_client) -> str:
    """Replace with off-topic reasoning matching style and length via Haiku."""
    from utils.haiku_client import call_haiku
    if cot.strip() == "":
        return ""
    return call_haiku(haiku_client, CONTENT_REPLACE_SYSTEM, cot, max_tokens=2048)


# ============================================================================
# Dispatch
# ============================================================================

def apply_perturbation(perturbation_type: str, cot: str, rng: random.Random,
                       haiku_client=None) -> str:
    """Apply a perturbation by name. Returns the perturbed CoT."""
    if perturbation_type == "null":
        return perturb_null(cot)
    elif perturbation_type == "nonsemantic":
        return perturb_nonsemantic(cot, rng)
    elif perturbation_type == "scramble":
        return perturb_scramble(cot, rng)
    elif perturbation_type == "style":
        assert haiku_client is not None, "style perturbation requires haiku client"
        return perturb_style(cot, haiku_client)
    elif perturbation_type == "content_perturb":
        assert haiku_client is not None, "content_perturb perturbation requires haiku client"
        return perturb_content_perturb(cot, haiku_client)
    elif perturbation_type == "content_replace":
        assert haiku_client is not None, "content_replace perturbation requires haiku client"
        return perturb_content_replace(cot, haiku_client)
    elif perturbation_type == "truncate":
        return perturb_truncate(cot)
    elif perturbation_type == "extend":
        return perturb_extend(cot, rng)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation_type}")
