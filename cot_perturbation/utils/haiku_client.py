"""Anthropic Haiku client for paraphrasing, transfer experiments, and evaluation."""

import time

import anthropic


def get_haiku_client() -> anthropic.Anthropic:
    """Get Anthropic client (uses ANTHROPIC_API_KEY env var)."""
    return anthropic.Anthropic()


def call_haiku(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    max_tokens: int = 2048,
    max_retries: int = 4,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Call Claude Haiku with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"  Haiku error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Haiku failed after {max_retries} attempts: {e}")
                return f"ERROR: {e}"
