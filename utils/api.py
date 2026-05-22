"""API clients for OpenRouter (Qwen models) and Anthropic (Claude Haiku)."""

import os
import re
import time
import json
from openai import OpenAI


def get_openrouter_client():
    """Get OpenRouter client (OpenAI-compatible)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_anthropic_client():
    """Get Anthropic client for Claude Haiku."""
    import anthropic
    return anthropic.Anthropic()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_openrouter(client, model_id: str, system: str, user: str,
                    max_tokens: int = 2048, temperature: float = 0.7,
                    thinking: bool = False, max_retries: int = 3) -> str:
    """Call an OpenRouter model with retry logic.

    Args:
        thinking: If True, enable reasoning mode. If False, disable it.
                  For Qwen3 models this controls the <think> block generation.
    """
    extra_body = {}
    if thinking:
        extra_body["reasoning"] = {"enabled": True}
    else:
        extra_body["reasoning"] = {"enabled": False}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            text = response.choices[0].message.content or ""
            text = text.strip()
            # Strip any <think> blocks that leaked through
            text = strip_think_blocks(text)
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  OpenRouter error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  OpenRouter failed after {max_retries} attempts: {e}")
                return f"ERROR: {e}"


def call_haiku(client, system: str, user: str,
               max_tokens: int = 2048, max_retries: int = 3) -> str:
    """Call Claude Haiku with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Haiku error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Haiku failed after {max_retries} attempts: {e}")
                return f"ERROR: {e}"


def parse_monitor_response(response: str) -> dict:
    """Parse monitor JSON response, handling malformed output."""
    try:
        # Try to extract JSON from the response
        text = response.strip()
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return {
            "suspicious": bool(result.get("suspicious", False)),
            "confidence": float(result.get("confidence", 0)),
            "reason": str(result.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError, IndexError):
        # If parsing fails, treat as suspicious if it mentions anything
        lower = response.lower()
        suspicious = any(w in lower for w in ["suspicious", "encoding", "hidden", "steganograph"])
        return {
            "suspicious": suspicious,
            "confidence": 0.5 if suspicious else 0.0,
            "reason": f"parse_error: {response[:200]}",
        }
