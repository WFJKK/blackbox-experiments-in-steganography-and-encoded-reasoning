"""vLLM client for baseline generation and pre-filled conditioned generation.

Expects a vLLM OpenAI-compatible server running locally (default port 8000).
Start it with e.g.:
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8000
    vllm serve Qwen/Qwen3-8B --port 8000
"""

import os
import time
from openai import OpenAI


def get_vllm_client(port: int = 8000) -> OpenAI:
    """Get a client pointing at the local vLLM server."""
    return OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",  # vLLM server doesn't check
    )


def get_served_model(client: OpenAI) -> str:
    """Ask the vLLM server which model it's serving."""
    models = client.models.list()
    return models.data[0].id


def generate_baseline(
    client: OpenAI,
    model_name: str,
    problem: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Generate a baseline response (including the CoT) for one problem.

    Returns the raw assistant text containing <think>...</think> + final answer,
    with BPE artifacts decoded.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": problem}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return strip_bpe_artifacts(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  vLLM error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  vLLM failed after {max_retries} attempts: {e}")
                return f"ERROR: {e}"


def generate_conditioned(
    client: OpenAI,
    model_name: str,
    problem: str,
    perturbed_cot: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Generate a response conditioned on a pre-filled (perturbed) CoT.

    Uses vLLM's continue_final_message to inject the partial assistant turn
    containing the perturbed think block, then lets the model continue.

    Returns just the newly generated text (the final answer, with no <think>).
    """
    # If the perturbed CoT is empty (null perturbation), use an empty think block
    if perturbed_cot.strip() == "":
        partial = "<think>\n\n</think>\n\n"
    else:
        partial = f"<think>\n{perturbed_cot}\n</think>\n\n"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": partial},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                },
            )
            return strip_bpe_artifacts(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  vLLM error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  vLLM failed after {max_retries} attempts: {e}")
                return f"ERROR: {e}"


def strip_bpe_artifacts(text: str) -> str:
    """Some vLLM versions return raw BPE tokens with Ġ (space) and Ċ (newline)
    instead of properly decoded text. Decode these markers."""
    if not text:
        return text
    # Only convert if we see the BPE markers - don't touch normal text
    if "Ġ" not in text and "Ċ" not in text:
        return text
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def parse_cot_and_answer(response_text: str) -> tuple[str, str]:
    """Split a baseline response into (cot, answer).

    Looks for <think>...</think> tags. If not found, returns empty CoT and
    the full text as the answer.
    """
    text = strip_bpe_artifacts(response_text).strip()
    # Handle case where model emits </think> without opening <think>
    if "</think>" in text:
        if "<think>" in text:
            # Standard case
            before, rest = text.split("<think>", 1)
            if "</think>" in rest:
                cot, after = rest.split("</think>", 1)
                return cot.strip(), after.strip()
            return rest.strip(), ""
        else:
            # Closing tag only (some chat templates auto-prepend <think>)
            cot, after = text.split("</think>", 1)
            return cot.strip(), after.strip()
    return "", text
