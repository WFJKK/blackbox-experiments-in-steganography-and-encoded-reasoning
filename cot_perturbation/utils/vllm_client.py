"""vLLM client for baseline generation and pre-filled conditioned generation.

Expects a vLLM OpenAI-compatible server running locally (default port 8000).
Start it with e.g.:
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8000
    vllm serve Qwen/Qwen3-8B --port 8000
"""

import os
import time
import requests
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
    max_tokens: int = 8192,
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
                wait = 2**attempt
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
    prompt_template: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    port: int = 8000,
    max_retries: int = 3,
) -> str:
    """Generate a response conditioned on a pre-filled (perturbed) CoT.

    Uses the raw /v1/completions endpoint with a hand-built prompt that
    includes the chat template tokens + the pre-filled CoT. This avoids
    the continue_final_message incompatibility with some chat templates.

    Args:
        prompt_template: a format string with {problem} and {cot} placeholders,
            containing the model's chat template tokens. Defined in configs.py.

    Returns just the newly generated text (the final answer after </think>).
    """
    # Build the prompt: user turn + assistant turn pre-filled up to after </think>
    if perturbed_cot.strip() == "":
        # Null perturbation: empty think block
        cot_content = ""
    else:
        cot_content = perturbed_cot

    prompt = prompt_template.format(problem=problem, cot=cot_content)

    for attempt in range(max_retries):
        try:
            # Use requests directly for /v1/completions (text completion, not chat)
            resp = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": [
                        "<|eot_id|>",
                        "<|im_end|>",
                    ],  # stop at end-of-turn for either model
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["text"]
            return strip_bpe_artifacts(text).strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"  vLLM completions error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  vLLM completions failed after {max_retries} attempts: {e}")
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
