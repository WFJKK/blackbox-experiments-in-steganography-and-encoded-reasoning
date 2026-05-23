"""Configuration for CoT perturbation experiments.

Models tested: two ~8B reasoning models accessed via vLLM running locally on GPU.
Sample sizes chosen for statistical power: detecting +/-10% accuracy changes with
reasonable confidence requires N>=50 per condition.
"""

MODELS = {
    "r1-distill-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "name": "R1-Distill-Llama-8B",
        "max_tokens": 8192,
        "temperature": 0.7,
        # Llama-3.1 chat template for raw /v1/completions pre-fill
        "prompt_template": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "<think>\n{cot}\n</think>\n\n"
        ),
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "name": "Qwen3-8B",
        "max_tokens": 8192,
        "temperature": 0.7,
        # Qwen/ChatML template for raw /v1/completions pre-fill
        "prompt_template": (
            "<|im_start|>user\n"
            "{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n{cot}\n</think>\n\n"
        ),
    },
}


# The eight perturbations, grouped by category.
# Each maps to a function in perturbations.py
PERTURBATIONS = [
    # Incoherent: destroy both content and structure
    ("null", "incoherent", "Empty CoT"),
    ("nonsemantic", "incoherent", "Filler words matching original length"),
    ("scramble", "incoherent", "Words shuffled within each sentence"),
    # Coherent correct: preserve content, change surface form
    ("style", "coherent_correct", "Full paraphrase with different vocabulary"),
    # Coherent but wrong: preserve surface, corrupt reasoning
    ("content_perturb", "coherent_wrong", "Subtle errors injected"),
    ("content_replace", "coherent_wrong", "Off-topic reasoning, matching style"),
    # Length modifications
    ("truncate", "length", "Half-length truncation"),
    ("extend", "length", "Double-length with filler padding"),
]


# Sample sizes
# N_BASELINES: number of problems per model. Each gets all 8 perturbations applied.
# Total conditioned generations per model = N_BASELINES * 8
# Total per model + baseline eval = N_BASELINES * 9
N_BASELINES = 50  # Gives +/-14% 95% CI on baseline accuracy
N_TRANSFER_PER_COND = 50  # Cross-model transfer (E_2): same N as conditioned
N_RESPONSE_PREDICTION_RUNS = 5  # E_3.1: independent runs with different few-shot
N_RESPONSE_PREDICTION_PER_RUN = 30  # Predictions per run
N_ENTANGLEMENT = 50  # E_4.1: rubric scoring of baseline CoT-answer pairs


# Path conventions
def baseline_path(model_key: str) -> str:
    return f"results/{model_key}_baselines.jsonl"


def perturbed_cot_path(model_key: str) -> str:
    return f"results/{model_key}_perturbed_cots.jsonl"


def conditioned_path(model_key: str) -> str:
    return f"results/{model_key}_conditioned.jsonl"


def evaluation_path(model_key: str) -> str:
    return f"results/{model_key}_evaluations.jsonl"


def transfer_path(model_key: str) -> str:
    return f"results/{model_key}_transfer.jsonl"


def response_prediction_path(model_key: str) -> str:
    return f"results/{model_key}_response_prediction.jsonl"


def entanglement_path(model_key: str) -> str:
    return f"results/{model_key}_entanglement.jsonl"
