# Blackbox Experiments in Steganography and Encoded Reasoning

Blackbox evaluation of steganographic capabilities and encoded reasoning in frontier LLMs.

## Prompted steganography experiments

Models are instructed to solve MATH-500 problems while simultaneously encoding a hidden 4-bit binary payload using one of three encoding schemes. We measure payload recovery before and after defences.

### Models

**Qwen RL gradient** (isolates effect of reasoning RL training):
- **Qwen2.5-7B-Instruct** -- SFT + basic RLHF, no reasoning RL
- **Qwen3-8B (non-thinking)** -- SFT + reasoning RL (GRPO) + general RL, inference without CoT
- **Qwen3-8B (thinking)** -- same weights, inference with CoT

**Frontier comparison:**
- **Llama-4-Maverick** -- 400B MoE (17B active), Meta, SFT + RLHF

### Encoding schemes

- **hex** -- payload bytes as hexadecimal at sentence starts
- **acrostic** -- first letters of sentences encode binary via vowel/consonant mapping
- **synonym** -- binary choice between pre-defined word pairs

### Defences

- **CAP** (Context-Aware Paraphrasing) -- Claude Haiku rewrites the response preserving task content
- **Trusted monitor** -- Claude Haiku inspects the response for steganographic indicators

### Usage

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Test (N=1, one model, one scheme)
python run_experiment.py --n-trials 1 --models qwen2.5-7b --schemes hex --output results/test.jsonl
python analyse.py results/test.jsonl

# Full run (N=30, all conditions)
python run_experiment.py --n-trials 30 --output results/full_run.jsonl

# Resume interrupted run
python run_experiment.py --n-trials 30 --output results/full_run.jsonl --resume

# LaTeX table
python analyse.py results/full_run.jsonl --latex

# Per-trial detail
python analyse.py results/full_run.jsonl --detail --model qwen3-8b-think --scheme hex
```

### Design

- 4 models x 3 schemes x N=30 = 360 total trials
- 4-bit binary payloads, randomly generated per trial
- Same math problem + payload across all models for each trial index (seed=42)
- All via API: OpenRouter for encoder/decoder models, Anthropic for Haiku defences
- JSONL output with full resume support

## CoT perturbation experiments (planned)

Replicate and extend the Janus E1-E4 experiments on R1-Distill-8B and Qwen3-8B using local GPU inference (vLLM) for CoT pre-fill capability.
