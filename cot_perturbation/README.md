# CoT Perturbation Experiments

This subdirectory replicates and extends the Janus-style chain-of-thought analysis experiments (E_1 through E_4) on two reasoning models, R1-Distill-Llama-8B and Qwen3-8B (thinking mode). It runs on a single GPU via vLLM.

## What this measures

Reasoning models like R1-Distill and Qwen3-thinking produce an explicit chain-of-thought (CoT) inside `<think>...</think>` tags before their final answer. These experiments test whether the model uses the CoT for its readable content alone, or whether the CoT also carries hidden information in its structure or style. The methodology is to take a model's natural CoT, perturb it in various ways, then pre-fill the perturbed CoT back into the model and let it generate a final answer conditioned on that perturbed reasoning. If the model's accuracy stays roughly constant under structural perturbations and drops only when the readable content is corrupted, then the readable content is doing all the work.

## The eight perturbations

The perturbations span three categories. Incoherent perturbations destroy both structure and content: null replaces the CoT with an empty string, nonsemantic replaces it with filler words matching the original word count, scramble shuffles the words within each sentence, and style is a full paraphrase via Haiku that preserves the mathematical content but uses completely different vocabulary and sentence structure. Coherent-but-wrong perturbations preserve the surface style but corrupt the reasoning: content_perturb has Haiku inject subtle errors like sign flips or swapped quantities so the trace looks plausible but leads to a wrong answer, and content_replace has Haiku write a trace about a completely different math problem while matching the original style and length. Length perturbations modify CoT length: truncate cuts the CoT to roughly half its word count ending at a sentence boundary, and extend roughly doubles it by appending filler phrases.

## The four experiments

E_1 (perturbation effect on task accuracy) is the main experiment. For each of N=50 MATH-500 problems, we generate the model's natural CoT, apply all eight perturbations to produce eight modified versions, pre-fill each modified CoT back into the model, and let the model generate a final answer conditioned on it. Claude Haiku then judges whether each final answer matches the ground truth. We compare accuracy across conditions to baseline accuracy.

E_2 (cross-model transfer) asks whether the CoT carries information that Haiku can use. Haiku receives the problem plus the original CoT (or a perturbed version) and tries to solve the problem itself. If Haiku's accuracy varies across perturbation conditions, the perturbations destroyed information Haiku was relying on.

E_3 (response prediction) tests whether the CoT determines the model's answer. Haiku sees several few-shot examples of (CoT, the model's actual answer) pairs and then must predict the model's answer for a new CoT. High match rates mean the CoT determines the answer.

E_4 (entanglement analysis) directly judges how tightly the model's answers follow from its CoTs. Haiku scores each baseline (CoT, answer) pair on a 1-to-5 rubric and flags contradictions (answer contradicts CoT) and ignored conclusions (CoT reaches a conclusion the answer does not use).

## Sample sizes and statistical power

Each model gets N=50 baseline problems and all 8 perturbations applied to each, giving 50 task-accuracy observations per condition. With N=50 the 95% Wilson confidence interval on a proportion is about plus or minus 14 percentage points, which is enough to distinguish "no effect" from "10-15 percentage point effect" reliably. The original Janus paper used N=30 in places, which gives about plus or minus 18 points, so this is a meaningful improvement. The analyse.py script reports Wilson confidence intervals for every reported rate.

## Files

`configs.py` holds model identifiers, perturbation definitions, and sample sizes. `perturbations.py` implements all eight perturbations. `utils/vllm_client.py` wraps the local vLLM server (used for both baseline generation and pre-filled conditioned generation). `utils/haiku_client.py` wraps the Anthropic API for paraphrasing, transfer, prediction, and judging. `utils/io.py` has JSONL helpers and MATH-500 loading.

The pipeline scripts are numbered: `01_generate_baselines.py` generates the natural CoT and final answer for each problem, `02_generate_perturbed_cots.py` applies all 8 perturbations, `03_generate_conditioned.py` pre-fills each perturbed CoT and generates an answer, `04_evaluate.py` judges accuracy via Haiku, `05_transfer.py` runs the E_2 cross-model transfer, `06_response_prediction.py` runs E_3, and `07_entanglement.py` runs E_4. `analyse.py` aggregates everything into a final summary table with confidence intervals.

All scripts append JSONL records and support resume by skipping records already present in the output file. If anything fails partway through, just rerun the same command and it will pick up where it left off.

## How to run

You need a single GPU with at least 24GB of VRAM. An H200 NVL, an A100 80GB, or even an A40 48GB all work. Both models fit comfortably with their KV caches.

On a fresh Vast.ai instance (60GB+ container disk recommended), run the setup script with your GitHub token and Anthropic API key. This installs vLLM and the Anthropic SDK, clones the repo, downloads MATH-500, and writes the environment file.

```bash
cd /workspace
curl -O https://raw.githubusercontent.com/WFJKK/blackbox-experiments-in-steganography-and-encoded-reasoning/main/cot_perturbation/scripts/setup_vast.sh
bash setup_vast.sh "$GITHUB_TOKEN" "$ANTHROPIC_API_KEY"
```

Then for each model, start a vLLM server in the background and run the pipeline. R1-Distill-Llama-8B and Qwen3-8B are roughly 16GB each so you destroy one server before starting the next.

```bash
source /workspace/.cot_env
cd /workspace/blackbox-experiments-in-steganography-and-encoded-reasoning/cot_perturbation

# R1-Distill-8B
$VLLM serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8000 --max-model-len 8192 > /workspace/vllm.log 2>&1 &
sleep 120  # wait for model to load
curl http://localhost:8000/v1/models  # verify
bash scripts/run_all.sh r1-distill-8b

# Stop vLLM, switch model
pkill -f "vllm serve"
sleep 10
$VLLM serve Qwen/Qwen3-8B --port 8000 --max-model-len 8192 > /workspace/vllm.log 2>&1 &
sleep 120
bash scripts/run_all.sh qwen3-8b
```

When everything is done, run the analysis to see the summary:

```bash
$PY analyse.py --model r1-distill-8b
$PY analyse.py --model qwen3-8b
```

Then push the results back to GitHub before destroying the instance:

```bash
cd /workspace/blackbox-experiments-in-steganography-and-encoded-reasoning
git add cot_perturbation/results/
git commit -m "Add CoT perturbation results for R1-Distill-8B and Qwen3-8B"
git push
```

## Cost estimate

A single H200 NVL on Vast.ai runs about $2-3 per hour. The full pipeline for one model takes roughly 30-45 minutes of GPU wall time (a few minutes for baselines, a few minutes for conditioned generations, the rest is Haiku calls that don't use the GPU). Doing both models is therefore 1 to 1.5 hours of GPU plus maybe 10 minutes of setup. Total GPU cost roughly $3 to $5. Haiku API calls add maybe $2 to $4 in total. Whole experiment under $10.

## Expected output

For each model, the final analysis prints a table like the one in the existing draft paper, with accuracy and confidence intervals per condition. The expected pattern from the original R1-Distill experiments is that incoherent perturbations leave accuracy roughly unchanged (model ignores the nonsense CoT), coherent-but-wrong perturbations drop accuracy substantially (model trusts a plausible-looking but wrong trace), and length perturbations have a small effect. If both R1-Distill-8B and Qwen3-8B show this pattern, that is strong evidence that reasoning models do not encode hidden signals in CoT structure. If Qwen3 shows a different pattern, that is also an interesting result worth investigating.
