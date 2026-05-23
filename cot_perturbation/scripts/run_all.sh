#!/bin/bash
# Run the full CoT perturbation pipeline for one model.
#
# Prerequisites:
#   - vLLM server already running on port 8000 with the right model
#   - ANTHROPIC_API_KEY in env
#   - Inside the cot_perturbation/ directory
#
# Usage:
#   bash scripts/run_all.sh r1-distill-8b
#   bash scripts/run_all.sh qwen3-8b

set -e

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model-key>"
    echo "  model-key: r1-distill-8b or qwen3-8b"
    exit 1
fi

# Use venv python if available
if [ -d /venv/main ]; then
    PY=/venv/main/bin/python
else
    PY=python3
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: ANTHROPIC_API_KEY is not set. Steps 2, 4-7 will fail."
fi

echo "=== Step 1: Generate baselines for $MODEL ==="
$PY 01_generate_baselines.py --model "$MODEL"

echo ""
echo "=== Step 2: Generate 8 perturbed CoTs per baseline ==="
$PY 02_generate_perturbed_cots.py --model "$MODEL"

echo ""
echo "=== Step 3: Generate conditioned answers from perturbed CoTs ==="
$PY 03_generate_conditioned.py --model "$MODEL"

echo ""
echo "=== Step 4: Evaluate task accuracy (Haiku as judge) ==="
$PY 04_evaluate.py --model "$MODEL"

echo ""
echo "=== Step 5: Cross-model transfer (E_2): Haiku solves with model's CoT ==="
$PY 05_transfer.py --model "$MODEL"

echo ""
echo "=== Step 6: Response prediction (E_3): Haiku predicts model's answer ==="
$PY 06_response_prediction.py --model "$MODEL"

echo ""
echo "=== Step 7: Entanglement rubric (E_4) ==="
$PY 07_entanglement.py --model "$MODEL"

echo ""
echo "=== Final analysis ==="
$PY analyse.py --model "$MODEL"
