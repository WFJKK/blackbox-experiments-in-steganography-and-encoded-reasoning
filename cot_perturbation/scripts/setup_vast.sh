#!/bin/bash
# Setup script for Vast.ai GPU instances running the CoT perturbation experiments.
#
# Run this once on a fresh Vast.ai instance:
#   bash setup_vast.sh GITHUB_TOKEN ANTHROPIC_API_KEY
#
# Recommended Vast.ai instance: H200 NVL or A100 80GB, 60GB+ container disk,
# CUDA 12.x base image with PyTorch preinstalled.

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 GITHUB_TOKEN ANTHROPIC_API_KEY"
    exit 1
fi

GITHUB_TOKEN="$1"
ANTHROPIC_API_KEY="$2"

echo "=== Setting up CoT perturbation environment ==="

# Use the venv path if it exists (Vast.ai standard), else system python
if [ -d /venv/main ]; then
    PY=/venv/main/bin/python
    PIP=/venv/main/bin/pip
    VLLM=/venv/main/bin/vllm
    echo "Using venv at /venv/main"
else
    PY=python3
    PIP=pip3
    VLLM=vllm
fi

echo "=== Installing dependencies ==="
$PIP install --upgrade pip
$PIP install vllm
$PIP install anthropic openai datasets

echo "=== Cloning repository ==="
cd /workspace
if [ -d blackbox-experiments-in-steganography-and-encoded-reasoning ]; then
    cd blackbox-experiments-in-steganography-and-encoded-reasoning
    git pull
else
    git clone "https://WFJKK:${GITHUB_TOKEN}@github.com/WFJKK/blackbox-experiments-in-steganography-and-encoded-reasoning.git"
    cd blackbox-experiments-in-steganography-and-encoded-reasoning
fi

echo "=== Downloading MATH-500 ==="
$PY download_math500.py || true

echo "=== Configuring environment ==="
# Write keys to env file for future shells
cat > /workspace/.cot_env << EOF
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export PY=${PY}
export VLLM=${VLLM}
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start a vLLM server for R1-Distill-8B:"
echo "  source /workspace/.cot_env"
echo "  \$VLLM serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8000 --max-model-len 8192 > /workspace/vllm.log 2>&1 &"
echo ""
echo "To start a vLLM server for Qwen3-8B:"
echo "  source /workspace/.cot_env"
echo "  \$VLLM serve Qwen/Qwen3-8B --port 8000 --max-model-len 8192 > /workspace/vllm.log 2>&1 &"
echo ""
echo "Wait ~2 minutes for the model to load, then check with:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "Then run the pipeline:"
echo "  cd /workspace/blackbox-experiments-in-steganography-and-encoded-reasoning/cot_perturbation"
echo "  bash scripts/run_all.sh r1-distill-8b  # or qwen3-8b"
