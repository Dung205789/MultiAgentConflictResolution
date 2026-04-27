#!/bin/bash
set -e

# Docker Entrypoint Script for Multi-Agent Memory Benchmark
# This script ensures models are downloaded before running the benchmark

echo "=========================================="
echo "Multi-Agent Memory Benchmark Container"
echo "=========================================="

# Check if HF token is available
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set."
    echo "You may not be able to download gated models."
    echo "Set HF_TOKEN with: -e HF_TOKEN=your_token"
fi

# Login to Hugging Face if token is provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" || true
fi

# Check if models are already cached
echo "Checking model cache..."
AGENT1_MODEL="${AGENT1_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
AGENT2_MODEL="${AGENT2_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

check_model() {
    local model=$1
    local cache_dir="/root/.cache/huggingface/hub/models--$(echo $model | sed 's/\//--/g')"
    if [ -d "$cache_dir" ] && [ "$(ls -A $cache_dir 2>/dev/null)" ]; then
        echo "✓ Model $model is already cached"
        return 0
    else
        echo "✗ Model $model not found in cache, will download..."
        return 1
    fi
}

check_model "$AGENT1_MODEL"
check_model "$AGENT2_MODEL"

# Download models if needed (lightweight check only)
echo ""
echo "Preparing to run benchmark..."
echo "Agent 1 model: $AGENT1_MODEL"
echo "Agent 2 model: $AGENT2_MODEL"
echo ""

# Execute the command
exec "$@"
