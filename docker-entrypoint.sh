#!/bin/bash
set -e

echo "=========================================="
echo "ProjectMem Benchmark Container"
echo "=========================================="

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set."
    echo "Public models and datasets can still download without it."
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" || true
fi

echo "Checking model cache..."
AGENT1_MODEL="${AGENT1_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
AGENT2_MODEL="${AGENT2_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

check_model() {
    local model=$1
    local cache_dir="/root/.cache/huggingface/hub/models--$(echo "$model" | sed 's/\//--/g')"
    if [ -d "$cache_dir" ] && [ "$(ls -A "$cache_dir" 2>/dev/null)" ]; then
        echo "[ok] Model $model is already cached"
        return 0
    fi

    echo "[pending] Model $model will download on first transformer run"
    return 1
}

check_model "$AGENT1_MODEL" || true
check_model "$AGENT2_MODEL" || true

echo ""
echo "Preparing to run benchmark..."
echo "Agent 1 model: $AGENT1_MODEL"
echo "Agent 2 model: $AGENT2_MODEL"
echo ""

exec "$@"
