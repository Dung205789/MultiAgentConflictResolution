# Multi-Agent Memory Layer Benchmark with Gemma Models
# Supports both CPU and GPU inference

FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        transformers>=4.35.0 \
        accelerate>=0.24.0 \
        sentencepiece \
        safetensors \
        huggingface-hub>=0.19.0 \
        bitsandbytes

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p reports data .claude

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Accept Gemma license terms (for downloading Gemma models)
# Note: Models are only downloaded when using real transformer agents with --use-dummy flag not set.
# Default runs with dummy agents, no model downloads needed.

# Default agent models (can be overridden via environment variables)
ENV AGENT1_MODEL=Qwen/Qwen2.5-3B-Instruct
ENV AGENT2_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV AGENT1_RELIABILITY=0.85
ENV AGENT2_RELIABILITY=0.75

# Expose port for potential monitoring (optional)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command runs the unified benchmark runner
# Fast test with dummy agents (default):
#   Uses rule-based agents, no heavy model downloads
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py", "--benchmark", "memae", "--max-scenarios", "10", "--use-dummy"]
