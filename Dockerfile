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
# Users should provide their own HF token via --hf-token flag or environment variable
RUN echo "Note: Models require accepting license on Hugging Face"
RUN echo "Using Qwen models by default (publicly available)"

# Default agent models (can be overridden via environment variables)
ENV AGENT1_MODEL=Qwen/Qwen2.5-3B-Instruct
ENV AGENT2_MODEL=Qwen/Qwen2.5-7B-Instruct
ENV AGENT1_RELIABILITY=0.85
ENV AGENT2_RELIABILITY=0.75

# Pre-download models during build to avoid runtime downloads
# This requires HF_TOKEN to be passed as build arg
ARG HF_TOKEN
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "Pre-downloading models..." && \
        python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" && \
        python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('$AGENT1_MODEL'); AutoModelForCausalLM.from_pretrained('$AGENT1_MODEL', low_cpu_mem_usage=True)" && \
        python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('$AGENT2_MODEL'); AutoModelForCausalLM.from_pretrained('$AGENT2_MODEL', low_cpu_mem_usage=True)" && \
        echo "Models downloaded successfully!"; \
    else \
        echo "HF_TOKEN not provided during build, models will be downloaded at runtime."; \
    fi

# Expose port for potential monitoring (optional)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command runs the agent comparison benchmark
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run_agent_memory_comparison.py", "--num-scenarios", "10"]
