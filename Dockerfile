FROM python:3.10-slim

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

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p reports data/raw data/processed dist

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV AGENT1_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV AGENT2_MODEL=Qwen/Qwen2.5-1.5B-Instruct
ENV AGENT1_RELIABILITY=0.85
ENV AGENT2_RELIABILITY=0.75

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "main.py", "--benchmark", "mab_conflict", "--max-scenarios", "4", "--use-dummy"]
