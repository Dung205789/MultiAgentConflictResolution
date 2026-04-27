# Multi-Agent Shared Memory with Conflict Resolution

This project implements a multi-agent shared memory system with advanced conflict resolution capabilities. The system enables multiple agents to read from and write to a shared memory store while handling conflicts through principled arbitration mechanisms.

## Core Features

- **Shared Memory Store**: Structured memory entries with metadata for version tracking, conflict detection, and visibility control
- **Conflict-Aware Writing**: Sophisticated conflict detection and resolution with multiple resolution strategies:
  - `overwrite`: Supersede latest active candidate based on confidence/provenance/recency
  - `merge`: Combine information from conflicting entries
  - `keep_multiple_versions`: Maintain concurrent active branches with explicit metadata
  - `reject`: Decline to commit conflicting information
  - `defer`: Mark as tentative pending further review
- **Staleness Detection**: Identify and handle stale-read scenarios
- **Model-Backed Extraction**: Extract structured information from text using models or rules
- **Benchmark-Driven Development**: Comprehensive evaluation framework comparing conflict-aware approach against baselines

## Quick Start

### Run Benchmark with Dummy Agents (Fast)

```bash
# Test with 20 scenarios (no heavy models)
python run_agent_memory_comparison.py --use-dummy --num-scenarios 20
```

### Run with Real Models (Qwen 3B & 7B)

```bash
# Set your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Run with lightweight Qwen models (publicly available)
python run_agent_memory_comparison.py \
  --num-scenarios 35 \
  --agent1-model Qwen/Qwen2.5-3B-Instruct \
  --agent2-model Qwen/Qwen2.5-7B-Instruct
```

### Docker Usage

See [Docker Usage Guide](docs/DOCKER_USAGE.md) for complete instructions.

Quick example:

```bash
# Set your HF token
export HF_TOKEN="your_token"

# Build and run with dummy agents (fast)
docker compose up

# Run with real Qwen models
docker compose run --rm memory-benchmark \
  python run_agent_memory_comparison.py \
  --num-scenarios 35 \
  --agent1-model Qwen/Qwen2.5-3B-Instruct \
  --agent2-model Qwen/Qwen2.5-7B-Instruct
```

## System Architecture

              ┌─────────────────┐
              │  Agent Runtime  │
              └────────┬────────┘
                       │
                       ▼
┌──────────────┐    ┌─────────────┐    ┌───────────────────┐
│  Extractor   │◄───┤ Multi-Agent │───►│ Staleness Detector│
│ (Model/Rule) │    │  Pipeline   │    └───────────────────┘
└──────────────┘    └──────┬──────┘
                       │
                       ▼
┌──────────────┐    ┌─────────────────────┐    ┌───────────────┐
│  Retriever   │◄───┤ Shared Memory Store │───►│    Writers    │
│(Embedding/KW)│    │  (Version-Aware)    │    │(Conflict-Aware│
└──────────────┘    └─────────────────────┘    │   /LWW/Naive) │
                                               └───────────────┘

## Modes of Operation

- **Conflict-Aware**: Full conflict detection, arbitration, and resolution
- **Last-Write-Wins (LWW)**: Simple overwrite policy based on recency
- **Naive Append**: No conflict handling, all writes active

## Model Integration

The system supports two operational modes:

- **research_strict**: Required models must load; fail fast if unavailable
- **debug_fallback**: Allows fallback behavior for local debugging

## Evaluation

The system is evaluated using a synthetic benchmark that tests different conflict scenarios:

- Concurrent updates
- Stale reads
- Direct contradictions
- Semantic overlaps

Metrics include:
- Conflict detection precision/recall/F1
- End-to-end scenario accuracy
- Per-conflict-type metrics
- Action quality metrics

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run memory layer comparison (recommended)
python run_agent_memory_comparison.py --use-dummy --num-scenarios 20
```

## Repository Structure

- `src/`: Core implementation
  - `memory/shared_memory_store.py`: Structured shared-memory entries with metadata
  - `conflict/conflict_aware_writer.py`: Conflict detection + arbitration + write effects
  - `conflict/staleness_detector.py`: Detects stale-read risk
  - `conflict/baselines.py`: LWW and naive baseline writers
  - `conflict/conflict_detector.py`: Conflict type detection
  - `agents/agent_runtime.py`: Agent-level read/retrieve/propose abstraction
  - `pipeline/multi_agent_pipeline.py`: Orchestrates scenario execution
  - `utils/retriever.py`, `utils/extractor.py`: Retrieval and extraction utilities
  - `local_models/runner.py`: Local LLM agent implementations (dummy & transformer)
- `data/`: Benchmark scenarios
  - `enhanced_multi_agent_benchmark.jsonl`: Primary benchmark (35 scenarios)
  - `multi_agent_benchmark.jsonl`: Original benchmark (20 scenarios)
- `reports/`: Generated evaluation reports (auto-created)
- `docs/`: Documentation (BENCHMARK_RESULTS.md, BENCHMARK_SUMMARY.md)

## Expected Results

Running with dummy agents on enhanced benchmark (35 scenarios):

```
Scenario Accuracy:  17.14% (naive) → 80.00% (memory-aware) [+62.86%]
Conflict F1:        0.0000 → 1.0000 [+1.0000]
Action Accuracy:    2.86% → 44.29% [+41.43%]
Avg Branch Count:   2.000 → 1.229 [-0.771]
```

With real Gwen models, expect:
- Action accuracy: 40-60%
- Scenario accuracy: 70-90%
- Better conflict resolution

## Documentation

- [BENCHMARK_SUMMARY.md](BENCHMARK_SUMMARY.md) - Vietnamese summary & results
- [BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) - Detailed benchmark results
- [Docker Usage](docs/DOCKER_USAGE.md) - Docker guide (if available)

## Legacy Code

The original single-agent memory pipeline has been archived in `src/legacy/` and is no longer maintained. The project now focuses exclusively on the multi-agent shared memory approach.
