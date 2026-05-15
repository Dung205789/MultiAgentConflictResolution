# Multi-Agent Shared Memory with Rule-Based Conflict Resolution

**Research project: Rule-based multi-agent memory conflict resolution under constraints.**

This project implements a cost-effective multi-agent shared memory system that resolves conflicts using rule-based arbitration instead of LLMs. The current focus is benchmark-aligned research on accepted external datasets, especially `MemoryAgentBench / Conflict_Resolution`, rather than synthetic headline metrics.

## Key Features

- **Rule-Based Arbitration**: No LLM dependency in the critical path - all conflict resolution uses deterministic rules with configurable weights
- **Unified Internal Standard Format (ISF)**: Single format for all benchmark datasets via adapters
- **Dynamic Contextual Weights**: Scenario-specific arbitration strategies (factual disputes, temporal updates, etc.)
- **Comprehensive Evaluation**: Supports 5+ benchmarks with detailed metrics
- **Single Entry Point**: `main.py` handles all use cases

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For full functionality, you may need optional dependencies for certain benchmarks. Fast accepted-benchmark runs can reuse adapter-structured proposals without local model re-extraction.

### Run Evaluation (Fast Structured Mode - Recommended for Testing)

```bash
# Test with 20 scenarios from MemAE
python main.py --benchmark memae --max-scenarios 20 --use-dummy

# Run all available benchmarks (lightweight)
python main.py --benchmark all --max-scenarios 50 --use-dummy
```

### Run with Real Models (Slower, Re-extracts From Raw Benchmark Text)

```bash
# Set Hugging Face token if needed
export HF_TOKEN="your_token_here"

# Run with a Colab-friendly Qwen model
python main.py --benchmark mab_conflict --max-scenarios 4 \
  --agent1-model Qwen/Qwen2.5-1.5B-Instruct \
  --agent2-model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

---

## Running on Google Colab

We provide `colab_runner.ipynb` for a Colab-first workflow:

- clone or pull the repo from GitHub
- install the exact Python runtime dependencies
- download Qwen on first transformer run
- auto-download Hugging Face benchmarks that do not require local parquet files
- save benchmark outputs under `reports/colab/`

### Setup Steps

1. **Push code to GitHub** (recommended):
   ```bash
   git add .
   git commit -m "Prepare for Colab"
   git push origin main
   ```

   If you do not want to use GitHub, create a clean upload bundle locally:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\export_clean_bundle.ps1
   ```

2. **Open Colab**:
   - Go to https://colab.research.google.com
   - Upload `colab_runner.ipynb`
   - Or open it directly from your GitHub repo

3. **Runtime Setup**:
   - Set Runtime type to **GPU (T4)**
   - Optional: set a Hugging Face token in Colab secrets as `HF_TOKEN`

4. **Enter Configuration**:
   - GitHub repo URL
   - Branch name
   - Benchmarks to run, for example: `mab_conflict,longmemeval,locomo,safeflow`
   - Model: `Qwen/Qwen2.5-1.5B-Instruct`
   - Scenarios: start with `1` or `2`

5. **Run All Cells**:
   - The notebook clones or updates the repo automatically
   - Qwen downloads on first transformer run
   - Supported Hugging Face benchmarks download automatically
   - Results are zipped at the end for download from Colab

### Important Notes for Colab

- **Memory**: T4 GPU has ~12GB. Use 1.5B or 2.7B models. **Avoid 7B** (OOM).
- **Time Limit**: Colab limits runtime to ~12 hours. Plan accordingly.
- **Automatic benchmarks**: `mab_conflict`, `longmemeval`, `locomo`, and `safeflow` can download on demand.
- **Local-only benchmark**: `memae` still expects `data/raw/memab/Conflict_Resolution-00000-of-00001.parquet`.
- **Model Cache**: Downloaded to `/root/.cache/huggingface` - persists across sessions.

### Troubleshooting Colab

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `MAX_SCENARIOS` to 1-2 and keep both agents on `Qwen/Qwen2.5-1.5B-Instruct` |
| Timeout | Run the `smoke` preset first, then increase scenarios |
| `memae` missing data | Switch to `mab_conflict` or upload the parquet file manually |

---

## System Architecture

```
┌──────────────┐
│   main.py    │  Single entry point for all evaluation modes
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              Unified Loader (unified_loader.py)         │
│  Converts any dataset → Internal Standard Format (ISF) │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│           Multi-Agent Pipeline (multi_agent_pipeline) │
│  ┌─────────────┐      ┌──────────────────┐            │
│  │   Agents    │─────▶│ Shared Memory    │            │
│  │ (AgentRuntime)│   │  (Version-Aware) │            │
│  └─────────────┘      └────────┬─────────┘            │
│                               │                        │
│                     ┌─────────▼─────────┐              │
│                     │  Conflict Engine  │              │
│                     │  • Staleness      │              │
│                     │  • Detection      │              │
│                     │  • Arbitration    │              │
│                     └───────────────────┘              │
└────────────────────────────────────────────────────────┘
```

### Core Components

1. **SharedMemoryStore** (`src/memory/shared_memory_store.py`)
   - Version-controlled memory entries with bi-temporal tracking
   - Status management (active, superseded, tentative, archived)
   - Persistence support

2. **Conflict Detection** (`src/conflict/conflict_detector.py`)
   - Tiered approach: rule-based → semantic similarity
   - Classifies conflicts: mutually_exclusive, stale_read_conflict, semantic_overlap, etc.
   - No LLM required - uses lexical overlap as fallback

3. **ConflictAwareWriter** (`src/conflict/conflict_aware_writer.py`)
   - Arbitration engine with configurable weights
   - Actions: overwrite, merge, keep_multiple_versions, defer, reject, append
   - Dynamic context weights via `scenario_id` parameter
   - Uncertainty-aware decision making

4. **Baselines** (`src/conflict/baselines.py`)
   - `LastWriteWinsWriter`: Simple recency-based overwrite
   - `NaiveAppendWriter`: No conflict handling (all writes active)

## Internal Standard Format (ISF)

All data converted to this canonical format:

```python
@dataclass
class Scenario:
    scenario_id: str
    agents: List[str]
    ordered_events: List[Event]      # Timeline of reads/writes
    gold_conflict_exists: bool
    gold_conflict_type: str          # Conflict taxonomy
    gold_resolution_action: str      # Expected action
    gold_reconciled_memory_state: List[MemoryEntry]
    gold_visible_shared_state_after_commit: List[MemoryEntry]
    scenario_type: str
    queries: List[Query] = []        # Optional retrieval queries
```

See `PROJECT_DOCUMENTATION.md` for full ISF specification.

## Supported Benchmarks

| Benchmark | Source | Scenarios | Conflict Types | Notes |
|-----------|--------|-----------|----------------|-------|
| **MemAE** | Local parquet | 8 | Conflict resolution | Fact consolidation |
| **MemoryAgentBench** | HuggingFace (`ai-hyz/MemoryAgentBench`) | 8 (Conflict_Resolution split) | Diverse conflicts | Only Conflict_Resolution split suitable |
| **Real Conflicts** | Combined | ~16 | Multiple | `--benchmark real_conflicts` (MemAE + MAB_Conflict) |
| **LoCoMo** | HuggingFace (`Aman279/Locomo`) | 35 | Conversation memory | Multi-party, long dialogues |

**Note**: The core conflict resolution benchmarks are small (~16 scenarios total) because publicly available datasets with ground-truth conflict labels are limited. This is a research gap in the field.

## Usage Reference

### Command-Line Options

```
python main.py [OPTIONS]

Required:
  --benchmark {memae,mab_conflict,real_conflicts,longmemeval,safeflow,mab,locomo,custom,all}
                Which benchmark to evaluate
                - real_conflicts: MemAE + MAB Conflict_Resolution (recommended accepted conflict bundle)
                - mab_conflict: MemoryAgentBench Conflict_Resolution only
                - all: All accepted built-in benchmarks
```

Data Options:
  --max-scenarios N      Limit scenarios (for testing)
  --num-samples N        Alias for max-scenarios
  --subset SUBSET        Subset for benchmarks (e.g. "Conflict_Resolution", "oracle", "s", "m", "test")
  --custom-path PATH     Path to custom JSONL (required for --benchmark custom)
  --cache-scenarios DIR  Save loaded scenarios as JSONL cache

Agent Options:
  --use-dummy            Use adapter-structured proposals with reliability priors; no model re-extraction
  --agent1-model NAME    Local model for agent 1; enables strict raw-text re-extraction
  --agent2-model NAME    Local model for agent 2; enables strict raw-text re-extraction
  --agent1-reliability X Reliability prior for agent 1 (0.0-1.0)
  --agent2-reliability X Reliability prior for agent 2 (0.0-1.0)

Output Options:
  --output-dir DIR       Output directory (default: "reports")
```

### Examples

```bash
# Quick test with 2 scenarios from real conflict benchmarks
python main.py --benchmark real_conflicts --max-scenarios 2 --use-dummy

# Full evaluation on all real conflict benchmarks
python main.py --benchmark real_conflicts

# Evaluate only MemoryAgentBench conflict subset
python main.py --benchmark mab_conflict --max-scenarios 20

# Run LongMemEval oracle subset
python main.py --benchmark longmemeval --subset oracle --max-scenarios 20

# Custom benchmark
python main.py --benchmark custom --custom-path data/my_benchmark.jsonl
```

## Output

Reports saved to `--output-dir` (default: `reports/`):

```
reports/
├── real_conflicts_report.json  # Combined results (if using real_conflicts)
├── memae_report.json           # MemAE results
├── mab_conflict_report.json    # MAB Conflict_Resolution results
├── summary.json                # Overall run summary
└── ... other benchmark reports
```

### Report Structure

```json
{
  "benchmark": "memae",
  "num_scenarios": 100,
  "timestamp": "2026-05-09T...",
  "execution": {
    "proposal_source": "structured",
    "strict_agent_execution": false,
    "mode_label": "dummy_structured"
  },
  "results": {
    "conflict_aware": {
      "scenario_accuracy": 0.85,
      "conflict_f1": 0.92,
      "action_accuracy": 0.78,
      "final_memory_f1": 0.88,
      "avg_branch_count": 1.15,
      "stale_handling_accuracy": 0.90,
      ...
    },
    "lww": { ... },
    "naive": { ... }
  },
  "deltas": {
    "conflict_aware_minus_lww": { ... }
  },
  "per_scenario_type": { ... }
}
```

## Configuration

### Arbitration Settings

Edit `configs/arbitration.yaml` to tune performance:

```yaml
arbitration:
  weights:
    confidence: 0.4     # Importance of confidence score
    provenance: 0.3     # Quality of source (explicit > inferred)
    recency: 0.2        # Preference for newer info
    authority: 0.1      # Agent authority score
  recency_half_life: 3600.0  # Half-life in seconds

thresholds:
  overwrite_margin: 0.12             # Min score diff to overwrite
  keep_multiple_versions_margin: 0.08  # Score margin too close → keep both
  defer_below_score: 0.40            # Score below this → defer
  semantic_duplicate: 0.95           # Duplicate detection threshold
  semantic_overlap: 0.70             # Overlap detection threshold

context_weights:  # Scenario-specific overrides
  factual_dispute:
    confidence: 0.5   # Heavier confidence for factual conflicts
    provenance: 0.25
    recency: 0.15
    authority: 0.1
  temporal_update:
    recency: 0.4      # Higher recency weight for temporal updates
```

Dynamic context: Pass `scenario_id` to `writer.write()` to activate context-specific weights.

## Performance Targets

| Mode | Cost | Latency | Scenario Acc | Conflict F1 |
|------|------|---------|---------------|-------------|
| **Conflict-aware (rule-based)** | Low | Fast | 80-90% | 0.90+ |
| LWW baseline | Minimal | Fastest | 40-60% | 0.30-0.50 |
| Naive baseline | Minimal | Fast | 10-30% | 0.00 |

**Advantage**: Near-SOTA performance at 10-20x lower cost than LLM-based approaches.

## File Structure (Optimized)

```
.
├── main.py                      # Single entry point (use this)
├── configs/
│   └── arbitration.yaml        # Arbitration configuration
├── data/
│   ├── raw/                    # Original dataset files
│   │   ├── memab/
│   │   └── longmemeval/
│   └── processed/              # ISF conversions
├── reports/                    # All output (single folder)
├── src/
│   ├── format.py              # ISF definitions (Scenario, MemoryEntry)
│   ├── benchmarks/
│   │   ├── unified_loader.py  # Main loader interface
│   │   ├── adapters/          # Dataset-specific converters
│   │   │   ├── memab_adapter.py
│   │   │   ├── longmemeval_adapter.py
│   │   │   └── safeflow_adapter.py
│   │   ├── memoryagentbench_loader.py  # Optional
│   │   ├── lococo_loader.py           # Optional
│   │   ├── generator_core.py          # Scenario generation utility
│   │   └── scenario_types.py          # Type definitions
│   ├── memory/
│   │   └── shared_memory_store.py
│   ├── conflict/
│   │   ├── conflict_aware_writer.py   # Main arbitration engine
│   │   ├── conflict_detector.py       # Tiered conflict detection
│   │   ├── staleness_detector.py
│   │   └── baselines.py               # LWW and naive
│   ├── pipeline/
│   │   └── multi_agent_pipeline.py
│   ├── agents/
│   │   └── agent_runtime.py
│   ├── utils/
│   │   ├── retriever.py
│   │   └── extractor.py              # Rule-based extraction
│   └── local_models/
│       └── runner.py                 # Dummy & transformer agents
├── requirements.txt
├── Dockerfile
├── PROJECT_DOCUMENTATION.md          # Comprehensive documentation
└── README.md                         # This file
```

**Removed files** (consolidated into `main.py`):
- `run_agent_memory_comparison.py`
- `run_all.py`
- `run_benchmarks.py`
- `run_local_comparison.py`
- `demo_runner.py`
- `ablation_study.py`

## Adding a New Benchmark

1. Create adapter in `src/benchmarks/adapters/`:

```python
class MyAdapter:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def convert_all_to_scenarios(self, num_agents: int = 2) -> List[Scenario]:
        # Convert your dataset to ISF Scenario objects
        scenarios = []
        # ... conversion logic ...
        return scenarios
```

2. Register in `src/benchmarks/unified_loader.py`:

```python
from .adapters.my_adapter import MyAdapter

def load_benchmark(benchmark_name: str, **kwargs):
    # ...
    elif benchmark_name == "mydataset":
        adapter = MyAdapter('data/raw/mydataset/')
        return adapter.convert_all_to_scenarios(max_scenarios=kwargs.get('max_scenarios'))
```

3. Use: `python main.py --benchmark mydataset`

## Technical Details

### Conflict Detection Tiers

1. **Rule-based** (fast, deterministic):
   - Exact duplicate check
   - Stale read detection (read_snapshot_time < latest commit)
   - Mutually exclusive predicate classification
   - Additive predicate detection

2. **Semantic** (when rule-based inconclusive):
   - Embedding similarity (sentence-transformers)
   - Lexical overlap fallback (Jaccard)
   - JSON object mergeability detection

### Arbitration Factors

Each memory entry scored on:
- **Confidence** (0.0-1.0): Explicit source reliability
- **Provenance** (weighted): explicit=1.0, behavioral=0.85, inferred=0.7, llm_inferred=0.6, unknown=0.4
- **Recency** (exponential decay): `2^(-time_diff / half_life)`
- **Authority** (agent-specific): Default 1.0

Weighted sum decides action with threshold margins.

### Uncertainty-Aware Decisions

For stale-read and contradiction scenarios, uncertainty is calculated:
```
certainty = 0.35*confidence + 0.25*provenance + 0.25*recency + 0.15*authority
uncertainty = 1.0 - certainty
```

High uncertainty favors newer entries in mutually exclusive conflicts.

## Dependencies

See `requirements.txt`. Core requirements:
- `torch`, `transformers` (optional, for agent models)
- `sentence-transformers` (optional, for semantic similarity)
- `datasets` (optional, for HuggingFace benchmarks)
- `pyyaml` (for config loading)

For pure rule-based evaluation, transformers and sentence-transformers are not strictly required - the system falls back to lexical similarity.

## License

[Specify license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[To be added]
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
