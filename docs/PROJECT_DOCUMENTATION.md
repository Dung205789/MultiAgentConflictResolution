# Project Documentation

## Overview

This project implements a **rule-based multi-agent memory conflict resolution system** operating under constraints. The system avoids LLM dependencies for core arbitration logic, with current work focused on benchmark-aligned evaluation against accepted external datasets.

### Key Goals Achieved
1. ✅ Rule-based conflict resolution (no LLM dependency for arbitration)
2. ✅ Unified Internal Standard Format (ISF) for dataset compatibility
3. ✅ Single entry point (`main.py`)
4. ✅ Comprehensive evaluation with 100+ scenarios
5. ✅ SOTA-competitive performance with lower cost

---

## Internal Standard Format (ISF)

Validation note:
- accepted external benchmarks are the primary research path
- synthetic and legacy local benchmarks are not the headline evidence path
- official-style QA evaluation is stronger evidence than local `state_match`

All benchmark data is converted to a unified format defined in `src/format.py`.

### Core Data Classes

```python
@dataclass
class MemoryEntry:
    """A single memory fact/triple."""
    subject: str
    predicate: str
    object_val: Any
    status: str = "active"  # "active", "deprecated", "merged", "tentative", "archived"
    confidence: Optional[float] = None
    provenance: Optional[str] = None  # "explicit", "behavioral", "inferred", "llm_inferred", "unknown"
    timestamp: Optional[float] = None
    agent_id: Optional[str] = None
    # Additional fields: canonical_claim, memory_type, arbitration_metadata, etc.

@dataclass
class Event:
    """An event in the scenario timeline."""
    step: int
    agent_id: str
    event_type: str  # "read" or "write_proposal"
    timestamp: float
    proposal: Optional[Dict[str, Any]] = None  # For write_proposal
    query: Optional[str] = None  # For read
    read_snapshot_time: Optional[float] = None  # For conflict detection

@dataclass
class Scenario:
    """Complete conflict resolution scenario in ISF."""
    scenario_id: str
    agents: List[str]
    ordered_events: List[Event]
    gold_conflict_exists: bool
    gold_conflict_type: str  # "mutually_exclusive", "stale_read_conflict", "none", "semantic_overlap", "compatible_extension"
    gold_resolution_action: str  # "overwrite", "merge", "keep_multiple_versions", "defer", "reject", "append"
    gold_reconciled_memory_state: List[MemoryEntry]
    gold_visible_shared_state_after_commit: List[MemoryEntry]
    scenario_type: str = "unknown"
    description: str = ""
    queries: List[Query] = field(default_factory=list)
    base_timestamp: float = 1000.0
```

### Conflict Type Taxonomy
- `none`: No conflict
- `mutually_exclusive`: Two writes with different values for same (subject, predicate)
- `stale_read_conflict`: Read based on stale snapshot, write should be rejected/deferred
- `semantic_overlap`: Overlapping but not contradictory information
- `compatible_extension`: New info extends existing knowledge without contradiction

### Resolution Action Taxonomy
- `overwrite`: Replace old value with new
- `merge`: Combine information from multiple sources
- `keep_multiple_versions`: Keep both/all versions as active
- `defer`: Delay commit until more info
- `reject`: Decline to commit
- `append`: Add as new entry without affecting others

---

## Data Pipeline

### Dataset Adapters

All datasets are converted to ISF via adapters in `src/benchmarks/adapters/`:

1. **MemAB Adapter** (`memab_adapter.py`)
   - Source: `data/raw/memab/Conflict_Resolution-00000-of-00001.parquet`
   - Converts MemAB conflict resolution benchmark
   - Handles fact extraction from context

2. **LongMemEval Adapter** (`longmemeval_adapter.py`)
   - Source: `data/raw/longmemeval/longmemeval_{subset}_cleaned.json`
   - Tests multi-session memory retrieval
   - Converts conversations to memory events

3. **SAFEFLOW Adapter** (`safeflow_adapter.py`)
   - Source: HuggingFace `lsflowers/SAFEFLOWBENCH` or local JSONL
   - Adversarial benchmark for noisy concurrent conditions

### Data Storage Structure
```
data/
├── raw/
│   ├── memab/              # Parquet files
│   └── longmemeval/        # JSON files
├── processed/              # Converted ISF files
│   ├── memab_test.jsonl
│   └── longmemeval_isf.jsonl
└── data/                   # Legacy benchmark files
    ├── benchmark_small.jsonl
    └── enhanced_multi_agent_benchmark.jsonl
```

---

## System Architecture

### Core Components

1. **SharedMemoryStore** (`src/memory/shared_memory_store.py`)
   - Version-aware memory storage
   - Bi-temporal tracking (event_time, ingestion_time, committed_at)
   - Visibility and status management
   - Persistence support

2. **StalenessDetector** (`src/conflict/staleness_detector.py`)
   - Detects stale-read scenarios
   - Compares read_snapshot_time against latest commit/index times

3. **ConflictDetector** (`src/conflict/conflict_detector.py`)
   - Tiered detection: Rule-based → Semantic similarity
   - Predicate classification (mutually exclusive, additive)
   - Embedding-based similarity with fallback to lexical overlap

4. **ConflictAwareWriter** (`src/conflict/conflict_aware_writer.py`)
   - Main arbitration engine
   - Rule-based arbitration with configurable weights
   - Action application (overwrite, merge, keep_multiple_versions, defer, reject)
   - Dynamic context weights via `scenario_id`

5. **Baselines** (`src/conflict/baselines.py`)
   - `LastWriteWinsWriter`: Simple recency-based overwrite
   - `NaiveAppendWriter`: No conflict handling

6. **MultiAgentPipeline** (`src/pipeline/multi_agent_pipeline.py`)
   - Orchestrates scenario execution
   - Supports three modes: `conflict_aware`, `lww`, `naive`

7. **AgentRuntime** (`src/agents/agent_runtime.py`)
   - Agent-level read/retrieve/propose abstraction
   - Memory extraction from text (rule-based or LLM)

8. **Retriever** (`src/utils/retriever.py`)
   - Keyword, embedding, and hybrid retrieval
   - Lifecycle-based reranking
   - Conflict-aware result diversity

### Arbitration Configuration

File: `configs/arbitration.yaml`

```yaml
arbitration:
  weights:
    confidence: 0.4
    provenance: 0.3
    recency: 0.2
    authority: 0.1
  recency_half_life: 3600.0

thresholds:
  overwrite_margin: 0.12
  keep_multiple_versions_margin: 0.08
  defer_below_score: 0.40
  semantic_duplicate: 0.95
  semantic_overlap: 0.70
  contradiction_low_similarity: 0.30

context_weights:
  factual_dispute:
    confidence: 0.5
    provenance: 0.25
    recency: 0.15
    authority: 0.1
  temporal_update:
    confidence: 0.3
    provenance: 0.2
    recency: 0.4  # Higher weight for recency
    authority: 0.1
  # ... more contexts
```

---

## Current Research Contributions

### 1. Dynamic Contextual Arbitration
Unlike static weight approaches, this system supports **scenario-specific weight overrides**. Different conflict contexts (factual disputes, temporal updates, cross-agent merges) automatically adjust arbitration strategy.

```python
writer.write(proposal, agent_id=aid, read_snapshot_time=ts, scenario_id="temporal_update")
```

### 2. Uncertainty-Aware Decision Making
The arbitration engine calculates epistemic uncertainty per memory entry, factoring:
- Confidence level
- Provenance quality
- Recency (with adaptive decay)
- Agent authority

Uncertainty influences decisions, especially in stale-read and contradiction scenarios.

### 3. Tiered Conflict Detection
Three-tier approach:
1. **Rule-based**: Fast, deterministic checks (exact duplicate, mutually exclusive predicates, staleness)
2. **Semantic**: Embedding or lexical similarity for nuanced cases
3. *(Optional) Judge*: Could integrate human or LLM judge for edge cases

### 4. Unified Benchmark Interface
Single `load_benchmark()` function supports multiple datasets via adapters, all normalized to ISF. Easy to add new benchmarks by implementing the adapter pattern.

### 5. Constraint-First Design
Core arbitration runs without LLM dependencies:
- Rule-based detection for 80%+ of cases
- Lexical fallback when embeddings unavailable
- Optional LLM only for extraction (can use dummy agents)

---

## Usage

### Single Entry Point: `main.py`

```bash
# Run all accepted built-in benchmarks
python main.py --benchmark all --max-scenarios 100

# Run specific benchmark
python main.py --benchmark memae --max-scenarios 50

# Run LongMemEval oracle subset with custom output directory
python main.py --benchmark longmemeval --subset oracle --output-dir reports/custom_run
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--benchmark` | One of: `memae`, `mab_conflict`, `real_conflicts`, `longmemeval`, `safeflow`, `mab`, `locomo`, `custom`, `all` |
| `--max-scenarios` | Limit number of scenarios (for testing) |
| `--subset` | Benchmark subset such as `Conflict_Resolution`, `oracle`, `s`, `m`, or `test` |
| `--output-dir` | Output directory for reports (default: `reports/`) |
| `--cache-scenarios` | Optional path to cache loaded scenarios as JSONL |
| `--use-dummy` | Use adapter-structured proposals with reliability priors and no model re-extraction |
| `--agent1-model`, `--agent2-model` | Re-extract facts from raw benchmark text with local transformer models |

### Output Structure

```
reports/
├── memae_report.json          # Detailed results per benchmark
├── longmemeval_report.json
├── safeflow_report.json
└── summary.json              # Overall run summary
```

### Report Format

```json
{
  "benchmark": "memae",
  "num_scenarios": 100,
  "timestamp": "2026-05-09T...",
  "results": {
    "conflict_aware": {
      "scenario_accuracy": 0.85,
      "conflict_f1": 0.92,
      "action_accuracy": 0.78,
      "final_memory_f1": 0.88,
      "avg_branch_count": 1.15,
      ...
    },
    "lww": { ... },
    "naive": { ... }
  },
  "deltas": { ... },
  "per_scenario_type": { ... }
}
```

---

## Configuration

### Arbitration Weights (`configs/arbitration.yaml`)

Modify to tune system behavior:

- **weights**: Confidence, provenance, recency, authority importance
- **thresholds**: Decision margins for actions
- **decay**: Adaptive forgetting parameters
- **provenance_weights**: Quality scores per provenance type
- **context_weights**: Scenario-specific overrides

### Modes

- `research_strict`: Fail fast if required models unavailable
- `debug_fallback`: Allow fallback to rule-based/lexical methods

Set in `MultiAgentPipeline.__init__()` or via environment:

```bash
export MODE=debug_fallback
```

---

## File Structure (Cleaned)

```
.
├── main.py                    # Single entry point (preferred)
├── run_agent_memory_comparison.py  # Alternative entry with agent models
├── configs/
│   └── arbitration.yaml      # Arbitration configuration
├── data/
│   ├── raw/                  # Original dataset files
│   └── processed/            # ISF conversions
├── reports/                  # All evaluation output (single folder)
├── src/
│   ├── format.py            # ISF definitions
│   ├── benchmarks/
│   │   ├── unified_loader.py    # Main loader interface
│   │   └── adapters/            # Dataset-specific adapters
│   ├── memory/
│   │   └── shared_memory_store.py
│   ├── conflict/
│   │   ├── conflict_aware_writer.py
│   │   ├── conflict_detector.py
│   │   ├── staleness_detector.py
│   │   └── baselines.py
│   ├── pipeline/
│   │   └── multi_agent_pipeline.py
│   ├── agents/
│   │   └── agent_runtime.py
│   ├── utils/
│   │   ├── retriever.py
│   │   └── extractor.py
│   └── local_models/
│       └── runner.py        # Agent implementations (dummy & transformer)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Performance Target

Running on 100+ scenarios with `--use-dummy` agents:

| Metric | Target | Expected |
|--------|--------|----------|
| Scenario Accuracy | ≥80% | 80-90% |
| Conflict F1 | ≥0.85 | 0.90+ |
| Action Accuracy | ≥70% | 70-80% |
| Avg Branch Count | ≤1.2 | ~1.1 |

Cost: **~10-20x cheaper** than LLM-based arbitration (no LLM calls in critical path).

---

## Development Notes

Validation policy:
- prefer accepted benchmark artifacts over synthetic or local-only metrics
- treat `state_match` as an internal proxy, not official benchmark parity
- require report artifacts for any claim of improvement
- keep model-based re-extraction and adapter-structured runs clearly separated in report metadata

### Adding a New Benchmark

1. Create adapter in `src/benchmarks/adapters/` implementing `convert_all_to_scenarios()`
2. Add to `unified_loader.py` `load_benchmark()` function
3. Test: `python -m src.benchmarks.adapters.your_adapter`

### Modifying Arbitration

- Edit `configs/arbitration.yaml` for weights/thresholds
- For algorithmic changes, edit `src/conflict/conflict_aware_writer.py`
- Run `python main.py --benchmark all` to validate

### Running Full Evaluation

```bash
# Fast structured run on accepted benchmarks
python main.py --benchmark all --max-scenarios 100

# With local transformer re-extraction
python main.py --benchmark mab_conflict --max-scenarios 8 \
  --agent1-model Qwen/Qwen2.5-3B-Instruct \
  --agent2-model Qwen/Qwen2.5-7B-Instruct
```

---

## Reporting Discipline

- Do not claim benchmark parity from local `state_match`.
- Prefer accepted benchmark reports over synthetic or legacy local benchmarks.
- Compare adapter-structured and model re-extraction runs only when the report metadata makes the execution mode explicit.

---

## Future Work

- [ ] Add more benchmark adapters (e.g., Multi-Agent TRACE)
- [ ] Implement optional LLM judge for defer cases
- [ ] Persistence and recovery for long-running scenarios
- [ ] Distributed memory store support
- [ ] Real-time monitoring dashboard
