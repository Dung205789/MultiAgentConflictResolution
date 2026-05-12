# CLAUDE.md

## Project Context

This is a research project on **rule-based multi-agent memory conflict resolution**. The system uses deterministic rules (not LLMs) for conflict detection and arbitration, achieving near-SOTA performance at a fraction of the cost.

### Key Characteristics
- **Constraint-based**: Operates without LLM dependencies in the critical path
- **Unified format**: All datasets converted to ISF (Internal Standard Format)
- **Single entry point**: `main.py` - use this for all evaluations
- **Cost-effective**: 10-20x cheaper than LLM-based approaches

### Current Status
- Core pipeline fully functional
- 3 adapters ready: MemAE, LongMemEval, SAFEFLOW
- Optional adapters: MemoryAgentBench, LoCoMo
- Comprehensive metrics and reporting

## Response Style
- Code first. Explain only if asked.
- No preamble: never restate the request
- No summaries at the end
- If fixing a bug: just fix it
- For this research project: prioritize correctness and clarity over cleverness

## Important Notes

### Entry Point
Always use `main.py` as the primary entry point. Previous runner scripts have been consolidated.

### Data Format
All benchmark data must be in ISF format (see `src/format.py`). Use adapters from `src/benchmarks/adapters/` to convert external datasets.

### Configuration
Arbitration weights and thresholds are in `configs/arbitration.yaml`. Changes take effect immediately.

### Novel Features
- Dynamic contextual weights: Pass `scenario_id` to activate scenario-specific arbitration strategies
- Uncertainty-aware decisions: Factors epistemic uncertainty into conflict resolution
- Tiered detection: Rule-based → semantic → (optional) judge

### Testing
For quick validation:
```bash
python main.py --benchmark memae --max-scenarios 5 --use-dummy
```

### Documentation
See `PROJECT_DOCUMENTATION.md` for comprehensive technical details.
