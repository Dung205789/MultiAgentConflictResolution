# ProjectMem

Rule-Based Multi-Agent Shared Memory Conflict Resolution for Multi-Hop QA

## What This Repo Is
ProjectMem is a symbolic shared-memory system for conflict-heavy multi-agent QA settings.

The locked research direction is:
- primary track: `oracle_structured`
- secondary track: `end_to_end_extract`
- LLMs may propose or extract
- LLMs must not decide `overwrite`, `reject`, `merge`, or `commit`
- `ConflictAwareWriter` is the only arbitration authority

This repo is currently paper-aligned, not paper-faithful.

## Current Status
Latest full code-state result:
- `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
- `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
- `conflict_aware qa_exact_match = 0.73000`
- `lww qa_exact_match = 0.68750`
- `conflict_aware qa_subem = 0.73000`
- `lww qa_subem = 0.69000`
- `conflict_aware FC-SH = 0.85460`
- `lww FC-SH = 0.83976`
- `conflict_aware FC-MH = 0.69613`
- `lww FC-MH = 0.59669`
- `scenario_accuracy = 1.0`
- `action_accuracy = 1.0`

Latest 2-scenario pipeline/public verification:
- `reports/paper_mode_mab2_pipeline_public_conflictonly_v2/mab_conflict_report.json`
- `reports/paper_mode_mab2_pipeline_public_lwwonly_v1/mab_conflict_report.json`
- `conflict_aware qa_exact_match = 0.875`
- `lww qa_exact_match = 0.795`
- `conflict_aware FC-SH = 1.0000`
- `lww FC-SH = 0.8125`
- `conflict_aware FC-MH = 0.8667`
- `lww FC-MH = 0.7933`
- `conflict_aware wrong_anchor_resolution = 2`
- `lww wrong_anchor_resolution = 20`

Extra surface smoke artifact:
- `reports/paper_mode_mab2_accurate_retrieval_public_v1/mab_report.json`
- this proves the pipeline runs on another MemoryAgentBench surface
- current zero QA on that surface reflects evaluator mismatch, not a broken pipeline

Query-aware isolation on the same code-state:
- `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
- `conflict_aware_no_query_support qa_exact_match = 0.68750`
- `conflict_aware_no_query_support qa_subem = 0.69000`
- `conflict_aware_no_query_support FC-SH = 0.83976`
- `conflict_aware_no_query_support FC-MH = 0.59669`

Interpretation:
- the latest repaired path is benchmark-safe and now beats `lww`
- query-aware metadata now has measured gain beyond both `lww` and `no_query_support`
- lineage is still neutral: `full` and `no_lineage_edges` match on the latest ablations

## Main Entry Points
- runner: `app/main.py`
- pipeline: `src/pipeline/multi_agent_pipeline.py`
- writer: `src/conflict/conflict_aware_writer.py`
- store: `src/memory/shared_memory_store.py`
- QA: `src/evaluation/qa_reasoner.py`
- evaluation: `src/evaluation/run_evaluation.py`

## Recommended Commands
Full 8-scenario `conflict_aware` headline run:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_conflictonly --enable-error-analysis --emit-scenario-bundles
```

Full 8-scenario `lww` baseline:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes lww --output-dir reports\paper_mode_mab8_queryaware_gain_v2_lwwonly --enable-error-analysis --emit-scenario-bundles
```

Full 8-scenario `no_query_support` isolation run:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_query_support --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_noquery --enable-error-analysis --emit-scenario-bundles
```

Full 8-scenario `no_lineage_edges` isolation run:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_lineage_edges --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_nolineage --enable-error-analysis --emit-scenario-bundles
```

2-scenario current ablation:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --include-conflict-aware-ablations --output-dir reports\paper_mode_mab2_queryaware_gain_v5 --enable-error-analysis --emit-scenario-bundles
```

Replay QA from saved scenario artifacts:
```powershell
python scripts/replay_qa_from_report.py reports\paper_mode_mab8_queryaware_gain_v2_conflictonly\mab_conflict_report.json.scenarios
```

Finalize a report from saved `partial + scenario bundles`:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --finalize-report-from-artifacts reports\paper_mode_mab8_queryaware_gain_v2_conflictonly\mab_conflict_report.json
```

Real-model secondary track, pure `end_to_end_extract`:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_realmodel_v1 --enable-error-analysis --emit-scenario-bundles
```

Real-model secondary track with explicit structured fallback label:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --allow-structured-fallback-in-end-to-end --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_fallback_v1 --enable-error-analysis --emit-scenario-bundles
```

Run tests:
```powershell
python -m unittest discover -s tests -p "test*.py"
```

## Public Claim Boundary
Safe claims:
- symbolic, judge-free arbitration
- benchmark-safe repaired pipeline
- explicit track separation
- explicit track reporting:
  - `oracle_structured`
  - `end_to_end_extract`
  - `end_to_end_extract__structured_fallback`
- shared-memory lifecycle and conflict authority
- stronger shared QA layer than earlier repaired builds
- `conflict_aware > lww` on the latest full code-state
- query-aware metadata has measured gain beyond `no_query_support`

Unsafe claims unless new evidence appears:
- lineage currently gives a measured benchmark advantage over `no_lineage_edges`
- full paper-faithful replication of the original paper

## Key Docs
- lock: `ZERO_CONTEXT_PROJECT_LOCK.md`
- lock copy: `docs/PROJECT_LOCK.md`
- results: `docs/RESEARCH_RESULTS.md`
- acceptance criteria: `docs/PUBLICATION_ACCEPTANCE_CRITERIA.md`
- implementation backlog: `docs/CORE_CODE_AND_CLAIM_TODO.md`
