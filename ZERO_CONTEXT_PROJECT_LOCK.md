# ZERO_CONTEXT_PROJECT_LOCK

## Project
Rule-Based Multi-Agent Shared Memory Conflict Resolution for Multi-Hop QA

## Hard Lock
- Primary benchmark: `MemoryAgentBench / Conflict_Resolution`
- Primary track: `oracle_structured`
- Secondary track: `end_to_end_extract`
- LLMs are not allowed to decide `overwrite`, `reject`, `merge`, or `commit`
- `src/conflict/conflict_aware_writer.py` is the only arbitration authority
- The memory system itself is part of the contribution

## What This Repo Is Now
- A symbolic multi-agent shared-memory system with:
  - a real runtime memory store
  - rule-based conflict arbitration
  - query-aware proposal annotation
  - symbolic QA over final visible memory
- It is not a paper-faithful reproduction of the original MemoryAgentBench paper setup.
- It is a paper-aligned symbolic alternative with a strict non-LLM arbitration core.

## Code-Verified Architecture
- Entry point: `app/main.py`
- Pipeline: `src/pipeline/multi_agent_pipeline.py`
- Store: `src/memory/shared_memory_store.py`
- Canonical memory schema: `src/memory/schema.py`
- Conflict writer: `src/conflict/conflict_aware_writer.py`
- Query-aware annotation: `src/conflict/query_aware_context.py`
- Conflict detector: `src/conflict/conflict_detector.py`
- QA: `src/evaluation/qa_reasoner.py`
- Evaluation: `src/evaluation/run_evaluation.py`
- Benchmark loader: `src/benchmarks/memoryagentbench_loader.py`

## What Works
- Shared memory store and lifecycle API exist.
- Extractor proposal contract is explicit.
- Track split is explicit.
- Fallback contamination is reported.
- Local `SubEM` is implemented.
- Long-running benchmark runs now emit:
  - `*.progress.json`
  - `*.partial.json`
- Reports now expose track separation fields:
  - `reported_track_name`
  - `pure_end_to_end_extract`
  - `structured_fallback_present`

## Best Verified Full Result By QA Advantage
- Source:
  - `reports/paper_mode_mab8_fc_refresh/mab_conflict_report.json`
- Track:
  - `oracle_structured`
- Variant:
  - `no_lineage_edges`
- Result:
  - `conflict_aware qa_exact_match = 0.44125`
  - `lww qa_exact_match = 0.43125`
  - `conflict_aware qa_subem = 0.44750`
  - `lww qa_subem = 0.43750`
  - `conflict_aware FC-SH = 0.51905`
  - `lww FC-SH = 0.49524`
  - `conflict_aware FC-MH = 0.43595`
  - `lww FC-MH = 0.43021`
  - `scenario_accuracy = 1.0`
  - `action_accuracy = 1.0`

## Latest Verified Full Code-State
- 2-scenario latest ablation:
  - `reports/paper_mode_mab2_pipeline_public_conflictonly_v2/mab_conflict_report.json`
  - `reports/paper_mode_mab2_pipeline_public_lwwonly_v1/mab_conflict_report.json`
- extra evaluation surface:
  - `reports/paper_mode_mab2_accurate_retrieval_public_v1/mab_report.json`
- 8-scenario latest full runs:
  - `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_nolineage/mab_conflict_report.json`
- Key truth:
  - overwrite leakage on the benchmark path is removed
  - `scenario_accuracy = 1.0`
  - `action_accuracy = 1.0`
  - `fallback_contamination_detected = false`
  - latest code-state now beats both `lww` and `no_query_support`
  - query-aware gain is now measured cleanly
  - lineage is still neutral
- Current full metrics:
  - `conflict_aware_full qa_exact_match = 0.73000`
  - `lww qa_exact_match = 0.68750`
  - `conflict_aware_no_query_support qa_exact_match = 0.68750`
  - `conflict_aware_no_lineage_edges qa_exact_match = 0.73000`
  - `conflict_aware_full qa_subem = 0.73000`
  - `lww qa_subem = 0.69000`
  - `conflict_aware_full FC-SH = 0.85460`
  - `lww FC-SH = 0.83976`
  - `conflict_aware_full FC-MH = 0.69613`
  - `lww FC-MH = 0.59669`

## Main Open Problems
- `lineage` still has no measured gain over `no_lineage_edges`.
- The dominant remaining error category is still `wrong_anchor_resolution`.
- On the latest 2-scenario public-ready split, `wrong_anchor_resolution` is down to `2` for `conflict_aware`.
- Symbolic QA is still the main shared bottleneck, even after the new query-aware lift.
- `end_to_end_extract` remains secondary evidence only, but its report path is now explicit and cleanly separated from `oracle_structured`.

## Paper-Aligned Vs Paper-Faithful
- Paper-aligned:
  - symbolic conflict core
  - multi-hop QA on final memory
  - local `SubEM`
  - local `FC-SH` / `FC-MH`
- Not paper-faithful:
  - main track is `oracle_structured`
  - evaluator is local rather than official
  - repo contribution is a symbolic arbitration system, not a 1:1 paper replication

## Do Not Undo
- Do not let the LLM become the final judge.
- Do not blur `oracle_structured` with `end_to_end_extract`.
- Do not blur query-aware gain with lineage gain.
- Do not report lineage as a verified contribution; it is still neutral.
- Do not remove the new progress/partial reporting for long runs.

## Rerun Commands
- 2-scenario latest ablation:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes conflict_aware --output-dir reports\paper_mode_mab2_pipeline_public_conflictonly_v2 --enable-error-analysis --emit-scenario-bundles`
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes lww --output-dir reports\paper_mode_mab2_pipeline_public_lwwonly_v1 --enable-error-analysis --emit-scenario-bundles`
- 8-scenario `conflict_aware_full`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_conflictonly --enable-error-analysis --emit-scenario-bundles`
- 8-scenario `lww`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes lww --output-dir reports\paper_mode_mab8_queryaware_gain_v2_lwwonly --enable-error-analysis --emit-scenario-bundles`
- 8-scenario `no_query_support`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_query_support --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_noquery --enable-error-analysis --emit-scenario-bundles`
- 8-scenario `no_lineage_edges`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_lineage_edges --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_nolineage --enable-error-analysis --emit-scenario-bundles`
- Secondary-track pure `end_to_end_extract`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_realmodel_v1 --enable-error-analysis --emit-scenario-bundles`
- Secondary-track explicit fallback labeling:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --allow-structured-fallback-in-end-to-end --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_fallback_v1 --enable-error-analysis --emit-scenario-bundles`
- Extra surface smoke run:
  - `python app/main.py --benchmark mab --subset Accurate_Retrieval --max-scenarios 2 --use-dummy --modes conflict_aware,lww --enable-error-analysis --emit-scenario-bundles --output-dir reports\paper_mode_mab2_accurate_retrieval_public_v1`

## Next Real Task
Keep the benchmark-safe overwrite path stable.
Then focus on the still-dominant `wrong_anchor_resolution` bucket.
Treat `query-aware` as a live claim.
Treat `lineage` as exploratory until it beats `no_lineage_edges`.
