# PROJECT_LOCK

## Research Direction
This repository is locked to:
`Rule-Based Multi-Agent Shared Memory Conflict Resolution for Multi-Hop QA`.

The core claim is:
- a symbolic shared-memory system can arbitrate conflicting writes
- without using an LLM as the final judge
- while preserving downstream multi-hop QA utility

## Non-Negotiable Rules
- LLMs may emit proposals and supporting metadata only.
- LLMs must not decide `overwrite`, `reject`, `merge`, or `commit`.
- `src/conflict/conflict_aware_writer.py` remains the only arbitration authority.
- Main track: `oracle_structured`
- Secondary track: `end_to_end_extract`
- Headline benchmark: `MemoryAgentBench / Conflict_Resolution`
- Headline metrics: `SubEM`, `FC-SH`, `FC-MH`, then supporting QA/system metrics

## Current Architecture
- Entry point: `app/main.py`
- Main orchestration: `src/pipeline/multi_agent_pipeline.py`
- Shared memory store: `src/memory/shared_memory_store.py`
- Canonical runtime memory schema: `src/memory/schema.py`
- Conflict writer: `src/conflict/conflict_aware_writer.py`
- Query-aware annotation: `src/conflict/query_aware_context.py`
- Conflict detection: `src/conflict/conflict_detector.py`
- Symbolic QA: `src/evaluation/qa_reasoner.py`
- Evaluation harness: `src/evaluation/run_evaluation.py`
- Benchmark loader: `src/benchmarks/memoryagentbench_loader.py`

## What Already Works
- Real shared-memory lifecycle and store API exist.
- Extractor proposals are restricted to proposal-only metadata.
- Track naming is locked at the research-facing surface.
- Fallback contamination is explicit in reporting.
- Local `SubEM` is implemented.
- Full benchmark reporting exists with QA, `SubEM`, `FC-SH`, and `FC-MH`.
- Long-running evaluations now emit:
  - `*.progress.json`
  - `*.partial.json`
- Public-ready debug artifacts now exist:
  - `*.failure_bundle.json`
  - `*.scenarios/` per-scenario snapshots
  - `scripts/replay_qa_from_report.py` for QA-only replay from saved visible-memory artifacts
- Secondary-track reporting is now explicit in report metrics:
  - `reported_track_name`
  - `pure_end_to_end_extract`
  - `structured_fallback_present`

## Best Verified Full Result By QA Advantage
- Source of truth:
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
  - `fallback_contamination_detected = false`

## Latest Verified Full Code-State
- Source artifacts:
  - `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_nolineage/mab_conflict_report.json`
- Why it matters:
  - latest repaired path is benchmark-safe
  - latest code-state restores a full-run advantage over `lww`
  - query-aware gain is now isolated against `no_query_support`
  - lineage is still neutral against `no_lineage_edges`
- Current full results:
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
  - `scenario_accuracy = 1.0`
  - `action_accuracy = 1.0`
  - `fallback_contamination_detected = false`
- Interpretation:
  - this is now the latest benchmark-safe code-state
  - `conflict_aware_full > lww`
  - `conflict_aware_full > no_query_support`
  - `conflict_aware_full == no_lineage_edges`
  - claim update:
    - `query-aware` is now supported by measured full-run evidence
    - `lineage` is still not supported as a separate gain

## Latest 2-Scenario Repair Ablation
- Sources:
  - `reports/paper_mode_mab2_pipeline_public_conflictonly_v2/mab_conflict_report.json`
  - `reports/paper_mode_mab2_pipeline_public_lwwonly_v1/mab_conflict_report.json`
- Result:
  - `conflict_aware qa_exact_match = 0.875`
  - `lww qa_exact_match = 0.795`
  - `conflict_aware FC-SH = 1.0000`
  - `lww FC-SH = 0.8125`
  - `conflict_aware FC-MH = 0.8667`
  - `lww FC-MH = 0.7933`
  - `conflict_aware wrong_anchor_resolution = 2`
  - `lww wrong_anchor_resolution = 20`
- Interpretation:
  - the public-ready split-mode pipeline is stable
  - the latest anchor-resolution pass improved the small run materially

## Extra Evaluation Surface
- Source:
  - `reports/paper_mode_mab2_accurate_retrieval_public_v1/mab_report.json`
- Interpretation:
  - the pipeline runs on another MemoryAgentBench surface without code changes
  - current `QA-EM = 0` on that artifact is not a broken pipeline result
  - it reflects that the current symbolic QA/evaluation surface is still tuned to `Conflict_Resolution`
  - do not use this artifact as a headline quality claim yet

## Current Main Gaps
- `lineage` is still neutral and should not be overclaimed.
- The main unresolved question is no longer overwrite leakage or query-aware parity.
- The dominant remaining bottleneck is still symbolic QA, especially `wrong_anchor_resolution`.
- `end_to_end_extract` is still secondary evidence only, but its report path is now cleanly separated from `oracle_structured`.
- A second evaluation surface exists, but it is not yet headline-usable with the current CR-focused symbolic QA scorer.
- Evaluation is still paper-aligned, not paper-faithful.

## Paper-Aligned Vs Paper-Faithful
- Paper-aligned:
  - symbolic conflict core
  - multi-hop QA over final shared memory
  - local `SubEM`
  - local `FC-SH` / `FC-MH`
- Not paper-faithful:
  - main track is `oracle_structured`
  - evaluator is still local, not official paper scorer
  - repo contribution is a symbolic arbitration system, not a 1:1 reproduction of the original agent setups

## Do Not Undo
- Do not let the LLM become the final judge.
- Do not report structured fallback as true `end_to_end_extract`.
- Do not replace headline QA conflict metrics with only internal memory-state metrics.
- Do not collapse the distinction between query-aware gain and lineage gain.
- Do not claim lineage beats `no_lineage_edges`; it currently does not.
- Do not present the historical `mab8_fc_refresh` win as if it were the current code-state result.

## Exact Rerun Commands
- 2-scenario current ablation:
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
- Extra evaluation surface smoke run:
  - `python app/main.py --benchmark mab --subset Accurate_Retrieval --max-scenarios 2 --use-dummy --modes conflict_aware,lww --enable-error-analysis --emit-scenario-bundles --output-dir reports\paper_mode_mab2_accurate_retrieval_public_v1`

## Immediate Priority Queue
1. Keep the overwrite-safe path stable and do not regress `scenario_accuracy` / `action_accuracy`.
2. Treat `query-aware` as a supported claim.
3. Treat `lineage` as exploratory until it beats `no_lineage_edges`.
4. Continue reducing `wrong_anchor_resolution`.
5. Keep separating arbitration correctness from QA bottlenecks in research-facing summaries.
