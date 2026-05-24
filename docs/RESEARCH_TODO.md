# Research TODO

## Done
- [x] Lock the project to rule-based shared-memory conflict resolution.
- [x] Keep `oracle_structured` as the main track and `end_to_end_extract` as secondary.
- [x] Enforce LLM proposal-only behavior.
- [x] Unify the canonical memory model and lifecycle/store API.
- [x] Add `SubEM`, `FC-SH`, and `FC-MH` to local evaluation.
- [x] Remove silent fallback contamination.
- [x] Add progress, partial, failure-bundle, and scenario-bundle artifacts.
- [x] Add report-level track separation for:
  - `oracle_structured`
  - `end_to_end_extract`
  - `end_to_end_extract__structured_fallback`
- [x] Repair benchmark-safe overwrite behavior.
- [x] Restore `conflict_aware > lww` on the latest full 8-scenario code-state.
- [x] Isolate `query-aware` gain beyond `no_query_support`.
- [x] Freeze a pre-LLM hardening code-state with verified 2-scenario ablations before spending on secondary-track model runs.
- [x] Expose explicit `raw_state_match`, `raw_memory_f1`, `canonical_state_match`, and `canonical_memory_f1` in research-facing reports.

## Remaining Must-Have
- [ ] Decide the final claim for `lineage`.
  Current evidence: `full == no_lineage_edges`, so lineage is still exploratory.
- [ ] Reduce the remaining `wrong_anchor_resolution` burden.
  Current evidence from latest full run: `full = 211`, `no_query_support = 244`, `lww = 244`.
  Current evidence from latest 2-scenario public-ready split runs: `conflict_aware = 2`, `lww = 20`.
- [ ] Add a research-facing summary that cleanly separates:
  - arbitration correct / QA wrong
  - arbitration wrong / QA wrong
  - answer-type mismatch
  - anchor-resolution failure
- [x] Make `end_to_end_extract` a clean secondary evidence track at the reporting/contract layer.
- [ ] Run and publish real-model `end_to_end_extract` artifacts as secondary evidence.
- [x] Add at least one extra evaluation surface beyond the local `MemoryAgentBench / Conflict_Resolution` split.
  Current artifact: `reports\paper_mode_mab2_accurate_retrieval_public_v1\mab_report.json`
  Caveat: the current symbolic QA/eval layer is still CR-shaped, so this is portability evidence, not a headline score claim.

## Remaining Nice-To-Have
- [ ] Add lifecycle-specific ablations:
  - supersession off
  - visibility gating off
  - snapshot/version semantics off
- [ ] Add a relation registry/config layer to reduce remaining benchmark-shaped logic.
- [ ] Add a smaller QA replay / per-scenario debug summary that aggregates top anchor-failure patterns automatically.
- [ ] Move local `SubEM` / `FC-*` scorers closer to paper-faithful official semantics.

## Claim Boundary
- Safe to claim:
  - symbolic judge-free arbitration
  - benchmark-safe pipeline
  - `conflict_aware > lww` on latest full code-state
  - `query-aware > no_query_support`
  - secondary-track high memory scores are often canonicalized, not raw-state-perfect
- Not yet safe to claim:
  - lineage gives measured gain
  - full paper-faithful replication
  - broad generalization outside current benchmark surface
  - strict `end_to_end_extract` portability to `LongMemEval` or `LoCoMo` on the current loader/eval contract

## Current Best Commands
- Full `conflict_aware`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_conflictonly --enable-error-analysis --emit-scenario-bundles`
- Full `lww`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes lww --output-dir reports\paper_mode_mab8_queryaware_gain_v2_lwwonly --enable-error-analysis --emit-scenario-bundles`
- Full `no_query_support`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_query_support --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_noquery --enable-error-analysis --emit-scenario-bundles`
- Current 2-scenario ablation:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes conflict_aware --output-dir reports\paper_mode_mab2_pipeline_public_conflictonly_v2 --enable-error-analysis --emit-scenario-bundles`
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes lww --output-dir reports\paper_mode_mab2_pipeline_public_lwwonly_v1 --enable-error-analysis --emit-scenario-bundles`
- Pre-LLM freeze:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes conflict_aware --output-dir reports\paper_mode_mab2_prellm_hardening_v1 --enable-error-analysis --emit-scenario-bundles`
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --include-conflict-aware-ablations --enable-error-analysis --emit-scenario-bundles --output-dir reports\paper_mode_mab2_prellm_ablation_v1`
- Secondary-track pure `end_to_end_extract`:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_realmodel_v1 --enable-error-analysis --emit-scenario-bundles`
- Secondary-track fallback-labeled:
  - `python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --allow-structured-fallback-in-end-to-end --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_fallback_v1 --enable-error-analysis --emit-scenario-bundles`
- Extra surface smoke run:
  - `python app/main.py --benchmark mab --subset Accurate_Retrieval --max-scenarios 2 --use-dummy --modes conflict_aware,lww --enable-error-analysis --emit-scenario-bundles --output-dir reports\paper_mode_mab2_accurate_retrieval_public_v1`
