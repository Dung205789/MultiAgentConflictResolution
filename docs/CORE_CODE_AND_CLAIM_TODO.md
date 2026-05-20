# Core Code And Claim TODO

## Goal
Turn the current analysis into an execution queue that is honest about:
- what code was improved
- what benchmark evidence was gained
- what still blocks a publication-grade result

## What Was Completed In This Pass
- Fixed query-aware edge propagation in `src/conflict/query_aware_context.py`.
- Made `compatible_extension` reachable in `src/conflict/conflict_detector.py`.
- Reduced the worst benchmark-parser corruption in:
  - `src/benchmarks/memoryagentbench_loader.py`
  - `src/benchmarks/adapters/memab_adapter.py`
- Added runtime observability:
  - scenario start logs
  - `*.progress.json`
  - `*.partial.json`
- Added scenario-vs-QA diagnostics in `src/evaluation/run_evaluation.py`.
- Added parser regression tests in `tests/test_memoryagentbench_loader.py`.
- Repaired benchmark overwrite behavior in `src/conflict/conflict_aware_writer.py` for:
  - `compatible_extension`
  - `potential_contradiction`
  under `memoryagentbench_Conflict_Resolution`
- Strengthened the general QA reasoner in `src/evaluation/qa_reasoner.py` by:
  - removing overbroad relation keyword matches
  - using relation-chain alignment during fallback path scoring
  - improving expected-type inference
- Verified full test suite:
  - `python -m unittest discover -s tests -p "test*.py"`
  - current status: `36` tests passing

## Current Benchmark Truth

### Historical best full symbolic artifact by QA advantage
- `reports/paper_mode_mab8_fc_refresh/mab_conflict_report.json`
- `conflict_aware qa_exact_match = 0.44125`
- `lww qa_exact_match = 0.43125`
- `conflict_aware qa_subem = 0.44750`
- `lww qa_subem = 0.43750`
- `scenario_accuracy = 1.0`
- `action_accuracy = 1.0`

### Latest full code-state now beats `lww`
- `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
- `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
- `conflict_aware_full qa_exact_match = 0.73000`
- `lww qa_exact_match = 0.68750`
- `conflict_aware_full qa_subem = 0.73000`
- `lww qa_subem = 0.69000`
- `conflict_aware_full fc_sh_accuracy = 0.85460`
- `lww fc_sh_accuracy = 0.83976`
- `conflict_aware_full fc_mh_accuracy = 0.69613`
- `lww fc_mh_accuracy = 0.59669`
- `conflict_aware_full scenario_accuracy = 1.0`
- `conflict_aware_full action_accuracy = 1.0`

### Query-aware isolation is now measured
- `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
- `conflict_aware_no_query_support qa_exact_match = 0.68750`
- `conflict_aware_no_query_support qa_subem = 0.69000`
- `conflict_aware_no_query_support fc_sh_accuracy = 0.83976`
- `conflict_aware_no_query_support fc_mh_accuracy = 0.59669`

### Latest repaired 2-scenario ablation
- `reports/paper_mode_mab2_queryaware_gain_v5/mab_conflict_report.json`
- `conflict_aware_full qa_exact_match = 0.775`
- `conflict_aware_no_lineage_edges qa_exact_match = 0.775`
- `conflict_aware_no_query_support qa_exact_match = 0.685`
- `lww qa_exact_match = 0.685`

### Latest lineage isolation run
- `reports/paper_mode_mab8_queryaware_gain_v2_nolineage/mab_conflict_report.json`
- `conflict_aware_no_lineage_edges qa_exact_match = 0.73000`
- `conflict_aware_no_lineage_edges qa_subem = 0.73000`
- `conflict_aware_no_lineage_edges fc_sh_accuracy = 0.85460`
- `conflict_aware_no_lineage_edges fc_mh_accuracy = 0.69613`

## Priority 0: Keep Benchmark Safety Stable
- [x] Eliminate the remaining `potential_contradiction -> reject` leakage on MAB CR.
- [x] Eliminate the remaining `compatible_extension -> keep_multiple_versions` leakage on MAB CR.
- [x] Trace and fix the highest-volume residual predicates enough to restore benchmark-safe overwrite behavior.
- [x] Add benchmark-specific tests that assert repaired MAB CR conflict paths prefer overwrite.

## Priority 1: Prove Or Downgrade Query-Aware Claims
- [x] Make `full` beat or at least match `no_lineage_edges` on a repaired 2-scenario run.
- [x] Remove lineage-edge contamination from live QA graph construction.
- [x] Keep `query-aware` as a live claim because it now produces measured gain beyond `no_query_support`.
- [ ] Stop treating lineage preservation as a main claim unless it beats `no_lineage_edges`.

## Priority 2: Separate Arbitration Failure From QA Failure
- [x] Add `scenario_diagnostics` to the report.
- [ ] Add a research-facing summary that explicitly distinguishes:
  - final memory mismatch
  - arbitration mismatch
  - QA failure on otherwise correct memory
- [x] Re-run error analysis on the repaired benchmark path after the next core fix.
- [ ] Cut the remaining `wrong_anchor_resolution` burden, which is still `211` on the latest full query-aware run.

## Priority 3: Throughput And Observability
- [x] Progress file during long runs.
- [x] Partial report during long runs.
- [ ] Add per-scenario partial result snapshots, not only per-mode partials.
- [ ] Reduce `naive` full-run latency, which is still dominated by branch explosion in the QA path.

## Priority 4: Claim Discipline
- [x] Main track remains `oracle_structured`.
- [x] Secondary track remains `end_to_end_extract`.
- [x] Fallback contamination remains explicit.
- [x] Keep the paper claim anchored to verified artifacts, not chat narrative.
- [x] Update claim boundary:
  - `query-aware` is now supported by measured gain
  - `lineage` is still exploratory

## Working Conclusion
This repo is now easier to debug, more truthful in how it ingests benchmark facts, and stronger on the latest benchmark-safe path.

The main symbolic benchmark path now beats `lww` again, and the measured gain comes from `query-aware` metadata rather than from lineage.

The next real code task is now unambiguous:
- keep the repaired overwrite path stable
- keep `query-aware` as a supported claim
- demote `lineage` unless it gains evidence
- reduce the remaining `wrong_anchor_resolution` failures
