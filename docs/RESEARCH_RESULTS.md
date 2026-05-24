# Research Results

## Historical Best Full Artifact By QA Advantage
- Source: `reports/paper_mode_mab8_fc_refresh/mab_conflict_report.json`
- Track: `oracle_structured`
- Variant: `no_lineage_edges`
- Why it still matters:
  - it remains the strongest verified full 8-scenario symbolic result
  - it preserves `scenario_accuracy = 1.0`
  - it preserves `action_accuracy = 1.0`

| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_aware | 0.44125 | 0.44750 | 0.51905 | 0.43595 | 1.000 | 1.000 |
| lww | 0.43125 | 0.43750 | 0.49524 | 0.43021 | 1.000 | 1.000 |
| naive | 0.04125 | 0.05125 | 0.10000 | 0.01912 | 0.000 | 0.000 |

## Latest Full Code-State
- Sources:
  - `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
  - `reports/paper_mode_mab8_queryaware_gain_v2_nolineage/mab_conflict_report.json`
- Purpose:
  - validate the repaired, benchmark-safe code-state
  - measure query-aware gain directly against `lww` and `no_query_support`
  - check whether lineage helps beyond `no_lineage_edges`

| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_aware_full | 0.73000 | 0.73000 | 0.85460 | 0.69613 | 1.000 | 1.000 |
| conflict_aware_no_lineage_edges | 0.73000 | 0.73000 | 0.85460 | 0.69613 | 1.000 | 1.000 |
| conflict_aware_no_query_support | 0.68750 | 0.69000 | 0.83976 | 0.59669 | 1.000 | 1.000 |
| lww | 0.68750 | 0.69000 | 0.83976 | 0.59669 | 1.000 | 1.000 |

## Latest 2-Scenario Repair Ablation
- Sources:
  - `reports/paper_mode_mab2_pipeline_public_conflictonly_v2/mab_conflict_report.json`
  - `reports/paper_mode_mab2_pipeline_public_lwwonly_v1/mab_conflict_report.json`
- Purpose:
  - verify the public-ready pipeline path after the latest anchor-resolution fixes
  - measure the new QA bottleneck counts on clean split-mode artifacts

| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_aware | 0.87500 | 0.87500 | 1.00000 | 0.86667 | 1.000 | 1.000 |
| lww | 0.79500 | 0.79500 | 0.81250 | 0.79330 | 1.000 | 1.000 |

## Pre-LLM Hardening Freeze
- Sources:
  - `reports/paper_mode_mab2_prellm_hardening_v1/mab_conflict_report.json`
  - `reports/paper_mode_mab2_prellm_ablation_v1/mab_conflict_report.json`
- Purpose:
  - freeze the symbolic code-state before expensive `end_to_end_extract` LLM runs
  - verify that the latest generalization sweep does not regress benchmark correctness
  - re-check that `query-aware` still beats `no_query_support` after rule and QA hardening

| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict_aware_full | 0.87500 | 0.87500 | 1.00000 | 0.86667 | 1.000 | 1.000 |
| conflict_aware_no_lineage_edges | 0.87500 | 0.87500 | 1.00000 | 0.86667 | 1.000 | 1.000 |
| conflict_aware_no_query_support | 0.80000 | 0.80000 | 0.87500 | 0.79444 | 1.000 | 1.000 |
| lww | 0.80000 | 0.80000 | 0.87500 | 0.79444 | 1.000 | 1.000 |

- Code-state changes in this freeze:
  - generalized overwrite-first handling for terse entity-slot conflicts in `src/conflict/conflict_aware_writer.py`
  - stronger anchor inference, edge deduplication, and relation-follow ranking in `src/evaluation/qa_reasoner.py`
  - explicit query-aware graph edges now outrank lower-priority duplicate edges during answer-time traversal

## What The Repair Changed
- The benchmark parser no longer truncates facts at the first `.` inside abbreviations or initials.
- Example classes fixed:
  - `Apple Inc.`
  - `Ursula K. Le Guin`
  - `Robert A. Heinlein`
  - `Washington, D.C.`
- The runner now writes:
  - `mab_conflict_report.json.progress.json`
  - `mab_conflict_report.json.partial.json`
  so long runs no longer appear completely opaque.

## Current Diagnostic Picture
- The overwrite-heavy benchmark regression is repaired on the current path.
- Key evidence from the latest full runs:
  - `scenario_accuracy = 1.0`
  - `action_accuracy = 1.0`
  - `fallback_contamination_detected = false`
  - `conflict_aware_full > lww`
  - `conflict_aware_full > no_query_support`
  - `conflict_aware_full == no_lineage_edges`
- Failure-bundle summary now shows:
  - `full`: `wrong_anchor_resolution = 211`
  - `no_query_support`: `wrong_anchor_resolution = 244`
  - `lww`: `wrong_anchor_resolution = 244`
- Latest 2-scenario public-ready split artifacts show:
  - `conflict_aware`: `wrong_anchor_resolution = 2`
  - `lww`: `wrong_anchor_resolution = 20`
- Remaining unresolved anchor cases on the latest 2-scenario `conflict_aware` split are now only:
  - `What is the name of the political entity that was established from the successor of the Sasanian Empire?`
  - `Who currently holds the title of head of state of the country where David Graveney was born?`
- The current blocker is no longer parity. The current blocker is the remaining anchor-resolution burden and the still-neutral lineage path.
- Public-facing debug artifacts are now available from benchmark runs when enabled:
  - `mab_conflict_report.json.failure_bundle.json`
  - `mab_conflict_report.json.scenarios/`
  - `python scripts/replay_qa_from_report.py <report>.scenarios`
  - `reported_track_name`, `pure_end_to_end_extract`, and `structured_fallback_present` in mode metrics

## Secondary-Track Reading Discipline
- Real-model `end_to_end_extract` reports now expose both:
  - `raw_state_match` / `raw_memory_f1`
  - `canonical_state_match` / `canonical_memory_f1`
- Interpretation rule:
  - treat `scenario_accuracy` as an alias of `canonical_state_match`
  - treat `final_memory_f1` as an alias of `canonical_memory_f1`
  - do not describe high secondary-track memory scores as raw-state-perfect unless `raw_state_match = 1.0`
- Current CR rerun evidence shows why this matters:
  - `s0`: `raw_state_match = 0.0`, `canonical_state_match = 1.0`
  - `s4`: `raw_state_match = 0.0`, `canonical_state_match = 1.0`
  - `s1`: `raw_state_match = 0.0`, `canonical_state_match = 0.0`, but `canonical_memory_f1 = 0.9996592845`
  - `s2`: `raw_state_match = 0.0`, `canonical_state_match = 0.0`, but `canonical_memory_f1 = 0.9998263587`
  - `s5`: `raw_state_match = 0.0`, `canonical_state_match = 0.0`, but `canonical_memory_f1 = 0.9996592845`
- Therefore:
  - the secondary track is not globally broken
  - arbitration is usually holding up
  - raw exactness is still materially below canonical exactness on the larger CR families
  - QA bottlenecks still dominate the non-passing slices

## Extra-Surface Contract Boundary
- `LongMemEval` is not currently clean strict-`end_to_end_extract` evidence:
  - loader emits `write_proposal` events with predicate `utterance`
  - gold visible state is the utterance surface itself
  - a strict extractor would be graded against a different state representation than what it is asked to produce
- `LoCoMo` is not currently clean strict-`end_to_end_extract` evidence:
  - loader writes conversation turns as `utterance` plus `session_date`
  - gold visible state mixes `session_date` with `session_event` summaries
  - strict extraction would again be evaluated against a mismatched target state surface
- Conclusion:
  - keep these surfaces as loader/runtime portability work for now
  - do not use them as headline strict secondary evidence until the extraction target and gold-state contract are aligned

## Interpretation
- The parser repair was worth doing:
  - runtime became observable
  - benchmark text ingestion became materially cleaner
  - the repaired full path is benchmark-safe again
- The lineage-edge QA fix was also necessary:
  - it removed spurious multi-hop shortcuts from overwritten-candidate metadata
  - it restored `full` to parity with `no_lineage_edges` and `lww`
- The generalized QA reasoner update was also worth doing:
  - `QA-EM` on the repaired full run rose from `0.47000` to `0.65000`
  - `FC-SH` rose from `0.51429` to `0.79399`
  - `FC-MH` rose from `0.46654` to `0.61972`
- The latest code-state now does beat `lww`, and it does so without losing benchmark correctness.
- The query-aware story is now experimentally isolated:
  - `full > no_query_support`
  - `full > lww`
- The lineage story is still not isolated:
  - `full == no_lineage_edges`
- A second evaluation surface now exists in the repo:
  - `reports/paper_mode_mab2_accurate_retrieval_public_v1/mab_report.json`
  - current `QA-EM = 0` there should be read as a benchmark/evaluator mismatch, not as pipeline failure
  - the current symbolic QA evaluator remains optimized for `Conflict_Resolution`-style compositional questions
- The main unresolved blocker is now:
  - reduce the remaining `wrong_anchor_resolution` burden without regressing the new query-aware gain
