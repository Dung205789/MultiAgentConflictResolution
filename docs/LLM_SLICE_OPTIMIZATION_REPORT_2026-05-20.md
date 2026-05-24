# LLM Slice Optimization Report - 2026-05-20

## Scope

Target slice before any full 8-scenario run:

- `s0` = `memoryagentbench_Conflict_Resolution_0`
- `s1` = `memoryagentbench_Conflict_Resolution_1`
- `s4` = `memoryagentbench_Conflict_Resolution_4`
- `s5` = `memoryagentbench_Conflict_Resolution_5`

Execution track:

- `end_to_end_extract`
- model: `gpt-4o-mini` for both agents
- benchmark surface: `MemoryAgentBench / Conflict_Resolution`

## Baseline Artifacts

- `reports/openai_gpt4omini_s0_baseline/custom_report.json`
- `reports/openai_gpt4omini_s1_baseline/custom_report.json`
- `reports/openai_gpt4omini_s4_baseline/custom_report.json`
- `reports/openai_gpt4omini_s5_baseline/custom_report.json`

Baseline summary:

| Scenario | Writes | QA-EM | Action Acc | FC-SH | FC-MH | Final Memory F1 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s0` | 455 | 0.83 | 1.0000 | 1.0000 | 0.8298 | 0.4538 | QA mostly answer-selection failures |
| `s1` | 2310 | 0.73 | 0.9987 | 0.9167 | 0.7093 | 0.4314 | one `reject` under `potential_contradiction` |
| `s4` | 455 | 0.87 | 1.0000 | 0.9324 | 1.0000 | 0.4493 | dominated by direct-question anchor failures |
| `s5` | 2310 | 0.85 | 0.9987 | 0.8780 | 0.8889 | 0.4259 | one `reject` under `potential_contradiction` |

Main baseline failure channels across the slice:

- `overwrite_correct_but_qa_unused = 30`
- `wrong_anchor_resolution = 16`
- `wrong_reverse_relation = 14`
- `parser_no_edge = 8`

## Code Changes

### 1. QA logic

File:

- `src/evaluation/qa_reasoner.py`

Changes:

- added direct-question templates for high-frequency missing forms:
  - `current head of the ... government`
  - `Who performed ...`
  - `Who is the developer of ...`
  - `Who is the employer of ...`
  - `Who is the author of ...`
  - `Who is ... married to`
  - `What is the country of citizenship of ...`
  - `What language does ... speak`
  - `Which religion is ... affiliated with`
  - `Which country was ... created in`
  - `Which city did ... work in`
  - `Which city did ... die in`
  - `Which city is the headquarter of ... located in`
  - `What is the capital of ...`
  - `Who is the chief executive officer of ...`
- added raw statement parsing for:
  - `The name of the current head of the ... government is ...`
- added symbolic relation support for `government_head`
- mapped `head` to `government_head` as a narrow recovery alias

### 2. Arbitration logic

File:

- `src/conflict/conflict_aware_writer.py`

Change:

- in `potential_contradiction`, near-tie handling now uses:
  - `abs(margin) <= keep_margin + 1e-9`

Reason:

- the failing `reject` cases in `s1` and `s5` had:
  - `margin = -0.020000000000000018`
  - `keep_multiple_versions_margin = 0.02`
- without the epsilon, these fell just outside the near-tie branch because of floating-point drift.

## Fast Probe Validation

Before full reruns, direct symbolic probes on baseline bundles confirmed that the patched reasoner now resolves these previously failing questions correctly:

- `What is the name of the current head of the Tucson government?`
- `Who performed Meredith Grey?`
- `Who is the developer of Windows Phone?`
- `Who is the author of David Copperfield?`
- `Who is Igor of Kiev married to?`
- `What is the country of citizenship of Wilhelm von Humboldt?`

## Full Verification Runs

Verified rerun artifacts:

- `reports/openai_gpt4omini_s4_verify_optv1/custom_report.json`
- `reports/openai_gpt4omini_s5_verify_optv1/custom_report.json`
- `reports/openai_gpt4omini_s1_verify_optv1/custom_report.json`

### Verified deltas

| Scenario | QA-EM Before | QA-EM After | Delta | Action Acc Before | Action Acc After | Key outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `s1` | 0.73 | 0.74 | +0.01 | 0.9987 | 1.0000 | writer fix removed the `reject`, QA still bottlenecked by reverse-relation / answer-selection |
| `s4` | 0.87 | 0.96 | +0.09 | 1.0000 | 1.0000 | direct-question patch worked strongly |
| `s5` | 0.85 | 0.92 | +0.07 | 0.9987 | 1.0000 | direct-question patch + writer fix both helped |

Additional verified deltas:

- `s4`
  - `FC-SH: 0.9324 -> 0.9545`
  - `Final Memory F1: 0.4493 -> 0.4515`
  - remaining error summary:
    - `overwrite_correct_but_qa_unused = 3`
    - `answer_type_mismatch = 1`
- `s5`
  - `FC-SH: 0.8780 -> 0.9213`
  - `FC-MH: 0.8889 -> 1.0000`
  - `Final Memory F1: 0.4259 -> 0.4293`
  - remaining error summary:
    - `overwrite_correct_but_qa_unused = 7`
    - `wrong_anchor_resolution = 1`
- `s1`
  - `FC-SH: 0.9167 -> 1.0000`
  - `FC-MH: 0.7093 -> 0.7059`
  - `Final Memory F1: 0.4314 -> 0.4283`
  - remaining error summary:
    - `overwrite_correct_but_qa_unused = 11`
    - `wrong_reverse_relation = 7`
    - `parser_no_edge = 5`

## Interpretation

### What is fixed well

- The direct-question QA surface is materially better.
- The `potential_contradiction` threshold bug that created the repeated `reject` in `s1` and `s5` is resolved.
- `s4` and `s5` now show clear end-to-end gains under the real LLM extraction path.

### What is still weak

- `scenario_accuracy` remains `0.0` on all verified runs.
- `final_memory_f1` is still low across the slice, around `0.428` to `0.452`.
- `s1` remains the hardest case and is still bottlenecked by:
  - reverse-relation recovery
  - multi-hop answer selection
  - parser / graph edge misses

## Recommendation

Do **not** run the full 8-scenario headline pass yet.

Reason:

- `s4` and `s5` are meaningfully better, but `s1` is still not strong enough.
- The current patch cleaned direct-template failures and the floating-threshold writer bug, but it did not solve the deeper multi-hop reverse-relation weakness.
- Running all 8 now would likely spend a lot of time and money while still producing a headline artifact with an avoidable weak spot.

## Suggested next step before full 8

Focus one more optimization loop on `s1`-style failures:

- improve reverse-relation handling for direct factual questions that should not fall into wrong-direction traversal
- tighten answer selection on multi-hop chains where the correct relation path exists but the wrong reachable answer wins
- reduce `parser_no_edge` on founder / religion / government-chain questions

Practical gate before full 8:

- rerun `s1` after the next QA pass
- only move to full 8 when `s1` materially improves beyond the current `QA-EM = 0.74`

