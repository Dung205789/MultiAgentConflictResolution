## ProjectMem Session Handoff - 2026-05-22

### What changed

- Fixed long-running benchmark report/finalize instability on Windows redirected runs.
  - File: `src/evaluation/run_evaluation.py`
  - Change: `_use_tqdm()` now disables tqdm when stderr is not a TTY.
  - Reason: redirected/background benchmark runs could end in invalid console/handle states.

- Implemented persistent extraction cache for the secondary `end_to_end_extract` path.
  - File: `src/local_models/runner.py`
  - Cache key: `model_name + prompt_version + raw_text`
  - Backends covered: `transformer`, `gemini_api`, `openai_api`
  - Storage: append-only JSONL
  - Default cache path when not provided:
    - `reports/extraction_cache/openai_api_gpt-4o-mini.jsonl`
    - similar pattern for other models

- Implemented 2-phase secondary workflow in CLI.
  - File: `app/main.py`
  - New flags:
    - `--extraction-cache-path`
    - `--warm-extraction-cache-only`
    - `--warm-extraction-cache-before-run`
  - Added unique-text collection across scenario writes before warmup.

- Added tests for cache persistence and unique-text dedupe.
  - File: `tests/test_local_model_runner.py`
  - Verified:
    - `python -m pytest tests/test_local_model_runner.py -q` -> `5 passed`
    - `python -m py_compile app/main.py src/local_models/runner.py` -> pass

### Verified benchmark state

- Dummy 8-scenario `conflict_aware` final report now exists and is the current verified symbolic reference:
  - `reports/dummy_mab8_metricprobe_v1/mab_conflict_report.json`

- Key verified dummy metrics:
  - `scenario_accuracy = 0.25`
  - `conflict_f1 = 1.0`
  - `action_accuracy = 0.9993985565`
  - `qa_exact_match = 0.815`
  - `final_memory_f1 = 0.9994250303`

- Interpretation:
  - arbitration/action logic is not globally broken
  - remaining gap is mainly QA/state-match mismatch, not conflict detection collapse

### Important runtime facts

- MAB Conflict_Resolution write counts:
  - `s0 = 455`
  - `s1 = 2310`
  - `s2 = 4580`
  - `s3 = 18332`
  - `s4 = 455`
  - `s5 = 2310`
  - `s6 = 4580`
  - `s7 = 18332`

- Scenario pairs are exact duplicate raw-text families:
  - `s0 <-> s4`
  - `s1 <-> s5`
  - `s2 <-> s6`
  - `s3 <-> s7`

- Measured dedupe leverage:
  - full 8 scenarios: `51354` writes but only `18336` unique `raw_text`
  - `s0,s4,s1,s5`: `5530` writes but only `2310` unique `raw_text`

### Blocker hit today

- Fresh OpenAI rerun is blocked by quota, not by repo code.
- Fresh `s0` run failed on 2026-05-22 with:
  - `HTTP 429 rate_limit_exceeded`
  - message indicates `gpt-4o-mini` requests-per-day limit reached
- Evidence:
  - `reports/openai_gpt4omini_s0_verifylogic_v1/run.err.log`
  - `reports/openai_gpt4omini_s0_verifylogic_v1/summary.json`

### Old single-scenario artifacts worth comparing after fresh rerun

- `reports/openai_gpt4omini_s1_overfitfix_v1/custom_report.json`
- `reports/openai_gpt4omini_s4_verify_optv1/custom_report.json`
- `reports/openai_gpt4omini_s5_verify_optv1/custom_report.json`

### Exact next steps for tomorrow

1. Warm cache for `s0` only, then run `s0`.

```powershell
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s0_statefix_v2\mab_conflict_s0.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --warm-extraction-cache-only --output-dir reports\openai_gpt4omini_s0_warm_v1
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s0_statefix_v2\mab_conflict_s0.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --output-dir reports\openai_gpt4omini_s0_rerun_v1
```

2. Run `s4` next. It should mostly hit the same cache family as `s0`.

```powershell
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s4_verify_optv1\mab_conflict_s4.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --output-dir reports\openai_gpt4omini_s4_rerun_v1
```

3. Warm/run `s1`, then run `s5`.

```powershell
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s1_overfitfix_v1\mab_conflict_s1.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --warm-extraction-cache-only --output-dir reports\openai_gpt4omini_s1_warm_v1
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s1_overfitfix_v1\mab_conflict_s1.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --output-dir reports\openai_gpt4omini_s1_rerun_v1
python app/main.py --benchmark custom --custom-path reports\openai_gpt4omini_s5_verify_optv1\mab_conflict_s5.jsonl --track end_to_end_extract --agent1-model gpt-4o-mini --agent2-model gpt-4o-mini --output-dir reports\openai_gpt4omini_s5_rerun_v1
```

4. After `s0,s4,s1,s5` finish, compare:
  - `qa_exact_match`
  - `qa_subem`
  - `fc_mh_accuracy`
  - `final_memory_f1`
  - `action_accuracy`
  - `fallback_contamination_detected`
  - error channels in `error_analysis`

5. Only after those 4 look good, extend to additional benchmark families.
  - Goal: show improvements are not only benchmark-pair specific.

### Do not forget

- For `openai_api`, local GPU does not materially speed up runtime.
- The big speed win here is persistent cache reuse across duplicate scenario families.
- If quota is still tight tomorrow, do not run `s3/s7` early.
