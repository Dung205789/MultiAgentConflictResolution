# Public Release Guide

## Goal
Ship the repo in a way that is:
- truthful
- reproducible
- hard to misread

## What To Point Readers To First
1. `README.md`
2. `ZERO_CONTEXT_PROJECT_LOCK.md`
3. `docs/PROJECT_LOCK.md`
4. `docs/RESEARCH_RESULTS.md`

## Which Artifact To Cite
Use two different labels and keep them separate.

Historical best by QA advantage:
- `reports/paper_mode_mab8_fc_refresh/mab_conflict_report.json`

Latest full code-state:
- `reports/paper_mode_mab8_queryaware_gain_v2_conflictonly/mab_conflict_report.json`
- `reports/paper_mode_mab8_queryaware_gain_v2_lwwonly/mab_conflict_report.json`
- `reports/paper_mode_mab8_queryaware_gain_v2_noquery/mab_conflict_report.json`
- `reports/paper_mode_mab8_queryaware_gain_v2_nolineage/mab_conflict_report.json`

Latest public-ready 2-scenario split artifacts:
- `reports/paper_mode_mab2_pipeline_public_conflictonly_v2/mab_conflict_report.json`
- `reports/paper_mode_mab2_pipeline_public_lwwonly_v1/mab_conflict_report.json`
- `reports/paper_mode_mab2_accurate_retrieval_public_v1/mab_report.json`

Do not merge the historical-best story with the latest-code-state story.

## What To Claim
Allowed:
- symbolic shared-memory conflict resolution
- judge-free arbitration
- benchmark-safe pipeline
- `conflict_aware > lww` on the latest full code-state
- measured query-aware gain beyond `no_query_support`
- strong QA improvements over earlier repaired builds

Not allowed:
- lineage is already proven as the source of gain
- full paper-faithful replication

## Public Repro Commands
Main full `conflict_aware` run:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_conflictonly --enable-error-analysis --emit-scenario-bundles
```

LWW baseline:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant full --modes lww --output-dir reports\paper_mode_mab8_queryaware_gain_v2_lwwonly --enable-error-analysis --emit-scenario-bundles
```

Query-aware isolation:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_query_support --modes conflict_aware --output-dir reports\paper_mode_mab8_queryaware_gain_v2_noquery --enable-error-analysis --emit-scenario-bundles
```

Ablation run:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes conflict_aware --output-dir reports\paper_mode_mab2_pipeline_public_conflictonly_v2 --enable-error-analysis --emit-scenario-bundles
python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --modes lww --output-dir reports\paper_mode_mab2_pipeline_public_lwwonly_v1 --enable-error-analysis --emit-scenario-bundles
```

Extra-surface portability run:
```powershell
python app/main.py --benchmark mab --subset Accurate_Retrieval --max-scenarios 2 --use-dummy --modes conflict_aware,lww --enable-error-analysis --emit-scenario-bundles --output-dir reports\paper_mode_mab2_accurate_retrieval_public_v1
```

Replay QA from artifact:
```powershell
python scripts/replay_qa_from_report.py reports\paper_mode_mab8_queryaware_gain_v2_conflictonly\mab_conflict_report.json.scenarios
```

Finalize from saved scenario bundles:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --finalize-report-from-artifacts reports\paper_mode_mab8_queryaware_gain_v2_conflictonly\mab_conflict_report.json
```

Secondary-track real-model runs:
```powershell
python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_realmodel_v1 --enable-error-analysis --emit-scenario-bundles
python app/main.py --benchmark mab_conflict --max-scenarios 2 --track end_to_end_extract --agent1-model Qwen/Qwen2.5-3B-Instruct --agent2-model Qwen/Qwen2.5-3B-Instruct --allow-structured-fallback-in-end-to-end --modes conflict_aware --output-dir reports\paper_mode_mab2_end_to_end_extract_fallback_v1 --enable-error-analysis --emit-scenario-bundles
```

## Release Checklist
- root `README.md` present
- lock docs current
- report artifact paths current
- scenario bundles emitted
- failure bundle emitted
- replay script works
- mode metrics expose:
  - `reported_track_name`
  - `pure_end_to_end_extract`
  - `structured_fallback_present`
- tests pass
