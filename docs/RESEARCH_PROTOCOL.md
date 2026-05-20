# Research Protocol

## Locked Tracks
- Primary track: `oracle_structured`
- Secondary track: `end_to_end_extract`
- Only `ConflictAwareWriter` may decide `overwrite`, `reject`, `merge`, or `commit`.

## Headline Benchmark
- `MemoryAgentBench / Conflict_Resolution`
- Headline metrics: `FC-SH`, `FC-MH`, `SubEM`, then `QA-EM` as supporting QA summary.

## Supporting Metrics
- `scenario_accuracy`
- `action_accuracy`
- `final_memory_f1`
- `fallback_contamination_detected`

## Main Commands
- `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_lineage_edges --output-dir reports\paper_mode_mab8_fc_refresh`
- `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --include-conflict-aware-ablations --enable-error-analysis --output-dir reports\paper_mode_mab2_research_bundle_v1`

## Paper Alignment Boundary
- This repo is paper-aligned, not paper-faithful 1:1.
- `oracle_structured` is the main research contribution track and is intentionally more structured than the original raw end-to-end paper setup.
- `end_to_end_extract` must always be reported separately and may not absorb structured fallback silently.
