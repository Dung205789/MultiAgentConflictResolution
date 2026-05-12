# PROJECT STATUS SUMMARY
# Generated for session handoff - new agent should read this and continue

## Current Date
2026-05-11

## Project Overview
Rule-based multi-agent memory conflict resolution system using deterministic rules (no LLMs in critical path).

## Key Components
- **Entry point**: `main.py`
- **ISF format**: `src/format.py` (Scenario, Event, MemoryEntry, Query)
- **Arbitration config**: `configs/arbitration.yaml`
- **Adapters**: `src/benchmarks/adapters/` (memab_adapter.py, memoryagentbench_loader.py)

---

## Changes Made in This Session

### 1. MemoryAgentBench Loader Fix (`src/benchmarks/memoryagentbench_loader.py`)
- **Problem**: Used non-existent dataset ID `THUDM/MemoryAgentBench`
- **Fix**: Changed to correct ID `ai-hyz/MemoryAgentBench`
- **Added**: Entity extraction function `_extract_entity_and_predicate()` to extract (entity, predicate, object) from facts
- **Added**: Conflict detection `_detect_conflicts_in_facts()` using entity-predicate grouping

### 2. Main.py Benchmark Selection (`main.py`)
- **Added new benchmark options**:
  - `real_conflicts` = MemAE + MAB Conflict_Resolution
  - `mab_conflict` = MemoryAgentBench Conflict_Resolution only
- **Changed default** from `"all"` to `"real_conflicts"`
- **"all" now excludes adversarial** benchmark (kept for ablation studies but not used)

### 3. MemAB Adapter Fix (`src/benchmarks/adapters/memab_adapter.py`)
- **Problem**: Memory entries used "fact_0", "fact_1" as subjects → conflict detection always 0%
- **Fix**: Implemented `_extract_entity_and_predicate()` to extract real entities from fact text
- **Updated**: `convert_row_to_scenario()` to:
  - Build `fact_entities` list with (entity, predicate, object)
  - Use entity as subject in events instead of "fact_X"
  - Build gold state with proper superseding for conflicts (latest wins per entity-predicate)
- **Fixed**: `_analyze_facts_for_conflicts()` signature to accept `fact_entities` instead of `events`

---

## Current Status

### Completed Tasks
- ✅ Revamp benchmark system to use online datasets
- ✅ Fix MemoryAgentBench loader
- ✅ Update main.py benchmark selection
- ✅ Fix memab_adapter entity extraction for conflict detection

### In Progress
- ⏳ Verify MemAE and MAB conflict metrics are now non-zero
- ⏳ Run full evaluation on `real_conflicts`

### Pending Tasks
- [ ] Review and improve LoCoMo adapter (if needed)
- [ ] Tune rule-based detection thresholds if conflict detection still low
- [ ] Document findings and update README if necessary

---

## Known Issues

### From Previous Run (memae benchmark)
```
Results showed:
- Conflict F1: 0.000
- Action accuracy: 0.000
- total_conflicts: 0
```

**Root cause**: Memory entries had unique subjects ("fact_0", "fact_1", ...) so conflict detector couldn't group facts by entity-predicate pairs.

**Fix applied**: Entity extraction now extracts real subjects from fact text (e.g., "Thomas Kyd" from "Thomas Kyd was born in London").

### Expected After Fix
- Conflict detection F1 should be > 0 for scenarios that have actual conflicts
- Action accuracy should improve (system should take conflict actions)
- Gold memory state should have "superseded" entries where appropriate

---

## Testing Commands

### Quick validation (2 scenarios):
```bash
python main.py --benchmark real_conflicts --max-scenarios 2
```

### Full evaluation:
```bash
python main.py --benchmark real_conflicts
```

### MemoryAgentBench Conflict only:
```bash
python main.py --benchmark mab_conflict --max-scenarios 10
```

### Check reports:
- `reports/memae_report.json` - MemAE detailed results
- `reports/mab_report.json` - MemoryAgentBench results
- `reports/summary.json` - Aggregated comparison

---

## Key Datasets
1. **MemAE**: Local parquet file at `data/raw/memae/`
2. **MemoryAgentBench**: Downloaded from HuggingFace `ai-hyz/MemoryAgentBench`
   - Conflict_Resolution split: ~200 scenarios
   - Long_Range_Understanding: ~200 scenarios
   - Test_Time_Learning: ~200 scenarios
   - Accurate_Retrieval: ~200 scenarios

---

## File Locations
- Main entry: `D:\ProjectMem\main.py`
- MemAB adapter: `D:\ProjectMem\src\benchmarks\adapters\memab_adapter.py`
- MAB loader: `D:\ProjectMem\src\benchmarks\memoryagentbench_loader.py`
- Format definitions: `D:\ProjectMem\src\format.py`
- Config: `D:\ProjectMem\configs\arbitration.yaml`

---

## Notes for Next Agent
1. The core fix is in place - entity extraction now works in both adapters
2. Run the test commands above to verify conflict metrics are non-zero
3. If conflict detection is still low, may need to:
   - Improve entity extraction heuristics
   - Tune conflict thresholds in `_detect_conflicts_in_facts()`
   - Debug conflict_detector.py logic
4. LoCoMo adapter is lower priority - user focused on real conflicts from online datasets
5. Adversarial benchmark code remains but is not included in any standard benchmark run

---

## Next Steps (in order)
1. Run: `python main.py --benchmark real_conflicts --max-scenarios 5`
2. Check `reports/summary.json` for conflict_f1 and action_accuracy values
3. If still 0, debug conflict detection pipeline:
   - Print sample fact_entities from adapters
   - Check if conflict_detector.py properly receives entities
   - Verify predicate matching in conflict grouping
4. Once metrics are working, run full evaluation and document results
