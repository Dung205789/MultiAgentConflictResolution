# Project Progress Summary

**Date**: 2026-05-12
**Status**: Code cleanup + Colab notebook ready for full benchmark run

---

## Recent Changes (This Session)

### 1. Entity Extraction Fix (MemAB Adapter)
- **File**: `src/benchmarks/adapters/memab_adapter.py`
- Improved `_extract_entity_and_predicate()` to handle raw_statements better
- Added pattern matching for "X of Y is Z" structure
- Entity extraction now uses full context (e.g., "The chairperson Fatah" instead of just "The chairperson")
- **Result**: Gold state alignment improved (expected active: 278, actual: 276 - only 2 difference)

### 2. Code Cleanup
Removed temporary/debug files:
- Debug scripts: `debug_entity_extraction.py`, `inspect_*.py`, `count_*.py`
- Test outputs: `*_output.txt`, `test_*.json`, `sample_*.json`
- Log files: `*.log` (memae_*, full_*, eval_*)
- Test scripts: `run_tests.py`, `test_mab_splits.py`
- Unused: `cloudflared.exe`, `tmp_pipeline_store.jsonl`

Created `.gitignore` to prevent committing temporary files.

### 3. Colab Notebook
- **File**: `colab_runner.ipynb`
- Full automated pipeline:
  1. Mount Google Drive
  2. Auto git clone/pull from GitHub
  3. Install dependencies
  4. Check/download benchmark data
  5. Configure models (default: Qwen2.5-1.5B-Instruct for Colab T4)
  6. Run benchmark with real-time output
  7. Display results
  8. Download reports as ZIP
- Supports both GitHub clone and manual upload options

---

## Current Repository Structure

```
ProjectMem/
├── src/                    # Source code
│   ├── benchmarks/        # Adapters & loaders
│   ├── conflict/          # Conflict detection & arbitration
│   ├── evaluation/        # Metrics & reporting
│   ├── format.py          # ISF definitions
│   ├── pipeline/          # Multi-agent pipeline
│   └── agents/            # Agent implementations
├── configs/
│   └── arbitration.yaml   # Arbitration thresholds
├── data/raw/              # Benchmark data (must be uploaded separately)
│   ├── memab/            # MemAB parquet files
│   └── longmemeval/      # LongMemEval JSON files
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── Dockerfile
├── docker-compose.yml
├── CLAUDE.md
├── README.md
├── PROJECT_DOCUMENTATION.md
├── colab_runner.ipynb     # Google Colab runner
└── .gitignore
```

---

## Running on Colab

1. **Push code to GitHub**
2. **Open Colab**: Upload `colab_runner.ipynb`
3. **Run cells sequentially**:
   - Mount Drive
   - Enter GitHub repo URL
   - Install dependencies
   - Configure model/scenarios (default: 1.5B, 8 scenarios)
   - Run benchmark
4. **Download reports** from Drive

**Recommended Colab settings:**
- Runtime type: GPU (T4)
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (fits in 12GB)
- Max scenarios: Start with 4, then 8 if time permits
- Estimated time: 4-8 hours for 8 scenarios

---

## Benchmark Results (Previous Run - Before Entity Fix)

**Adversarial Benchmark (50 scenarios):**
| Mode | Scenario Acc | Action Acc | Conflict F1 | Mem F1 |
|------|--------------|------------|-------------|---------|
| conflict_aware | 0.280 | 0.725 | 1.000 | 0.422 |
| lww | 0.460 | 0.424 | 1.000 | 0.555 |
| naive | 0.200 | 0.000 | 0.000 | 0.534 |

**Per-type action accuracy:**
- stale_read_conflict: 0.735
- concurrent_update: **1.000** ✓
- compatible_extension: 0.530
- potential_contradiction: **1.000** ✓
- semantic_overlap: 0.280 (still low)
- mutually_exclusive: **1.000** ✓

---

## Next Steps

1. **Push code to GitHub**
2. **Run full benchmark on Colab** with entity fix
3. **Analyze results** - expect higher conflict F1 for MemAE/MAB
4. If semantic_overlap still low, consider adjusting detector thresholds

---

## Commands

```bash
# Local quick test (2 scenarios)
python main.py --benchmark real_conflicts --max-scenarios 2

# Full run on Colab - use colab_runner.ipynb
```

---

---

## 1. Mục tiêu dự án (Project Goals)

- Xây dựng hệ thống **rule-based multi-agent memory conflict resolution**
- Không phụ thuộc vào LLM trong critical path → tiết kiệm chi phí (10-20x cheaper)
- Đạt performance gần SOTA trên benchmarks
- Dựa trên cơ sở học thuật với benchmark validation

---

## 2. Những đã làm (Completed Work)

### 2.1 Core Architecture & Existing Code
- [x] Internal Standard Format (ISF) trong `src/format.py`
- [x] SharedMemoryStore với versioning và bi-temporal tracking
- [x] Conflict detection: tiered (rule-based → semantic similarity)
- [x] Six resolution actions: overwrite, merge, keep_multiple_versions, defer, reject, append
- [x] Arbitration với multi-factor scoring (confidence, provenance, recency, authority)
- [x] Dynamic contextual weights support

### 2.2 Benchmarks & Adapters
- [x] **MemAE adapter**: `src/benchmarks/adapters/memab_adapter.py`
  - Added `_analyze_facts_for_conflicts()` với NLP heuristics để detect real conflicts
- [x] **Adversarial benchmark generator**: `src/benchmarks/adapters/adversarial_adapter.py`
  - Tạo 50 scenarios với 6 conflict types controlled
  - Templates cho mỗi type, gold action được xác định rõ
- [x] **Unified loader**: `src/benchmarks/unified_loader.py` → thêm `load_adversarial_benchmark()`
- [x] **Main entry point**: `main.py` → thêm `adversarial` vào choices

### 2.3 Evaluation & Metrics
- [x] `src/evaluation/run_evaluation.py`
  - Action accuracy, scenario accuracy, conflict F1
  - Per-type breakdown
  - Action appropriateness score (weighted)
  - Judge-free rate, stale handling accuracy
- [x] Reports saved to JSON

### 2.4 Configuration Tuning
- `configs/arbitration.yaml` - thresholds lowered multiple times:
  ```yaml
  overwrite_margin: 0.05           # từ 0.12 → 0.08 → 0.05
  keep_multiple_versions_margin: 0.02  # từ 0.08 → 0.04 → 0.02
  defer_below_score: 0.30          # từ 0.40 → 0.35 → 0.30
  ```

### 2.5 Bug Fixes (THIS SESSION)
- [x] **Timestamp propagation bug**: Proposal timestamps were 0 because pipeline wasn't passing event timestamps
  - Added `event_timestamp` parameter to `ConflictAwareWriter.write()`
  - Updated `_arbitrate()` to accept and use `proposal_timestamp`
  - Fixed pipeline (`multi_agent_pipeline.py`) to pass `ev["timestamp"]` as `event_timestamp`
  - Set `entry.timestamp = proposal_timestamp` in write flow
- [x] **Arbitration logic for mutually_exclusive & concurrent_update**:
  - Changed from score-margin-based to **timestamp-priority (last-write-wins)**
  - Newer entry with higher timestamp → always overwrite
  - Older or same timestamp → reject (or tie-break by confidence)
- [x] **Semantic overlap handling**:
  - Changed to always `merge` when conflict_type is semantic_overlap
  - Previously had threshold (similarity >= 0.7) causing many to be overwritten
- [x] Removed debug logging after fix

---

## 3. Kết quả benchmark hiện tại (Current Benchmark Results)

### Adversarial Benchmark (50 scenarios) - AFTER FIX

| Mode | Scenario Acc | Action Acc | Conflict F1 | Mem F1 |
|------|--------------|------------|-------------|---------|
| conflict_aware | 0.280 | **0.725** | 1.000 | 0.422 |
| lww | 0.460 | 0.424 | 1.000 | 0.555 |
| naive | 0.200 | 0.000 | 0.000 | 0.534 |

**Action distribution (predicted vs gold):**
- Gold: overwrite=190, merge=175, reject=83
- Pred: overwrite=238, merge=74, reject=127

**Per-type action accuracy:**
- stale_read_conflict: 0.735 (was 0.952 - dropped!)
- concurrent_update: **1.000** ✓ (was 0.12)
- compatible_extension: 0.530 (unchanged)
- potential_contradiction: **1.000** ✓ (was 0.15)
- semantic_overlap: 0.280 (unchanged - still low)
- mutually_exclusive: **1.000** ✓ (was 0.05)

**Cải thiện lớn:**
- Action accuracy tăng từ **39.5% → 72.5%** (vượt LWW 42.4%)
- `mutually_exclusive` và `concurrent_update` đã hoạt động đúng (last-write-wins)
- `potential_contradiction` cũng hoạt động tốt

**Vấn đề còn lại:**
1. `semantic_overlap` vẫn chỉ 28% (gold 175 merges, pred chỉ 74)
2. `stale_read_conflict` accuracy giảm từ 95% → 73.5%
3. Tổng reject=127 (gold là 83) → vẫn reject hơn cần thiết

---

## 4. Vấn đề đang phân tích (Current Issues)

### 4.1 Semantic Overlap Under-performance
- Gold action: merge (175)
- Predicted: merge (74), overwrite (238) → nhiều overwrite khi nên merge
- Nguyên nhân: Conflict detector đang classify một số semantic_overlap thực tế thành `potential_contradiction` hoặc `compatible_extension`?
- Cần kiểm tra: conflict type distribution trong report

### 4.2 Stale Read Regression
- Trước đây: stale_read accuracy 95% (sau khi fix reject)
- Bây giờ: 73.5%
- Cần kiểm tra: Có phải vì pipeline change hay logic mới?

### 4.3 Over-rejection
- Pred reject=127 vs gold reject=83
- Một số scenario nên append/merge đang bị reject
- Có thể do `potential_contradiction` vẫn đang reject nhiều

---

## 5. Files Modified (Chi tiết thay đổi)

### Core Files
- `configs/arbitration.yaml` - lowered thresholds
- `src/conflict/conflict_aware_writer.py` - timestamp fix, mutually_exclusive/concurrent_update → LWW, semantic_overlap → always merge
- `src/pipeline/multi_agent_pipeline.py` - pass event_timestamp to writer

### Evaluation
- `src/evaluation/run_evaluation.py` - added action_appropriateness_score, counterfactual_accuracy, judge_free_rate

### New Files
- `src/benchmarks/adapters/adversarial_adapter.py` - adversarial generator (full implementation)
- `src/benchmarks/adapters/memab_adapter.py` - enhanced with `_analyze_facts_for_conflicts()`

### Entry Points
- `main.py` - added adversarial benchmark support

---

## 6. Next Steps / Cần làm tiếp

### 6.1 Investigate Semantic Overlap Mismatch (PRIORITY 1)
- Kiểm tra conflict type distribution trong report: có bao nhiêu scenario được classify là semantic_overlap?
- Nếu ít, có thể detector thresholds quá cao (0.7) → nên điều chỉnh từ config
- Hoặc adversarial generator tạo semantic_overlap với similarity thấp hơn 0.7

### 6.2 Investigate Stale Read Regression (PRIORITY 2)
- So sánh logs cũ và mới để xem stale_read decisions có đúng không
- Kiểm tra staleness_detector logic

### 6.3 Optimize Action Distribution (PRIORITY 3)
- Nếu semantic_overlap đã đúng, xem xem giảm reject bằng cách:
  - Điều chỉnh `potential_contradiction` handling: ưu tiên newer timestamp
  - Hoặc giảm overwrite_margin thêm nữa

### 6.4 Validate on Other Benchmarks
- Chạy lại MemAE với improved conflict detection
- Chạy LongMemEval, SAFEFLOW để xem improvements có generalize không

---

## 7. Commands để tiếp tục

```bash
# Chạy adversarial benchmark (50 scenarios)
python main.py --benchmark adversarial --max-scenarios 50

# Chạy MemAE
python main.py --benchmark memae --max-scenarios 10

# Chạy tất cả
python main.py --benchmark all --max-scenarios 100
```

---

## 8. Key Observations

1. **Timestamp fix là then chốt**: Trước đó proposal timestamps = 0 → last-write-wins không hoạt động
2. **Last-write-wins cho factual conflicts**: `mutually_exclusive` và `concurrent_update` cần ưu tiên timestamp
3. **Semantic overlap nên merge ngay**: Khi detector đã xác định là overlap, merge là hợp lý
4. **Stale read accuracy drop** cần được investigate - có thể do pipeline change

---

## 9. CONTINUATION CHECKLIST (for new session)

- [ ] Read this `PROJECT_PROGRESS.md` file first
- [ ] Current state: Action accuracy 72.5%, but semantic_overlap only 28%
- [ ] Check conflict type distribution in latest report
- [ ] Investigate why semantic_overlap scenarios are not merging
- [ ] Check if detector threshold (0.7) needs adjustment or generator needs fix
- [ ] Investigate stale_read regression (accuracy dropped from 95% to 73.5%)
- [ ] Consider tweaking potential_contradiction to reduce false rejects
- [ ] Rerun and aim for action accuracy >80%

---

**Hypothesis for semantic_overlap**: The adversarial generator creates overlapping sets like ["Python","ML"] vs ["Python","AI"]. The embedding similarity may be below 0.7 threshold, causing detector to classify as `compatible_extension` or `potential_contradiction` instead of `semantic_overlap`. Need to verify by checking the actual similarity scores in logs or lowering the semantic_overlap threshold to 0.6 or 0.5.

**Next debugging**: Add logging to see which conflict types semantic_overlap scenarios are being classified as, and adjust accordingly.
