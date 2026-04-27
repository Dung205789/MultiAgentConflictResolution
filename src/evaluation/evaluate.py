"""
Enhanced evaluation with support for multiple benchmark sources.
Adds MemoryAgentBench integration and richer metrics.
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.shared_memory_store import SharedMemoryStore
from src.conflict.baselines import LastWriteWinsWriter, NaiveAppendWriter
from src.conflict.conflict_aware_writer import ConflictAwareWriter
from src.conflict.staleness_detector import StalenessDetector
from src.benchmarks.memoryagentbench_loader import load_memoryagentbench


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_state(records: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    out = []
    for r in records:
        subj = r.get("subject", "")
        pred = r.get("predicate", "")
        obj = str(r.get("object_val", r.get("object", "")))
        out.append((subj, pred, obj))
    return sorted(out)


def run_scenario_with_writer(
    scenario: Dict[str, Any],
    writer_type: str,
    enable_retrieval_metrics: bool = False
) -> Dict[str, Any]:
    store = SharedMemoryStore(
        persistence_path="tmp_eval_store.jsonl",
        enable_persistence=False,
    )
    store.records = []

    if writer_type == "lww":
        writer = LastWriteWinsWriter(store)
    elif writer_type == "naive":
        writer = NaiveAppendWriter(store)
    elif writer_type == "conflict_aware":
        writer = ConflictAwareWriter(store, StalenessDetector())
    else:
        raise ValueError(f"Unknown writer_type: {writer_type}")

    interception_count = 0
    total_writes = 0
    detected_conflicts = 0
    event_results = []
    retrieval_metrics: List[Dict[str, Any]] = []

    for ev in scenario.get("ordered_events", []):
        if ev.get("event_type") != "write_proposal":
            continue

        total_writes += 1
        proposal = ev["proposal"]
        agent_id = ev["agent_id"]
        read_snapshot_time = ev.get("read_snapshot_time", time.time())

        existing = [
            r for r in store.records
            if r.subject == proposal["subject"] and r.predicate == proposal["predicate"] and r.status == "active"
        ]
        if existing:
            interception_count += 1

        if writer_type == "conflict_aware":
            res = writer.write(proposal, agent_id=agent_id, read_snapshot_time=read_snapshot_time)
        else:
            res = writer.write(proposal, agent_id=agent_id)

        if res.get("conflict_detected"):
            detected_conflicts += 1

        event_results.append({
            "pred_conflict_type": res.get("conflict_type", "none"),
            "pred_action": res.get("resolution_action", "append"),
            "gold_conflict_type": scenario.get("gold_conflict_type", "none"),
            "gold_action": scenario.get("gold_resolution_action", "append"),
            "scenario_type": scenario.get("scenario_type", "unknown"),
        })

        # Optional retrieval evaluation after each write
        if enable_retrieval_metrics and scenario.get("queries"):
            for query_info in scenario["queries"]:
                query_text = query_info["query_text"]
                gold_answers = query_info["gold_answers"]
                # Retrieve from store
                all_visible = [r.to_dict() for r in store.get_all_visible()]
                # Simple retrieval using keyword overlap for now
                retrieved = _simple_retrieve(all_visible, query_text, k=5)
                retrieved_objs = [r.get("object_val") for r, _ in retrieved]
                # Compute recall@k
                recall = len(set(retrieved_objs) & set(gold_answers)) / len(gold_answers) if gold_answers else 0.0
                retrieval_metrics.append({
                    "step": ev.get("step"),
                    "query": query_text,
                    "retrieved": retrieved_objs,
                    "gold": gold_answers,
                    "recall_at_k": recall
                })

    final_visible = [r.to_dict() for r in store.get_all_visible()]
    gold_visible = scenario.get("gold_visible_shared_state_after_commit", [])
    state_match = normalize_state(final_visible) == normalize_state(gold_visible)

    gold_conflict = 1 if scenario.get("gold_conflict_exists") else 0
    pred_conflict = 1 if detected_conflicts > 0 else 0

    result = {
        "scenario_id": scenario.get("scenario_id"),
        "writer_type": writer_type,
        "state_match": state_match,
        "gold_conflict": gold_conflict,
        "pred_conflict": pred_conflict,
        "interception_rate": (interception_count / total_writes) if total_writes else 0.0,
        "visibility_indexing_correct": state_match,
        "final_visible": normalize_state(final_visible),
        "gold_visible": normalize_state(gold_visible),
        "event_results": event_results,
    }

    if retrieval_metrics:
        result["retrieval_metrics"] = retrieval_metrics

    return result


def _simple_retrieve(
    memories: List[Dict[str, Any]],
    query: str,
    k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """Simple keyword-based retrieval for evaluation."""
    query_tokens = set(query.lower().split())
    scored = []
    for mem in memories:
        text = f"{mem.get('subject','')} {mem.get('predicate','')} {mem.get('object_val','')}".lower()
        mem_tokens = set(text.split())
        overlap = len(query_tokens & mem_tokens) / len(query_tokens) if query_tokens else 0.0
        scored.append((mem, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def compute_per_conflict_type_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    all_events = [e for r in results for e in r.get("event_results", [])]
    labels = sorted(set([e["gold_conflict_type"] for e in all_events] + [e["pred_conflict_type"] for e in all_events]))
    metrics = {}

    for label in labels:
        if label == "none":
            continue
        tp = sum(1 for e in all_events if e["gold_conflict_type"] == label and e["pred_conflict_type"] == label)
        fp = sum(1 for e in all_events if e["gold_conflict_type"] != label and e["pred_conflict_type"] == label)
        fn = sum(1 for e in all_events if e["gold_conflict_type"] == label and e["pred_conflict_type"] != label)
        metrics[label] = _classification_prf(tp, fp, fn)

    return metrics


def compute_action_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_events = [e for r in results for e in r.get("event_results", [])]
    non_none_events = [e for e in all_events if e["gold_conflict_type"] != "none"]

    overall = (
        sum(1 for e in non_none_events if e["pred_action"] == e["gold_action"]) / len(non_none_events)
        if non_none_events else 0.0
    )

    by_conflict_type = defaultdict(list)
    for e in non_none_events:
        by_conflict_type[e["gold_conflict_type"]].append(e)

    per_type_acc = {}
    for ct, rows in by_conflict_type.items():
        per_type_acc[ct] = sum(1 for e in rows if e["pred_action"] == e["gold_action"]) / len(rows)

    action_distribution = defaultdict(int)
    for e in all_events:
        action_distribution[e["pred_action"]] += 1

    return {
        "resolution_action_accuracy": overall,
        "resolution_action_accuracy_per_conflict_type": dict(per_type_acc),
        "predicted_action_distribution": dict(action_distribution),
    }


def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregated retrieval metrics."""
    all_retrieval = [m for r in results for m in r.get("retrieval_metrics", [])]
    if not all_retrieval:
        return {"retrieval_recall_avg": 0.0, "retrieval_tasks_evaluated": 0}

    avg_recall = sum(m["recall_at_k"] for m in all_retrieval) / len(all_retrieval)
    return {
        "retrieval_recall_avg": avg_recall,
        "retrieval_tasks_evaluated": len(all_retrieval)
    }


def compute_stale_handling_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute metrics about stale read handling."""
    stale_events = []
    for r in results:
        for ev in r.get("event_results", []):
            if ev["gold_conflict_type"] == "stale_read_conflict":
                stale_events.append(ev)

    if not stale_events:
        return {"stale_handling_accuracy": 0.0, "stale_events": 0}

    correct = sum(1 for ev in stale_events if ev["pred_action"] == ev["gold_action"])
    return {
        "stale_handling_accuracy": correct / len(stale_events),
        "stale_events": len(stale_events)
    }


def compute_branch_explosion_metric(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Measure how many active versions per entity exist at end."""
    # This would need store access; approximate via action distribution
    total_events = sum(len(r.get("event_results", [])) for r in results)
    merge_actions = sum(
        1 for r in results for ev in r.get("event_results", [])
        if ev["pred_action"] == "merge"
    )
    keep_versions_actions = sum(
        1 for r in results for ev in r.get("event_results", [])
        if ev["pred_action"] == "keep_multiple_versions"
    )
    return {
        "merge_action_rate": merge_actions / total_events if total_events else 0.0,
        "keep_multiple_versions_rate": keep_versions_actions / total_events if total_events else 0.0,
    }


def _classification_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(
    results: List[Dict[str, Any]],
    enable_retrieval: bool = False
) -> Dict[str, Any]:
    tp = sum(1 for r in results if r["gold_conflict"] == 1 and r["pred_conflict"] == 1)
    fp = sum(1 for r in results if r["gold_conflict"] == 0 and r["pred_conflict"] == 1)
    fn = sum(1 for r in results if r["gold_conflict"] == 1 and r["pred_conflict"] == 0)

    base = _classification_prf(tp, fp, fn)
    scenario_acc = sum(1 for r in results if r["state_match"]) / len(results) if results else 0.0
    vis_correct = sum(1 for r in results if r["visibility_indexing_correct"]) / len(results) if results else 0.0
    interception_rate = sum(r["interception_rate"] for r in results) / len(results) if results else 0.0

    out: Dict[str, Any] = {
        "conflict_precision": base["precision"],
        "conflict_recall": base["recall"],
        "conflict_f1": base["f1"],
        "end_to_end_scenario_accuracy": scenario_acc,
        "visibility_indexing_correctness": vis_correct,
        "conflict_writer_interception_rate": interception_rate,
    }

    out["per_conflict_type"] = compute_per_conflict_type_metrics(results)
    out["action_quality"] = compute_action_quality_metrics(results)
    out["stale_handling"] = compute_stale_handling_metrics(results)
    out["branch_explosion"] = compute_branch_explosion_metric(results)

    if enable_retrieval:
        out["retrieval"] = compute_retrieval_metrics(results)

    return out


def evaluate(
    benchmark_path: Optional[str] = None,
    use_memoryagentbench: bool = False,
    mab_subset: str = "all",
    enable_retrieval_metrics: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate writers on a benchmark.

    Args:
        benchmark_path: Path to custom JSONL benchmark (if not using MAB)
        use_memoryagentbench: If True, load from Hugging Face MemoryAgentBench
        mab_subset: Subset filter for MemoryAgentBench (e.g., "conflict", "temporal")
        enable_retrieval_metrics: Whether to evaluate retrieval performance

    Returns:
        Dictionary with metrics for each writer type and per-scenario analysis.
    """
    if use_memoryagentbench:
        scenarios = load_memoryagentbench(subset=mab_subset)
    else:
        if not benchmark_path:
            benchmark_path = "data/enhanced_multi_agent_benchmark.jsonl"
        scenarios = load_jsonl(benchmark_path)

    print(f"Loaded {len(scenarios)} scenarios")

    conflict_aware_results = [
        run_scenario_with_writer(s, writer_type="conflict_aware", enable_retrieval_metrics=enable_retrieval_metrics)
        for s in scenarios
    ]
    lww_results = [
        run_scenario_with_writer(s, writer_type="lww", enable_retrieval_metrics=enable_retrieval_metrics)
        for s in scenarios
    ]
    naive_results = [
        run_scenario_with_writer(s, writer_type="naive", enable_retrieval_metrics=enable_retrieval_metrics)
        for s in scenarios
    ]

    conflict_aware_metrics = compute_metrics(conflict_aware_results, enable_retrieval=enable_retrieval_metrics)
    lww_metrics = compute_metrics(lww_results, enable_retrieval=enable_retrieval_metrics)
    naive_metrics = compute_metrics(naive_results, enable_retrieval=enable_retrieval_metrics)

    scenario_type_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for s, c, l, n in zip(scenarios, conflict_aware_results, lww_results, naive_results):
        st = s.get("scenario_type", "unknown")
        scenario_type_groups.setdefault(st, {"conflict_aware": [], "lww": [], "naive": []})
        scenario_type_groups[st]["conflict_aware"].append(c)
        scenario_type_groups[st]["lww"].append(l)
        scenario_type_groups[st]["naive"].append(n)

    per_type_delta = {}
    for st, grp in scenario_type_groups.items():
        c_acc = sum(1 for r in grp["conflict_aware"] if r["state_match"]) / len(grp["conflict_aware"]) if grp["conflict_aware"] else 0.0
        l_acc = sum(1 for r in grp["lww"] if r["state_match"]) / len(grp["lww"]) if grp["lww"] else 0.0
        n_acc = sum(1 for r in grp["naive"] if r["state_match"]) / len(grp["naive"]) if grp["naive"] else 0.0
        per_type_delta[st] = {
            "conflict_aware_minus_lww_scenario_acc": c_acc - l_acc,
            "conflict_aware_minus_naive_scenario_acc": c_acc - n_acc,
            "lww_minus_naive_scenario_acc": l_acc - n_acc,
        }

    return {
        "conflict_aware": conflict_aware_metrics,
        "lww": lww_metrics,
        "naive": naive_metrics,
        "per_scenario_type_delta": per_type_delta,
        "num_scenarios": len(scenarios),
        "benchmark_source": "MemoryAgentBench" if use_memoryagentbench else benchmark_path,
    }


if __name__ == "__main__":
    # Example: evaluate on custom benchmark
    report = evaluate(benchmark_path="data/enhanced_multi_agent_benchmark.jsonl", enable_retrieval_metrics=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Or evaluate on MemoryAgentBench:
    # report = evaluate(use_memoryagentbench=True, mab_subset="conflict", enable_retrieval_metrics=True)
