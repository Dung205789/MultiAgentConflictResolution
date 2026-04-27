"""
Benchmark collection and utilities.
Generates the complete benchmark suite from scenario types.
"""
import json
from typing import Dict, List, Any
from src.benchmarks.generator_core import generate_scenario
from src.benchmarks.scenario_types import (
    generate_overwrite_scenario,
    generate_keep_multiple_versions_scenario,
    generate_merge_scenario,
    generate_defer_scenario,
    generate_reject_scenario,
    generate_stale_read_scenario,
    generate_multi_agent_scenario,
    generate_structured_merge_scenario,
)


def _make_agent_profiles(agents: List[str], scenario_type: str) -> Dict[str, Dict[str, Any]]:
    """Generate agent profiles for a scenario based on type and agents."""
    profiles = {}
    if len(agents) == 2:
        a, b = agents
        if scenario_type == "overwrite_correct":
            profiles[a] = {"role": "novice", "reliability": 0.6}
            profiles[b] = {"role": "expert", "reliability": 0.9}
        elif scenario_type in ["keep_multiple_versions_correct", "merge_correct", "structured_merge_correct"]:
            profiles[a] = {"role": "specialist", "reliability": 0.8}
            profiles[b] = {"role": "analyst", "reliability": 0.8}
        elif scenario_type == "defer_correct":
            profiles[a] = {"role": "fast_writer", "reliability": 0.7}
            profiles[b] = {"role": "slow_writer", "reliability": 0.7}
        elif scenario_type == "reject_correct":
            profiles[a] = {"role": "authoritative", "reliability": 0.95}
            profiles[b] = {"role": "speculative", "reliability": 0.4}
        elif scenario_type == "stale_read_conflict":
            profiles[a] = {"role": "primary", "reliability": 0.8}
            profiles[b] = {"role": "secondary", "reliability": 0.9}
        else:
            for aid in agents:
                profiles[aid] = {"role": "agent", "reliability": 0.7}
    elif len(agents) == 3:
        roles = ["generalist", "specialist", "expert"]
        reliabilities = [0.7, 0.8, 0.9]
        for idx, aid in enumerate(agents):
            profiles[aid] = {"role": roles[idx], "reliability": reliabilities[idx]}
    else:
        for aid in agents:
            profiles[aid] = {"role": "agent", "reliability": 0.7}
    return profiles


def _make_queries(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate retrieval queries based on gold visible state."""
    gold_visible = scenario.get("gold_visible_shared_state_after_commit", [])
    if not gold_visible:
        return []
    queries_map = {}
    for mem in gold_visible:
        subj = mem.get("subject")
        pred = mem.get("predicate")
        obj = mem.get("object_val")
        key = (subj, pred)
        if key not in queries_map:
            queries_map[key] = []
        queries_map[key].append(obj)
    queries = []
    for (subj, pred), objs in queries_map.items():
        query_text = f"What is the {subj}'s {pred}?"
        queries.append({
            "query_text": query_text,
            "gold_answers": objs,
            "expected_retrieval_style": "all" if len(objs) > 1 else "best"
        })
    return queries


def generate_benchmark_scenarios() -> List[Dict[str, Any]]:
    """Generate a diverse set of benchmark scenarios."""
    scenarios = []

    mutually_exclusive_predicates = ["study_time", "city", "location", "current_task", "current_status"]
    additive_predicates = ["language", "skill", "interest", "hobby", "focus_area", "known_for", "likes"]
    neutral_predicates = ["career_goal", "profile", "summary", "description"]

    for i in range(5):
        subject = ["user", "agent", "system", "profile"][i % 4]

        # Overwrite scenarios
        pred_overwrite = mutually_exclusive_predicates[i % len(mutually_exclusive_predicates)]
        scenario = generate_overwrite_scenario(f"enhanced_overwrite_{i+1}", subject, pred_overwrite)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

        # Keep multiple versions scenarios
        pred_keep = additive_predicates[i % len(additive_predicates)]
        scenario = generate_keep_multiple_versions_scenario(f"enhanced_keep_versions_{i+1}", subject, pred_keep)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

        # Merge scenarios
        pred_merge = neutral_predicates[i % len(neutral_predicates)]
        scenario = generate_merge_scenario(f"enhanced_merge_{i+1}", subject, pred_merge)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

        # Defer scenarios
        pred_defer = mutually_exclusive_predicates[(i+1) % len(mutually_exclusive_predicates)]
        scenario = generate_defer_scenario(f"enhanced_defer_{i+1}", subject, pred_defer)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

        # Reject scenarios
        pred_reject = mutually_exclusive_predicates[(i+2) % len(mutually_exclusive_predicates)]
        scenario = generate_reject_scenario(f"enhanced_reject_{i+1}", subject, pred_reject)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

        # Stale read scenarios
        pred_stale = mutually_exclusive_predicates[(i+3) % len(mutually_exclusive_predicates)]
        scenario = generate_stale_read_scenario(f"enhanced_stale_read_{i+1}", subject, pred_stale)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

    # Add complex multi-agent scenarios
    for i, pred in enumerate(additive_predicates[:2]):
        subject = ["user", "system"][i]
        scenario = generate_multi_agent_scenario(f"enhanced_multi_agent_{i+1}", subject, pred)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

    # Add structured merge scenarios
    for i, pred in enumerate(neutral_predicates[:2]):
        subject = ["user", "system"][i]
        scenario = generate_structured_merge_scenario(f"enhanced_structured_merge_{i+1}", subject, pred)
        scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
        scenario["queries"] = _make_queries(scenario)
        scenarios.append(scenario)

    # Add a no-conflict scenario
    scenario = generate_scenario(
        scenario_id="enhanced_no_conflict_1",
        scenario_type="no_conflict",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": "user",
                    "predicate": "name",
                    "object_val": "John",
                    "confidence": 1.0,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": "user",
                    "predicate": "age",
                    "object_val": "30",
                    "confidence": 1.0,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=False,
        gold_conflict_type="none",
        gold_resolution_action="append",
        gold_reconciled_memory_state=[
            {"subject": "user", "predicate": "name", "object_val": "John", "status": "active"},
            {"subject": "user", "predicate": "age", "object_val": "30", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": "user", "predicate": "name", "object_val": "John"},
            {"subject": "user", "predicate": "age", "object_val": "30"}
        ],
        description="No conflict scenario with different predicates",
        base_timestamp=1000.0
    )
    scenario["agent_profiles"] = _make_agent_profiles(scenario["agents"], scenario["scenario_type"])
    scenario["queries"] = _make_queries(scenario)
    scenarios.append(scenario)

    return scenarios


def save_benchmark(scenarios: List[Dict[str, Any]], output_path: str) -> None:
    """Save benchmark scenarios to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    scenarios = generate_benchmark_scenarios()
    save_benchmark(scenarios, "data/enhanced_multi_agent_benchmark.jsonl")
    print(f"Generated {len(scenarios)} benchmark scenarios")
