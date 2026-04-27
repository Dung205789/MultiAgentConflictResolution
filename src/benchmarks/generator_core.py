"""
Core scenario generation function.
"""
from typing import Dict, List, Any


def generate_scenario(
    scenario_id: str,
    scenario_type: str,
    agents: List[str],
    ordered_events: List[Dict[str, Any]],
    gold_conflict_exists: bool,
    gold_conflict_type: str,
    gold_resolution_action: str,
    gold_reconciled_memory_state: List[Dict[str, Any]],
    gold_visible_shared_state_after_commit: List[Dict[str, Any]],
    description: str = "",
    agent_profiles: Dict[str, Dict[str, Any]] = None,
    queries: List[Dict[str, Any]] = None,
    base_timestamp: float = 1000.0
) -> Dict[str, Any]:
    """Generate a benchmark scenario with the given parameters."""
    scenario = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "description": description,
        "agents": agents,
        "ordered_events": ordered_events,
        "gold_conflict_exists": gold_conflict_exists,
        "gold_conflict_type": gold_conflict_type,
        "gold_resolution_action": gold_resolution_action,
        "gold_reconciled_memory_state": gold_reconciled_memory_state,
        "gold_visible_shared_state_after_commit": gold_visible_shared_state_after_commit,
    }

    if agent_profiles:
        scenario["agent_profiles"] = agent_profiles

    if queries:
        scenario["queries"] = queries

    if base_timestamp is not None:
        time_delta = 60.0
        for i, ev in enumerate(ordered_events):
            if "timestamp" not in ev:
                ev["timestamp"] = base_timestamp + i * time_delta

    return scenario
