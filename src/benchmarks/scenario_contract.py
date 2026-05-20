"""
Canonical benchmark-facing scenario access helpers.

This keeps the runtime pipeline, evaluation, and adapters on one scenario
surface so benchmark-specific loaders only need to satisfy the ISF schema once.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


SCENARIO_CONTRACT_VERSION = "isf_scenario_v1"


def scenario_to_dict(scenario: Any) -> Dict[str, Any]:
    """
    Normalize a scenario-like object to a plain ISF dictionary.
    """
    if hasattr(scenario, "to_dict"):
        data = scenario.to_dict()
    elif isinstance(scenario, dict):
        data = dict(scenario)
    else:
        raise TypeError(f"Unsupported scenario type: {type(scenario)!r}")

    data.setdefault("queries", [])
    data.setdefault("ordered_events", [])
    data.setdefault("agents", [])
    data.setdefault("scenario_type", "unknown")
    data.setdefault("description", "")
    data.setdefault("base_timestamp", 1000.0)
    data.setdefault("agent_profiles", {})
    data.setdefault("_scenario_contract_version", SCENARIO_CONTRACT_VERSION)
    return data


def scenarios_to_dicts(scenarios: Iterable[Any]) -> List[Dict[str, Any]]:
    return [scenario_to_dict(scenario) for scenario in scenarios]


def scenario_identifier(scenario: Any, fallback_idx: int) -> str:
    scenario_dict = scenario_to_dict(scenario)
    return str(scenario_dict.get("scenario_id", f"scenario_{fallback_idx}"))
