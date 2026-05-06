"""
Scenario type generators for the multi-agent memory benchmark.

Each function generates a specific type of conflict scenario with appropriate
proposals, gold labels, and expected resolution actions.
"""
from typing import Dict, List, Any, Optional
from src.benchmarks.generator_core import generate_scenario


def generate_overwrite_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario where overwrite is the correct action."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="overwrite_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "initial_value",
                    "confidence": 0.6,
                    "provenance": "inferred"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "updated_value",
                    "confidence": 0.9,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="mutually_exclusive",
        gold_resolution_action="overwrite",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "updated_value", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "updated_value"}
        ],
        description="Higher confidence and more recent entry should overwrite the older one",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_keep_multiple_versions_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario where keeping multiple versions is the correct action."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="keep_multiple_versions_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "enjoys outdoor activities like hiking and camping",
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "prefers indoor activities such as painting and chess",
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="compatible_extension",
        gold_resolution_action="keep_multiple_versions",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "enjoys outdoor activities like hiking and camping", "status": "active"},
            {"subject": subject, "predicate": predicate, "object_val": "prefers indoor activities such as painting and chess", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "enjoys outdoor activities like hiking and camping"},
            {"subject": subject, "predicate": predicate, "object_val": "prefers indoor activities such as painting and chess"}
        ],
        description="Similar confidence entries with different but valid perspectives should both be kept",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_merge_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario where merging is the correct action."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="merge_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": '{"name": "Alice", "age": 25}',
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": '{"city": "Paris", "occupation": "Data Scientist"}',
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="semantic_overlap",
        gold_resolution_action="merge",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate,
             "object_val": '{"name": "Alice", "age": 25, "city": "Paris", "occupation": "Data Scientist"}',
             "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate,
             "object_val": '{"name": "Alice", "age": 25, "city": "Paris", "occupation": "Data Scientist"}'}
        ],
        description="Complementary JSON objects that should be merged",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_defer_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario where deferring is the correct action (stale read with equal confidence)."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="defer_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "value_a",
                    "confidence": 0.7,
                    "provenance": "inferred"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "value_b",
                    "confidence": 0.7,
                    "provenance": "inferred"
                },
                "read_snapshot_time": 1.0  # Stale read: before agent_a's commit
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="stale_read_conflict",
        gold_resolution_action="defer",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "value_a", "status": "active"},
            {"subject": subject, "predicate": predicate, "object_val": "value_b", "status": "tentative"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "value_a"}
        ],
        description="Stale read with equal confidence should defer to manual review",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_reject_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario where rejection is the correct action."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="reject_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "established_value",
                    "confidence": 0.95,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "uncertain_value",
                    "confidence": 0.4,
                    "provenance": "inferred"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="mutually_exclusive",
        gold_resolution_action="reject",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "established_value", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "established_value"}
        ],
        description="Low confidence entry contradicting high confidence entry should be rejected",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_stale_read_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario with stale read conflict."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="stale_read_conflict",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "first_value",
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "second_value",
                    "confidence": 0.9,
                    "provenance": "explicit"
                },
                "read_snapshot_time": 90.0  # Read before agent_a's write
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="stale_read_conflict",
        gold_resolution_action="overwrite",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "second_value", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "second_value"}
        ],
        description="Stale read where the second agent reads before the first agent's write is visible",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_multi_agent_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario with multiple agents and complex interactions."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="multi_agent_complex",
        agents=["agent_a", "agent_b", "agent_c"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "perspective_a",
                    "confidence": 0.7,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "perspective_b",
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            },
            {
                "step": 3,
                "agent_id": "agent_c",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": "perspective_c",
                    "confidence": 0.9,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="compatible_extension",
        gold_resolution_action="keep_multiple_versions",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate, "object_val": "perspective_a", "status": "active"},
            {"subject": subject, "predicate": predicate, "object_val": "perspective_b", "status": "active"},
            {"subject": subject, "predicate": predicate, "object_val": "perspective_c", "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate, "object_val": "perspective_a"},
            {"subject": subject, "predicate": predicate, "object_val": "perspective_b"},
            {"subject": subject, "predicate": predicate, "object_val": "perspective_c"}
        ],
        description="Three agents with different perspectives that should all be preserved",
        agent_profiles=agent_profiles,
        queries=queries
    )


def generate_structured_merge_scenario(
    scenario_id: str,
    subject: str,
    predicate: str,
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Generate a scenario with structured data that should be merged."""
    return generate_scenario(
        scenario_id=scenario_id,
        scenario_type="structured_merge_correct",
        agents=["agent_a", "agent_b"],
        ordered_events=[
            {
                "step": 1,
                "agent_id": "agent_a",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": '{"name": "John", "age": 30}',
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            },
            {
                "step": 2,
                "agent_id": "agent_b",
                "event_type": "write_proposal",
                "proposal": {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": '{"city": "New York", "occupation": "Engineer"}',
                    "confidence": 0.8,
                    "provenance": "explicit"
                }
            }
        ],
        gold_conflict_exists=True,
        gold_conflict_type="semantic_overlap",
        gold_resolution_action="merge",
        gold_reconciled_memory_state=[
            {"subject": subject, "predicate": predicate,
             "object_val": '{"name": "John", "age": 30, "city": "New York", "occupation": "Engineer"}',
             "status": "active"}
        ],
        gold_visible_shared_state_after_commit=[
            {"subject": subject, "predicate": predicate,
             "object_val": '{"name": "John", "age": 30, "city": "New York", "occupation": "Engineer"}'}
        ],
        description="JSON objects that should be merged into a single object",
        agent_profiles=agent_profiles,
        queries=queries
    )
