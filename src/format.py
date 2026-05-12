"""
Internal Standard Format (ISF) for Multi-Agent Conflict Resolution Scenarios.

This module defines the canonical format used throughout the project.
All benchmark adapters should convert their data to this format.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class MemoryEntry:
    """A single memory fact/triple."""
    subject: str
    predicate: str
    object_val: Any
    status: str = "active"  # "active", "deprecated", "merged"
    confidence: Optional[float] = None
    provenance: Optional[str] = None
    timestamp: Optional[float] = None
    agent_id: Optional[str] = None


@dataclass
class Event:
    """An event in the scenario timeline."""
    step: int
    agent_id: str
    event_type: str  # "read" or "write_proposal"
    timestamp: float

    # For write_proposal
    proposal: Optional[Dict[str, Any]] = None  # Contains subject, predicate, object_val, confidence, provenance

    # For read
    query: Optional[str] = None

    # Snapshot time for conflict detection (used by conflict_aware writer)
    read_snapshot_time: Optional[float] = None


@dataclass
class Query:
    """Evaluation query for retrieval."""
    query_text: str
    gold_answers: List[Any]
    expected_retrieval_style: str = "best"  # "best", "any", "all"


@dataclass
class Scenario:
    """
    Internal Standard Format (ISF) for a conflict resolution scenario.

    This is the canonical format used by the evaluation pipeline.
    All benchmark adapters should produce instances of this class.
    """
    scenario_id: str
    agents: List[str]
    ordered_events: List[Event]
    gold_conflict_exists: bool
    gold_conflict_type: str  # e.g., "mutually_exclusive", "stale_read_conflict", "none", "semantic_overlap", "compatible_extension"
    gold_resolution_action: str  # e.g., "overwrite", "merge", "keep_multiple_versions", "defer", "reject", "append"

    gold_reconciled_memory_state: List[MemoryEntry]
    gold_visible_shared_state_after_commit: List[MemoryEntry]

    scenario_type: str = "unknown"
    description: str = ""
    agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    queries: List[Query] = field(default_factory=list)
    base_timestamp: float = 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary (for JSON serialization)."""
        data = {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "description": self.description,
            "agents": self.agents,
            "ordered_events": [asdict(ev) for ev in self.ordered_events],
            "gold_conflict_exists": self.gold_conflict_exists,
            "gold_conflict_type": self.gold_conflict_type,
            "gold_resolution_action": self.gold_resolution_action,
            "gold_reconciled_memory_state": [asdict(m) for m in self.gold_reconciled_memory_state],
            "gold_visible_shared_state_after_commit": [asdict(m) for m in self.gold_visible_shared_state_after_commit],
            "queries": [asdict(q) for q in self.queries] if self.queries else [],
        }
        if self.agent_profiles:
            data["agent_profiles"] = self.agent_profiles
        if self.base_timestamp is not None:
            data["base_timestamp"] = self.base_timestamp
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create Scenario from plain dictionary."""
        # Parse events
        events = [Event(**ev) for ev in data.get("ordered_events", [])]

        # Parse memory entries
        gold_reconciled = [MemoryEntry(**m) for m in data.get("gold_reconciled_memory_state", [])]
        gold_visible = [MemoryEntry(**m) for m in data.get("gold_visible_shared_state_after_commit", [])]

        # Parse queries
        queries = [Query(**q) for q in data.get("queries", [])]

        return cls(
            scenario_id=data["scenario_id"],
            scenario_type=data.get("scenario_type", "unknown"),
            description=data.get("description", ""),
            agents=data.get("agents", []),
            ordered_events=events,
            gold_conflict_exists=data["gold_conflict_exists"],
            gold_conflict_type=data["gold_conflict_type"],
            gold_resolution_action=data["gold_resolution_action"],
            gold_reconciled_memory_state=gold_reconciled,
            gold_visible_shared_state_after_commit=gold_visible,
            agent_profiles=data.get("agent_profiles"),
            queries=queries,
            base_timestamp=data.get("base_timestamp", 1000.0)
        )


# Conflict type taxonomy (standard set)
CONFLICT_TYPES = {
    "none",  # No conflict
    "mutually_exclusive",  # Two writes with different values for same (subject, predicate)
    "stale_read_conflict",  # Read based on stale snapshot, write should be rejected/deferred
    "semantic_overlap",  # Overlapping but not contradictory information
    "compatible_extension",  # New info extends existing knowledge without contradiction
    "exact_duplicate",  # Identical value (subtype of semantic overlap)
    "semantic_duplicate",  # Near-identical meaning (subtype of semantic overlap)
    "counterfactual_temporal",  # Out-of-order updates that contradict (newer but wrong order)
    "potential_contradiction",  # Low similarity, could be contradictory
}

# Resolution actions taxonomy (standard set)
RESOLUTION_ACTIONS = {
    "overwrite",  # Replace old value with new
    "merge",  # Combine information from multiple sources
    "keep_multiple_versions",  # Keep both/all versions
    "defer",  # Delay commit until more info
    "reject",  # Reject the write proposal
    "append",  # Add as new entry without affecting others
}
