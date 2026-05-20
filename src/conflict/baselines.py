from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.memory.shared_memory_store import SharedMemoryStore, MemoryEntry


class BaseWriter(ABC):
    def __init__(self, store: SharedMemoryStore):
        self.store = store

    @abstractmethod
    def write(self, proposal: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        pass


class LastWriteWinsWriter(BaseWriter):
    """
    Baseline A:
    - For same (subject, predicate), newest write supersedes previous active ones.
    - No semantic conflict handling.
    """

    def write(self, proposal: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        subject = proposal["subject"]
        predicate = proposal["predicate"]
        object_val = proposal["object_val"]

        entry = MemoryEntry.from_proposal(proposal, agent_id=agent_id)

        # Find active visible entries with same entity key and supersede them
        same_entity = [
            r for r in self.store.records
            if r.subject == subject and r.predicate == predicate and r.status == "active"
        ]
        parent_id: Optional[str] = same_entity[-1].memory_id if same_entity else None
        entry.parent_version_id = parent_id
        entry.version_id = (same_entity[-1].version_id + 1) if same_entity else 1

        self.store.propose(entry)
        self.store.commit(
            entry.memory_id,
            resolution_action="overwrite",
            conflict_type="concurrent_update_conflict" if same_entity else "none",
            arb_metadata={"writer": "last_write_wins", "superseded_count": len(same_entity)}
        )
        self.store.set_indexed(entry.memory_id, delay=0.0)

        # Supersede previous active after commit
        for r in same_entity:
            self.store.supersede(r.memory_id, superseded_by=entry.memory_id)

        return {
            "memory_id": entry.memory_id,
            "resolution_action": "overwrite",
            "conflict_detected": bool(same_entity),
            "conflict_type": "concurrent_update_conflict" if same_entity else "none",
            "candidate_count": len(same_entity),
        }


class NaiveAppendWriter(BaseWriter):
    """
    Baseline B:
    - Appends all writes as active.
    - No conflict handling, no supersede, no arbitration.
    """

    def write(self, proposal: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        same_entity = [
            r for r in self.store.records
            if r.subject == proposal["subject"] and r.predicate == proposal["predicate"] and r.status == "active"
        ]

        entry = MemoryEntry.from_proposal(proposal, agent_id=agent_id)

        self.store.propose(entry)
        self.store.commit(
            entry.memory_id,
            resolution_action="append",
            conflict_type="none",
            arb_metadata={"writer": "naive_append"},
        )
        self.store.set_indexed(entry.memory_id, delay=0.0)

        return {
            "memory_id": entry.memory_id,
            "resolution_action": "append",
            "conflict_detected": False,
            "conflict_type": "none",
            "candidate_count": len(same_entity),
        }
