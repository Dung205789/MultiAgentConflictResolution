from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
import uuid
from typing import Any, Dict, List, Optional

from src.memory.canonicalization import (
    build_canonical_claim,
    build_entity_id,
    canonicalize_memory_triplet,
)


MEMORY_LIFECYCLE_STAGES = {
    "proposal",
    "committed",
    "visible",
    "superseded",
    "rejected",
    "archived",
    "tentative",
}


def _now() -> float:
    return time.time()


def _new_memory_id() -> str:
    return str(uuid.uuid4())


@dataclass
class MemoryEntry:
    """
    Canonical shared-memory record used across:
    - benchmark adapters and gold states
    - runtime proposals
    - committed store state
    - retrieval
    - reporting

    The repo previously had separate scenario and runtime memory schemas.
    This dataclass is now the single source of truth.
    """

    subject: str
    predicate: str
    object_val: Any
    agent_id: Optional[str] = None

    confidence: Optional[float] = None
    provenance: Optional[str] = None
    raw_text: str = ""
    canonical_claim: Optional[str] = None
    memory_type: str = "fact"

    status: str = "active"
    lifecycle_stage: str = "committed"
    visibility_state: str = "visible"
    canonical_status: Optional[str] = None

    timestamp: Optional[float] = None
    event_time: Optional[float] = None
    ingestion_time: Optional[float] = None
    committed_at: Optional[float] = None
    indexed_at: Optional[float] = None
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None

    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    recall_count: int = 0
    last_recalled_at: Optional[float] = None

    version_id: int = 1
    parent_version_id: Optional[str] = None
    supersedes: Optional[str] = None
    merged_from: List[str] = field(default_factory=list)
    conflicts_with: Optional[str] = None

    resolution_action: Optional[str] = None
    conflict_type: Optional[str] = None
    arbitration_metadata: Optional[Dict[str, Any]] = None

    rationale: Optional[str] = None
    support_spans: List[Dict[str, Any]] = field(default_factory=list)
    extractor_id: Optional[str] = None
    challenger_metadata: Optional[Dict[str, Any]] = None

    memory_id: str = field(default_factory=_new_memory_id)
    entity_id: Optional[str] = None
    canonical_subject: Optional[str] = None
    canonical_predicate: Optional[str] = None
    canonical_object_val: Optional[Any] = None

    def __post_init__(self) -> None:
        canonical_subject, canonical_predicate, canonical_object_val = canonicalize_memory_triplet(
            self.subject,
            self.predicate,
            self.object_val,
            raw_text=self.raw_text,
        )
        self.subject = canonical_subject
        self.predicate = canonical_predicate
        self.object_val = canonical_object_val
        self.canonical_subject = canonical_subject
        self.canonical_predicate = canonical_predicate
        self.canonical_object_val = canonical_object_val
        self.entity_id = build_entity_id(
            self.subject,
            self.predicate,
            raw_text=self.raw_text,
            object_val=self.object_val,
        )

        if self.timestamp is None:
            self.timestamp = _now()
        if self.event_time is None:
            self.event_time = self.timestamp
        if self.ingestion_time is None:
            self.ingestion_time = self.timestamp
        if self.valid_from is None and self.status in {"active", "tentative"}:
            self.valid_from = self.event_time

        if self.confidence is None:
            self.confidence = 1.0
        if self.provenance is None:
            self.provenance = "inferred"

        self.canonical_claim = build_canonical_claim(
            self.subject,
            self.predicate,
            self.object_val,
            raw_text=self.raw_text,
        )
        if self.canonical_status is None:
            self.canonical_status = self.status

        # Keep visible / committed states coherent for benchmark gold records.
        if self.lifecycle_stage == "visible" and self.indexed_at is None:
            self.indexed_at = self.committed_at or self.timestamp
        if self.lifecycle_stage in {"committed", "visible"} and self.committed_at is None and self.status == "active":
            self.committed_at = self.timestamp

    @classmethod
    def from_proposal(cls, proposal: Dict[str, Any], agent_id: str, proposal_time: Optional[float] = None) -> "MemoryEntry":
        proposal_time = proposal_time if proposal_time is not None else _now()
        entry = cls(
            subject=proposal["subject"],
            predicate=proposal["predicate"],
            object_val=proposal["object_val"],
            agent_id=agent_id,
            confidence=float(proposal.get("confidence", 1.0)),
            provenance=proposal.get("provenance", "inferred"),
            raw_text=proposal.get("raw_text", ""),
            canonical_claim=proposal.get("canonical_claim"),
            memory_type=proposal.get("memory_type", "fact"),
            status="proposal",
            lifecycle_stage="proposal",
            visibility_state="pending_index",
            canonical_status="proposal",
            timestamp=proposal_time,
            event_time=proposal.get("event_time", proposal_time),
            ingestion_time=proposal.get("ingestion_time", proposal_time),
            valid_from=proposal.get("valid_from"),
            valid_until=proposal.get("valid_until"),
            session_id=proposal.get("session_id"),
            turn_index=proposal.get("turn_index"),
            recall_count=int(proposal.get("recall_count", 0)),
            last_recalled_at=proposal.get("last_recalled_at"),
            supersedes=proposal.get("supersedes"),
            merged_from=list(proposal.get("merged_from", [])),
            conflicts_with=proposal.get("conflicts_with"),
            rationale=proposal.get("rationale"),
            support_spans=list(proposal.get("support_spans", [])),
            extractor_id=proposal.get("extractor_id"),
            challenger_metadata=proposal.get("challenger_metadata"),
        )
        return entry

    @property
    def proposal_time(self) -> float:
        return float(self.timestamp or 0.0)

    def mark_committed(
        self,
        resolution_action: Optional[str] = None,
        conflict_type: Optional[str] = None,
        arbitration_metadata: Optional[Dict[str, Any]] = None,
        commit_time: Optional[float] = None,
        visible: bool = False,
    ) -> None:
        commit_time = commit_time if commit_time is not None else _now()
        self.status = "active"
        self.lifecycle_stage = "visible" if visible else "committed"
        self.visibility_state = "visible" if visible else "pending_index"
        self.canonical_status = "active"
        self.committed_at = commit_time
        self.valid_from = self.valid_from if self.valid_from is not None else (self.event_time or commit_time)
        self.resolution_action = resolution_action
        self.conflict_type = conflict_type
        if arbitration_metadata is not None:
            self.arbitration_metadata = arbitration_metadata
        if visible and self.indexed_at is None:
            self.indexed_at = commit_time

    def mark_visible(self, visible_time: Optional[float] = None) -> None:
        visible_time = visible_time if visible_time is not None else _now()
        self.lifecycle_stage = "visible"
        self.visibility_state = "visible"
        self.indexed_at = visible_time
        if self.committed_at is None:
            self.committed_at = visible_time
        if self.valid_from is None:
            self.valid_from = self.event_time or visible_time

    def mark_superseded(self, superseded_by: Optional[str] = None, when: Optional[float] = None) -> None:
        when = when if when is not None else _now()
        self.status = "superseded"
        self.lifecycle_stage = "superseded"
        self.visibility_state = "hidden"
        self.canonical_status = "superseded"
        self.valid_until = when
        metadata = dict(self.arbitration_metadata or {})
        if superseded_by:
            metadata["superseded_by"] = superseded_by
        self.arbitration_metadata = metadata

    def mark_rejected(self, reason: Optional[str] = None, when: Optional[float] = None) -> None:
        when = when if when is not None else _now()
        self.status = "rejected"
        self.lifecycle_stage = "rejected"
        self.visibility_state = "hidden"
        self.canonical_status = "rejected"
        self.valid_until = when
        metadata = dict(self.arbitration_metadata or {})
        if reason:
            metadata["rejected_reason"] = reason
        self.arbitration_metadata = metadata

    def is_visible_at(self, at_time: Optional[float]) -> bool:
        if self.status != "active":
            return False
        if self.visibility_state != "visible":
            return False
        if at_time is None:
            return True
        start = self.valid_from if self.valid_from is not None else (self.committed_at or self.timestamp or 0.0)
        end = self.valid_until
        if at_time < start:
            return False
        if end is not None and at_time >= end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**dict(data))
