from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime
import time

class MemoryEntry:
    """Structured shared memory entry schema."""
    def __init__(self,
                 subject: str,
                 predicate: str,
                 object_val: Any,
                 agent_id: str,
                 confidence: float = 1.0,
                 provenance: str = "inferred",
                 raw_text: Optional[str] = None,
                 canonical_claim: Optional[str] = None,
                 memory_type: str = "fact",
                 # Bi-temporal and lifecycle fields
                 event_time: Optional[float] = None,
                 ingestion_time: Optional[float] = None,
                 valid_from: Optional[float] = None,
                 valid_until: Optional[float] = None,
                 session_id: Optional[str] = None,
                 turn_index: Optional[int] = None,
                 recall_count: int = 0,
                 last_recalled_at: Optional[float] = None,
                 supersedes: Optional[str] = None,
                 merged_from: Optional[List[str]] = None,
                 conflicts_with: Optional[str] = None,
                 canonical_status: str = "tentative") -> None:
        self.memory_id = str(uuid.uuid4())
        self.entity_id = f"{subject}_{predicate}"
        self.subject = subject
        self.predicate = predicate
        self.object_val = object_val  # str/float/list/etc.
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.raw_text = raw_text or ""
        self.canonical_claim = canonical_claim or f"{subject} {predicate} {object_val}"
        self.confidence = confidence
        self.provenance = provenance  # explicit/inferred/behavioral

        # Timestamps
        self.timestamp = time.time()           # Creation time in this system
        self.event_time = event_time           # When the event occurred in the real world
        self.ingestion_time = ingestion_time   # When this fact was ingested/believed
        self.committed_at = None
        self.indexed_at = None

        # Versioning
        self.version_id = 1
        self.parent_version_id = None
        self.supersedes = supersedes           # Memory ID this entry supersedes
        self.merged_from = merged_from or []  # List of memory IDs merged into this

        # Status and visibility
        self.status = canonical_status         # active/superseded/tentative/archived/rejected/needs_review
        self.visibility_state = "pending_index"
        self.canonical_status = canonical_status

        # Conflict and arbitration
        self.conflict_type = None
        self.resolution_action = None
        self.arbitration_metadata = None
        self.conflicts_with = conflicts_with   # Memory ID of a conflicting entry

        # Recall and lifecycle
        self.recall_count = recall_count
        self.last_recalled_at = last_recalled_at

        # Session/turn context
        self.session_id = session_id
        self.turn_index = turn_index

    def commit(self):
        self.status = "active"
        self.committed_at = time.time()

    def set_visible(self):
        self.visibility_state = "visible"
        self.indexed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class SharedMemoryStore:
    def __init__(self, persistence_path: str = "shared_memory.jsonl", enable_persistence: bool = True):
        self.records: List[MemoryEntry] = []
        self.persistence_path = persistence_path
        self.enable_persistence = enable_persistence and bool(persistence_path)
        self._load()

    def _load(self):
        if not self.enable_persistence:
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    entry = MemoryEntry(
                        subject=data.get("subject", ""),
                        predicate=data.get("predicate", ""),
                        object_val=data.get("object_val"),
                        agent_id=data.get("agent_id", "unknown"),
                        confidence=float(data.get("confidence", 1.0)),
                        provenance=data.get("provenance", "inferred"),
                        raw_text=data.get("raw_text", ""),
                        canonical_claim=data.get("canonical_claim"),
                        memory_type=data.get("memory_type", "fact"),
                        event_time=data.get("event_time"),
                        ingestion_time=data.get("ingestion_time"),
                        valid_from=data.get("valid_from"),
                        valid_until=data.get("valid_until"),
                        session_id=data.get("session_id"),
                        turn_index=data.get("turn_index"),
                        recall_count=int(data.get("recall_count", 0)),
                        last_recalled_at=data.get("last_recalled_at"),
                        supersedes=data.get("supersedes"),
                        merged_from=data.get("merged_from", []),
                        conflicts_with=data.get("conflicts_with"),
                        canonical_status=data.get("canonical_status", "tentative"),
                    )

                    # Restore persisted fields
                    entry.memory_id = data.get("memory_id", entry.memory_id)
                    entry.entity_id = data.get("entity_id", entry.entity_id)
                    entry.timestamp = data.get("timestamp", entry.timestamp)
                    entry.version_id = data.get("version_id", entry.version_id)
                    entry.parent_version_id = data.get("parent_version_id", entry.parent_version_id)
                    entry.status = data.get("status", entry.status)
                    entry.visibility_state = data.get("visibility_state", entry.visibility_state)
                    entry.committed_at = data.get("committed_at", entry.committed_at)
                    entry.indexed_at = data.get("indexed_at", entry.indexed_at)
                    entry.conflict_type = data.get("conflict_type", entry.conflict_type)
                    entry.resolution_action = data.get("resolution_action", entry.resolution_action)
                    entry.arbitration_metadata = data.get("arbitration_metadata", entry.arbitration_metadata)

                    self.records.append(entry)
        except FileNotFoundError:
            pass

    def _save(self):
        if not self.enable_persistence:
            return
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + '\n')

    def propose_write(self, entry: MemoryEntry) -> str:
        """Tentative add (pre-conflict check)."""
        self.records.append(entry)
        self._save()
        return entry.memory_id

    def get_visible_by_predicate(self, predicate: str, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        """Retrieve currently visible active entries."""
        return [r for r in self.records 
                if (r.predicate == predicate and 
                    r.status == "active" and 
                    r.visibility_state == "visible" and
                    (agent_id is None or r.agent_id == agent_id))]

    def get_all_visible(self, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        return [r for r in self.records 
                if r.status == "active" and 
                   r.visibility_state == "visible" and
                   (agent_id is None or r.agent_id == agent_id)]

    def get_by_entity(self, entity_id: str) -> List[MemoryEntry]:
        return [r for r in self.records if r.entity_id == entity_id]

    def get_version_chain(self, entity_id: str) -> List[MemoryEntry]:
        chain = [r for r in self.records if r.entity_id == entity_id]
        chain.sort(key=lambda x: (x.version_id, x.committed_at or 0.0))
        return chain

    def get_active_versions(self, entity_id: str) -> List[MemoryEntry]:
        versions = [
            r for r in self.records
            if r.entity_id == entity_id
            and r.status == "active"
            and r.visibility_state == "visible"
        ]
        # Sort primarily by committed_at (most recent first), then by version_id as tiebreaker
        versions.sort(key=lambda x: ((x.committed_at or 0.0), x.version_id), reverse=True)
        return versions

    def commit(self, memory_id: str, resolution_action: str = None, conflict_type: str = None, arb_metadata: dict = None):
        """Post-conflict commit w/ metadata."""
        for r in self.records:
            if r.memory_id == memory_id:
                r.status = "active"
                r.committed_at = time.time()
                r.resolution_action = resolution_action
                r.conflict_type = conflict_type
                r.arbitration_metadata = arb_metadata
                # Supersede parents if overwrite/version
                if resolution_action in ["overwrite", "version"]:
                    if r.parent_version_id:
                        parent = next((p for p in self.records if p.memory_id == r.parent_version_id), None)
                        if parent:
                            parent.status = "superseded"
                self._save()
                break

    def set_indexed(self, memory_id: str, delay: float = 0.1):
        """Simulate indexing delay."""
        time.sleep(delay)
        for r in self.records:
            if r.memory_id == memory_id:
                r.set_visible()
                self._save()
                break

    def to_jsonl(self) -> List[str]:
        return [json.dumps(r.to_dict()) for r in self.records]
