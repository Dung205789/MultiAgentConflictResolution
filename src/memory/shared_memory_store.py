from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import time

from src.memory.schema import MemoryEntry


class SharedMemoryStore:
    """
    Canonical symbolic shared-memory store.

    The store is the runtime source of truth for proposal, commit, visibility,
    supersession, retrieval, and time-sliced snapshot semantics.
    """

    def __init__(self, persistence_path: str = "shared_memory.jsonl", enable_persistence: bool = True):
        self.records: List[MemoryEntry] = []
        self._records_by_entity: Dict[str, List[MemoryEntry]] = {}
        self.persistence_path = persistence_path
        self.enable_persistence = enable_persistence and bool(persistence_path)
        self._load()

    def _index_record(self, entry: MemoryEntry) -> None:
        self._records_by_entity.setdefault(entry.entity_id or f"{entry.subject}_{entry.predicate}", []).append(entry)

    def reset(self) -> None:
        self.records = []
        self._records_by_entity = {}

    def _load(self) -> None:
        if not self.enable_persistence:
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = MemoryEntry.from_dict(json.loads(line))
                    self.records.append(entry)
                    self._index_record(entry)
        except FileNotFoundError:
            pass

    def _save(self) -> None:
        if not self.enable_persistence:
            return
        with open(self.persistence_path, "w", encoding="utf-8") as handle:
            for entry in self.records:
                handle.write(json.dumps(entry.to_dict()) + "\n")

    def propose(self, entry: MemoryEntry) -> str:
        """
        Add a proposal record to the store without making it visible.
        """
        if entry.lifecycle_stage != "proposal":
            entry.lifecycle_stage = "proposal"
        if entry.status == "active":
            entry.status = "proposal"
        if entry.visibility_state == "visible":
            entry.visibility_state = "pending_index"
        entry.canonical_status = "proposal"
        self.records.append(entry)
        self._index_record(entry)
        self._save()
        return entry.memory_id

    def propose_write(self, entry: MemoryEntry) -> str:
        """
        Backward-compatible wrapper.
        """
        return self.propose(entry)

    def _find_record(self, memory_id: str) -> Optional[MemoryEntry]:
        for record in self.records:
            if record.memory_id == memory_id:
                return record
        return None

    def commit(
        self,
        memory_id: str,
        resolution_action: Optional[str] = None,
        conflict_type: Optional[str] = None,
        arb_metadata: Optional[Dict[str, Any]] = None,
        commit_time: Optional[float] = None,
    ) -> Optional[MemoryEntry]:
        """
        Commit a proposal record but do not automatically make it visible.
        """
        record = self._find_record(memory_id)
        if record is None:
            return None

        commit_time = commit_time if commit_time is not None else time.time()
        record.mark_committed(
            resolution_action=resolution_action,
            conflict_type=conflict_type,
            arbitration_metadata=arb_metadata,
            commit_time=commit_time,
            visible=False,
        )
        self._save()
        return record

    def reject(self, memory_id: str, reason: Optional[str] = None, when: Optional[float] = None) -> Optional[MemoryEntry]:
        record = self._find_record(memory_id)
        if record is None:
            return None
        record.mark_rejected(reason=reason, when=when)
        self._save()
        return record

    def supersede(self, memory_id: str, superseded_by: Optional[str] = None, when: Optional[float] = None) -> Optional[MemoryEntry]:
        record = self._find_record(memory_id)
        if record is None:
            return None
        record.mark_superseded(superseded_by=superseded_by, when=when)
        self._save()
        return record

    def set_indexed(self, memory_id: str, delay: float = 0.1) -> Optional[MemoryEntry]:
        """
        Materialize visibility after commit. This is the proposal -> committed ->
        visible lifecycle transition.
        """
        if delay > 0:
            time.sleep(delay)
        record = self._find_record(memory_id)
        if record is None:
            return None
        record.mark_visible(visible_time=time.time())
        self._save()
        return record

    def get_visible_by_predicate(self, predicate: str, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        return [
            record
            for record in self.records
            if record.predicate == predicate
            and record.status == "active"
            and record.visibility_state == "visible"
            and (agent_id is None or record.agent_id == agent_id)
        ]

    def visible_state(self, agent_id: Optional[str] = None, at_time: Optional[float] = None) -> List[MemoryEntry]:
        return [
            record
            for record in self.records
            if record.is_visible_at(at_time)
            and (agent_id is None or record.agent_id == agent_id)
        ]

    def get_all_visible(self, agent_id: Optional[str] = None) -> List[MemoryEntry]:
        return self.visible_state(agent_id=agent_id)

    def get_by_entity(self, entity_id: str) -> List[MemoryEntry]:
        return list(self._records_by_entity.get(entity_id, []))

    def get_visible_candidates(self, subject: str, predicate: str, at_time: Optional[float] = None) -> List[MemoryEntry]:
        entity_id = f"{subject}_{predicate}"
        candidates = [
            record
            for record in self._records_by_entity.get(entity_id, [])
            if record.is_visible_at(at_time)
        ]
        candidates.sort(key=lambda x: ((x.committed_at or 0.0), x.version_id))
        return candidates

    def get_version_chain(self, entity_id: str) -> List[MemoryEntry]:
        chain = list(self._records_by_entity.get(entity_id, []))
        chain.sort(key=lambda x: (x.version_id, x.committed_at or 0.0, x.timestamp or 0.0))
        return chain

    def get_active_versions(self, entity_id: str) -> List[MemoryEntry]:
        versions = [
            record
            for record in self._records_by_entity.get(entity_id, [])
            if record.status == "active" and record.visibility_state == "visible"
        ]
        versions.sort(key=lambda x: ((x.committed_at or 0.0), x.version_id), reverse=True)
        return versions

    def snapshot(self, at_time: Optional[float]) -> List[MemoryEntry]:
        return self.visible_state(at_time=at_time)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        at_time: Optional[float] = None,
        include_non_visible: bool = False,
    ) -> List[MemoryEntry]:
        records = self.records if include_non_visible else self.visible_state(at_time=at_time)
        out: List[MemoryEntry] = []
        for record in records:
            if subject is not None and record.subject != subject:
                continue
            if predicate is not None and record.predicate != predicate:
                continue
            if include_non_visible or record.is_visible_at(at_time):
                out.append(record)
        return out

    def lifecycle_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            counts[record.lifecycle_stage] = counts.get(record.lifecycle_stage, 0) + 1
        return counts

    def to_jsonl(self) -> List[str]:
        return [json.dumps(record.to_dict()) for record in self.records]
