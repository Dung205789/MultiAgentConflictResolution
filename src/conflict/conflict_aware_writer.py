"""
Enhanced conflict-aware writer with improved arbitration and action semantics.
"""
from typing import Dict, Any, List, Tuple, Optional
import time
import json
import os

from src.memory.shared_memory_store import SharedMemoryStore, MemoryEntry
from src.conflict.staleness_detector import StalenessDetector
from src.conflict.conflict_detector import detect_conflict_type

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class ConflictAwareWriter:
    """
    Enhanced conflict-aware writer with concrete store-level action effects:
    - overwrite: supersede latest active candidate
    - reject: do not commit
    - merge: commit a materially merged object value
    - keep_multiple_versions: retain concurrent active branches with metadata
    - defer: commit as tentative/pending review
    """

    def __init__(self, store: SharedMemoryStore, staleness_detector: StalenessDetector, mode: str = "debug_fallback", config_path: str = "configs/arbitration.yaml"):
        """
        Initialize the conflict-aware writer.

        Args:
            store: The shared memory store
            staleness_detector: Detector for stale reads
            mode: Either "research_strict" (fail if models required but unavailable) or
                  "debug_fallback" (allow fallback to rule-based)
            config_path: Path to arbitration configuration YAML
        """
        self.store = store
        self.staleness_detector = staleness_detector
        self.mode = mode

        # Load configuration
        self.config = self._load_config(config_path)

        # Set arbitration parameters from config
        self.arbitration_config = self.config.get("arbitration", {})
        self.arbitration_weights = self.arbitration_config.get("weights", {
            "confidence": 0.4,
            "provenance": 0.3,
            "recency": 0.2,
            "authority": 0.1,
        })
        self.arbitration_thresholds = self.config.get("thresholds", {})
        self.provenance_weights = self.config.get("provenance_weights", {
            "explicit": 1.0,
            "behavioral": 0.85,
            "inferred": 0.7,
            "llm_inferred": 0.6,
            "unknown": 0.4,
        })
        self.recency_half_life = self.arbitration_config.get("recency_half_life", 3600.0)
        self.decay_params = self.config.get("decay", {
            "lambda_base": 0.0001,
            "recall_boost_beta": 0.3,
            "recall_boost_gamma": 0.1,
        })

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load arbitration configuration from YAML file."""
        if not HAVE_YAML:
            print("Warning: PyYAML not installed, using default configuration")
            return {}

        # Try absolute path or relative to project root
        if not os.path.isabs(config_path):
            # Assume relative to project root (two levels up from this file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(project_root, config_path)

        try:
            with open(config_path, 'r') as f:
                import yaml as yaml_mod
                config = yaml_mod.safe_load(f)
                if config is None:
                    config = {}
                print(f"Loaded arbitration config from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}, using defaults")
            return {}

    def _retrieve_candidates(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve all visible candidates with the same subject and predicate."""
        visible = [r.to_dict() for r in self.store.get_all_visible()]
        return [
            r for r in visible
            if r.get("subject") == proposal.get("subject")
            and r.get("predicate") == proposal.get("predicate")
        ]

    def _normalize_recency(self, timestamp: float, reference_time: float) -> float:
        """
        Normalize recency using an exponential decay function.

        Args:
            timestamp: The timestamp to normalize
            reference_time: The reference time (usually the current time)

        Returns:
            Normalized recency score between 0 and 1
        """
        if reference_time <= 0 or timestamp >= reference_time:
            return 1.0

        time_diff = reference_time - timestamp
        # Exponential decay: score = 2^(-time_diff/half_life)
        return 2 ** (-time_diff / self.recency_half_life)

    def _score_memory(self, mem: Dict[str, Any], recency_ref: float) -> Dict[str, float]:
        """
        Score a memory entry based on confidence, provenance, recency, and authority.

        Args:
            mem: Memory entry to score
            recency_ref: Reference time for recency normalization

        Returns:
            Dictionary with individual scores and total score
        """
        # Extract and normalize individual factors
        confidence = float(mem.get("confidence", 0.0))

        provenance_type = str(mem.get("provenance", "unknown"))
        provenance_score = self.provenance_weights.get(provenance_type, 0.4)

        timestamp = float(mem.get("committed_at") or mem.get("timestamp") or 0.0)
        recency_score = self._normalize_recency(timestamp, recency_ref)

        # Use agent_authority if available, else default to 1.0 (neutral)
        authority_score = float(mem.get("agent_authority", 1.0))

        # Calculate weighted total score using arbitration weights
        total_score = (
            self.arbitration_weights.get("confidence", 0.4) * confidence +
            self.arbitration_weights.get("provenance", 0.3) * provenance_score +
            self.arbitration_weights.get("recency", 0.2) * recency_score +
            self.arbitration_weights.get("authority", 0.1) * authority_score
        )

        return {
            "confidence": confidence,
            "provenance": provenance_score,
            "recency": recency_score,
            "authority": authority_score,
            "total": total_score,
            "provenance_type": provenance_type,
        }

    def _arbitrate(
        self,
        conflict_type: str,
        conflict_details: Dict[str, Any],
        proposal: Dict[str, Any],
        candidates: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced arbitration logic with more principled decision making.
        
        Args:
            conflict_type: Type of conflict detected
            conflict_details: Additional details about the conflict
            proposal: The proposed memory entry
            candidates: Existing memory entries with the same subject and predicate
            
        Returns:
            Tuple of (action, details)
        """
        # No conflict case
        if conflict_type == "none":
            return "append", {"reason": "no_conflict"}
        
        # Duplicate handling
        if conflict_type in ["exact_duplicate", "semantic_duplicate"]:
            return "reject", {"reason": f"{conflict_type}_detected", "details": conflict_details}
        
        # Get the latest candidate for comparison
        latest = candidates[-1] if candidates else {}
        
        # Calculate reference time for recency normalization
        recency_ref = time.time()
        
        # Score both the proposal and the latest candidate
        proposal_scored = dict(proposal)
        proposal_scored["agent_authority"] = float(proposal.get("agent_authority", 0.5))
        proposal_scored["timestamp"] = time.time()
        
        new_scores = self._score_memory(proposal_scored, recency_ref)
        old_scores = self._score_memory(latest, recency_ref) if latest else {"total": 0.0}
        
        # Calculate score margin
        margin = new_scores["total"] - old_scores["total"]
        confidence_margin = new_scores["confidence"] - old_scores.get("confidence", 0.0)
        
        # Stale read conflict handling
        if conflict_type == "stale_read_conflict":
            # If the new entry has significantly higher confidence, overwrite despite staleness
            if confidence_margin > self.arbitration_thresholds.get("overwrite_margin", 0.15):
                return "overwrite", {
                    "reason": "high_confidence_despite_staleness",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "margin": margin,
                }
            # Otherwise, defer to manual review
            return "defer", {
                "reason": "stale_read_requires_review",
                "new_scores": new_scores,
                "old_scores": old_scores,
                "margin": margin,
            }
        
        # Semantic overlap handling - good candidate for merging
        if conflict_type == "semantic_overlap":
            # Special case: if both values are JSON objects, always merge
            if conflict_details.get("json_mergeable"):
                return "merge", {
                    "reason": "json_objects_mergeable",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                }
            similarity = conflict_details.get("similarity", 0.0)
            # High similarity but not duplicate - merge
            if similarity >= 0.7:
                return "merge", {
                    "reason": "high_semantic_overlap_suitable_for_merge",
                    "similarity": similarity,
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                }
            # Moderate similarity - keep both if confidence is similar
            elif abs(confidence_margin) < self.arbitration_thresholds.get("keep_multiple_versions_margin", 0.08):
                return "keep_multiple_versions", {
                    "reason": "moderate_overlap_similar_confidence",
                    "similarity": similarity,
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                }
            # Otherwise, prefer the higher confidence one
            elif confidence_margin > 0:
                return "overwrite", {
                    "reason": "higher_confidence_moderate_overlap",
                    "similarity": similarity,
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                }
            else:
                return "reject", {
                    "reason": "lower_confidence_moderate_overlap",
                    "similarity": similarity,
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                }
        
        # Compatible extension - keep both versions
        if conflict_type == "compatible_extension":
            return "keep_multiple_versions", {
                "reason": "compatible_information",
                "new_scores": new_scores,
                "old_scores": old_scores,
            }
        
        # Mutually exclusive or potential contradiction
        if conflict_type in ["mutually_exclusive", "potential_contradiction"]:
            # If new entry has significantly higher confidence, overwrite
            if confidence_margin > self.arbitration_thresholds.get("overwrite_margin", 0.15):
                return "overwrite", {
                    "reason": "higher_confidence_contradiction",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "margin": margin,
                }
            # If similar confidence, use recency to break tie for mutually_exclusive
            # (newer information is generally better)
            elif abs(confidence_margin) < self.arbitration_thresholds.get("keep_multiple_versions_margin", 0.08):
                if conflict_type == "mutually_exclusive":
                    # For mutually exclusive, prefer the newer entry (overwrite)
                    # The new entry is newer by construction (higher timestamp)
                    return "overwrite", {
                        "reason": "mutually_exclusive_newer_info_overwrites",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                    }
                else:  # potential_contradiction
                    return "keep_multiple_versions", {
                        "reason": "contradictory_similar_confidence",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "contradiction_type": conflict_type,
                    }
            # If lower confidence, reject
            else:
                return "reject", {
                    "reason": "lower_confidence_contradiction",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "margin": margin,
                }
        
        # Default case - defer to manual review
        return "defer", {
            "reason": "unhandled_conflict_type",
            "conflict_type": conflict_type,
            "new_scores": new_scores,
            "old_scores": old_scores,
        }

    def _build_entry(self, proposal: Dict[str, Any], agent_id: str, candidates: List[Dict[str, Any]]) -> MemoryEntry:
        """Build a new memory entry from the proposal."""
        entry = MemoryEntry(
            subject=proposal["subject"],
            predicate=proposal["predicate"],
            object_val=proposal["object_val"],
            agent_id=agent_id,
            confidence=float(proposal.get("confidence", 1.0)),
            provenance=proposal.get("provenance", "inferred"),
            raw_text=proposal.get("raw_text", ""),
            canonical_claim=proposal.get("canonical_claim"),
            memory_type=proposal.get("memory_type", "fact"),
        )
        if candidates:
            parent_id = candidates[-1].get("memory_id")
            parent_version = int(candidates[-1].get("version_id", 1))
            entry.parent_version_id = parent_id
            entry.version_id = parent_version + 1
        return entry

    def _apply_action_effects(
        self,
        entry: MemoryEntry,
        action: str,
        conflict_type: str,
        candidates: List[Dict[str, Any]],
        arb_metadata: Dict[str, Any],
    ) -> None:
        """Apply the effects of the chosen action on the memory store."""
        if action == "overwrite" and candidates:
            # Note: store.commit already supersedes the parent for "overwrite"
            # Here we only add relationship metadata to the old entry
            latest_id = candidates[-1].get("memory_id")
            for r in self.store.records:
                if r.memory_id == latest_id:
                    # Add relationship metadata (do not change status here, store.commit already did)
                    if r.arbitration_metadata is None:
                        r.arbitration_metadata = {}
                    r.arbitration_metadata.update({
                        "superseded_by": entry.memory_id,
                        "superseded_reason": conflict_type,
                    })
                    break

        elif action == "keep_multiple_versions":
            # Keep both active and mark branch metadata for explicit multi-version handling
            if entry.arbitration_metadata is None:
                entry.arbitration_metadata = {}

            # Add rich branch metadata
            entry.arbitration_metadata.update({
                "branch_mode": "parallel_active_versions",
                "branch_id": f"{entry.entity_id}_branch_{entry.version_id}",
                "conflict_type": conflict_type,
                "parent_memory_id": entry.parent_version_id,
            })

            # Add relationship to parent
            if entry.parent_version_id:
                for r in self.store.records:
                    if r.memory_id == entry.parent_version_id:
                        if r.arbitration_metadata is None:
                            r.arbitration_metadata = {}
                        r.arbitration_metadata.update({
                            "has_parallel_branch": True,
                            "branch_children": r.arbitration_metadata.get("branch_children", []) + [entry.memory_id],
                        })
                        break

        elif action == "merge" and candidates:
            # Enhanced merge logic beyond simple concatenation
            latest = candidates[-1]
            latest_obj = str(latest.get("object_val", "")).strip()
            new_obj = str(entry.object_val).strip()

            # Skip merge if values are identical
            if latest_obj == new_obj:
                entry.object_val = latest_obj
            else:
                # Try to perform a structured merge if possible
                try:
                    # Check if both values are JSON objects
                    latest_json = json.loads(latest_obj) if latest_obj.startswith("{") else None
                    new_json = json.loads(new_obj) if new_obj.startswith("{") else None

                    if isinstance(latest_json, dict) and isinstance(new_json, dict):
                        # Merge JSON objects
                        merged = {**latest_json, **new_json}
                        entry.object_val = json.dumps(merged)
                    elif isinstance(latest_json, list) and isinstance(new_json, list):
                        # Merge lists with deduplication
                        merged = list(set(latest_json + new_json))
                        entry.object_val = json.dumps(merged)
                    else:
                        # Fall back to text concatenation with separator
                        entry.object_val = f"{latest_obj} | {new_obj}"
                except (json.JSONDecodeError, TypeError):
                    # Fall back to text concatenation with separator
                    entry.object_val = f"{latest_obj} | {new_obj}"

            # Update canonical claim
            entry.canonical_claim = f"{entry.subject} {entry.predicate} {entry.object_val}"

            # Add merge metadata
            if entry.arbitration_metadata is None:
                entry.arbitration_metadata = {}
            entry.arbitration_metadata.update({
                "merge_source": entry.parent_version_id,
                "merge_type": "semantic_enrichment",
            })

            # Supersede the old entry to avoid duplicate visible entries
            latest_id = latest.get("memory_id")
            for r in self.store.records:
                if r.memory_id == latest_id:
                    r.status = "superseded"
                    if r.arbitration_metadata is None:
                        r.arbitration_metadata = {}
                    r.arbitration_metadata.update({
                        "superseded_by": entry.memory_id,
                        "superseded_reason": conflict_type,
                    })
                    break

        elif action == "defer":
            # Mark as tentative and add detailed metadata
            entry.status = "tentative"
            if entry.arbitration_metadata is None:
                entry.arbitration_metadata = {}
            entry.arbitration_metadata.update({
                "defer_reason": conflict_type,
                "requires_review": True,
                "review_priority": "high" if conflict_type in ["mutually_exclusive", "stale_read_conflict"] else "medium",
            })

    def write(self, proposal: Dict[str, Any], agent_id: str, read_snapshot_time: float) -> Dict[str, Any]:
        """
        Write a proposal to the memory store with conflict awareness.
        
        Args:
            proposal: The proposed memory entry
            agent_id: ID of the agent making the proposal
            read_snapshot_time: When the agent read the memory before making this proposal
            
        Returns:
            Result of the write operation
        """
        # Retrieve candidates and detect conflicts
        candidates = self._retrieve_candidates(proposal)
        conflict_type, conflict_details = detect_conflict_type(
            proposal, candidates, read_snapshot_time, self.staleness_detector, mode=self.mode
        )
        
        # Arbitrate to decide action
        action, arbitration_details = self._arbitrate(conflict_type, conflict_details, proposal, candidates)

        # Handle rejection without creating an entry
        if action == "reject":
            return {
                "committed": False,
                "conflict_detected": conflict_type != "none",
                "conflict_type": conflict_type,
                "resolution_action": action,
                "memory_id": None,
                "arbitration_details": arbitration_details,
                "conflict_details": conflict_details,
            }

        # Build and propose the new entry
        entry = self._build_entry(proposal, agent_id, candidates)

        # Add conflict metadata to the entry
        if entry.arbitration_metadata is None:
            entry.arbitration_metadata = {}
        entry.arbitration_metadata.update({
            "conflict_type": conflict_type,
            "resolution_action": action,
        })

        # Propose the write to the store
        self.store.propose_write(entry)
        
        # Commit with metadata
        self.store.commit(
            entry.memory_id,
            resolution_action=action,
            conflict_type=conflict_type,
            arb_metadata={
                "writer": "enhanced_conflict_aware",
                "candidate_count": len(candidates),
                "arbitration_details": arbitration_details,
                "conflict_details": conflict_details,
            },
        )

        # Apply action-specific effects
        self._apply_action_effects(entry, action, conflict_type, candidates, arbitration_details)
        
        # Save and index
        self.store._save()
        self.store.set_indexed(entry.memory_id, delay=0.0)

        # Return detailed result
        return {
            "committed": True,
            "memory_id": entry.memory_id,
            "conflict_detected": conflict_type != "none",
            "conflict_type": conflict_type,
            "resolution_action": action,
            "candidate_count": len(candidates),
            "arbitration_details": arbitration_details,
            "conflict_details": conflict_details,
        }
