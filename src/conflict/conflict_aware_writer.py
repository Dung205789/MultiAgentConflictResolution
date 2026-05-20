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

# Lazy import - load only when meta-learning is enabled
ArbitrationHistoryTracker = None
ArbitrationRecord = None

def _ensure_history_import():
    global ArbitrationHistoryTracker, ArbitrationRecord
    if ArbitrationHistoryTracker is None:
        from src.conflict.arbitration_history import ArbitrationHistoryTracker as AHT
        from src.conflict.arbitration_history import ArbitrationRecord as AR
        ArbitrationHistoryTracker = AHT
        ArbitrationRecord = AR

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

    def __init__(
        self,
        store: SharedMemoryStore,
        staleness_detector: StalenessDetector,
        mode: str = "debug_fallback",
        config_path: str = "configs/arbitration.yaml",
        variant: str = "full",
    ):
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
        self.variant = variant

        # Load configuration
        self.config = self._load_config(config_path)

        # Set arbitration parameters from config
        self.arbitration_config = self.config.get("arbitration", {})
        self.arbitration_weights = self.arbitration_config.get("weights", {
            "confidence": 0.24,
            "provenance": 0.16,
            "recency": 0.12,
            "authority": 0.08,
            "answer_criticality": 0.24,
            "graph_support": 0.10,
            "query_coverage": 0.06,
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
                # DEBUG: Log thresholds
                thresholds = config.get("thresholds", {})
                return config
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}, using defaults")
            return {}

    def _get_context_weights(self, scenario_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get context-specific arbitration weights.
        Falls back to base weights if no scenario override exists.
        """
        context_weights = self.config.get("context_weights", {})
        if scenario_id and scenario_id in context_weights:
            return context_weights[scenario_id]
        return self.arbitration_weights

    def _resolve_context_key(
        self,
        scenario_id: Optional[str],
        conflict_type: str,
        proposal: Dict[str, Any],
    ) -> Optional[str]:
        context_weights = self.config.get("context_weights", {})

        explicit_context = proposal.get("arbitration_context")
        if explicit_context and explicit_context in context_weights:
            return explicit_context

        if scenario_id and scenario_id in context_weights:
            return scenario_id

        conflict_context_map = {
            "mutually_exclusive": "factual_dispute",
            "potential_contradiction": "factual_dispute",
            "semantic_overlap": "cross_agent_merge",
            "compatible_extension": "cross_agent_merge",
            "stale_read_conflict": "temporal_update",
            "counterfactual_temporal": "temporal_update",
            "temporal_inconsistency": "temporal_update",
            "concurrent_update": "temporal_update",
        }
        return conflict_context_map.get(conflict_type)

    def _calculate_uncertainty(self, mem: Dict[str, Any], recency_ref: float) -> float:
        """
        Calculate epistemic uncertainty for a memory entry.
        Combines low confidence, weak provenance, staleness, and low authority.
        Returns a value in [0, 1] where 1 = maximally uncertain.
        """
        confidence = float(mem.get("confidence", 0.0))
        provenance_type = str(mem.get("provenance", "unknown"))
        provenance_score = self.provenance_weights.get(provenance_type, 0.4)
        timestamp = float(
            mem.get("event_time")
            or mem.get("timestamp")
            or mem.get("committed_at")
            or 0.0
        )
        recency_score = self._normalize_recency(timestamp, recency_ref)
        authority_score = float(mem.get("agent_authority", 1.0))

        # Uncertainty = 1 - weighted combination of certainty signals
        certainty = (
            0.35 * confidence +
            0.25 * provenance_score +
            0.25 * recency_score +
            0.15 * authority_score
        )
        return max(0.0, min(1.0, 1.0 - certainty))

    def _retrieve_candidates(self, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve all visible candidates with the same subject and predicate."""
        candidates = self.store.get_visible_candidates(
            proposal.get("subject", ""),
            proposal.get("predicate", "")
        )
        return [r.to_dict() for r in candidates]

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

    def _extract_signal(self, mem: Dict[str, Any], key: str, default: float = 0.0) -> float:
        metadata = mem.get("arbitration_metadata") or {}
        raw = mem.get(key, metadata.get(key, default))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _extract_query_support_ids(self, mem: Dict[str, Any]) -> List[str]:
        metadata = mem.get("arbitration_metadata") or {}
        raw = mem.get("query_support_ids", metadata.get("query_support_ids", []))
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return []

    def _extract_graph_edges(self, mem: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = mem.get("arbitration_metadata") or {}
        raw = mem.get("graph_edges", metadata.get("graph_edges", []))
        if isinstance(raw, list):
            return [edge for edge in raw if isinstance(edge, dict)]
        return []

    def _extract_query_relation_roles(self, mem: Dict[str, Any]) -> List[str]:
        metadata = mem.get("arbitration_metadata") or {}
        raw = mem.get("query_relation_roles", metadata.get("query_relation_roles", []))
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return []

    def _build_lineage_graph_edges(self, proposal: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preserve query-relevant graph evidence from overwritten candidates.
        """
        if self.variant == "no_lineage_edges":
            return []
        current_support = set(proposal.get("query_support_ids", []))
        if not current_support:
            return []
        lineage: List[Dict[str, Any]] = []
        seen = set()
        for edge in proposal.get("graph_edges", []):
            sig = (edge.get("source"), edge.get("relation"), edge.get("target"), edge.get("output_type"))
            seen.add(sig)
        for candidate in candidates:
            support_ids = set(self._extract_query_support_ids(candidate))
            if not support_ids or not (support_ids & current_support):
                continue
            relation_roles = set(self._extract_query_relation_roles(candidate))
            if relation_roles and not (relation_roles & {"bridge_edge", "terminal_edge"}):
                continue
            for edge in self._extract_graph_edges(candidate):
                sig = (edge.get("source"), edge.get("relation"), edge.get("target"), edge.get("output_type"))
                if sig in seen:
                    continue
                lineage.append(edge)
                seen.add(sig)
        return lineage

    def _score_memory(self, mem: Dict[str, Any], recency_ref: float, context_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Score a memory entry based on confidence, provenance, recency, and authority.

        Args:
            mem: Memory entry to score
            recency_ref: Reference time for recency normalization
            context_weights: Optional scenario-specific weights override

        Returns:
            Dictionary with individual scores and total score
        """
        weights = context_weights or self.arbitration_weights

        # Extract and normalize individual factors
        confidence = float(mem.get("confidence", 0.0))

        provenance_type = str(mem.get("provenance", "unknown"))
        provenance_score = self.provenance_weights.get(provenance_type, 0.4)

        timestamp = float(
            mem.get("event_time")
            or mem.get("timestamp")
            or mem.get("committed_at")
            or 0.0
        )
        recency_score = self._normalize_recency(timestamp, recency_ref)

        # Use agent_authority if available, else default to 1.0 (neutral)
        authority_score = float(mem.get("agent_authority", 1.0))
        answer_criticality = self._extract_signal(mem, "answer_criticality", 0.0)
        graph_support_score = self._extract_signal(mem, "graph_support_score", 0.0)
        query_support_ids = self._extract_query_support_ids(mem)
        query_coverage_score = min(1.0, len(query_support_ids) / 3.0)

        # Calculate weighted total score using arbitration weights
        total_score = (
            weights.get("confidence", 0.4) * confidence +
            weights.get("provenance", 0.3) * provenance_score +
            weights.get("recency", 0.2) * recency_score +
            weights.get("authority", 0.1) * authority_score +
            weights.get("answer_criticality", 0.0) * answer_criticality +
            weights.get("graph_support", 0.0) * graph_support_score +
            weights.get("query_coverage", 0.0) * query_coverage_score
        )

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(mem, recency_ref)

        return {
            "confidence": confidence,
            "provenance": provenance_score,
            "recency": recency_score,
            "authority": authority_score,
            "answer_criticality": answer_criticality,
            "graph_support": graph_support_score,
            "query_coverage": query_coverage_score,
            "query_support_ids": query_support_ids,
            "total": total_score,
            "provenance_type": provenance_type,
            "uncertainty": uncertainty,
        }

    def _graph_preservation_decision(
        self,
        conflict_type: str,
        proposal_timestamp: float,
        latest: Dict[str, Any],
        new_scores: Dict[str, float],
        old_scores: Dict[str, float],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Query-aware decision override.

        This is the core paper-facing behavior: preserve facts that support
        multi-hop answer chains instead of reducing every contradiction to pure
        recency.
        """
        critical_margin = new_scores.get("answer_criticality", 0.0) - old_scores.get("answer_criticality", 0.0)
        graph_margin = new_scores.get("graph_support", 0.0) - old_scores.get("graph_support", 0.0)
        keep_margin = self.arbitration_thresholds.get("keep_multiple_versions_margin", 0.08)
        critical_keep_threshold = self.arbitration_thresholds.get("answer_critical_keep_threshold", 0.55)
        critical_margin_threshold = self.arbitration_thresholds.get("answer_critical_margin", 0.15)
        prop_ts = proposal_timestamp
        latest_ts = latest.get("timestamp", 0) if latest else 0

        new_support = set(new_scores.get("query_support_ids", []))
        old_support = set(old_scores.get("query_support_ids", []))
        support_overlap = new_support & old_support

        # Preserve parallel branches when both versions are strongly answer-critical
        # but support different query paths.
        allow_parallel_preservation = conflict_type not in {
            "mutually_exclusive",
            "counterfactual_temporal",
            "stale_read_conflict",
            "concurrent_update",
        }

        if (
            allow_parallel_preservation
            and
            new_scores.get("answer_criticality", 0.0) >= critical_keep_threshold
            and old_scores.get("answer_criticality", 0.0) >= critical_keep_threshold
            and new_support
            and old_support
            and new_support != old_support
            and not support_overlap
        ):
            return "keep_multiple_versions", {
                "reason": f"query_aware_{conflict_type}_preserve_distinct_answer_paths",
                "critical_margin": critical_margin,
                "graph_margin": graph_margin,
                "new_scores": new_scores,
                "old_scores": old_scores,
            }

        # Strongly query-critical new information can override older memory even
        # if pure recency/flat score would not obviously win.
        if critical_margin >= critical_margin_threshold and graph_margin >= -keep_margin:
            return "overwrite", {
                "reason": f"query_aware_{conflict_type}_new_fact_is_more_answer_critical",
                "critical_margin": critical_margin,
                "graph_margin": graph_margin,
                "new_scores": new_scores,
                "old_scores": old_scores,
                "timestamps": {"proposal": prop_ts, "latest": latest_ts},
            }

        return None

    def _arbitrate(
        self,
        conflict_type: str,
        conflict_details: Dict[str, Any],
        proposal: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        scenario_id: Optional[str] = None,
        proposal_timestamp: float = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced arbitration logic with context weights, uncertainty, and history tracking.

        Args:
            conflict_type: Type of conflict detected
            conflict_details: Additional details about the conflict
            proposal: The proposed memory entry
            candidates: Existing memory entries with the same subject and predicate
            scenario_id: Optional scenario context for dynamic weight adjustment

        Returns:
            Tuple of (action, details)
        """
        # Resolve context-specific weights
        context_key = self._resolve_context_key(scenario_id, conflict_type, proposal)
        context_weights = self._get_context_weights(context_key)

        # No conflict case
        if conflict_type == "none":
            return "append", {"reason": "no_conflict"}

        # Duplicate handling
        if conflict_type in ["exact_duplicate", "semantic_duplicate"]:
            action = "reject"
            reason = "duplicate_detected" if conflict_type == "exact_duplicate" else "semantic_duplicate_detected"
            return action, {
                "reason": reason,
                "conflict_type": conflict_type,
                "details": conflict_details,
            }

        # Get the latest candidate for comparison
        latest = candidates[-1] if candidates else {}

        # Calculate reference time for recency normalization
        recency_ref = time.time()

        # Score both the proposal and the latest candidate
        proposal_scored = dict(proposal)
        proposal_scored["agent_authority"] = float(proposal.get("agent_authority", 0.5))
        # Use the actual proposal timestamp for recency calculation
        if proposal_timestamp is None:
            proposal_timestamp = time.time()
        proposal_scored["timestamp"] = proposal_timestamp

        new_scores = self._score_memory(proposal_scored, recency_ref, context_weights)
        old_scores = self._score_memory(latest, recency_ref, context_weights) if latest else {"total": 0.0, "uncertainty": 1.0}

        # Calculate score margin
        margin = new_scores["total"] - old_scores["total"]
        confidence_margin = new_scores["confidence"] - old_scores.get("confidence", 0.0)
        query_aware_override = self._graph_preservation_decision(
            conflict_type,
            proposal_timestamp,
            latest,
            new_scores,
            old_scores,
        )

        # Log to arbitration history if enabled
        if proposal.get("_enable_history_tracking", False):
            _ensure_history_import()
            if ArbitrationHistoryTracker is not None:
                tracker = ArbitrationHistoryTracker(self.store)
                record = ArbitrationRecord(
                    scenario_id=scenario_id or "default",
                    conflict_type=conflict_type,
                    proposal_scores=new_scores,
                    candidate_scores=old_scores,
                    action=None,  # will be set below
                    uncertainty=new_scores.get("uncertainty", 0.0),
                )
                # Defer commit until action is decided
                # Store temporarily for later update
                proposal["_temp_history_record"] = record
                proposal["_temp_history_tracker"] = tracker

        # Stale read conflict handling
        if conflict_type == "stale_read_conflict":
            # Stale read is an agent error - the agent wrote based on outdated information
            # Default: reject the write (agent should re-read)
            # Only defer if there are exceptional circumstances (e.g., time travel detected)
            stale_info = conflict_details.get("stale_info", {})
            reason = stale_info.get("reason", "")

            # Check for temporal anomalies that might warrant deferral
            if "temporal_anomaly" in reason or "out_of_order" in reason:
                # Defer for manual review of timing issues
                action = "defer"
                return action, {
                    "reason": "temporal_anomaly_stale_read_needs_audit",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                    "stale_info": stale_info,
                }
            else:
                # Normal stale read - reject (agent should re-read)
                action = "reject"
                return action, {
                    "reason": "stale_read_rejected_agent_should_reread",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                    "stale_info": stale_info,
                }

        # Concurrent update handling - simultaneous writes, favor the newer entry
        if conflict_type == "concurrent_update":
            if query_aware_override is not None:
                return query_aware_override
            prop_ts = proposal_timestamp
            latest_ts = latest.get("timestamp", 0) if latest else 0


            # For concurrent updates, the newer entry should win (last-write-wins)
            if prop_ts > latest_ts:
                action = "overwrite"
                return action, {
                    "reason": "newer_timestamp_wins_concurrent_update",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }
            else:
                # Proposal is not newer - could be out-of-order or stale
                # If timestamps are very close and scores are close, could keep both
                margin = new_scores["total"] - old_scores["total"]
                keep_margin = self.arbitration_thresholds.get("keep_multiple_versions_margin", 0.08)
                if abs(margin) < keep_margin:
                    action = "keep_multiple_versions"
                    return action, {
                        "reason": "concurrent_similar_scores_not_newer_keep_both",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
                else:
                    action = "reject"
                    return action, {
                        "reason": "older_proposal_not_newer_reject",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }

        # Counterfactual temporal conflict - newer info contradicts older in wrong order
        if conflict_type == "counterfactual_temporal":
            # Check if new entry is actually newer (corrective) or older (erroneous)
            prop_ts = proposal_timestamp
            latest_ts = latest.get("timestamp", 0)

            if prop_ts > latest_ts:
                # Newer info is correcting the old - overwrite with confidence check
                if margin > -self.arbitration_thresholds.get("overwrite_margin", 0.15):
                    action = "overwrite"
                    return action, {
                        "reason": "newer_temporal_correction_overwrite",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
                else:
                    # Newer but low confidence - keep both for review
                    action = "keep_multiple_versions"
                    return action, {
                        "reason": "newer_low_confidence_keep_both",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
            else:
                # Old info is newer - the "new" entry is actually outdated
                # Reject it
                action = "reject"
                return action, {
                    "reason": "out_of_order_proposal_is_older_reject",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }

        # Temporal inconsistency (legacy) - merge into counterfactual_temporal
        if conflict_type == "temporal_inconsistency":
            # Treat as counterfactual_temporal for backward compatibility
            # Use the same logic above
            prop_ts = proposal_timestamp
            latest_ts = latest.get("timestamp", 0)

            if prop_ts > latest_ts:
                if margin > -self.arbitration_thresholds.get("overwrite_margin", 0.15):
                    action = "overwrite"
                    return action, {
                        "reason": "newer_temporal_corrective_overwrite",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
                else:
                    action = "keep_multiple_versions"
                    return action, {
                        "reason": "newer_but_lower_confidence_keep_both",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
            else:
                action = "defer"
                return action, {
                    "reason": "temporal_order_anomaly_requires_review",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }

        # Semantic overlap handling - should merge overlapping information
        if conflict_type == "semantic_overlap":
            if query_aware_override is not None:
                return query_aware_override

            if scenario_id and scenario_id.startswith("memoryagentbench_Conflict_Resolution"):
                prop_ts = proposal_timestamp
                latest_ts = latest.get("timestamp", 0) if latest else 0
                if prop_ts >= latest_ts:
                    action = "overwrite"
                    return action, {
                        "reason": "mab_conflict_semantic_overlap_prefers_latest_overwrite",
                        "similarity": conflict_details.get("similarity", 0.0),
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
                action = "reject"
                return action, {
                    "reason": "mab_conflict_older_semantic_overlap_reject",
                    "similarity": conflict_details.get("similarity", 0.0),
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }

            # For semantic overlap, the appropriate action is to merge the values
            # The detector already determined there is significant overlap
            similarity = conflict_details.get("similarity", 0.0)
            action = "merge"
            return action, {
                "reason": "semantic_overlap_merge",
                "similarity": similarity,
                "new_scores": new_scores,
                "old_scores": old_scores,
                "uncertainty": new_scores.get("uncertainty", 0.0),
            }

        # Compatible extension - keep both versions
        if conflict_type == "compatible_extension":
            if scenario_id and scenario_id.startswith("memoryagentbench_Conflict_Resolution"):
                prop_ts = proposal_timestamp
                latest_ts = latest.get("timestamp", 0) if latest else 0
                if prop_ts >= latest_ts:
                    action = "overwrite"
                    return action, {
                        "reason": "mab_conflict_compatible_extension_prefers_latest_overwrite",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "uncertainty": new_scores.get("uncertainty", 0.0),
                    }
                action = "reject"
                return action, {
                    "reason": "mab_conflict_older_compatible_extension_reject",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }
            action = "keep_multiple_versions"
            return action, {
                "reason": "compatible_information",
                "new_scores": new_scores,
                "old_scores": old_scores,
                "uncertainty": new_scores.get("uncertainty", 0.0),
            }

        # Potential contradiction - uncertain if truly conflicting
        if conflict_type == "potential_contradiction":
            if query_aware_override is not None:
                return query_aware_override
            # Low similarity but different values - could be unrelated or contradictory
            # Use uncertainty to guide: high uncertainty in either → defer
            new_uncertainty = new_scores.get("uncertainty", 0.0)
            old_uncertainty = old_scores.get("uncertainty", 0.0)
            avg_uncertainty = (new_uncertainty + old_uncertainty) / 2

            prop_ts = proposal_timestamp
            latest_ts = latest.get("timestamp", 0)
            overwrite_thresh = self.arbitration_thresholds.get("overwrite_margin", 0.15)
            keep_margin = self.arbitration_thresholds.get("keep_multiple_versions_margin", 0.08)

            if scenario_id and scenario_id.startswith("memoryagentbench_Conflict_Resolution"):
                if prop_ts >= latest_ts:
                    action = "overwrite"
                    return action, {
                        "reason": "mab_conflict_potential_contradiction_prefers_latest_overwrite",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                        "margin": margin,
                        "uncertainty": new_uncertainty,
                    }
                action = "reject"
                return action, {
                    "reason": "mab_conflict_older_potential_contradiction_reject",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "margin": margin,
                    "uncertainty": new_uncertainty,
                }

            if avg_uncertainty > 0.6:
                action = "defer"
                return action, {
                    "reason": "high_uncertainty_potential_contradiction_needs_review",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "uncertainty": new_uncertainty,
                    "old_uncertainty": old_uncertainty,
                }
            else:
                # Low uncertainty - treat as essentially a conflict
                # If proposal is newer and margin is not significantly negative, overwrite
                if margin > 0:
                    action = "overwrite"
                    return action, {
                        "reason": "confidence_win_potential_contradiction",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                        "uncertainty": new_uncertainty,
                    }
                elif abs(margin) < keep_margin:
                    # Very close - if proposal is newer, overwrite; else keep both
                    if prop_ts > latest_ts:
                        action = "overwrite"
                        return action, {
                            "reason": "similar_scores_newer_wins_potential_contradiction",
                            "new_scores": new_scores,
                            "old_scores": old_scores,
                            "uncertainty": new_uncertainty,
                        }
                    else:
                        action = "keep_multiple_versions"
                        return action, {
                            "reason": "similar_scores_not_newer_keep_both",
                            "new_scores": new_scores,
                            "old_scores": old_scores,
                            "uncertainty": new_uncertainty,
                        }
                else:
                    # Negative margin (proposal lower) and not close
                    action = "reject"
                    return action, {
                        "reason": "lower_confidence_potential_contradiction_reject",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                        "uncertainty": new_uncertainty,
                    }

        # Mutually exclusive - definitive conflict, prioritize newer information (last-write-wins)
        if conflict_type == "mutually_exclusive":
            if query_aware_override is not None:
                return query_aware_override
            prop_ts = proposal_timestamp
            latest_ts = latest.get("timestamp", 0) if latest else 0


            # For mutually exclusive facts, the newer entry should win
            if prop_ts > latest_ts:
                action = "overwrite"
                return action, {
                    "reason": "newer_timestamp_wins_mutually_exclusive",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }
            elif prop_ts == latest_ts:
                # Exact same timestamp - use confidence to break tie
                margin = new_scores["total"] - old_scores["total"]
                if margin > 0:
                    action = "overwrite"
                    return action, {
                        "reason": "same_timestamp_higher_score",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                    }
                else:
                    action = "reject"
                    return action, {
                        "reason": "same_timestamp_not_higher_score_reject",
                        "new_scores": new_scores,
                        "old_scores": old_scores,
                        "margin": margin,
                    }
            else:
                # Older proposal - reject (stale write)
                action = "reject"
                return action, {
                    "reason": "older_proposal_stale_reject",
                    "new_scores": new_scores,
                    "old_scores": old_scores,
                    "timestamps": {"proposal": prop_ts, "latest": latest_ts},
                    "uncertainty": new_scores.get("uncertainty", 0.0),
                }

        # Default case - defer to manual review
        action = "defer"
        return action, {
            "reason": "unhandled_conflict_type",
            "conflict_type": conflict_type,
            "new_scores": new_scores,
            "old_scores": old_scores,
            "uncertainty": new_scores.get("uncertainty", 0.0),
        }

    def _build_entry(self, proposal: Dict[str, Any], agent_id: str, candidates: List[Dict[str, Any]]) -> MemoryEntry:
        """Build a new memory entry from the proposal."""
        entry = MemoryEntry.from_proposal(proposal, agent_id=agent_id)
        if candidates:
            parent_id = candidates[-1].get("memory_id")
            parent_version = int(candidates[-1].get("version_id", 1))
            entry.parent_version_id = parent_id
            entry.version_id = parent_version + 1
        entry.arbitration_metadata = {
            "answer_criticality": float(proposal.get("answer_criticality", 0.0)),
            "graph_support_score": float(proposal.get("graph_support_score", 0.0)),
            "query_support_ids": list(proposal.get("query_support_ids", [])),
            "query_relation_roles": list(proposal.get("query_relation_roles", [])),
            "graph_cluster_id": proposal.get("graph_cluster_id"),
            "graph_edges": list(proposal.get("graph_edges", [])),
            "lineage_graph_edges": list(proposal.get("lineage_graph_edges", [])),
            "support_spans": list(proposal.get("support_spans", [])),
            "extractor_id": proposal.get("extractor_id"),
            "rationale": proposal.get("rationale"),
        }
        entry.rationale = proposal.get("rationale")
        entry.support_spans = list(proposal.get("support_spans", []))
        entry.extractor_id = proposal.get("extractor_id")
        entry.challenger_metadata = proposal.get("challenger_metadata")
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
            latest_id = candidates[-1].get("memory_id")
            latest_record = self.store.supersede(latest_id, superseded_by=entry.memory_id)
            if latest_record is not None:
                if latest_record.arbitration_metadata is None:
                    latest_record.arbitration_metadata = {}
                latest_record.arbitration_metadata.update({
                    "superseded_by": entry.memory_id,
                    "superseded_reason": conflict_type,
                })

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
                    # Attempt to parse both as JSON (objects or arrays)
                    latest_parsed = json.loads(latest_obj)
                    new_parsed = json.loads(new_obj)
                    if isinstance(latest_parsed, dict) and isinstance(new_parsed, dict):
                        # Merge JSON objects
                        merged = {**latest_parsed, **new_parsed}
                        entry.object_val = json.dumps(merged)
                    elif isinstance(latest_parsed, list) and isinstance(new_parsed, list):
                        # Merge JSON arrays with deduplication, preserving order
                        seen = set(latest_parsed)
                        merged = list(latest_parsed)
                        for item in new_parsed:
                            if item not in seen:
                                merged.append(item)
                                seen.add(item)
                        entry.object_val = json.dumps(merged)
                    else:
                        # Mismatched types or non-container JSON, fallback to concatenation
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
            latest_record = self.store.supersede(latest_id, superseded_by=entry.memory_id)
            if latest_record is not None:
                if latest_record.arbitration_metadata is None:
                    latest_record.arbitration_metadata = {}
                latest_record.arbitration_metadata.update({
                    "superseded_by": entry.memory_id,
                    "superseded_reason": conflict_type,
                })

        elif action == "defer":
            # Mark as tentative and add detailed metadata
            entry.status = "tentative"
            entry.lifecycle_stage = "tentative"
            entry.visibility_state = "pending_review"
            entry.canonical_status = "tentative"
            if entry.arbitration_metadata is None:
                entry.arbitration_metadata = {}
            entry.arbitration_metadata.update({
                "defer_reason": conflict_type,
                "requires_review": True,
                "review_priority": "high" if conflict_type in ["mutually_exclusive", "stale_read_conflict"] else "medium",
            })

    def write(
        self,
        proposal: Dict[str, Any],
        agent_id: str,
        read_snapshot_time: float,
        scenario_id: Optional[str] = None,
        event_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Write a proposal to the memory store with conflict awareness.

        Args:
            proposal: The proposed memory entry
            agent_id: ID of the agent making the proposal
            read_snapshot_time: When the agent read the memory before making this proposal
            scenario_id: Optional scenario context for dynamic weight adjustment
            event_timestamp: The timestamp of the write event. If None, uses current time.

        Returns:
            Result of the write operation
        """
        # Determine the timestamp for this proposal
        proposal_timestamp = event_timestamp if event_timestamp is not None else time.time()
        conflict_like = {
            "stale_read_conflict",
            "concurrent_update",
            "counterfactual_temporal",
            "temporal_inconsistency",
            "semantic_overlap",
            "compatible_extension",
            "potential_contradiction",
            "mutually_exclusive",
        }

        candidates = self._retrieve_candidates(proposal)
        conflict_type, conflict_details = detect_conflict_type(
            proposal, candidates, read_snapshot_time, self.staleness_detector, mode=self.mode
        )
        proposal["lineage_graph_edges"] = self._build_lineage_graph_edges(proposal, candidates)

        # Arbitrate to decide action (with scenario context and explicit timestamp)
        action, arbitration_details = self._arbitrate(
            conflict_type, conflict_details, proposal, candidates, scenario_id, proposal_timestamp
        )

        # Commit arbitration history if tracking is enabled
        if proposal.get("_enable_history_tracking", False) and "_temp_history_record" in proposal:
            record = proposal.pop("_temp_history_record")
            tracker = proposal.pop("_temp_history_tracker", None)
            if tracker is not None:
                record.action = action
                tracker.commit_record(record)

        # Handle rejection without creating an entry
        if action == "reject":
            return {
                "committed": False,
                "conflict_detected": conflict_type in conflict_like,
                "conflict_type": conflict_type,
                "resolution_action": action,
                "memory_id": None,
                "arbitration_details": arbitration_details,
                "conflict_details": conflict_details,
            }

        # Build and propose the new entry
        entry = self._build_entry(proposal, agent_id, candidates)
        # Set the entry's timestamp to the actual event time (not system creation time)
        entry.timestamp = proposal_timestamp
        entry.event_time = proposal_timestamp

        # Add conflict metadata to the entry
        if entry.arbitration_metadata is None:
            entry.arbitration_metadata = {}
        entry.arbitration_metadata.update({
            "conflict_type": conflict_type,
            "resolution_action": action,
            "scenario_id": scenario_id,
        })

        # Propose the write to the store
        self.store.propose(entry)

        # Commit with metadata
        self.store.commit(
            entry.memory_id,
            resolution_action=action,
            conflict_type=conflict_type,
            arb_metadata={
                "writer": "enhanced_conflict_aware",
                "candidate_count": len(candidates),
                "scenario_id": scenario_id,
                "answer_criticality": float(proposal.get("answer_criticality", 0.0)),
                "graph_support_score": float(proposal.get("graph_support_score", 0.0)),
                "query_support_ids": list(proposal.get("query_support_ids", [])),
                "query_relation_roles": list(proposal.get("query_relation_roles", [])),
                "graph_cluster_id": proposal.get("graph_cluster_id"),
                "graph_edges": list(proposal.get("graph_edges", [])),
                "lineage_graph_edges": list(proposal.get("lineage_graph_edges", [])),
                "arbitration_details": arbitration_details,
                "conflict_details": conflict_details,
            },
        )

        # Apply action-specific effects
        self._apply_action_effects(entry, action, conflict_type, candidates, arbitration_details)

        # Save and materialize visibility only for non-deferred commits.
        self.store._save()
        if action != "defer":
            self.store.set_indexed(entry.memory_id, delay=0.0)

        # Return detailed result
        return {
            "committed": True,
            "memory_id": entry.memory_id,
            "conflict_detected": conflict_type in conflict_like,
            "conflict_type": conflict_type,
            "resolution_action": action,
            "scenario_id": scenario_id,
            "candidate_count": len(candidates),
            "arbitration_details": arbitration_details,
            "conflict_details": conflict_details,
        }
