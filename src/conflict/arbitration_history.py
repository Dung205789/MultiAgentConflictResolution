"""
Arbitration History Tracker for Meta-Learning.

Tracks arbitration decisions and their outcomes to enable:
1. Learning optimal weights for different conflict types
2. Detecting systematic biases in arbitration
3. Providing data for meta-learning algorithms
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class ArbitrationRecord:
    """Single arbitration decision record."""

    def __init__(
        self,
        scenario_id: str,
        conflict_type: str,
        proposal: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        action_taken: str,
        scores: Dict[str, Any],
        uncertainty: Dict[str, Any],
        outcome: Optional[str] = None,  # correct | incorrect | unknown
        ground_truth: Optional[Any] = None,
    ):
        self.record_id = f"arb_{int(time.time() * 1000)}_{hash(conflict_type) % 10000}"
        self.timestamp = time.time()
        self.scenario_id = scenario_id
        self.conflict_type = conflict_type
        self.proposal = proposal
        self.candidates = candidates
        self.action_taken = action_taken
        self.scores = scores
        self.uncertainty = uncertainty
        self.outcome = outcome
        self.ground_truth = ground_truth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "datetime": datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z",
            "scenario_id": self.scenario_id,
            "conflict_type": self.conflict_type,
            "proposal": {
                "subject": self.proposal.get("subject"),
                "predicate": self.proposal.get("predicate"),
                "object_val": str(self.proposal.get("object_val", ""))[:200],
                "confidence": self.proposal.get("confidence", 0.0),
                "provenance": self.proposal.get("provenance", "unknown"),
            },
            "candidates_count": len(self.candidates),
            "action_taken": self.action_taken,
            "scores": self.scores,
            "uncertainty": self.uncertainty,
            "outcome": self.outcome,
            "ground_truth": str(self.ground_truth)[:200] if self.ground_truth else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArbitrationRecord":
        record = cls(
            scenario_id=data.get("scenario_id", "unknown"),
            conflict_type=data.get("conflict_type", "none"),
            proposal=data.get("proposal", {}),
            candidates=data.get("candidates", []),
            action_taken=data.get("action_taken", "unknown"),
            scores=data.get("scores", {}),
            uncertainty=data.get("uncertainty", {}),
            outcome=data.get("outcome"),
            ground_truth=data.get("ground_truth"),
        )
        record.record_id = data.get("record_id", record.record_id)
        record.timestamp = data.get("timestamp", record.timestamp)
        return record


class ArbitrationHistoryTracker:
    """
    Tracks arbitration decisions and enables meta-learning.

    Features:
    - Persistent storage to JSONL
    - Outcome tracking (correct/incorrect/unknown)
    - Weight optimization suggestions based on historical performance
    """

    def __init__(self, history_path: str = "data/arbitration_history.jsonl", enabled: bool = False):
        self.history_path = history_path
        self.enabled = enabled
        self.records: List[ArbitrationRecord] = []
        self._load()

    def _ensure_dir(self):
        """Ensure the directory for history file exists."""
        if self.history_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.history_path)), exist_ok=True)

    def _load(self):
        """Load historical records from disk."""
        if not self.enabled:
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = ArbitrationRecord.from_dict(data)
                        self.records.append(record)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Failed to load record: {e}")
        except FileNotFoundError:
            pass

    def _save(self):
        """Save records to disk."""
        if not self.enabled or not self.history_path:
            return
        self._ensure_dir()
        with open(self.history_path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    def log_decision(
        self,
        scenario_id: str,
        conflict_type: str,
        proposal: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        action_taken: str,
        scores: Dict[str, Any],
        uncertainty: Dict[str, Any],
    ) -> str:
        """
        Log an arbitration decision.

        Returns:
            record_id of the created record
        """
        if not self.enabled:
            return ""

        record = ArbitrationRecord(
            scenario_id=scenario_id,
            conflict_type=conflict_type,
            proposal=proposal,
            candidates=candidates,
            action_taken=action_taken,
            scores=scores,
            uncertainty=uncertainty,
        )
        self.records.append(record)
        self._save()
        return record.record_id

    def update_outcome(self, record_id: str, outcome: str, ground_truth: Any = None):
        """
        Update the outcome of a previous arbitration decision.

        Args:
            record_id: ID of the record to update
            outcome: "correct" | "incorrect" | "unknown"
            ground_truth: Optional ground truth value for verification
        """
        for record in self.records:
            if record.record_id == record_id:
                record.outcome = outcome
                record.ground_truth = ground_truth
                self._save()
                return True
        return False

    def get_performance_by_conflict_type(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics grouped by conflict type.

        Returns:
            Dict mapping conflict_type to performance metrics
        """
        stats = {}

        for record in self.records:
            ct = record.conflict_type
            if ct not in stats:
                stats[ct] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "unknown": 0,
                    "actions": {},
                }

            stats[ct]["total"] += 1

            if record.outcome == "correct":
                stats[ct]["correct"] += 1
            elif record.outcome == "incorrect":
                stats[ct]["incorrect"] += 1
            else:
                stats[ct]["unknown"] += 1

            action = record.action_taken
            if action not in stats[ct]["actions"]:
                stats[ct]["actions"][action] = 0
            stats[ct]["actions"][action] += 1

        # Calculate accuracy
        for ct in stats:
            total_with_outcome = stats[ct]["correct"] + stats[ct]["incorrect"]
            if total_with_outcome > 0:
                stats[ct]["accuracy"] = stats[ct]["correct"] / total_with_outcome
            else:
                stats[ct]["accuracy"] = 0.0

        return stats

    def suggest_weights(self, conflict_type: str) -> Optional[Dict[str, float]]:
        """
        Suggest optimal weights based on historical performance.
        Uses a simple gradient-based approach: increase weights for features that
        were more reliable in correct decisions.

        Returns:
            Suggested weights dict or None if not enough data
        """
        # Get all records for this conflict type with known outcomes
        relevant = [
            r for r in self.records
            if r.conflict_type == conflict_type and r.outcome in ("correct", "incorrect") and r.scores
        ]

        if len(relevant) < 10:  # Need minimum samples
            return None

        correct_records = [r for r in relevant if r.outcome == "correct"]
        incorrect_records = [r for r in relevant if r.outcome == "incorrect"]

        if not correct_records:
            return None

        # Calculate average weights for correct vs incorrect
        features = ["confidence", "provenance", "recency", "authority"]

        correct_avg = {}
        incorrect_avg = {}
        for f in features:
            correct_avg[f] = sum(r.scores.get("weights_used", {}).get(f, 0.25) for r in correct_records) / len(correct_records)
            incorrect_avg[f] = sum(r.scores.get("weights_used", {}).get(f, 0.25) for r in incorrect_records) / len(incorrect_records) if incorrect_records else correct_avg[f]

        # Suggest weights: increase weight for features that were higher in correct decisions
        suggested = {}
        for f in features:
            if incorrect_avg[f] > 0:
                ratio = correct_avg[f] / incorrect_avg[f]
                # Bound the adjustment
                suggested[f] = min(0.6, max(0.05, correct_avg[f] * (1 + 0.1 * (ratio - 1))))
            else:
                suggested[f] = correct_avg[f]

        # Normalize to sum to 1.0
        total = sum(suggested.values())
        if total > 0:
            suggested = {k: v / total for k, v in suggested.items()}

        return suggested if total > 0 else None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the arbitration history."""
        return {
            "total_records": len(self.records),
            "enabled": self.enabled,
            "history_path": self.history_path,
            "performance_by_type": self.get_performance_by_conflict_type(),
            "records_with_outcome": sum(1 for r in self.records if r.outcome in ("correct", "incorrect")),
        }
