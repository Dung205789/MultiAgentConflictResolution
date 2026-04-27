from typing import Dict, List, Any
import time


class StalenessDetector:
    """
    Stage A staleness simulator/detector:
    - if a read snapshot time is before latest commit/index for same (subject,predicate),
      mark stale_read_conflict risk.
    """

    def __init__(self, staleness_window_sec: float = 0.0):
        self.staleness_window_sec = staleness_window_sec

    def is_stale_read(
        self,
        read_time: float,
        latest_commit_time: float,
        latest_index_time: float,
    ) -> bool:
        # stale if read occurs before index catches up; optional margin via staleness window
        return read_time + self.staleness_window_sec < max(latest_commit_time, latest_index_time)

    def detect_stale_for_proposal(
        self,
        proposal: Dict[str, Any],
        read_snapshot_time: float,
        candidate_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not candidate_records:
            return {"stale": False, "reason": "no_candidates"}

        subject = proposal.get("subject")
        predicate = proposal.get("predicate")

        relevant = [
            r for r in candidate_records
            if r.get("subject") == subject and r.get("predicate") == predicate
        ]
        if not relevant:
            return {"stale": False, "reason": "no_relevant_candidates"}

        latest_commit = max(float(r.get("committed_at") or 0.0) for r in relevant)
        latest_index = max(float(r.get("indexed_at") or 0.0) for r in relevant)

        stale = self.is_stale_read(read_snapshot_time, latest_commit, latest_index)
        return {
            "stale": stale,
            "reason": "read_before_latest_commit_or_index" if stale else "fresh_enough",
            "latest_commit": latest_commit,
            "latest_index": latest_index,
            "read_snapshot_time": read_snapshot_time,
        }
