import json
import time
from typing import Dict, Any, List

from src.memory.shared_memory_store import SharedMemoryStore
from src.agents.agent_runtime import AgentRuntime
from src.conflict.staleness_detector import StalenessDetector
from src.conflict.conflict_aware_writer import ConflictAwareWriter
from src.conflict.baselines import LastWriteWinsWriter, NaiveAppendWriter


class MultiAgentPipeline:
    """
    End-to-end orchestration for Stage A.
    Supports modes: conflict_aware, lww, naive.
    """

    def __init__(
        self,
        mode: str = "conflict_aware",
        persistence_path: str = "tmp_pipeline_store.jsonl",
        enable_persistence: bool = True,
    ):
        assert mode in {"conflict_aware", "lww", "naive"}
        self.mode = mode
        self.store = SharedMemoryStore(
            persistence_path=persistence_path,
            enable_persistence=enable_persistence,
        )
        self.enable_persistence = enable_persistence
        self.staleness_detector = StalenessDetector()
        self.conflict_writer = None
        self.lww_writer = None
        self.naive_writer = None

        if self.mode == "conflict_aware":
            self.conflict_writer = ConflictAwareWriter(self.store, self.staleness_detector)
        elif self.mode == "lww":
            self.lww_writer = LastWriteWinsWriter(self.store)
        else:
            self.naive_writer = NaiveAppendWriter(self.store)

    def _build_agents(self, agent_ids: List[str]) -> Dict[str, AgentRuntime]:
        return {aid: AgentRuntime(self.store, aid) for aid in agent_ids}

    def run_scenario(self, scenario: Dict[str, Any], enable_retrieval_eval: bool = False) -> Dict[str, Any]:
        """
        Run a complete scenario through the pipeline.

        Args:
            scenario: Benchmark scenario dict
            enable_retrieval_eval: Whether to evaluate retrieval on provided queries

        Returns:
            Detailed logs and metrics
        """
        # reset store for scenario
        self.store.records = []
        if self.enable_persistence and self.store.persistence_path:
            with open(self.store.persistence_path, "w", encoding="utf-8") as f:
                f.write("")

        agents = self._build_agents(scenario.get("agents", []))

        logs = {
            "scenario_id": scenario.get("scenario_id"),
            "mode": self.mode,
            "agent_reads": [],
            "write_proposals": [],
            "detected_conflicts": [],
            "arbitration_decisions": [],
            "final_committed_state": [],
            "final_visible_state": [],
            "retrieval_results": [] if enable_retrieval_eval else None,
        }

        # Process events
        for ev in scenario.get("ordered_events", []):
            aid = ev.get("agent_id")
            if aid not in agents:
                continue

            if ev.get("event_type") == "read":
                snapshot = agents[aid].read()
                logs["agent_reads"].append({
                    "step": ev.get("step"),
                    "agent_id": aid,
                    "snapshot": snapshot,
                })

            if ev.get("event_type") == "write_proposal":
                proposal = ev.get("proposal", {})
                logs["write_proposals"].append({
                    "step": ev.get("step"),
                    "agent_id": aid,
                    "proposal": proposal,
                })

                if self.mode == "conflict_aware":
                    read_snapshot_time = float(ev.get("read_snapshot_time", time.time()))
                    scenario_id = scenario.get("scenario_id")
                    result = self.conflict_writer.write(
                        proposal, agent_id=aid, read_snapshot_time=read_snapshot_time, scenario_id=scenario_id
                    )
                elif self.mode == "lww":
                    result = self.lww_writer.write(proposal, agent_id=aid)
                else:
                    result = self.naive_writer.write(proposal, agent_id=aid)

                if result.get("conflict_detected"):
                    logs["detected_conflicts"].append({
                        "step": ev.get("step"),
                        "agent_id": aid,
                        "conflict_type": result.get("conflict_type"),
                    })

                logs["arbitration_decisions"].append({
                    "step": ev.get("step"),
                    "agent_id": aid,
                    "resolution_action": result.get("resolution_action", result.get("action")),
                    "result": result,
                })

                # Optional retrieval evaluation
                if enable_retrieval_eval and scenario.get("queries"):
                    visible = [r.to_dict() for r in self.store.get_all_visible()]
                    for query_info in scenario["queries"]:
                        query_text = query_info["query_text"]
                        gold_answers = query_info["gold_answers"]
                        retrieved = self._retrieve_for_eval(visible, query_text, k=5)
                        retrieved_objs = [r.get("object_val") for r in retrieved]
                        recall = len(set(retrieved_objs) & set(gold_answers)) / len(gold_answers) if gold_answers else 0.0
                        logs["retrieval_results"].append({
                            "step": ev.get("step"),
                            "query": query_text,
                            "retrieved": retrieved_objs,
                            "gold": gold_answers,
                            "recall_at_k": recall
                        })

        logs["final_committed_state"] = [r.to_dict() for r in self.store.records]
        logs["final_visible_state"] = [r.to_dict() for r in self.store.get_all_visible()]

        # Compute final metrics
        logs["metrics"] = self._compute_scenario_metrics(scenario, logs)

        return logs

    def _retrieve_for_eval(self, memories: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple retrieval for evaluation."""
        query_tokens = set(query.lower().split())
        scored = []
        for mem in memories:
            text = f"{mem.get('subject','')} {mem.get('predicate','')} {mem.get('object_val','')}".lower()
            mem_tokens = set(text.split())
            overlap = len(query_tokens & mem_tokens) / len(query_tokens) if query_tokens else 0.0
            scored.append((mem, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]

    def _compute_scenario_metrics(self, scenario: Dict[str, Any], logs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics for this scenario."""
        gold_visible = scenario.get("gold_visible_shared_state_after_commit", [])
        final_visible = logs["final_visible_state"]

        # Normalize states for comparison
        def norm(records):
            out = []
            for r in records:
                subj = r.get("subject", "")
                pred = r.get("predicate", "")
                obj = str(r.get("object_val", r.get("object", "")))
                out.append((subj, pred, obj))
            return sorted(out)

        state_match = norm(final_visible) == norm(gold_visible)

        metrics = {
            "state_match": state_match,
            "num_writes": len(logs["write_proposals"]),
            "num_conflicts": len(logs["detected_conflicts"]),
        }

        if logs["retrieval_results"]:
            recalls = [r["recall_at_k"] for r in logs["retrieval_results"]]
            metrics["avg_retrieval_recall"] = sum(recalls) / len(recalls) if recalls else 0.0

        return metrics


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
