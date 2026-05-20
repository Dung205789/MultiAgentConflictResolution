import json
import time
from typing import Dict, Any, List, Optional

from src.benchmarks.scenario_contract import scenario_to_dict
from src.memory.shared_memory_store import SharedMemoryStore
from src.agents.agent_runtime import AgentRuntime
from src.conflict.staleness_detector import StalenessDetector
from src.conflict.conflict_aware_writer import ConflictAwareWriter
from src.conflict.baselines import LastWriteWinsWriter, NaiveAppendWriter
from src.conflict.query_aware_context import (
    annotate_proposal_with_query_context,
    build_query_plan,
    summarize_query_plan,
)
from src.evaluation.qa_reasoner import answer_question_from_memories, score_answers
from src.memory.proposal_contract import normalize_proposal


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
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        proposal_source: str = "structured",
        strict_agent_execution: bool = False,
        conflict_aware_variant: str = "full",
        track_name: str = "oracle_structured",
        allow_structured_fallback_in_end_to_end: bool = False,
    ):
        assert mode in {"conflict_aware", "lww", "naive"}
        assert proposal_source in {"structured", "agent_extract"}
        assert conflict_aware_variant in {"full", "no_lineage_edges", "no_query_support"}
        assert track_name in {"oracle_structured", "end_to_end_extract"}
        self.mode = mode
        self.store = SharedMemoryStore(
            persistence_path=persistence_path,
            enable_persistence=enable_persistence,
        )
        self.enable_persistence = enable_persistence
        self.agent_configs = agent_configs or {}
        self.proposal_source = proposal_source
        self.strict_agent_execution = strict_agent_execution
        self.conflict_aware_variant = conflict_aware_variant
        self.track_name = track_name
        self.allow_structured_fallback_in_end_to_end = allow_structured_fallback_in_end_to_end
        self.staleness_detector = StalenessDetector()
        self.conflict_writer = None
        self.lww_writer = None
        self.naive_writer = None

        if self.mode == "conflict_aware":
            self.conflict_writer = ConflictAwareWriter(
                self.store,
                self.staleness_detector,
                variant=self.conflict_aware_variant,
            )
        elif self.mode == "lww":
            self.lww_writer = LastWriteWinsWriter(self.store)
        else:
            self.naive_writer = NaiveAppendWriter(self.store)

    def _resolve_agent_config(self, agent_id: str, position: int) -> Dict[str, Any]:
        if agent_id in self.agent_configs:
            return dict(self.agent_configs[agent_id])
        slot_key = f"__slot_{position}__"
        if slot_key in self.agent_configs:
            return dict(self.agent_configs[slot_key])
        return {}

    def _build_agents(self, agent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        agents: Dict[str, Dict[str, Any]] = {}
        for idx, aid in enumerate(agent_ids):
            cfg = self._resolve_agent_config(aid, idx)
            runtime = AgentRuntime(self.store, aid, mode=cfg.get("runtime_mode", "debug_fallback"))
            extractor = None
            if cfg.get("model_type") == "transformer":
                from src.local_models.runner import create_agent
                extractor = create_agent(
                    agent_id=aid,
                    model_type="transformer",
                    model_name=cfg.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                    device=cfg.get("device", "cpu"),
                    strict_loading=self.strict_agent_execution,
                )
            agents[aid] = {
                "runtime": runtime,
                "config": cfg,
                "extractor": extractor,
            }
        return agents

    def _prepare_proposal(
        self,
        agent_info: Dict[str, Any],
        proposal: Dict[str, Any],
        event_payload: Dict[str, Any],
        reliability: Optional[float],
    ) -> Dict[str, Any]:
        prepared = dict(proposal or {})
        prepared.setdefault("agent_id", event_payload.get("agent_id"))

        source_used = "oracle_structured"
        if self.proposal_source == "agent_extract":
            seed_text = (
                prepared.get("raw_text")
                or event_payload.get("text")
                or f"{prepared.get('subject', '')} {prepared.get('predicate', '')} {prepared.get('object_val', '')}".strip()
            )
            extractor = agent_info.get("extractor")
            extracted: List[Dict[str, Any]] = []
            if extractor and seed_text:
                extracted = extractor.extract_memories(seed_text)

            if extracted:
                merged = dict(extracted[0])
                merged.setdefault("raw_text", prepared.get("raw_text", seed_text))
                for key in [
                    "canonical_claim",
                    "memory_type",
                    "session_id",
                    "turn_index",
                    "provenance",
                    "event_time",
                    "ingestion_time",
                ]:
                    if key in prepared and key not in merged:
                        merged[key] = prepared[key]
                prepared = merged
                source_used = "end_to_end_extract"
            elif self.strict_agent_execution:
                if self.allow_structured_fallback_in_end_to_end and prepared.get("predicate") and prepared.get("object_val"):
                    prepared.setdefault("raw_text", seed_text)
                    prepared["_agent_extract_fallback"] = "structured_proposal"
                    source_used = "end_to_end_extract__structured_fallback"
                else:
                    raise RuntimeError(
                        f"Agent extraction failed for {event_payload.get('agent_id')} under strict agent execution mode."
                    )

        if reliability is not None:
            prepared["agent_authority"] = float(reliability)
            prepared.setdefault("confidence", float(reliability))

        if source_used == "oracle_structured":
            prepared = normalize_proposal(
                prepared,
                source_label=source_used,
                default_extractor_id="oracle_structured_adapter",
                default_provenance=prepared.get("provenance", "benchmark_structured"),
                default_confidence=float(prepared.get("confidence", reliability or 1.0)),
                default_rationale="benchmark_adapter_structured_proposal",
            )
        else:
            extractor = agent_info.get("extractor")
            prepared = normalize_proposal(
                prepared,
                source_label=source_used,
                default_extractor_id=getattr(extractor, "model_name", None) or event_payload.get("agent_id") or "end_to_end_extractor",
                default_provenance=prepared.get("provenance", "llm_inferred"),
                default_confidence=float(prepared.get("confidence", reliability or 0.7)),
                default_rationale="extractor_generated_proposal",
            )

        prepared["_proposal_source"] = source_used
        prepared["_track_name"] = self.track_name
        return prepared

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
        self.store.reset()
        if self.enable_persistence and self.store.persistence_path:
            with open(self.store.persistence_path, "w", encoding="utf-8") as f:
                f.write("")

        scenario_dict = scenario_to_dict(scenario)
        agents_list = scenario_dict.get("agents", [])
        events_list = scenario_dict.get("ordered_events", [])
        queries_list = scenario_dict.get("queries", [])
        scenario_id = scenario_dict.get("scenario_id")
        agent_profiles = scenario_dict.get("agent_profiles", {})

        agents = self._build_agents(agents_list)

        logs = {
            "scenario_id": scenario_id,
            "mode": self.mode,
            "agent_reads": [],
            "write_proposals": [],
            "detected_conflicts": [],
            "arbitration_decisions": [],
            "final_committed_state": [],
            "final_visible_state": [],
            "retrieval_results": [] if enable_retrieval_eval else None,
            "qa_results": [] if enable_retrieval_eval else None,
            "execution": {
                "track_name": self.track_name,
                "proposal_source": self.proposal_source,
                "strict_agent_execution": self.strict_agent_execution,
                "conflict_aware_variant": self.conflict_aware_variant,
                "allow_structured_fallback_in_end_to_end": self.allow_structured_fallback_in_end_to_end,
            },
        }
        query_plan = build_query_plan(queries_list)
        logs["execution"]["query_plan"] = summarize_query_plan(query_plan)

        # Process events
        for ev in events_list:
            # Handle both Event objects and dicts
            if hasattr(ev, 'agent_id'):
                aid = ev.agent_id
                ev_step = ev.step
                ev_type = ev.event_type
                ev_proposal = ev.proposal
                ev_query = ev.query
                ev_read_snapshot_time = ev.read_snapshot_time
            else:
                aid = ev.get("agent_id")
                ev_step = ev.get("step")
                ev_type = ev.get("event_type")
                ev_proposal = ev.get("proposal", {})
                ev_query = ev.get("query")
                ev_read_snapshot_time = ev.get("read_snapshot_time")

            if aid not in agents:
                continue

            agent_info = agents[aid]
            agent_runtime = agent_info["runtime"]
            agent_config = agent_info["config"]
            agent_profile = agent_profiles.get(aid, {}) if isinstance(agent_profiles, dict) else {}
            reliability = agent_config.get("reliability", agent_profile.get("reliability"))

            if ev_type == "read":
                snapshot = agent_runtime.read()
                logs["agent_reads"].append({
                    "step": ev_step,
                    "agent_id": aid,
                    "snapshot": snapshot,
                })

            if ev_type == "write_proposal":
                event_payload = {
                    "agent_id": aid,
                    "step": ev_step,
                    "event_type": ev_type,
                    "proposal": ev_proposal,
                    "query": ev_query,
                    "read_snapshot_time": ev_read_snapshot_time,
                    "text": getattr(ev, "text", None) if hasattr(ev, "agent_id") else ev.get("text"),
                }
                proposal = self._prepare_proposal(agent_info, ev_proposal, event_payload, reliability)
                proposal = annotate_proposal_with_query_context(
                    proposal,
                    query_plan,
                    variant=self.conflict_aware_variant,
                )
                logs["write_proposals"].append({
                    "step": ev_step,
                    "agent_id": aid,
                    "proposal": proposal,
                    "proposal_source": proposal.get("_proposal_source", "structured"),
                })

                # Get read_snapshot_time, default to current time if None
                raw_snapshot = ev_read_snapshot_time
                read_snapshot_time = float(raw_snapshot) if raw_snapshot is not None else time.time()

                # Get the event timestamp for proper ordering
                event_timestamp = ev.timestamp if hasattr(ev, 'timestamp') else ev.get("timestamp", read_snapshot_time)

                if self.mode == "conflict_aware":
                    result = self.conflict_writer.write(
                        proposal, agent_id=aid, read_snapshot_time=read_snapshot_time,
                        scenario_id=scenario_id, event_timestamp=event_timestamp
                    )
                elif self.mode == "lww":
                    result = self.lww_writer.write(proposal, agent_id=aid)
                else:
                    result = self.naive_writer.write(proposal, agent_id=aid)

                if result.get("conflict_detected"):
                    logs["detected_conflicts"].append({
                        "step": ev_step,
                        "agent_id": aid,
                        "conflict_type": result.get("conflict_type"),
                    })

                logs["arbitration_decisions"].append({
                    "step": ev_step,
                    "agent_id": aid,
                    "proposal": {
                        "subject": proposal.get("subject"),
                        "predicate": proposal.get("predicate"),
                        "object_val": proposal.get("object_val"),
                        "raw_text": proposal.get("raw_text"),
                        "proposal_source": proposal.get("_proposal_source", "structured"),
                    },
                    "resolution_action": result.get("resolution_action", result.get("action")),
                    "result": result,
                })

        logs["final_committed_state"] = [r.to_dict() for r in self.store.records]
        logs["final_visible_state"] = [r.to_dict() for r in self.store.get_all_visible()]

        # Retrieval evaluation on the final visible state is enough for benchmark reporting
        # and avoids quadratic blowups on long conflict scenarios.
        if enable_retrieval_eval and queries_list:
            visible = logs["final_visible_state"]
            qa_enabled = self._supports_symbolic_qa(visible)
            for query_info in queries_list:
                if hasattr(query_info, "query_text"):
                    query_text = query_info.query_text
                    gold_answers = query_info.gold_answers
                else:
                    query_text = query_info["query_text"]
                    gold_answers = query_info["gold_answers"]
                retrieved = self._retrieve_for_eval(visible, query_text, k=5)
                retrieved_objs = [r.get("object_val") for r in retrieved]
                recall = len(set(retrieved_objs) & set(gold_answers)) / len(gold_answers) if gold_answers else 0.0
                logs["retrieval_results"].append({
                    "step": "final",
                    "query": query_text,
                    "retrieved": retrieved_objs,
                    "gold": gold_answers,
                    "recall_at_k": recall
                })
                if qa_enabled:
                    qa_result = answer_question_from_memories(query_text, visible)
                    qa_score = score_answers(qa_result.get("predicted_answers", []), gold_answers)
                    logs["qa_results"].append({
                        "step": "final",
                        "query": query_text,
                        "gold": gold_answers,
                        **qa_result,
                        **qa_score,
                    })

        # Compute final metrics
        logs["metrics"] = self._compute_scenario_metrics(scenario_dict, logs)

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

    def _supports_symbolic_qa(self, memories: List[Dict[str, Any]]) -> bool:
        """
        Run the symbolic QA reasoner only when the memory surface looks like
        graph facts rather than raw conversational utterances.
        """
        non_graph_predicates = {"utterance", "statement", "info"}
        return any(
            str(mem.get("predicate", "")).strip().lower() not in non_graph_predicates
            for mem in memories
        )

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
            "scenario_contract_version": scenario.get("_scenario_contract_version"),
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
