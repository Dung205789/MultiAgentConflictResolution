"""
Local model runner for CPU-only environments.
Supports loading quantized or smaller models for multi-agent evaluation.
"""
from typing import Dict, Any, List, Optional, Tuple
import time
from abc import ABC, abstractmethod
import os

_PIPELINE_CACHE: Dict[tuple, Any] = {}


def _resolve_pipeline_device(device: str) -> Tuple[int, bool]:
    """Map a device string into a transformers pipeline device index."""
    normalized = (device or "cpu").strip().lower()
    if normalized == "cpu":
        return -1, False
    if normalized == "cuda":
        return 0, True
    if normalized.startswith("cuda:"):
        try:
            return int(normalized.split(":", 1)[1]), True
        except ValueError:
            return 0, True
    return 0, True


class LocalAgent(ABC):
    """Abstract base class for a local agent."""
    def __init__(self, agent_id: str, model_name: str):
        self.agent_id = agent_id
        self.model_name = model_name

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the agent."""
        pass

    @abstractmethod
    def extract_memories(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured memories from text."""
        pass


class DummyLocalAgent(LocalAgent):
    """Simple rule-based agent for testing without heavy models."""
    def __init__(self, agent_id: str, reliability: float = 0.7):
        super().__init__(agent_id, f"dummy_reliability_{reliability}")
        self.reliability = reliability

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Return a simple response based on prompt keywords."""
        # Very simple rule-based response
        if "city" in prompt.lower():
            return "The user lives in New York"
        elif "language" in prompt.lower():
            return "The user likes Python"
        elif "study" in prompt.lower():
            return "The user prefers morning study"
        else:
            return f"Agent {self.agent_id} observed: {prompt[:50]}..."

    def extract_memories(self, text: str) -> List[Dict[str, Any]]:
        """Extract memories using simple keyword matching."""
        memories = []
        t = text.lower()

        # City extraction
        cities = ["new york", "los angeles", "chicago", "paris", "london"]
        for city in cities:
            if city in t:
                memories.append({
                    "subject": "user",
                    "predicate": "city",
                    "object_val": city.title(),
                    "confidence": self.reliability,
                    "provenance": "inferred"
                })
                break

        # Language extraction
        langs = ["python", "javascript", "java", "sql", "go"]
        for lang in langs:
            if lang in t:
                memories.append({
                    "subject": "user",
                    "predicate": "language",
                    "object_val": lang,
                    "confidence": self.reliability,
                    "provenance": "inferred"
                })
                break

        # Study time extraction
        if "morning" in t or "sáng" in t:
            memories.append({
                "subject": "user",
                "predicate": "study_time",
                "object_val": "morning",
                "confidence": self.reliability,
                "provenance": "inferred"
            })
        elif "night" in t or "tối" in t or "đêm" in t:
            memories.append({
                "subject": "user",
                "predicate": "study_time",
                "object_val": "night",
                "confidence": self.reliability,
                "provenance": "inferred"
            })

        return memories


class TransformerAgent(LocalAgent):
    """Agent that uses a local transformer model for generation and extraction."""
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        device: str = "cpu",
        quantization_mode: Optional[str] = None,
        strict_loading: bool = False,
    ):
        super().__init__(agent_id, model_name)
        self.device = device
        self.quantization_mode = quantization_mode or ("4bit" if str(device).startswith("cuda") else "none")
        self.strict_loading = strict_loading
        self.generator = None
        self._load_model()

    def _load_model(self):
        """Load the model into memory."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
            cache_key = (self.model_name, self.device, self.quantization_mode)
            if cache_key in _PIPELINE_CACHE:
                self.generator = _PIPELINE_CACHE[cache_key]
                print(f"Reusing cached model {self.model_name} on {self.device}.")
                return

            pipeline_device, use_cuda = _resolve_pipeline_device(self.device)
            print(f"Loading model {self.model_name} on {self.device}...")
            if use_cuda:
                model_kwargs: Dict[str, Any] = {
                    "trust_remote_code": True,
                    "device_map": {"": pipeline_device},
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                }
                if self.quantization_mode == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quantization_mode == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                self.generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                )
            else:
                model_kwargs = {
                    "model": self.model_name,
                    "device": pipeline_device,
                    "trust_remote_code": True,
                }
                self.generator = pipeline(
                    "text-generation",
                    **model_kwargs
                )
            _PIPELINE_CACHE[cache_key] = self.generator
            print(f"Model loaded successfully.")
        except Exception as e:
            if self.strict_loading:
                raise RuntimeError(f"Could not load model {self.model_name} on {self.device}: {e}") from e
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Falling back to dummy agent behavior.")
            self.generator = None

    def generate_response(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        if self.generator is None:
            return f"[{self.agent_id}] Model not available: {prompt[:50]}..."

        try:
            output = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                return_full_text=True,
            )
            return output[0]["generated_text"][len(prompt):].strip()
        except Exception as e:
            return f"[{self.agent_id}] Generation error: {e}"

    def extract_memories(self, text: str) -> List[Dict[str, Any]]:
        """Extract memories using the local model with a prompting approach."""
        if self.generator is None:
            return []

        prompt = (
            "Extract structured facts from the input text. "
            "Return ONLY a JSON array. Each item must contain keys: "
            "subject, predicate, object_val, confidence, provenance, rationale, support_spans, extractor_id. "
            "Preserve subject and predicate exactly from the input when explicit. "
            "Use provenance='llm_inferred'. Confidence must be between 0.5 and 0.95.\n"
            f"Input: {text}\n"
            "Output JSON:"
        )

        try:
            response = self.generate_response(prompt, max_new_tokens=96)
            # Parse JSON from response
            import json
            response = response.replace("```json", "").replace("```", "").strip()
            # Find first [ and last ]
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                memories = json.loads(json_str)
                # Validate and normalize
                valid_memories = []
                for m in memories:
                    if isinstance(m, dict) and m.get("predicate") and m.get("object_val"):
                        m.setdefault("subject", "unknown")
                        m.setdefault("confidence", 0.7)
                        m.setdefault("provenance", "llm_inferred")
                        m.setdefault("rationale", "local_transformer_extraction")
                        m.setdefault("support_spans", [{"span_text": text[:200], "span_index": 0}])
                        m.setdefault("extractor_id", self.model_name)
                        m.setdefault("challenger_metadata", None)
                        valid_memories.append(m)
                return valid_memories
        except Exception:
            pass

        return []


def create_agent(agent_id: str, model_type: str = "dummy", reliability: float = None, **kwargs) -> LocalAgent:
    """
    Factory function to create agents.

    Args:
        agent_id: Unique agent identifier
        model_type: "dummy" or "transformer"
        reliability: For dummy agents, the reliability score (0.0-1.0)
        **kwargs: For transformer agents: model_name, device

    Returns:
        LocalAgent instance
    """
    if model_type == "dummy":
        if reliability is None:
            reliability = 0.7
        return DummyLocalAgent(agent_id, reliability=reliability)
    elif model_type == "transformer":
        model_name = kwargs.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        device = kwargs.get("device", "cpu")
        quantization_mode = kwargs.get("quantization_mode")
        strict_loading = kwargs.get("strict_loading", False)
        return TransformerAgent(
            agent_id,
            model_name=model_name,
            device=device,
            quantization_mode=quantization_mode,
            strict_loading=strict_loading,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class MultiAgentLocalRunner:
    """
    Runner for evaluating multi-agent scenarios with local models.

    Supports before/after memory layer comparisons.
    """
    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        memory_store,
        writer_type: str = "conflict_aware",
        force_model_extraction: bool = False,
        show_event_progress: bool = False,
    ):
        """
        Initialize runner.

        Args:
            agent_configs: List of dicts with keys: agent_id, model_type, reliability/model_name
            memory_store: SharedMemoryStore instance
            writer_type: One of "conflict_aware", "lww", "naive"
            force_model_extraction: If True, run model extraction for each write event
                even when structured proposals already exist in the scenario.
            show_event_progress: If True, show per-write progress bars inside each scenario.
        """
        self.memory_store = memory_store
        self.writer_type = writer_type
        self.force_model_extraction = force_model_extraction
        self.show_event_progress = show_event_progress
        self.agents = [
            create_agent(**config) for config in agent_configs
        ]
        self.agent_map = {a.agent_id: a for a in self.agents}

        from src.conflict.staleness_detector import StalenessDetector
        from src.conflict.conflict_aware_writer import ConflictAwareWriter
        from src.conflict.baselines import LastWriteWinsWriter, NaiveAppendWriter

        if writer_type == "conflict_aware":
            self.writer = ConflictAwareWriter(self.memory_store, StalenessDetector())
        elif writer_type == "lww":
            self.writer = LastWriteWinsWriter(self.memory_store)
        elif writer_type == "naive":
            self.writer = NaiveAppendWriter(self.memory_store)
        else:
            raise ValueError(f"Unknown writer_type: {writer_type}")

    def run_scenario(self, scenario: Dict[str, Any], enable_retrieval_eval: bool = True) -> Dict[str, Any]:
        """
        Run a scenario with local agents.

        Args:
            scenario: Benchmark scenario dict
            enable_retrieval_eval: Whether to evaluate retrieval on provided queries

        Returns:
            Results dict with metrics and logs
        """
        # Reset store
        self.memory_store.records = []
        if getattr(self.memory_store, "enable_persistence", True) and self.memory_store.persistence_path:
            with open(self.memory_store.persistence_path, "w", encoding="utf-8") as f:
                f.write("")

        logs = {
            "scenario_id": scenario.get("scenario_id"),
            "writer_type": self.writer_type,
            "arbitration_decisions": [],  # renamed from agent_actions for compatibility
            "final_visible_state": [],
            "retrieval_results": [],
            "metrics": {}
        }

        ordered_events = scenario.get("ordered_events", [])
        write_events_total = sum(1 for ev in ordered_events if ev.get("event_type") == "write_proposal")

        event_progress = None
        if self.show_event_progress and write_events_total > 0:
            try:
                from tqdm import tqdm
                event_progress = tqdm(
                    total=write_events_total,
                    desc=f"{scenario.get('scenario_id', 'scenario')} writes",
                    unit="write",
                    leave=False
                )
            except Exception:
                event_progress = None

        for ev in ordered_events:
            agent_id = ev.get("agent_id")
            if agent_id not in self.agent_map:
                continue

            agent = self.agent_map[agent_id]
            event_type = ev.get("event_type", "write_proposal")

            if event_type == "write_proposal":
                # Get the proposal (possibly extracted from text)
                proposal = ev.get("proposal")
                if self.force_model_extraction:
                    seed_text = ev.get("text", "")
                    if not seed_text and proposal:
                        seed_text = (
                            f"{proposal.get('subject', 'user')} "
                            f"{proposal.get('predicate', 'info')} "
                            f"{proposal.get('object_val', '')}"
                        ).strip()
                    if seed_text:
                        extracted = agent.extract_memories(seed_text)
                        if extracted:
                            proposal = extracted[0]

                if not proposal:
                    # Need to extract from text
                    text = ev.get("text", "")
                    if text:
                        extracted = agent.extract_memories(text)
                        if extracted:
                            proposal = extracted[0]
                        else:
                            # Skip this event if no proposal could be extracted
                            continue

                read_snapshot_time = ev.get("read_snapshot_time", time.time())

                if self.writer_type == "conflict_aware":
                    result = self.writer.write(proposal, agent_id=agent_id, read_snapshot_time=read_snapshot_time)
                else:
                    result = self.writer.write(proposal, agent_id=agent_id)

                logs["arbitration_decisions"].append({
                    "step": ev.get("step"),
                    "agent_id": agent_id,
                    "resolution_action": result.get("resolution_action", result.get("action", "append")),
                    "conflict_detected": result.get("conflict_detected", False),
                    "conflict_type": result.get("conflict_type"),
                    "candidate_count": result.get("candidate_count", 0),
                    "result": result,
                })
                if event_progress is not None:
                    event_progress.update(1)
                    event_progress.set_postfix({
                        "step": ev.get("step"),
                        "agent": agent_id
                    })

        if event_progress is not None:
            event_progress.close()

        logs["final_visible_state"] = [r.to_dict() for r in self.memory_store.get_all_visible()]

        # Compute basic metrics
        gold_visible = scenario.get("gold_visible_shared_state_after_commit", [])
        def norm(records):
            out = []
            for r in records:
                subj = r.get("subject", "")
                pred = r.get("predicate", "")
                obj = str(r.get("object_val", r.get("object", "")))
                out.append((subj, pred, obj))
            return sorted(out)

        logs["metrics"]["state_match"] = norm(logs["final_visible_state"]) == norm(gold_visible)
        logs["metrics"]["num_writes"] = len(logs["arbitration_decisions"])
        logs["metrics"]["num_conflicts"] = sum(1 for a in logs["arbitration_decisions"] if a["conflict_detected"])

        # Optional retrieval evaluation
        if enable_retrieval_eval and scenario.get("queries"):
            visible = logs["final_visible_state"]
            for query_info in scenario["queries"]:
                query_text = query_info["query_text"]
                gold_answers = query_info["gold_answers"]
                retrieved = self._retrieve_for_eval(visible, query_text, k=5)
                retrieved_objs = [r.get("object_val") for r in retrieved]
                recall = len(set(retrieved_objs) & set(gold_answers)) / len(gold_answers) if gold_answers else 0.0
                logs["retrieval_results"].append({
                    "query": query_text,
                    "retrieved": retrieved_objs,
                    "gold": gold_answers,
                    "recall_at_k": recall
                })

        return logs

    def _retrieve_for_eval(self, memories: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval for evaluation."""
        query_tokens = set(query.lower().split())
        scored = []
        for mem in memories:
            text = f"{mem.get('subject','')} {mem.get('predicate','')} {mem.get('object_val','')}".lower()
            mem_tokens = set(text.split())
            overlap = len(query_tokens & mem_tokens) / len(query_tokens) if query_tokens else 0.0
            scored.append((mem, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]
