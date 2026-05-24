"""
Local model runner for CPU-only environments.
Supports loading quantized or smaller models for multi-agent evaluation.
"""
from typing import Dict, Any, List, Optional, Tuple
import time
from abc import ABC, abstractmethod
import os
import json
import hashlib
import re
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

from src.memory.canonicalization import (
    canonicalize_memory_triplet,
    canonicalize_object_value as canonicalize_slot_object_value,
    canonicalize_predicate_name as canonicalize_slot_predicate_name,
)

_PIPELINE_CACHE: Dict[tuple, Any] = {}
_API_LAST_CALL: Dict[str, float] = {}
_EXTRACTION_CACHE_REGISTRY: Dict[str, "_PersistentExtractionCache"] = {}


def _load_env_value(key: str) -> Optional[str]:
    def _is_placeholder(raw: Optional[str]) -> bool:
        text = str(raw or "").strip().lower()
        return text in {
            "",
            "your-api-key",
            "your_openai_api_key_here",
            "your_gemini_api_key_here",
            "your_huggingface_token_here",
        }

    value = os.getenv(key)
    if value and not _is_placeholder(value):
        return value.strip()

    project_root = Path(__file__).resolve().parents[2]
    for candidate in (project_root / ".env", project_root / ".env.example"):
        if not candidate.exists():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                lhs, rhs = line.split("=", 1)
                lhs = lhs.lstrip("\ufeff")
                if lhs.strip() == key:
                    parsed = rhs.strip().strip("\"' ")
                    if parsed and not _is_placeholder(parsed):
                        os.environ[key] = parsed
                        return parsed
        except Exception:
            continue
    return None


def _canonicalize_predicate(predicate: str, object_val: str) -> str:
    return canonicalize_slot_predicate_name(predicate)


def _canonicalize_object_value(predicate: str, object_val: str) -> str:
    return str(canonicalize_slot_object_value(object_val))


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


def _clone_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class _PersistentExtractionCache:
    """Append-only JSONL cache for extracted write proposals."""

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path).resolve()
        self._loaded = False
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return
            if self.cache_path.exists():
                with self.cache_path.open("r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        cache_key = str(payload.get("key", "")).strip()
                        items = payload.get("items")
                        if cache_key and isinstance(items, list):
                            self._entries[cache_key] = payload
            self._loaded = True

    def get_items(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        self._ensure_loaded()
        payload = self._entries.get(cache_key)
        if not payload:
            return None
        return _clone_json_value(payload.get("items", []))

    def put_items(self, payload: Dict[str, Any]) -> None:
        cache_key = str(payload.get("key", "")).strip()
        items = payload.get("items")
        if not cache_key or not isinstance(items, list):
            return

        self._ensure_loaded()
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._entries[cache_key] = _clone_json_value(payload)


def _get_persistent_extraction_cache(cache_path: Optional[str]) -> Optional[_PersistentExtractionCache]:
    if not cache_path:
        return None
    resolved = str(Path(cache_path).resolve())
    cache = _EXTRACTION_CACHE_REGISTRY.get(resolved)
    if cache is None:
        cache = _PersistentExtractionCache(resolved)
        _EXTRACTION_CACHE_REGISTRY[resolved] = cache
    return cache


class ExtractionCacheMixin:
    """Shared persistent cache helpers for real extraction agents."""

    EXTRACTION_PROMPT_VERSION = "extract_v1"

    def _init_extraction_cache(self, extraction_cache_path: Optional[str]) -> None:
        self.extraction_cache_path = str(Path(extraction_cache_path).resolve()) if extraction_cache_path else None
        self.extraction_cache = _get_persistent_extraction_cache(self.extraction_cache_path)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0

    def _build_extraction_cache_key(self, text: str) -> Optional[str]:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        payload = {
            "model_name": self.model_name,
            "prompt_version": self.EXTRACTION_PROMPT_VERSION,
            "raw_text": normalized,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _load_cached_extraction(self, text: str) -> Optional[List[Dict[str, Any]]]:
        if self.extraction_cache is None:
            return None
        cache_key = self._build_extraction_cache_key(text)
        if not cache_key:
            return None
        cached = self.extraction_cache.get_items(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        return None

    def _store_cached_extraction(
        self,
        text: str,
        items: List[Dict[str, Any]],
        *,
        raw_response: str = "",
    ) -> None:
        if self.extraction_cache is None or not items:
            return
        cache_key = self._build_extraction_cache_key(text)
        if not cache_key:
            return
        self.extraction_cache.put_items(
            {
                "key": cache_key,
                "model_name": self.model_name,
                "prompt_version": self.EXTRACTION_PROMPT_VERSION,
                "raw_text": str(text),
                "items": items,
                "raw_response": raw_response,
                "cached_at": time.time(),
            }
        )
        self.cache_writes += 1

    def get_extraction_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_path": self.extraction_cache_path,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_writes": self.cache_writes,
        }


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


class TransformerAgent(ExtractionCacheMixin, LocalAgent):
    """Agent that uses a local transformer model for generation and extraction."""
    EXTRACTION_PROMPT_VERSION = "transformer_extract_v1"

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        device: str = "cpu",
        quantization_mode: Optional[str] = None,
        strict_loading: bool = False,
        extraction_cache_path: Optional[str] = None,
    ):
        super().__init__(agent_id, model_name)
        self.device = device
        self.quantization_mode = quantization_mode or ("4bit" if str(device).startswith("cuda") else "none")
        self.strict_loading = strict_loading
        self.generator = None
        self.last_extraction_response: str = ""
        self._init_extraction_cache(extraction_cache_path)
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
        cached = self._load_cached_extraction(text)
        if cached is not None:
            return cached
        if self.generator is None:
            return []

        prompt = (
            "Extract structured facts from the input text.\n"
            "Return ONLY JSON, preferably a JSON array. A single JSON object is also acceptable.\n"
            "Each item must contain keys: subject, predicate, object_val, confidence, provenance, rationale, support_spans, extractor_id.\n"
            "Preserve explicit entities and relation names from the input when possible.\n"
            "Use provenance='llm_inferred'. Confidence must be between 0.5 and 0.95.\n"
            "Example Input: Thomas Kyd was born in the city of London\n"
            "Example Output: [{\"subject\":\"Thomas Kyd\",\"predicate\":\"birth_place\",\"object_val\":\"London\",\"confidence\":0.9,\"provenance\":\"llm_inferred\",\"rationale\":\"direct stated fact\",\"support_spans\":[{\"span_text\":\"Thomas Kyd was born in the city of London\",\"span_index\":0}],\"extractor_id\":\"extractor\"}]\n"
            f"Input: {text}\n"
            "Output JSON:"
        )

        try:
            response = self.generate_response(prompt, max_new_tokens=96)
            self.last_extraction_response = response or ""
            items = self._parse_extraction_response(response, text)
            if items:
                self._store_cached_extraction(text, items, raw_response=response or "")
            return items
        except Exception:
            pass

        return []

    def _parse_extraction_response(self, response: str, source_text: str) -> List[Dict[str, Any]]:
        cleaned = (response or "").replace("```json", "").replace("```", "").strip()
        payload_candidates: List[Any] = []

        array_match = re.search(r"\[[\s\S]*\]", cleaned)
        if array_match:
            try:
                payload_candidates.append(json.loads(array_match.group(0)))
            except Exception:
                pass

        object_match = re.search(r"\{[\s\S]*\}", cleaned)
        if object_match:
            try:
                payload_candidates.append(json.loads(object_match.group(0)))
            except Exception:
                pass

        line_fields = self._parse_line_fields(cleaned)
        if line_fields:
            payload_candidates.append(line_fields)

        valid_memories: List[Dict[str, Any]] = []
        for payload in payload_candidates:
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                normalized = self._normalize_extracted_item(item, source_text)
                if normalized:
                    valid_memories.append(normalized)

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in valid_memories:
            key = (item["subject"], item["predicate"], item["object_val"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _parse_line_fields(self, text: str) -> Optional[Dict[str, Any]]:
        patterns = {
            "subject": r"subject\s*[:=]\s*(.+)",
            "predicate": r"predicate\s*[:=]\s*(.+)",
            "object_val": r"(?:object_val|object)\s*[:=]\s*(.+)",
        }
        out: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                out[key] = match.group(1).strip().strip('",')
        return out or None

    def _normalize_extracted_item(self, item: Any, source_text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        subject, predicate, object_val = canonicalize_memory_triplet(
            item.get("subject", "unknown"),
            item.get("predicate", ""),
            item.get("object_val", item.get("object", "")),
            raw_text=source_text,
        )
        if not predicate or not object_val:
            return None
        try:
            confidence = float(item.get("confidence", 0.7))
        except Exception:
            confidence = 0.7
        confidence = max(0.5, min(confidence, 0.95))
        normalized = {
            "subject": subject or "unknown",
            "predicate": predicate,
            "object_val": object_val,
            "confidence": confidence,
            "provenance": item.get("provenance", "llm_inferred"),
            "rationale": item.get("rationale", "local_transformer_extraction"),
            "support_spans": item.get("support_spans", [{"span_text": source_text[:200], "span_index": 0}]),
            "extractor_id": item.get("extractor_id", self.model_name),
            "challenger_metadata": item.get("challenger_metadata", None),
        }
        return normalized


class GeminiAPIAgent(ExtractionCacheMixin, LocalAgent):
    """Agent that uses the Gemini REST API for extraction."""
    EXTRACTION_PROMPT_VERSION = "gemini_extract_v1"

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        strict_loading: bool = False,
        extraction_cache_path: Optional[str] = None,
    ):
        super().__init__(agent_id, model_name)
        self.strict_loading = strict_loading
        self.api_key = _load_env_value("GEMINI_API_KEY")
        self.api_base = _load_env_value("GEMINI_API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/models"
        self.min_interval_seconds = float(_load_env_value("GEMINI_MIN_INTERVAL_SECONDS") or "7")
        self.last_extraction_response: str = ""
        self._init_extraction_cache(extraction_cache_path)
        if self.strict_loading and not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini API extraction.")

    def generate_response(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return ""

        url = f"{self.api_base}/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                self._wait_for_rate_limit()
                with urllib.request.urlopen(request, timeout=60) as response:
                    content = response.read().decode("utf-8", errors="replace")
                _API_LAST_CALL[self.model_name] = time.time()
                payload = json.loads(content)
                parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(str(part.get("text", "")) for part in parts)
                return text.strip()
            except urllib.error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body or exc.reason}")
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                sleep_seconds = self._retry_delay_seconds(error_body) or max(self.min_interval_seconds, 2 ** attempt)
                time.sleep(sleep_seconds)
                continue
            except Exception as exc:
                last_error = exc
            time.sleep(max(self.min_interval_seconds, 2 ** attempt))

        if self.strict_loading and last_error is not None:
            raise RuntimeError(f"Gemini API request failed: {last_error}") from last_error
        return ""

    def extract_memories(self, text: str) -> List[Dict[str, Any]]:
        cached = self._load_cached_extraction(text)
        if cached is not None:
            return cached
        prompt = (
            "Extract structured facts from the input text.\n"
            "Return ONLY JSON, preferably a JSON array. A single JSON object is also acceptable.\n"
            "Each item must contain keys: subject, predicate, object_val, confidence, provenance, rationale, support_spans, extractor_id.\n"
            "Preserve explicit entities and relation names from the input when possible.\n"
            "Use provenance='llm_inferred'. Confidence must be between 0.5 and 0.95.\n"
            "Example Input: Thomas Kyd was born in the city of London\n"
            f"Example Output: [{{\"subject\":\"Thomas Kyd\",\"predicate\":\"birth_place\",\"object_val\":\"London\",\"confidence\":0.9,"
            "\"provenance\":\"llm_inferred\",\"rationale\":\"direct stated fact\","
            "\"support_spans\":[{\"span_text\":\"Thomas Kyd was born in the city of London\",\"span_index\":0}],"
            f"\"extractor_id\":\"{self.model_name}\"}}]\n"
            f"Input: {text}\n"
            "Output JSON:"
        )
        response = self.generate_response(prompt)
        self.last_extraction_response = response or ""
        parser = TransformerAgent.__new__(TransformerAgent)
        parser.model_name = self.model_name
        items = TransformerAgent._parse_extraction_response(parser, response, text)
        if items:
            self._store_cached_extraction(text, items, raw_response=response or "")
        return items

    def _wait_for_rate_limit(self) -> None:
        last_call = _API_LAST_CALL.get(self.model_name)
        if last_call is None:
            return
        elapsed = time.time() - last_call
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _retry_delay_seconds(self, error_body: str) -> Optional[float]:
        if not error_body:
            return None
        match = re.search(r'"retryDelay"\s*:\s*"([0-9]+)s"', error_body)
        if match:
            try:
                return float(match.group(1)) + 1.0
            except Exception:
                return None
        return None


class OpenAIAPIAgent(ExtractionCacheMixin, LocalAgent):
    """Agent that uses the OpenAI Chat Completions API for extraction."""
    EXTRACTION_PROMPT_VERSION = "openai_extract_v1"

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        strict_loading: bool = False,
        extraction_cache_path: Optional[str] = None,
    ):
        super().__init__(agent_id, model_name)
        self.strict_loading = strict_loading
        self.api_key = _load_env_value("OPENAI_API_KEY") or _load_env_value("OPEN_API_KEY")
        self.api_base = _load_env_value("OPENAI_API_BASE_URL") or "https://api.openai.com/v1"
        self.last_extraction_response: str = ""
        self._init_extraction_cache(extraction_cache_path)
        if self.strict_loading and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY (or OPEN_API_KEY) is required for OpenAI API extraction.")

    def _is_transient_network_error(self, exc: Exception) -> bool:
        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            if isinstance(reason, socket.gaierror):
                return True
            if isinstance(reason, TimeoutError):
                return True
            if isinstance(reason, OSError):
                return True
        return isinstance(exc, TimeoutError)

    def _retry_sleep_seconds(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None and retry_after > 0:
            return retry_after
        # Back off more gently for early attempts, then widen to survive
        # transient DNS / routing issues during long benchmark runs.
        return min(120.0, 2.0 * (2 ** attempt))

    def generate_response(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return ""

        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured facts from short text. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    content = response.read().decode("utf-8", errors="replace")
                payload = json.loads(content)
                message = payload.get("choices", [{}])[0].get("message", {})
                text = message.get("content", "")
                if isinstance(text, list):
                    text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
                return str(text).strip()
            except urllib.error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body or exc.reason}")
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                retry_after: Optional[float] = None
                match = re.search(r'"retry_after"\s*:\s*([0-9]+)', error_body)
                if match:
                    try:
                        retry_after = float(match.group(1)) + 1.0
                    except Exception:
                        retry_after = None
            except Exception as exc:
                last_error = exc
                if not self._is_transient_network_error(exc):
                    break
                retry_after = None
            time.sleep(self._retry_sleep_seconds(attempt, retry_after))

        if self.strict_loading and last_error is not None:
            raise RuntimeError(f"OpenAI API request failed: {last_error}") from last_error
        return ""

    def extract_memories(self, text: str) -> List[Dict[str, Any]]:
        cached = self._load_cached_extraction(text)
        if cached is not None:
            return cached
        prompt = (
            "Extract structured facts from the input text.\n"
            "Return ONLY JSON. A JSON object with a top-level key `items` is preferred.\n"
            "Each item must contain keys: subject, predicate, object_val, confidence, provenance, rationale, support_spans, extractor_id.\n"
            "Preserve explicit entities and relation names from the input when possible.\n"
            "Use provenance='llm_inferred'. Confidence must be between 0.5 and 0.95.\n"
            "If there is exactly one fact, still return JSON with `items` as an array of one object.\n"
            f"Input: {text}\n"
            "Output JSON:"
        )
        response = self.generate_response(prompt)
        self.last_extraction_response = response or ""
        parser = TransformerAgent.__new__(TransformerAgent)
        parser.model_name = self.model_name
        items = TransformerAgent._parse_extraction_response(parser, response, text)
        if items:
            self._store_cached_extraction(text, items, raw_response=response or "")
            return items
        try:
            payload = json.loads(response or "{}")
            raw_items = payload.get("items", [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            normalized = []
            for item in raw_items:
                norm = TransformerAgent._normalize_extracted_item(parser, item, text)
                if norm is not None:
                    normalized.append(norm)
            if normalized:
                self._store_cached_extraction(text, normalized, raw_response=response or "")
            return normalized
        except Exception:
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
        extraction_cache_path = kwargs.get("extraction_cache_path")
        return TransformerAgent(
            agent_id,
            model_name=model_name,
            device=device,
            quantization_mode=quantization_mode,
            strict_loading=strict_loading,
            extraction_cache_path=extraction_cache_path,
        )
    elif model_type == "gemini_api":
        model_name = kwargs.get("model_name", "gemini-2.5-flash-lite")
        strict_loading = kwargs.get("strict_loading", False)
        extraction_cache_path = kwargs.get("extraction_cache_path")
        return GeminiAPIAgent(
            agent_id,
            model_name=model_name,
            strict_loading=strict_loading,
            extraction_cache_path=extraction_cache_path,
        )
    elif model_type == "openai_api":
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        strict_loading = kwargs.get("strict_loading", False)
        extraction_cache_path = kwargs.get("extraction_cache_path")
        return OpenAIAPIAgent(
            agent_id,
            model_name=model_name,
            strict_loading=strict_loading,
            extraction_cache_path=extraction_cache_path,
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
