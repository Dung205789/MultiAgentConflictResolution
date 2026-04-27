import json
import re
import os
from typing import Dict, List

try:
    from transformers import pipeline
except Exception:
    pipeline = None


MEMORY_PATTERNS = [
    (r"tôi là sinh viên năm (\d+)", ("user", "study_year", "profile")),
    (r"tôi là sinh viên cntt", ("user", "major", "profile"), "cntt"),
    (r"tôi sống ở (.+)", ("user", "city", "profile")),
    (r"tôi ở (.+)", ("user", "city", "profile")),
    (r"tôi thích học đêm|tôi thích học tối|tôi thích học tối muộn", ("user", "study_time", "preference"), "night"),
    (r"tôi thích học sáng", ("user", "study_time", "preference"), "morning"),
    (r"gần đây tôi chuyển sang học sáng|dạo này tôi chuyển sang học sáng|gần đây tôi học sáng tốt hơn", ("user", "study_time", "preference"), "morning"),
    (r"hiện tại tôi chuyển sang học tối|tôi chuyển sang thích học tối|bây giờ tôi học tối sẽ hiệu quả hơn", ("user", "study_time", "preference"), "night"),
    (r"tôi thích (python|javascript|java|sql|golang|node\.js|c\+\+|excel)", ("user", "language", "preference")),
    (r"tôi muốn làm (backend|frontend|data science|data engineer|research|mobile|ai engineer|security|data analyst|hệ thống nhúng|ai)", ("user", "career_goal", "goal")),
    (r"tôi đang tập trung vào (nlp|agent memory|cv)", ("user", "focus_area", "goal")),
    (r"học kỳ này tôi chuyển sang (agent memory)", ("user", "focus_area", "goal")),
    (r"thực ra học kỳ này tôi chuyển sang (agent memory)", ("user", "focus_area", "goal")),
    (r"tôi học thêm (fastapi|react|flutter|linux|postgresql|pytorch|spring boot|docker|sql|spark|transformers|linux hardening)", ("user", "focus_area", "goal")),
    (r"tôi bắt đầu học (spark)", ("user", "focus_area", "goal")),
]

_PREDICATE_NORMALIZE = {
    "prefers_study_time": "study_time",
    "study_time_pref": "study_time",
    "location": "city",
    "preferred_language": "language",
    "goal_area": "focus_area",
}

_OBJECT_NORMALIZE = {
    "buổi sáng": "morning",
    "sáng": "morning",
    "morning": "morning",
    "sáng sớm": "morning",
    "buổi tối": "night",
    "tối": "night",
    "đêm": "night",
    "night": "night",
    "ai memory": "agent memory",
}


_GEN = None


def _load_generator():
    global _GEN
    if _GEN is not None:
        return _GEN
    if pipeline is None:
        return None
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_qwen_15b = os.path.join(project_root, "models_cache", "Qwen2.5-3B-Instruct")
    local_qwen_7b = os.path.join(project_root, "models_cache", "Qwen2.5-7B-Instruct")

    models = [
        local_qwen_15b,
        local_qwen_7b,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ]
    for m in models:
        try:
            # Force CPU-only to avoid CUDA dependency issues
            kwargs = {"model": m, "device": -1, "trust_remote_code": True}
            _GEN = pipeline("text-generation", **kwargs)
            return _GEN
        except Exception:
            continue
    return None


def normalize_text(text: str) -> str:
    return text.strip().lower()


def normalize_memory(mem: Dict) -> Dict:
    m = dict(mem)
    pred = normalize_text(str(m.get("predicate", "")))
    obj = normalize_text(str(m.get("object_val", "")))
    pred = _PREDICATE_NORMALIZE.get(pred, pred)
    obj = _OBJECT_NORMALIZE.get(obj, obj)
    m["predicate"] = pred
    m["object_val"] = obj
    return m


def _rule_extract(text: str) -> List[Dict]:
    t = normalize_text(text)
    extracted = []

    for pattern in MEMORY_PATTERNS:
        if len(pattern) == 2:
            regex, (subject, predicate, memory_type) = pattern
            fixed_object = None
        else:
            regex, (subject, predicate, memory_type), fixed_object = pattern

        m = re.search(regex, t)
        if not m:
            continue

        if fixed_object is not None:
            obj = fixed_object
        elif m.groups():
            obj = m.group(1).strip()
        else:
            obj = t

        if predicate == "study_year":
            obj = f"year_{obj}"

        extracted.append(
            normalize_memory(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object_val": obj,
                    "memory_type": memory_type,
                    "confidence": 0.85 if fixed_object else 0.7,
                    "provenance": "explicit",
                    "evidence_span": text,
                }
            )
        )

    return extracted


def _llm_extract(text: str, mode: str = "debug_fallback") -> List[Dict]:
    gen = _load_generator()
    if gen is None:
        if mode == "research_strict":
            raise RuntimeError(
                "LLM extraction requested in research_strict mode but no model available. "
                "Install transformers and a supported model (Qwen2.5-1.5B-Instruct or SmolLM2-1.7B-Instruct)."
            )
        # In debug_fallback mode, return empty list when model is not available
        return []

    prompt = (
        "Trích xuất memory facts từ câu user. Trả về JSON list, mỗi item có: "
        "subject,predicate,object_val,memory_type,confidence,evidence_span. "
        "Chỉ trả JSON, không thêm text.\n"
        f"Input: {text}\nOutput:"
    )

    try:
        out = gen(prompt, max_new_tokens=180, do_sample=False, temperature=0.0)
        content = out[0]["generated_text"]
        if "Output:" in content:
            content = content.split("Output:", 1)[1].strip()
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1:
            return []
        arr = json.loads(content[start : end + 1])
        cleaned = []
        for x in arr:
            if not isinstance(x, dict):
                continue
            item = {
                "subject": x.get("subject", "user"),
                "predicate": x.get("predicate", ""),
                "object_val": x.get("object_val", x.get("object", "")),
                "memory_type": x.get("memory_type", "preference"),
                "confidence": float(x.get("confidence", 0.55)),
                "provenance": "llm_inferred",
                "evidence_span": x.get("evidence_span", text),
            }
            item = normalize_memory(item)
            if item["predicate"] and item["object_val"]:
                cleaned.append(item)
        return cleaned
    except Exception:
        return []


def extract_memories_from_text(text: str, use_llm: bool = False, mode: str = "debug_fallback") -> List[Dict]:
    """
    Extract structured memories from text.
    
    Args:
        text: Input text to extract from
        use_llm: Whether to use LLM-based extraction
        mode: Either "research_strict" (fail if models required but unavailable) or 
              "debug_fallback" (allow fallback to rule-based)
    
    Returns:
        List of extracted memory items
        
    Raises:
        RuntimeError: If mode="research_strict", use_llm=True, but LLM model unavailable
    """
    llm_items: List[Dict] = []
    if use_llm:
        if mode == "research_strict":
            gen = _load_generator()
            if gen is None:
                raise RuntimeError(
                    "LLM extraction requested in research_strict mode but no model available. "
                    "Install transformers and a supported model (Qwen2.5-1.5B-Instruct or SmolLM2-1.7B-Instruct)."
                )
        llm_items = _llm_extract(text, mode=mode)

    rule_items = _rule_extract(text)

    merged = {}
    for item in llm_items + rule_items:
        key = (item["predicate"], item["object_val"])
        if key not in merged or item.get("confidence", 0.0) > merged[key].get("confidence", 0.0):
            merged[key] = item

    return list(merged.values())
