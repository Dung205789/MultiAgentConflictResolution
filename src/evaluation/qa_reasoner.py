"""
Deterministic QA reasoner for graph-like memory benchmarks.

The current primary use case is MemoryAgentBench Conflict_Resolution, where the
benchmark provides factual contexts plus compositional questions. This module
builds a lightweight symbolic graph from final visible memories and answers
questions through bounded multi-hop traversal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _normalize(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = text.replace('"', " ").replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer(text: Any) -> str:
    text = _normalize(text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text


def _canonical_node_name(name: str) -> str:
    text = str(name or "").strip().rstrip(".")
    lowered = text.lower()
    wrappers = (
        "the city of ",
        "city of ",
        "the country of ",
        "country of ",
        "the continent of ",
        "continent of ",
        "the language of ",
        "language of ",
        "the religion of ",
        "religion of ",
        "the sport of ",
        "sport of ",
    )
    for wrapper in wrappers:
        if lowered.startswith(wrapper):
            return text[len(wrapper):].strip()
    return text


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    output_type: str


QUESTION_TEMPLATES: Sequence[Tuple[re.Pattern[str], Sequence[str]]] = (
    (re.compile(r"^what is the country of citizenship of the spouse of the author of (.+?)\??$", re.IGNORECASE), ("author", "spouse", "citizenship")),
    (re.compile(r"^in which location did the spouse of (.+?) pass away\??$", re.IGNORECASE), ("spouse", "death_place")),
    (re.compile(r"^which country is the birthplace of the sport associated with (.+?)\??$", re.IGNORECASE), ("sport", "origin_country")),
    (re.compile(r"^from which country does the sport, with which (.+?) is associated, come from\??$", re.IGNORECASE), ("sport", "origin_country")),
    (re.compile(r"^which city serves as the capital of the country of origin for the sport of (.+?)\??$", re.IGNORECASE), ("sport", "origin_country", "capital")),
    (re.compile(r"^what is the sport that (.+?) plays in which he is known for his position on the team\??$", re.IGNORECASE), ("position", "sport")),
    (re.compile(r"^what is the name of the educational institution where the partner of (.+?) was educated\??$", re.IGNORECASE), ("spouse", "educated_at")),
    (re.compile(r"^which country does the current ceo of (.+?) hold citizenship of\??$", re.IGNORECASE), ("ceo", "citizenship")),
    (re.compile(r"^on which continent was the place of birth of (.+?) located\??$", re.IGNORECASE), ("birth_place", "continent")),
    (re.compile(r"^to which continent does the country of citizenship of (.+?) pertain\??$", re.IGNORECASE), ("citizenship", "continent")),
    (re.compile(r"^what is the location of work of the founder of the religion to which (.+?) belongs\??$", re.IGNORECASE), ("religion", "founder", "work_site")),
    (re.compile(r"^what is the religious affiliation of the person who founded (.+?)\??$", re.IGNORECASE), ("founder", "religion")),
    (re.compile(r"^what was the religion of the person with the citizenship of the country to which (.+?) belonged\??$", re.IGNORECASE), ("citizenship", "rev_citizenship", "religion")),
    (re.compile(r"^what is the capital city of the country where the sport associated with (.+?) has its origin\??$", re.IGNORECASE), ("sport", "origin_country", "capital")),
    (re.compile(r"^which faith is the partner of (.+?) associated with\??$", re.IGNORECASE), ("spouse", "religion")),
    (re.compile(r"^what was the name of the head of state in the country where (.+?) was a citizen\??$", re.IGNORECASE), ("citizenship", "head_of_state")),
    (re.compile(r"^from which capital city did (.+?), a sports competition, originate\??$", re.IGNORECASE), ("sport", "origin_country", "capital")),
    (re.compile(r"^in which city is the capital of the country where (.+?) held a citizenship\??$", re.IGNORECASE), ("citizenship", "capital")),
    (re.compile(r"^what is the job title of the chairperson of the political organization (.+?)\??$", re.IGNORECASE), ("chairperson", "occupation")),
    (re.compile(r"^what is the work location of the person who founded the religion (.+?) belongs to\??$", re.IGNORECASE), ("religion", "founder", "work_site")),
    (re.compile(r"^which city is the capital of the country where the ceo of the developer of (.+?) holds a citizenship\??$", re.IGNORECASE), ("producer_company", "ceo", "citizenship", "capital")),
    (re.compile(r"^what is the name of the person who is the chief executive officer of the developer of (.+?)\??$", re.IGNORECASE), ("producer_company", "ceo")),
    (re.compile(r"^what city serves as the capital of the country where the performer who is the spouse of (.+?) holds citizenship\??$", re.IGNORECASE), ("spouse", "performer", "citizenship", "capital")),
    (re.compile(r"^what city or town is the headquarters of the educational institution, where the performer of the album (.+?) received education, located in\??$", re.IGNORECASE), ("performer", "educated_at", "headquarters_city")),
    (re.compile(r"^what is the name of the chief executive of the country where the sport associated with (.+?) was first developed\??$", re.IGNORECASE), ("sport", "origin_country", "head_of_state")),
    (re.compile(r"^what is the continent of origin of the chief executive officer for the developer of (.+?)\??$", re.IGNORECASE), ("producer_company", "ceo", "citizenship", "continent")),
    (re.compile(r"^what is the name of the significant creation associated with the creator of (.+?)\??$", re.IGNORECASE), ("creator", "known_for")),
)


RAW_PATTERNS: Sequence[Tuple[re.Pattern[str], str, str, str, str]] = (
    (re.compile(r"^the chairperson of (.+?) is (.+?)$", re.IGNORECASE), "chairperson", "org", "person", "person"),
    (re.compile(r"^the director of (.+?) is (.+?)$", re.IGNORECASE), "director", "work", "person", "person"),
    (re.compile(r"^the author of (.+?) is (.+?)$", re.IGNORECASE), "author", "work", "person", "person"),
    (re.compile(r"^the chief executive officer of (.+?) is (.+?)$", re.IGNORECASE), "ceo", "org", "person", "person"),
    (re.compile(r"^the capital of (.+?) is (.+?)$", re.IGNORECASE), "capital", "country", "city", "city"),
    (re.compile(r"^the name of the current head of state in (.+?) is (.+?)$", re.IGNORECASE), "head_of_state", "country", "person", "person"),
    (re.compile(r"^the prime minister of (.+?) is (.+?)$", re.IGNORECASE), "prime_minister", "country", "person", "person"),
    (re.compile(r"^the official language of (.+?) is (.+?)$", re.IGNORECASE), "official_language", "geo", "language", "language"),
    (re.compile(r"^(.+?) is affiliated with the religion of (.+?)$", re.IGNORECASE), "religion", "entity", "religion", "religion"),
    (re.compile(r"^(.+?) is associated with the sport of (.+?)$", re.IGNORECASE), "sport", "entity", "sport", "sport"),
    (re.compile(r"^(.+?) plays the position of (.+?)$", re.IGNORECASE), "position", "entity", "position", "position"),
    (re.compile(r"^the type of music that (.+?) plays is (.+?)$", re.IGNORECASE), "music_type", "entity", "music", "music"),
    (re.compile(r"^(.+?) was created in the country of (.+?)$", re.IGNORECASE), "origin_country", "entity", "country", "country"),
    (re.compile(r"^(.+?) was developed by (.+?)$", re.IGNORECASE), "producer_company", "entity", "org", "org"),
    (re.compile(r"^(.+?) was founded in the city of (.+?)$", re.IGNORECASE), "origin_city", "entity", "city", "city"),
    (re.compile(r"^(.+?) was founded by (.+?)$", re.IGNORECASE), "founder", "entity", "person", "person"),
    (re.compile(r"^the univeristy where (.+?) was educated is (.+?)$", re.IGNORECASE), "educated_at", "person", "institution", "institution"),
    (re.compile(r"^the university where (.+?) was educated is (.+?)$", re.IGNORECASE), "educated_at", "person", "institution", "institution"),
    (re.compile(r"^the headquarters of (.+?) is located in the city of (.+?)$", re.IGNORECASE), "headquarters_city", "org", "city", "city"),
    (re.compile(r"^(.+?) is employed by (.+?)$", re.IGNORECASE), "employer", "person", "org", "org"),
    (re.compile(r"^(.+?) worked in the city of (.+?)$", re.IGNORECASE), "work_location", "person", "city", "city"),
    (re.compile(r"^(.+?) was performed by (.+?)$", re.IGNORECASE), "performer", "work", "person", "person"),
    (re.compile(r"^(.+?) was created by (.+?)$", re.IGNORECASE), "creator", "entity", "person", "person"),
    (re.compile(r"^(.+?) is famous for (.+?)$", re.IGNORECASE), "known_for", "person", "work", "work"),
    (re.compile(r"^(.+?) is a citizen of (.+?)$", re.IGNORECASE), "citizenship", "person", "country", "country"),
    (re.compile(r"^(.+?) is located in the continent of (.+?)$", re.IGNORECASE), "continent", "geo", "continent", "continent"),
    (re.compile(r"^(.+?) is married to (.+?)$", re.IGNORECASE), "spouse", "person", "person", "person"),
    (re.compile(r"^(.+?) speaks the language of (.+?)$", re.IGNORECASE), "language", "person", "language", "language"),
    (re.compile(r"^(.+?) works in the field of (.+?)$", re.IGNORECASE), "occupation", "person", "occupation", "occupation"),
    (re.compile(r"^the company that produced (.+?) is (.+?)$", re.IGNORECASE), "producer_company", "work", "org", "org"),
    (re.compile(r"^the company that originally broadcasted (.+?) is (.+?)$", re.IGNORECASE), "original_broadcaster", "work", "org", "org"),
    (re.compile(r"^(.+?)'s child is (.+?)$", re.IGNORECASE), "child", "person", "person", "person"),
)


RELATION_KEYWORDS: Dict[str, Set[str]] = {
    "author": {"author"},
    "spouse": {"spouse", "partner", "married"},
    "citizenship": {"citizenship", "citizen", "country"},
    "rev_citizenship": {"citizenship", "citizen", "person"},
    "death_place": {"pass away", "died", "death", "place of death"},
    "birth_place": {"born", "birthplace", "place of birth"},
    "sport": {"sport", "discipline", "specialize"},
    "position": {"position", "plays in", "known for position"},
    "position_sport": {"sport", "position", "plays in", "specialize"},
    "origin_country": {"country of origin", "come from", "originate", "birthplace of the sport", "hail"},
    "origin_city": {"origin", "founded in", "came from"},
    "capital": {"capital", "capital city"},
    "continent": {"continent"},
    "religion": {"religion", "faith", "religious affiliation", "believed in"},
    "founder": {"founder", "founded", "created"},
    "head_of_state": {"head of state", "chief public representative"},
    "prime_minister": {"prime minister"},
    "official_language": {"official language", "language is recognized"},
    "work_location": {"location of work", "worked in"},
    "employer": {"employed by", "works at", "company"},
    "headquarters_city": {"headquarters", "originally broadcasted", "location"},
    "ceo": {"chief executive officer", "ceo"},
    "director": {"director", "managed"},
    "performer": {"performer", "performed by"},
    "creator": {"creator", "created by"},
    "educated_at": {"educational institution", "educated"},
    "language": {"language"},
    "occupation": {"occupation", "job title", "field"},
    "child": {"child"},
    "known_for": {"famous for", "made famous by", "significant creation"},
    "producer_company": {"produced", "company", "developer", "developed"},
    "original_broadcaster": {"broadcasted", "original broadcast"},
}


RELATION_OUTPUT_TYPES: Dict[str, str] = {
    "author": "person",
    "spouse": "person",
    "citizenship": "country",
    "rev_citizenship": "person",
    "death_place": "city",
    "birth_place": "city",
    "sport": "sport",
    "position": "position",
    "position_sport": "sport",
    "origin_country": "country",
    "origin_city": "city",
    "capital": "city",
    "continent": "continent",
    "religion": "religion",
    "founder": "person",
    "head_of_state": "person",
    "prime_minister": "person",
    "official_language": "language",
    "work_location": "city",
    "employer": "org",
    "headquarters_city": "city",
    "ceo": "person",
    "director": "person",
    "performer": "person",
    "creator": "person",
    "educated_at": "institution",
    "language": "language",
    "occupation": "occupation",
    "child": "person",
    "known_for": "work",
    "producer_company": "org",
    "original_broadcaster": "org",
}


REVERSE_RELATIONS: Dict[str, str] = {
    "citizenship": "rev_citizenship",
    "author": "rev_author",
    "founder": "rev_founder",
    "performer": "rev_performer",
    "creator": "rev_creator",
    "spouse": "rev_spouse",
    "child": "rev_child",
    "religion": "rev_religion",
    "sport": "rev_sport",
    "position": "rev_position",
    "educated_at": "rev_educated_at",
    "official_language": "rev_official_language",
}


def _extract_edges_from_raw_statement(text: str) -> List[Edge]:
    clean = str(text or "").strip().rstrip(".")
    edges: List[Edge] = []
    for pattern, relation, _, _, output_type in RAW_PATTERNS:
        match = pattern.match(clean)
        if not match:
            continue
        left = match.group(1).strip()
        right = match.group(2).strip()
        edges.append(Edge(left, relation, right, output_type))
        return edges
    return edges


def _extract_edges_from_memory(mem: Dict[str, Any]) -> List[Edge]:
    raw_text = str(mem.get("raw_text", "")).strip()
    if raw_text:
        raw_edges = _extract_edges_from_raw_statement(raw_text)
        if raw_edges:
            return [
                Edge(_canonical_node_name(edge.source), edge.relation, _canonical_node_name(edge.target), edge.output_type)
                for edge in raw_edges
            ]

    subject = str(mem.get("subject", "")).strip()
    predicate = str(mem.get("predicate", "")).strip()
    obj = str(mem.get("object_val", "")).strip()
    if not subject or not predicate or not obj:
        return []

    if predicate == "raw_statement":
        return _extract_edges_from_raw_statement(obj)

    if predicate == "capital" and _normalize(subject) == "the" and " is " in obj:
        left, right = obj.rsplit(" is ", 1)
        return [Edge(_canonical_node_name(left), "capital", _canonical_node_name(right), "city")]

    relation_map = {
        "birth_place": ("birth_place", "city"),
        "death_place": ("death_place", "city"),
        "spouse": ("spouse", "person"),
        "citizenship": ("citizenship", "country"),
        "location": ("continent", "continent"),
        "founder": ("founder", "person"),
        "performer": ("performer", "person"),
        "creator": ("creator", "person"),
        "language": ("language", "language"),
        "work_location": ("work_location", "city"),
        "capital": ("capital", "city"),
        "founder_location": ("origin_city", "city"),
        "occupation": ("occupation", "occupation"),
        "sport": ("sport", "sport"),
        "position": ("position", "position"),
        "religion": ("religion", "religion"),
        "ceo": ("ceo", "person"),
        "chairperson": ("chairperson", "person"),
        "head_of_state": ("head_of_state", "person"),
        "prime_minister": ("prime_minister", "person"),
        "official_language": ("official_language", "language"),
        "educated_at": ("educated_at", "institution"),
        "employer": ("employer", "org"),
        "headquarters_city": ("headquarters_city", "city"),
        "known_for": ("known_for", "work"),
        "music_type": ("music_type", "music"),
        "child": ("child", "person"),
        "origin_country": ("origin_country", "country"),
        "producer_company": ("producer_company", "org"),
        "original_broadcaster": ("original_broadcaster", "org"),
        "director": ("director", "person"),
        "author": ("author", "person"),
    }
    if predicate not in relation_map:
        return []

    relation, output_type = relation_map[predicate]
    return [Edge(_canonical_node_name(subject), relation, _canonical_node_name(obj), output_type)]


def build_memory_graph(memories: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, List[Edge]], Dict[str, Set[str]], Dict[str, str]]:
    graph: Dict[str, List[Edge]] = defaultdict(list)
    name_index: Dict[str, Set[str]] = defaultdict(set)
    canonical_name: Dict[str, str] = {}

    def register_node(name: str) -> None:
        norm = _normalize(name)
        if not norm:
            return
        name_index[norm].add(name)
        canonical_name.setdefault(norm, name)

    for mem in memories:
        for edge in _extract_edges_from_memory(mem):
            canonical_edge = Edge(
                _canonical_node_name(edge.source),
                edge.relation,
                _canonical_node_name(edge.target),
                edge.output_type,
            )
            graph[canonical_edge.source].append(canonical_edge)
            register_node(canonical_edge.source)
            register_node(canonical_edge.target)

            reverse_relation = REVERSE_RELATIONS.get(canonical_edge.relation)
            if reverse_relation:
                reverse_type = "person" if reverse_relation.startswith("rev_") and edge.relation == "citizenship" else "entity"
                reverse_out_type = {
                    "rev_citizenship": "person",
                    "rev_author": "work",
                    "rev_founder": "entity",
                    "rev_performer": "work",
                    "rev_creator": "entity",
                    "rev_spouse": "person",
                    "rev_child": "person",
                    "rev_religion": "entity",
                    "rev_sport": "entity",
                    "rev_position": "entity",
                    "rev_educated_at": "person",
                    "rev_official_language": "geo",
                }.get(reverse_relation, reverse_type)
                graph[canonical_edge.target].append(Edge(canonical_edge.target, reverse_relation, canonical_edge.source, reverse_out_type))

            register_node(_canonical_node_name(edge.source))
            register_node(_canonical_node_name(edge.target))

    return graph, name_index, canonical_name


def _infer_expected_types(question: str) -> Set[str]:
    q = _normalize(question)
    expected: Set[str] = set()
    if (
        "which country" in q
        or "from which country" in q
        or "country of citizenship" in q
        or "country of origin" in q
        or "come from" in q
        or "originate" in q
    ):
        expected.add("country")
    if "which city" in q or "in which city" in q or "capital city" in q or "what city" in q or "city or town" in q:
        expected.add("city")
    if "which continent" in q or "to which continent" in q or "on which continent" in q:
        expected.add("continent")
    if "which language" in q or "official language" in q or "language of the work" in q:
        expected.add("language")
    if q.startswith("who ") or "what is the name of" in q:
        expected.add("person")
    if "significant creation" in q or "created by the individual" in q or "work that was created" in q:
        expected.add("work")
    if "what is the sport" in q or "which sports discipline" in q or "specialize in" in q:
        expected.add("sport")
    if "religion" in q or "faith" in q or "religious affiliation" in q:
        expected.add("religion")
    if "educational institution" in q or "where" in q and "educated" in q:
        expected.add("institution")
    if "headquarters" in q and "city" not in expected:
        expected.add("city")
    if "job title" in q or "occupation" in q or "field" in q:
        expected.add("occupation")
    if "place of death" in q or "pass away" in q or "birthplace" in q or "place of birth" in q:
        expected.add("city")
    return expected


def _resolve_anchor_node(anchor: str, graph: Dict[str, List[Edge]]) -> Optional[str]:
    anchor_norm = _normalize(_canonical_node_name(anchor))
    if not anchor_norm:
        return None
    for node in graph.keys():
        if _normalize(node) == anchor_norm:
            return node
    for node in graph.keys():
        node_norm = _normalize(node)
        if anchor_norm in node_norm or node_norm in anchor_norm:
            return node
    return None


def _follow_relation(graph: Dict[str, List[Edge]], nodes: Sequence[str], relation: str) -> Tuple[List[str], List[str]]:
    next_nodes: List[str] = []
    path_fragments: List[str] = []
    for node in nodes:
        for edge in graph.get(node, []):
            if relation == "work_site":
                if edge.relation == "work_location":
                    next_nodes.append(edge.target)
                    path_fragments.append(f"{edge.source} --{edge.relation}--> {edge.target}")
                elif edge.relation == "employer":
                    for employer_edge in graph.get(edge.target, []):
                        if employer_edge.relation == "headquarters_city":
                            next_nodes.append(employer_edge.target)
                            path_fragments.append(f"{edge.source} --{edge.relation}--> {edge.target}")
                            path_fragments.append(f"{employer_edge.source} --{employer_edge.relation}--> {employer_edge.target}")
                continue
            if edge.relation == relation:
                next_nodes.append(edge.target)
                path_fragments.append(f"{edge.source} --{edge.relation}--> {edge.target}")
    deduped = []
    seen = set()
    for node in next_nodes:
        norm = _normalize(node)
        if norm and norm not in seen:
            deduped.append(node)
            seen.add(norm)
    return deduped, path_fragments


def _answer_with_templates(question: str, graph: Dict[str, List[Edge]]) -> Optional[Dict[str, Any]]:
    for pattern, chain in QUESTION_TEMPLATES:
        match = pattern.match(question.strip())
        if not match:
            continue
        anchor = _resolve_anchor_node(match.group(1).strip(), graph)
        if not anchor:
            return None
        current_nodes = [anchor]
        path_fragments: List[str] = []
        for relation in chain:
            current_nodes, fragments = _follow_relation(graph, current_nodes, relation)
            path_fragments.extend(fragments)
            if not current_nodes:
                return None
        answer = current_nodes[0]
        relation = chain[-1]
        output_type = RELATION_OUTPUT_TYPES.get(relation, "entity")
        return {
            "answer": answer,
            "path": path_fragments,
            "score": 100.0 - len(chain),
            "anchor": anchor,
            "hops": len(chain),
            "answer_type": output_type,
        }
    return None


def _find_anchor_entities(question: str, graph: Dict[str, List[Edge]]) -> List[str]:
    normalized_question = f" {_normalize(question)} "
    stop_nodes = {
        "the", "a", "an", "what", "which", "who", "where", "country", "city",
        "continent", "religion", "sport", "language", "person", "name",
    }
    matches: List[Tuple[int, str]] = []
    for node in graph.keys():
        norm = _normalize(node)
        if len(norm) < 3:
            continue
        if norm in stop_nodes:
            continue
        if f" {norm} " in normalized_question:
            matches.append((len(norm), node))
    matches.sort(key=lambda item: item[0], reverse=True)

    selected: List[str] = []
    selected_norms: List[str] = []
    for _, node in matches:
        norm = _normalize(node)
        if any(norm in prev or prev in norm for prev in selected_norms):
            continue
        selected.append(node)
        selected_norms.append(norm)
        if len(selected) >= 6:
            break
    return selected


def _path_score(question: str, anchor: str, path: Sequence[Edge], expected_types: Set[str]) -> float:
    q = _normalize(question)
    score = min(len(_normalize(anchor).split()), 6) * 0.5

    for edge in path:
        keywords = RELATION_KEYWORDS.get(edge.relation, set())
        for keyword in keywords:
            if _normalize(keyword) and _normalize(keyword) in q:
                score += 2.0 if " " in keyword else 1.0

    last_edge = path[-1]
    if not expected_types or last_edge.output_type in expected_types:
        score += 4.0
    else:
        score -= 6.0

    score += len(path) * 0.75

    answer_norm = normalize_answer(last_edge.target)
    if answer_norm and answer_norm in q:
        score -= 2.0

    if expected_types:
        intermediate_hit = any(edge.output_type in expected_types for edge in path[:-1])
        if intermediate_hit and last_edge.output_type not in expected_types:
            score -= 5.0

    if last_edge.relation.startswith("rev_") and "who " not in q and "name of" not in q:
        score -= 2.0

    if any(rel in q for rel in ("country of origin", "capital", "head of state", "official language", "chairperson")):
        score += 0.5

    return score


def _search_best_answer(question: str, graph: Dict[str, List[Edge]]) -> Optional[Dict[str, Any]]:
    anchors = _find_anchor_entities(question, graph)
    if not anchors:
        return None

    expected_types = _infer_expected_types(question)
    best: Optional[Dict[str, Any]] = None

    for anchor in anchors:
        queue: List[Tuple[str, List[Edge], Set[str]]] = [(anchor, [], {_normalize(anchor)})]
        for depth in range(4):
            next_queue: List[Tuple[str, List[Edge], Set[str]]] = []
            for node, path, visited in queue:
                for edge in graph.get(node, []):
                    next_norm = _normalize(edge.target)
                    if next_norm in visited:
                        continue
                    new_path = path + [edge]
                    candidate = {
                        "answer": edge.target,
                        "path": [f"{p.source} --{p.relation}--> {p.target}" for p in new_path],
                        "score": _path_score(question, anchor, new_path, expected_types),
                        "anchor": anchor,
                        "hops": len(new_path),
                        "answer_type": edge.output_type,
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate
                    if len(new_path) < 4:
                        next_queue.append((edge.target, new_path, visited | {next_norm}))
            queue = next_queue
            if not queue:
                break

    return best


def answer_question_from_memories(question: str, memories: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    graph, _, _ = build_memory_graph(memories)
    best = _answer_with_templates(question, graph)
    if best is None:
        best = _search_best_answer(question, graph)
    if best is None:
        return {
            "question": question,
            "predicted_answers": [],
            "path": [],
            "hops": 0,
            "score": 0.0,
            "answer_type": None,
        }
    return {
        "question": question,
        "predicted_answers": [best["answer"]],
        "path": best["path"],
        "hops": best["hops"],
        "score": best["score"],
        "answer_type": best["answer_type"],
        "anchor": best["anchor"],
    }


def score_answers(predicted_answers: Sequence[Any], gold_answers: Sequence[Any]) -> Dict[str, Any]:
    pred_norm = [normalize_answer(x) for x in predicted_answers if normalize_answer(x)]
    gold_norm = [normalize_answer(x) for x in gold_answers if normalize_answer(x)]
    exact_match = any(p == g for p in pred_norm for g in gold_norm) if pred_norm and gold_norm else False
    return {
        "predicted_normalized": pred_norm,
        "gold_normalized": gold_norm,
        "exact_match": exact_match,
    }
