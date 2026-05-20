#!/usr/bin/env python
"""
Replay symbolic QA from saved per-scenario artifacts.

This lets researchers re-score the QA layer from final visible memory without
rerunning the entire benchmark pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.qa_reasoner import answer_question_from_memories, score_answers


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_scenario_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            files.append(os.path.join(path, name))
    return files


def _replay_single(bundle: Dict[str, Any]) -> Dict[str, Any]:
    visible = bundle.get("final_visible_state", [])
    qa_failures = bundle.get("qa_failures", [])
    replays: List[Dict[str, Any]] = []
    exact = 0
    total = 0
    subem = 0
    for item in qa_failures:
        question = item.get("query", "")
        gold = item.get("gold", [])
        replay = answer_question_from_memories(question, visible)
        score = score_answers(replay.get("predicted_answers", []), gold)
        total += 1
        exact += int(bool(score.get("exact_match", False)))
        subem += int(bool(score.get("substring_exact_match", False)))
        replays.append(
            {
                "query": question,
                "gold": gold,
                "original_predicted_answers": item.get("predicted_answers", []),
                "replayed_predicted_answers": replay.get("predicted_answers", []),
                "original_path": item.get("path", []),
                "replayed_path": replay.get("path", []),
                "exact_match": bool(score.get("exact_match", False)),
                "substring_exact_match": bool(score.get("substring_exact_match", False)),
                "answer_type": replay.get("answer_type"),
                "hops": replay.get("hops"),
            }
        )
    return {
        "mode": bundle.get("mode"),
        "scenario_id": bundle.get("scenario_id"),
        "scenario_type": bundle.get("scenario_type"),
        "qa_failures_replayed": total,
        "exact_match_on_replayed_failures": exact / total if total else 0.0,
        "subem_on_replayed_failures": subem / total if total else 0.0,
        "replays": replays,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay QA from saved per-scenario artifacts.")
    parser.add_argument(
        "path",
        help="Path to a scenario artifact JSON file or to a directory such as <report>.scenarios",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to <path>.qa_replay.json",
    )
    args = parser.parse_args()

    scenario_files = _iter_scenario_files(args.path)
    if not scenario_files:
        raise FileNotFoundError(f"No scenario JSON files found at {args.path}")

    payload = {
        "source": args.path,
        "num_scenarios": len(scenario_files),
        "results": [],
    }
    for path in scenario_files:
        bundle = _load_json(path)
        payload["results"].append(_replay_single(bundle))

    output_path = args.output or f"{args.path}.qa_replay.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved QA replay report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
