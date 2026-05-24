import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.unified_loader import load_benchmark, save_scenarios_to_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run a strict real-model end_to_end_extract benchmark slice."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-index", type=int, default=1, help="0-based index within the first two Conflict_Resolution scenarios.")
    parser.add_argument("--load-max-scenarios", type=int, default=2, help="Number of source scenarios to load before slicing.")
    parser.add_argument("--agent1-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--agent2-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--allow-structured-fallback-in-end-to-end", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_benchmark("mab", subset="Conflict_Resolution", max_scenarios=args.load_max_scenarios)
    if not scenarios:
        raise RuntimeError("No Conflict_Resolution scenarios were loaded.")
    if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
        raise IndexError(f"scenario-index {args.scenario_index} is out of range for {len(scenarios)} loaded scenarios.")

    scenario = scenarios[args.scenario_index]
    slice_path = output_dir / f"mab_conflict_s{args.scenario_index}.jsonl"
    save_scenarios_to_jsonl([scenario], str(slice_path))

    manifest = {
        "source_benchmark": "mab",
        "source_subset": "Conflict_Resolution",
        "loaded_scenarios": len(scenarios),
        "selected_index": args.scenario_index,
        "selected_scenario_id": getattr(scenario, "scenario_id", None),
        "selected_scenario_type": getattr(scenario, "scenario_type", None),
        "custom_path": str(slice_path),
        "track": "end_to_end_extract",
        "agent1_model": args.agent1_model,
        "agent2_model": args.agent2_model,
        "allow_structured_fallback_in_end_to_end": args.allow_structured_fallback_in_end_to_end,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "app" / "main.py"),
        "--benchmark",
        "custom",
        "--custom-path",
        str(slice_path),
        "--track",
        "end_to_end_extract",
        "--agent1-model",
        args.agent1_model,
        "--agent2-model",
        args.agent2_model,
        "--device",
        args.device,
        "--modes",
        "conflict_aware",
        "--enable-error-analysis",
        "--emit-scenario-bundles",
        "--output-dir",
        str(output_dir),
    ]
    if args.allow_structured_fallback_in_end_to_end:
        cmd.append("--allow-structured-fallback-in-end-to-end")

    print("Prepared slice:", slice_path)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
