"""
Unified loader for all benchmarks.

Provides a single interface to load any benchmark and convert to ISF.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.format import Scenario

# Use the new adapters from src/benchmarks/adapters/
from .adapters.memab_adapter import MemABAdapter
from .adapters.longmemeval_adapter import LongMemEvalAdapter
from .adapters.safeflow_adapter import SAFEFLOWAdapter
# Legacy loaders for backward compatibility
from .longmemeval_loader import load_longmemeval
from .memoryagentbench_loader import load_memoryagentbench
from .lococo_loader import load_lococo


def load_benchmark(
    benchmark_name: str,
    **kwargs
) -> List[Scenario]:
    """
    Load a benchmark by name and convert to ISF.

    Args:
        benchmark_name: One of "memae", "longmemeval", "memoryagentbench", "conflict_res_parquet"
        **kwargs: Arguments passed to the specific loader

    Returns:
        List of Scenario objects (ISF)

    Examples:
        load_benchmark("memae", max_scenarios=100)
        load_benchmark("longmemeval", subset="s", num_samples=50)
        load_benchmark("memoryagentbench", subset="conflict")
        load_benchmark("conflict_res_parquet", max_rows=10)
    """
    if benchmark_name == "memae":
        # MemAB Conflict Resolution
        dataset_path = Path("data/raw/memab/Conflict_Resolution-00000-of-00001.parquet")
        if not dataset_path.exists():
            raise FileNotFoundError(
                "MemAE requires a local parquet file at "
                f"`{dataset_path.as_posix()}`. For fully automatic Colab runs, "
                "prefer `mab_conflict`, `longmemeval`, `locomo`, or `safeflow`."
            )
        adapter = MemABAdapter(str(dataset_path), 'conflict_resolution')
        scenarios = adapter.convert_all_to_scenarios(num_agents=2)
        if kwargs.get('max_scenarios'):
            scenarios = scenarios[:kwargs['max_scenarios']]
        return scenarios

    elif benchmark_name == "longmemeval":
        # LongMemEval
        subset = kwargs.get('subset', 's')
        limit = kwargs.get('num_samples') or kwargs.get('max_scenarios')
        if subset == "oracle":
            filename = "longmemeval_oracle.json"
        else:
            filename = f"longmemeval_{subset}_cleaned.json"

        local_path = Path("data/raw/longmemeval") / filename
        if not local_path.exists():
            try:
                from huggingface_hub import hf_hub_download

                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id="xiaowu0162/longmemeval-cleaned",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(local_path.parent),
                )
                local_path = Path(downloaded)
            except Exception:
                scenarios = load_longmemeval(subset=subset, num_samples=limit)
                if kwargs.get('max_scenarios'):
                    scenarios = scenarios[:kwargs['max_scenarios']]
                return scenarios

        adapter = LongMemEvalAdapter(str(local_path))
        return adapter.convert_all_to_scenarios(num_agents=2, limit=limit)

    elif benchmark_name == "mab":
        # MemoryAgentBench
        subset = kwargs.get('subset', 'all')
        num_samples = kwargs.get('num_samples') or kwargs.get('max_scenarios')
        scenarios = load_memoryagentbench(subset=subset, num_samples=num_samples)
        if kwargs.get('max_scenarios'):
            scenarios = scenarios[:kwargs['max_scenarios']]
        return scenarios

    elif benchmark_name in {"lococo", "locomo"}:
        # LoCoMo
        subset = kwargs.get('subset', 'test')
        num_samples = kwargs.get('num_samples') or kwargs.get('max_scenarios')
        scenarios = load_lococo(subset=subset, num_samples=num_samples)
        if kwargs.get('max_scenarios'):
            scenarios = scenarios[:kwargs['max_scenarios']]
        return scenarios

    elif benchmark_name == "custom":
        # Custom JSONL file
        custom_path = kwargs.get('custom_path')
        if not custom_path:
            raise ValueError("custom_path is required for custom benchmark")
        with open(custom_path, 'r', encoding='utf-8') as f:
            scenarios = [json.loads(line) for line in f if line.strip()]
        if kwargs.get('max_scenarios'):
            scenarios = scenarios[:kwargs['max_scenarios']]
        return scenarios

    elif benchmark_name == "safeflow":
        # SAFEFLOWBENCH - Adversarial benchmark for noisy concurrent conditions
        adapter = SAFEFLOWAdapter(use_huggingface=True)
        scenarios = adapter.convert_all_to_scenarios()
        if kwargs.get('max_scenarios'):
            scenarios = scenarios[:kwargs['max_scenarios']]
        return scenarios

    raise ValueError(f"Unsupported benchmark_name: {benchmark_name}")

def save_scenarios_to_jsonl(
    scenarios: List[Scenario],
    output_path: str
) -> None:
    """Save list of Scenario objects to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario.to_dict(), ensure_ascii=False) + '\n')
    print(f"Saved {len(scenarios)} scenarios to {output_path}")


def load_scenarios_from_jsonl(
    jsonl_path: str,
    max_scenarios: int = None
) -> List[Scenario]:
    """Load scenarios from JSONL file (ISF format)."""
    scenarios = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                data = json.loads(line)
                scenario = Scenario.from_dict(data)
                scenarios.append(scenario)
                if max_scenarios and len(scenarios) >= max_scenarios:
                    break
    print(f"Loaded {len(scenarios)} scenarios from {jsonl_path}")
    return scenarios


if __name__ == "__main__":
    # Test unified loader
    print("Testing unified loader...")

    print("\n1. MemAE (first 2 scenarios):")
    memae_scenarios = load_benchmark("memae", max_scenarios=2)
    if memae_scenarios:
        s = memae_scenarios[0]
        print(f"   ID: {s.scenario_id}, Type: {s.scenario_type}, Events: {len(s.ordered_events)}")

    print("\n2. LongMemEval oracle (first 2 scenarios):")
    longmemeval_scenarios = load_benchmark("longmemeval", subset="oracle", max_scenarios=2)
    if longmemeval_scenarios:
        s = longmemeval_scenarios[0]
        print(f"   ID: {s.scenario_id}, Type: {s.scenario_type}, Events: {len(s.ordered_events)}, Queries: {len(s.queries)}")
