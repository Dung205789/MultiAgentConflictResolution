import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.multi_agent_pipeline import MultiAgentPipeline, load_benchmark


def run_demo(benchmark_path: str = "data/enhanced_multi_agent_benchmark.jsonl", n: int = 3):
    scenarios = load_benchmark(benchmark_path)[:n]

    for mode in ["conflict_aware", "lww", "naive"]:
        print(f"\n=== DEMO MODE: {mode} ===")
        pipe = MultiAgentPipeline(mode=mode, persistence_path=f"tmp_demo_{mode}.jsonl")
        for s in scenarios:
            out = pipe.run_scenario(s, enable_retrieval_eval=bool(s.get("queries")))
            print(f"- scenario: {out['scenario_id']}")
            print(f"  conflicts: {len(out['detected_conflicts'])}")
            print(f"  final_visible: {[(x['predicate'], x.get('object_val')) for x in out['final_visible_state']]}")
            if 'metrics' in out:
                print(f"  state_match: {out['metrics']['state_match']}")
                if 'avg_retrieval_recall' in out['metrics']:
                    print(f"  retrieval_recall: {out['metrics']['avg_retrieval_recall']:.3f}")


if __name__ == "__main__":
    run_demo()
