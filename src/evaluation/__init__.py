"""
Evaluation and benchmarking package.

Provides:
- Multi-mode evaluation (conflict_aware, lww, naive)
- Benchmark loading from multiple sources
- Comprehensive metrics computation
- Reporting and comparison utilities
"""
from .evaluate import evaluate, run_scenario_with_writer, compute_metrics
from .run_evaluation import run_evaluation
from .reporter import generate_summary_report, save_comparison_report, export_json_report

__all__ = [
    "evaluate",
    "run_evaluation",
    "run_scenario_with_writer",
    "compute_metrics",
    "generate_summary_report",
    "save_comparison_report",
    "export_json_report",
]
