import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
FULL_REPORT = ROOT / "reports" / "paper_mode_mab8_fc_refresh" / "mab_conflict_report.json"
ABLATION_REPORT = ROOT / "reports" / "paper_mode_mab2_research_bundle_v1" / "mab_conflict_report.json"
OUTPUT_DIR = ROOT / "reports" / "research_package_v1"
RESULTS_DOC = ROOT / "docs" / "RESEARCH_RESULTS.md"
PROTOCOL_DOC = ROOT / "docs" / "RESEARCH_PROTOCOL.md"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_row(name: str, metrics: Dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics.get('qa_exact_match', 0.0):.5f} | "
        f"{metrics.get('qa_subem', 0.0):.5f} | "
        f"{metrics.get('fc_sh_accuracy', 0.0):.5f} | "
        f"{metrics.get('fc_mh_accuracy', 0.0):.5f} | "
        f"{metrics.get('scenario_accuracy', 0.0):.3f} | "
        f"{metrics.get('action_accuracy', 0.0):.3f} |"
    )


def _extract_error_summary(error_analysis: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for mode, payload in (error_analysis or {}).items():
        out[mode] = dict(payload.get("summary_counts", {}))
    return out


def build_markdown(full_report: Dict[str, Any], ablation_report: Dict[str, Any]) -> str:
    full_conflict = full_report["results"]["conflict_aware"]
    full_lww = full_report["results"]["lww"]
    ablation_results = ablation_report["results"]
    error_summary = _extract_error_summary(ablation_report.get("error_analysis", {}))

    lines = [
        "# Research Results",
        "",
        "## Headline Full Run",
        f"- Source: `reports/paper_mode_mab8_fc_refresh/mab_conflict_report.json`",
        f"- Track: `{full_report['execution']['track_name']}`",
        f"- Variant: `{full_report['execution']['conflict_aware_variant']}`",
        f"- Fallback contamination detected: `{full_conflict['fallback_contamination_detected']}`",
        "",
        "| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _mode_row("conflict_aware", full_conflict),
        _mode_row("lww", full_lww),
        _mode_row("naive", full_report["results"]["naive"]),
        "",
        "## 2-Scenario Research Bundle",
        "- Source: `reports/paper_mode_mab2_research_bundle_v1/mab_conflict_report.json`",
        "- Purpose: fast ablation and error-analysis bundle for protocol development",
        "",
        "| Mode | QA-EM | QA-SubEM | FC-SH | FC-MH | Scenario Acc | Action Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _mode_row("conflict_aware_full", ablation_results["conflict_aware_full"]),
        _mode_row("conflict_aware_no_lineage_edges", ablation_results["conflict_aware_no_lineage_edges"]),
        _mode_row("conflict_aware_no_query_support", ablation_results["conflict_aware_no_query_support"]),
        _mode_row("lww", ablation_results["lww"]),
        _mode_row("naive", ablation_results["naive"]),
        "",
        "## Error Analysis Summary",
        "- `conflict_aware_full`: "
        + ", ".join(f"{k}={v}" for k, v in error_summary.get("conflict_aware_full", {}).items()),
        "- `conflict_aware_no_lineage_edges`: "
        + ", ".join(f"{k}={v}" for k, v in error_summary.get("conflict_aware_no_lineage_edges", {}).items()),
        "- `conflict_aware_no_query_support`: "
        + ", ".join(f"{k}={v}" for k, v in error_summary.get("conflict_aware_no_query_support", {}).items()),
        "- `lww`: "
        + ", ".join(f"{k}={v}" for k, v in error_summary.get("lww", {}).items()),
        "",
        "## Interpretation",
        "- Headline full-run result stays stable: `conflict_aware > lww` on `QA-EM`, `QA-SubEM`, `FC-SH`, and `FC-MH`.",
        "- In the 2-scenario ablation bundle, `no_lineage_edges` is stronger than `full` on local QA metrics.",
        "- The current query-aware signals do not outperform `no_query_support` on the 2-scenario bundle, so the query-preservation story still needs a stronger full-scale ablation before paper claims.",
        "- Error analysis is dominated by symbolic QA issues such as `wrong_anchor_resolution`, not by arbitration collapse.",
    ]
    return "\n".join(lines) + "\n"


def build_protocol_doc() -> str:
    return "\n".join(
        [
            "# Research Protocol",
            "",
            "## Locked Tracks",
            "- Primary track: `oracle_structured`",
            "- Secondary track: `end_to_end_extract`",
            "- Only `ConflictAwareWriter` may decide `overwrite`, `reject`, `merge`, or `commit`.",
            "",
            "## Headline Benchmark",
            "- `MemoryAgentBench / Conflict_Resolution`",
            "- Headline metrics: `FC-SH`, `FC-MH`, `SubEM`, then `QA-EM` as supporting QA summary.",
            "",
            "## Supporting Metrics",
            "- `scenario_accuracy`",
            "- `action_accuracy`",
            "- `final_memory_f1`",
            "- `fallback_contamination_detected`",
            "",
            "## Main Commands",
            "- `python app/main.py --benchmark mab_conflict --max-scenarios 8 --use-dummy --conflict-aware-variant no_lineage_edges --output-dir reports\\paper_mode_mab8_fc_refresh`",
            "- `python app/main.py --benchmark mab_conflict --max-scenarios 2 --use-dummy --include-conflict-aware-ablations --enable-error-analysis --output-dir reports\\paper_mode_mab2_research_bundle_v1`",
            "",
            "## Paper Alignment Boundary",
            "- This repo is paper-aligned, not paper-faithful 1:1.",
            "- `oracle_structured` is the main research contribution track and is intentionally more structured than the original raw end-to-end paper setup.",
            "- `end_to_end_extract` must always be reported separately and may not absorb structured fallback silently.",
        ]
    ) + "\n"


def main() -> None:
    full_report = _load(FULL_REPORT)
    ablation_report = _load(ABLATION_REPORT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "headline_full_report": str(FULL_REPORT.relative_to(ROOT)),
        "ablation_report": str(ABLATION_REPORT.relative_to(ROOT)),
        "full_results": {
            "conflict_aware": full_report["results"]["conflict_aware"],
            "lww": full_report["results"]["lww"],
            "naive": full_report["results"]["naive"],
        },
        "ablation_results": {
            key: ablation_report["results"][key]
            for key in [
                "conflict_aware_full",
                "conflict_aware_no_lineage_edges",
                "conflict_aware_no_query_support",
                "lww",
                "naive",
            ]
        },
        "error_summary": _extract_error_summary(ablation_report.get("error_analysis", {})),
    }

    (OUTPUT_DIR / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    RESULTS_DOC.write_text(build_markdown(full_report, ablation_report), encoding="utf-8")
    PROTOCOL_DOC.write_text(build_protocol_doc(), encoding="utf-8")


if __name__ == "__main__":
    main()
