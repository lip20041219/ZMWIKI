"""Compare two multi-hop result files with different link sources.

The only intended difference between the two runs is ``link_source``
("manual" vs "auto"). This tool refuses to compare runs whose dataset,
model, endpoint, step budget, search budget, code hashes, or case sets
differ, so a comparison cannot silently mix configurations.

Usage:
    python benchmark/compare_multihop_runs.py \
        --manual benchmark/results/phase10_manual_retry.json \
        --auto benchmark/results/phase10_auto_links_retry.json \
        --output benchmark/results/phase10_comparison_retry.json

Without --output the summary is printed to stdout. The output file is never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC_KEYS = (
    "verified_answer_accuracy",
    "answer_accuracy",
    "unanswerable_handling",
    "mean_evidence_recall",
    "mean_source_precision",
    "verified_multi_hop_success_rate",
    "mean_steps",
    "mean_pages_read",
    "failure_rate",
    "budget_exhausted_rate",
)

_CASE_FIELDS = (
    "answer", "status", "answer_correct", "grounded_correct",
    "unsupported_correct", "evidence_recall", "source_precision",
    "steps", "pages_read", "sources", "accessed_pages", "required_pages",
    "error", "step_details",
)

#: Configuration fields that must match exactly; link_source is checked
#: separately because it is the one intended experimental variable.
_CONFIG_PATHS = (
    ("config.experiment.agent", ("config", "experiment", "agent")),
    ("config.experiment.live", ("config", "experiment", "live")),
    ("config.experiment.base_url", ("config", "experiment", "base_url")),
    ("config.experiment.model", ("config", "experiment", "model")),
    ("config.experiment.max_steps", ("config", "experiment", "max_steps")),
    ("config.experiment.search_top_k", ("config", "experiment", "search_top_k")),
    ("config.limit", ("config", "limit")),
    ("dataset", ("dataset",)),
    ("dataset_version", ("dataset_version",)),
    ("scoring_rule", ("scoring_rule",)),
    ("experiment_id", ("experiment_id",)),
)

_HASH_PATHS = (
    ("dataset_sha256_16", ("dataset_sha256_16",)),
    ("source_file_sha256_16", ("source_file_sha256_16",)),
)


def _get(payload: dict, path: tuple[str, ...]):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _experiment(payload: dict) -> dict:
    section = _get(payload, ("results", "experiment"))
    if not isinstance(section, dict):
        raise ValueError("missing results.experiment")
    return section


def comparability_issues(manual: dict, auto: dict) -> list[str]:
    """Return human-readable reasons the runs cannot be compared."""
    issues: list[str] = []

    for label, path in _CONFIG_PATHS:
        left, right = _get(manual, path), _get(auto, path)
        if left != right:
            issues.append(f"{label}: manual={left!r} auto={right!r}")

    for label, path in _HASH_PATHS:
        left, right = _get(manual, path), _get(auto, path)
        if left is None or right is None:
            issues.append(f"{label}: missing in one or both files")
        elif left != right:
            issues.append(f"{label}: manual={left!r} auto={right!r}")

    baseline_left = _get(manual, ("config", "baseline"))
    baseline_right = _get(auto, ("config", "baseline"))
    if baseline_left != baseline_right:
        issues.append("config.baseline differs between runs")

    manual_link = manual.get("link_source")
    auto_link = auto.get("link_source")
    if manual_link != "manual":
        issues.append(f"--manual file has link_source={manual_link!r}, expected 'manual'")
    if auto_link != "auto":
        issues.append(f"--auto file has link_source={auto_link!r}, expected 'auto'")
    if _get(manual, ("config", "experiment", "link_source")) != manual_link:
        issues.append("manual file: top-level link_source disagrees with config")
    if _get(auto, ("config", "experiment", "link_source")) != auto_link:
        issues.append("auto file: top-level link_source disagrees with config")

    for name, payload in (("manual", manual), ("auto", auto)):
        aggregate = _experiment(payload).get("aggregate")
        if not isinstance(aggregate, dict):
            issues.append(f"{name} file has no experiment aggregate (run not executed live?)")
            continue
        missing = [key for key in METRIC_KEYS if key not in aggregate]
        if missing:
            issues.append(f"{name} aggregate missing metrics: {missing}")

    manual_cases = _get(manual, ("results", "experiment", "per_case")) or []
    auto_cases = _get(auto, ("results", "experiment", "per_case")) or []
    manual_ids = [c.get("id") for c in manual_cases]
    auto_ids = [c.get("id") for c in auto_cases]
    if not manual_ids or not auto_ids:
        issues.append("one or both files have no experiment per-case results")
    elif manual_ids != auto_ids:
        issues.append("experiment case ids/order differ between runs")

    return issues


def _case_view(case: dict) -> dict:
    return {field: case.get(field) for field in _CASE_FIELDS}


def _case_summary(per_case: list[dict], field: str) -> dict:
    improved, regressed = [], []
    for case in per_case:
        delta = case["delta"][field]
        if delta > 0:
            improved.append(case["id"])
        elif delta < 0:
            regressed.append(case["id"])
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(per_case) - len(improved) - len(regressed),
    }


def build_comparison(manual: dict, auto: dict) -> dict:
    """Build a comparison report, refusing incomparable inputs."""
    issues = comparability_issues(manual, auto)
    if issues:
        raise ValueError("runs are not comparable:\n- " + "\n- ".join(issues))

    manual_agg = _experiment(manual)["aggregate"]
    auto_agg = _experiment(auto)["aggregate"]
    metrics = {}
    for key in METRIC_KEYS:
        left, right = manual_agg[key], auto_agg[key]
        metrics[key] = {
            "manual": left,
            "auto": right,
            "delta": right - left,
        }

    manual_cases = {c["id"]: c for c in _experiment(manual)["per_case"]}
    auto_cases = {c["id"]: c for c in _experiment(auto)["per_case"]}
    per_case = []
    for case_id, manual_case in manual_cases.items():
        auto_case = auto_cases[case_id]
        per_case.append({
            "id": case_id,
            "question": manual_case.get("question"),
            "manual": _case_view(manual_case),
            "auto": _case_view(auto_case),
            "delta": {
                field: int(auto_case.get(field) or 0) - int(manual_case.get(field) or 0)
                for field in ("answer_correct", "grounded_correct",
                              "unsupported_correct")
            } | {
                "steps": (auto_case.get("steps") or 0) - (manual_case.get("steps") or 0),
                "pages_read": (auto_case.get("pages_read") or 0)
                              - (manual_case.get("pages_read") or 0),
            },
        })

    status_changes = [
        {
            "id": case["id"],
            "manual": case["manual"]["status"],
            "auto": case["auto"]["status"],
        }
        for case in per_case
        if case["manual"]["status"] != case["auto"]["status"]
    ]

    return {
        "comparability": {
            "comparable": True,
            "intended_variable": "link_source",
            "link_source": {"manual": "manual", "auto": "auto"},
        },
        "provenance": {
            "dataset": manual.get("dataset"),
            "dataset_version": manual.get("dataset_version"),
            "dataset_sha256_16": manual.get("dataset_sha256_16"),
            "model": _get(manual, ("config", "experiment", "model")),
            "base_url": _get(manual, ("config", "experiment", "base_url")),
            "max_steps": _get(manual, ("config", "experiment", "max_steps")),
            "search_top_k": _get(manual, ("config", "experiment", "search_top_k")),
            "limit": _get(manual, ("config", "limit")),
            "source_file_sha256_16": manual.get("source_file_sha256_16"),
            "manual_timestamp": manual.get("timestamp"),
            "auto_timestamp": auto.get("timestamp"),
            "auto_link_resolution": auto.get("link_resolution"),
        },
        "metrics": metrics,
        "status_counts": {
            "manual": manual_agg.get("status_counts"),
            "auto": auto_agg.get("status_counts"),
            "changed_cases": status_changes,
        },
        "summary": {
            "total_cases": len(per_case),
            "answer_correct": _case_summary(per_case, "answer_correct"),
            "grounded_correct": _case_summary(per_case, "grounded_correct"),
            "unsupported_correct": _case_summary(per_case, "unsupported_correct"),
        },
        "per_case": per_case,
    }


def _print_summary(report: dict) -> None:
    prov = report["provenance"]
    print("Comparable: yes (only link_source differs)")
    print(f"  dataset hash: {prov['dataset_sha256_16']}")
    print(f"  model: {prov['model']}  endpoint: {prov['base_url']}")
    print(f"  max_steps: {prov['max_steps']}  search_top_k: {prov['search_top_k']}")
    print(f"\n{'metric':40s} {'manual':>10s} {'auto':>10s} {'delta':>10s}")
    for key, values in report["metrics"].items():
        print(f"{key:40s} {values['manual']:>10.4f} {values['auto']:>10.4f} "
              f"{values['delta']:>+10.4f}")
    print(f"\nstatus counts manual: {report['status_counts']['manual']}")
    print(f"status counts auto:   {report['status_counts']['auto']}")
    summary = report["summary"]
    print(f"answer_correct improved={summary['answer_correct']['improved']} "
          f"regressed={summary['answer_correct']['regressed']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare manual-link vs auto-link multi-hop result files."
    )
    parser.add_argument("--manual", required=True, help="Manual-link result JSON")
    parser.add_argument("--auto", required=True, help="Auto-link result JSON")
    parser.add_argument("--output", default=None,
                        help="Write the full JSON report here (must not exist). "
                             "Omit to print the summary only.")
    args = parser.parse_args()

    manual_path, auto_path = Path(args.manual), Path(args.auto)
    for path in (manual_path, auto_path):
        if not path.exists():
            parser.error(f"Result file does not exist: {path}")
    if args.output and Path(args.output).exists():
        parser.error(f"Output file already exists, refusing to overwrite: {args.output}")

    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    auto = json.loads(auto_path.read_text(encoding="utf-8"))
    try:
        report = build_comparison(manual, auto)
    except ValueError as exc:
        print(f"Refusing to compare: {exc}", file=sys.stderr)
        raise SystemExit(1)

    report["source_files"] = {"manual": str(manual_path), "auto": str(auto_path)}
    _print_summary(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
