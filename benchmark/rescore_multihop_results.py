"""Re-score an existing multi-hop benchmark result file without rerunning models.

Only answer correctness (keyword/refusal matching) and the aggregates derived
from it are recomputed, using the current dataset annotations. Retrieved-page
and evidence metrics stored in the input file are preserved verbatim. Metric
definitions are unchanged; see the ``rescoring`` block written to the output.

Usage:
    python benchmark/rescore_multihop_results.py \
        --input benchmark/results/phase09_full.json \
        --output benchmark/results/phase09_full_rescored.json

The output path is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import run_multihop_evaluation as m4  # noqa: E402


def _rescore_system(
    cases: list[dict], per_case: list[dict]
) -> tuple[list[m4.CaseResult], list[dict], list[dict]]:
    """Recompute score flags for one system's per-case records.

    Returns (results ordered by dataset order, matching case dicts, changes).
    """
    by_id = {c["id"]: c for c in cases}
    rescored: dict[str, m4.CaseResult] = {}
    changes: list[dict] = []
    for raw in per_case:
        case = by_id[raw["id"]]
        result = m4.CaseResult(**raw)
        before = (result.answer_correct, result.grounded_correct,
                  result.unsupported_correct)
        result.answer_correct = m4.score_answer(result.answer, case)
        result.grounded_correct = bool(
            result.answer_correct
            and result.status == m4.AgentStatus.ANSWERED
            and result.sources
        )
        result.unsupported_correct = (
            result.answer_correct and not result.grounded_correct
        )
        after = (result.answer_correct, result.grounded_correct,
                 result.unsupported_correct)
        if before != after:
            changes.append({
                "id": result.id,
                "answer_correct": {"before": before[0], "after": after[0]},
                "grounded_correct": {"before": before[1], "after": after[1]},
                "unsupported_correct": {"before": before[2], "after": after[2]},
            })
        rescored[result.id] = result

    ordered_cases = [c for c in cases if c["id"] in rescored]
    ordered_results = [rescored[c["id"]] for c in ordered_cases]
    return ordered_results, ordered_cases, changes


def rescore_payload(payload: dict, dataset: dict) -> dict:
    """Return a rescored copy of a result payload (input is not mutated)."""
    payload = json.loads(json.dumps(payload))  # deep copy
    changes: dict[str, list[dict]] = {}
    for system, section in payload.get("results", {}).items():
        per_case = section.get("per_case") or []
        if not per_case:
            continue
        results, run_cases, system_changes = _rescore_system(
            dataset["cases"], per_case
        )
        section["per_case"] = [r.to_dict() for r in results]
        section["aggregate"] = m4.aggregate_results(run_cases, results)
        changes[system] = system_changes

    current_hash = m4._dataset_hash()
    payload["rescoring"] = {
        "tool": "benchmark/rescore_multihop_results.py",
        "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_rerun": False,
        "note": "Only answer_correct / grounded_correct / unsupported_correct and "
                "aggregates derived from them were recomputed. Retrieval and "
                "evidence metrics are preserved from the source file.",
        "source_dataset_sha256_16": payload.get("dataset_sha256_16"),
        "dataset_sha256_16": current_hash,
        "scoring_rule": dataset.get("scoring_rule", "unknown"),
        "metric_definitions": {
            "answer_correct": "answerable: every answer_keyword_group matched; "
                              "unanswerable: at least one refusal_keyword matched",
            "grounded_correct": "answer_correct AND status=='answered' AND at "
                                "least one cited source",
            "unsupported_correct": "answer_correct AND NOT grounded_correct",
        },
        "changes": changes,
    }
    payload["dataset_sha256_16"] = current_hash
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score an existing multi-hop result file (no model calls)."
    )
    parser.add_argument("--input", required=True, help="Existing result JSON")
    parser.add_argument("--output", required=True,
                        help="New result JSON path (must not exist)")
    parser.add_argument("--dataset", default=str(m4.DATASET_PATH),
                        help="Dataset JSON used for re-scoring")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")
    if output_path.exists():
        parser.error(f"Output file already exists, refusing to overwrite: {output_path}")

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    rescored = rescore_payload(payload, dataset)
    rescored["rescoring"]["source_result_file"] = str(input_path)
    rescored["rescoring"]["source_result_sha256_16"] = hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()[:16]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rescored, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Rescored {input_path} -> {output_path}")
    for system, system_changes in rescored["rescoring"]["changes"].items():
        print(f"  {system}: {len(system_changes)} case(s) changed")


if __name__ == "__main__":
    main()
