"""Stage 12: multi-round stability runs for Traditional RAG vs WikiAgent Full.

Each round is a complete Stage 10 paired run with its own output, checkpoint,
and run_id.  The batch summary adds cross-round statistics and per-case
consistency without changing the Stage 10 runner.

Offline by default: without ``--live`` no model request is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK_DIR))

import run_multihop_evaluation as m4  # noqa: E402
import run_stage10_comparison as m10  # noqa: E402

EXPERIMENT_ID = "stage12_multiround_stability"
DEFAULT_ROUNDS = 3
GROUPS = ("traditional_rag", "wiki_agent")
Z_95 = 1.959963984540054
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")
_API_FAILURE_STATUSES = {
    "model_error",
    "protocol_error",
    "retrieval_error",
    "runtime_error",
    "budget_exhausted",
}
# Two-sided 95% t critical values for df=1..30.  Above 30, z=1.96 is used.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
STAGE12_SOURCE_SHA256_16 = m10._file_sha256(Path(__file__))[:16]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage12_config_hash(stage10_config_hash: str, rounds: int) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage10_config_hash": stage10_config_hash,
        "stage12_source_sha256_16": STAGE12_SOURCE_SHA256_16,
        "rounds": rounds,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical)


def _expected_source_hashes() -> dict[str, str]:
    hashes = m4._source_file_hashes()
    hashes["benchmark/run_stage10_comparison.py"] = m10._file_sha256(
        Path(m10.__file__)
    )[:16]
    return hashes


def _expected_metadata(
    config: m10.Stage10Config, dataset_sha256: str, rounds: int
) -> dict[str, Any]:
    stage10_hash = m10.config_hash(config, dataset_sha256)
    return {
        "dataset_sha256": dataset_sha256,
        "stage10_config_hash": stage10_hash,
        "stage12_config_hash": stage12_config_hash(stage10_hash, rounds),
        "git_sha": m4._git_sha(),
        "source_file_sha256_16": _expected_source_hashes(),
        "prompt_hashes": {
            "rag_prompt_version": m10.RAG_PROMPT_VERSION,
            "rag_prompt_sha256_16": m10._sha16(m10.RAG_SYSTEM_PROMPT),
            "agent_prompt_version": m10.AGENT_PROMPT_VERSION,
            "agent_prompt_sha256_16": m10._sha16(m10.wiki_agent._SYSTEM_PROMPT),
        },
        "scoring_hashes": {
            "score_answer_source_sha256_16": m10._sha16(
                inspect.getsource(m4.score_answer)
            ),
            "aggregate_results_source_sha256_16": m10._sha16(
                inspect.getsource(m4.aggregate_results)
            ),
        },
    }


def _batch_id(value: str | None) -> str:
    batch_id = value or f"stage12_multiround_{time.strftime('%Y%m%d_%H%M%S')}"
    if not _BATCH_ID_RE.fullmatch(batch_id):
        raise ValueError(
            "batch_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,120}"
        )
    return batch_id


def _round_paths(results_dir: Path, batch_id: str, round_index: int):
    run_id = f"{batch_id}_r{round_index}"
    return (
        run_id,
        results_dir / f"{run_id}.json",
        results_dir / f"{run_id}.checkpoint.jsonl",
    )


def _owner_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.name + ".owner.json")


def _owner_payload(
    *,
    batch_id: str,
    run_id: str,
    round_index: int,
    rounds: int,
    dataset_sha256: str,
    expected: dict[str, Any],
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "batch_id": batch_id,
        "run_id": run_id,
        "round_index": round_index,
        "rounds_total": rounds,
        "dataset_sha256": dataset_sha256,
        "stage10_config_hash": expected["stage10_config_hash"],
        "stage12_config_hash": expected["stage12_config_hash"],
    }


def _ensure_round_owner(
    checkpoint_path: Path, owner: dict, *, resume: bool
) -> None:
    """Bind a checkpoint file to exactly one round/batch.

    Stage 10 checkpoints contain config and dataset hashes but no run_id.
    This sidecar is the round identity; copying another round's checkpoint
    into this path is rejected instead of silently mixing rounds.
    """
    path = _owner_path(checkpoint_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != owner:
            raise RuntimeError(
                f"checkpoint owner mismatch: {path} belongs to "
                f"{existing.get('run_id')!r}, not {owner['run_id']!r}"
            )
        return
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"Checkpoint exists; use --resume or choose a new output: "
                f"{checkpoint_path}"
            )
        raise RuntimeError(
            f"checkpoint exists without owner; refusing cross-round resume: "
            f"{checkpoint_path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(owner, ensure_ascii=False))
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != owner:
            raise RuntimeError(
                f"checkpoint owner mismatch: {path} belongs to "
                f"{existing.get('run_id')!r}, not {owner['run_id']!r}"
            )


def _annotate_round_files(
    output_path: Path,
    *,
    run_id: str,
    round_index: int,
    rounds: int,
    batch_id: str,
    stage12_hash: str,
) -> None:
    payload = {
        "stage12_experiment_id": EXPERIMENT_ID,
        "batch_id": batch_id,
        "run_id": run_id,
        "round_index": round_index,
        "rounds_total": rounds,
        "stage12_config_hash": stage12_hash,
        "stage12_source_sha256_16": STAGE12_SOURCE_SHA256_16,
    }
    for path in (
        output_path,
        output_path.with_name(output_path.stem + "_metrics.json"),
        output_path.with_name(output_path.stem + "_run_info.json"),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(payload)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _verify_round_payload(
    payload: dict,
    expected: dict[str, Any],
    run_id: str,
    *,
    batch_id: str | None = None,
    round_index: int | None = None,
    rounds: int | None = None,
) -> None:
    checks = {
        "run_id": run_id,
        "dataset_sha256": expected["dataset_sha256"],
        "config_hash": expected["stage10_config_hash"],
        "stage12_config_hash": expected["stage12_config_hash"],
        "git_sha": expected["git_sha"],
        "source_file_sha256_16": expected["source_file_sha256_16"],
    }
    if batch_id is not None:
        checks["batch_id"] = batch_id
    if round_index is not None:
        checks["round_index"] = round_index
    if rounds is not None:
        checks["rounds_total"] = rounds
    for key, value in checks.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"round {run_id} {key} mismatch: {payload.get(key)!r} != {value!r}"
            )
    answer_generation = payload.get("answer_generation", {})
    for key, value in expected["prompt_hashes"].items():
        if answer_generation.get(key) != value:
            raise RuntimeError(
                f"round {run_id} prompt {key} mismatch: "
                f"{answer_generation.get(key)!r} != {value!r}"
            )
    scoring = payload.get("scoring", {})
    for key, value in expected["scoring_hashes"].items():
        if scoring.get(key) != value:
            raise RuntimeError(
                f"round {run_id} scoring {key} mismatch: "
                f"{scoring.get(key)!r} != {value!r}"
            )


def run_round(
    dataset: dict,
    cases: list[dict],
    index,
    config: m10.Stage10Config,
    *,
    round_index: int,
    rounds: int,
    batch_id: str,
    dataset_sha256: str,
    results_dir: Path,
    resume: bool = False,
    rag_transport=None,
    agent_transport=None,
    logger=None,
) -> dict:
    """Run or resume one round and return its annotated output payload."""
    run_id, output_path, checkpoint_path = _round_paths(
        results_dir, batch_id, round_index
    )
    expected = _expected_metadata(config, dataset_sha256, rounds)
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing round output: {output_path}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        _verify_round_payload(
            payload, expected, run_id,
            batch_id=batch_id, round_index=round_index, rounds=rounds,
        )
        if logger:
            logger(f"[{run_id}] already complete; loaded existing output")
        return payload

    _ensure_round_owner(
        checkpoint_path,
        _owner_payload(
            batch_id=batch_id,
            run_id=run_id,
            round_index=round_index,
            rounds=rounds,
            dataset_sha256=dataset_sha256,
            expected=expected,
        ),
        resume=resume,
    )
    pairs = m10.run_stage10(
        cases,
        index,
        config,
        dataset_sha256=dataset_sha256,
        rag_transport=rag_transport,
        agent_transport=agent_transport,
        checkpoint_path=checkpoint_path,
        resume=resume,
        logger=logger,
    )
    m10.serialize_stage10(
        dataset,
        config,
        pairs,
        output_path,
        dataset_sha256=dataset_sha256,
        checkpoint_path=checkpoint_path,
    )
    _annotate_round_files(
        output_path,
        run_id=run_id,
        round_index=round_index,
        rounds=rounds,
        batch_id=batch_id,
        stage12_hash=expected["stage12_config_hash"],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    _verify_round_payload(
        payload, expected, run_id,
        batch_id=batch_id, round_index=round_index, rounds=rounds,
    )
    return payload


def wilson_interval(successes: int, n: int, z: float = Z_95) -> dict:
    if n <= 0:
        return {"low": None, "high": None, "method": "wilson"}
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    ) / denominator
    return {
        "low": max(0.0, centre - half),
        "high": min(1.0, centre + half),
        "method": "wilson",
    }


def _summary_stats(values: list[float | None]) -> dict:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "n": 0, "mean": None, "std": None, "min": None,
            "max": None, "ci95": None,
        }
    mean = sum(present) / len(present)
    std = statistics.stdev(present) if len(present) > 1 else None
    ci95 = None
    if std is not None:
        t = _T_975.get(len(present) - 1, Z_95)
        half = t * std / math.sqrt(len(present))
        ci95 = {"low": mean - half, "high": mean + half, "method": "t"}
    return {
        "n": len(present),
        "mean": mean,
        "std": std,
        "min": min(present),
        "max": max(present),
        "ci95": ci95,
    }


def _rate(successes: int, denominator: int) -> dict:
    return {
        "numerator": successes,
        "denominator": denominator,
        "value": successes / denominator if denominator else None,
        "ci95": wilson_interval(successes, denominator),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round_metrics(payload: dict, group: str) -> dict:
    result = payload["results"][group]
    aggregate = result["aggregate"]
    per_case = result["per_case"]
    answer_quality = aggregate["answer_quality_metrics"]
    answerable = [case for case in per_case if case["answerable"]]
    usage_calls = [
        call
        for case in per_case
        for call in (case.get("model_calls") or [])
    ]
    missing_usage_calls = [
        call
        for call in usage_calls
        if (call.get("usage") or {}).get("total_tokens", 0) == 0
    ]
    missing_usage_cases = sorted({
        case["id"]
        for case in per_case
        if any(
            (call.get("usage") or {}).get("total_tokens", 0) == 0
            for call in (case.get("model_calls") or [])
        )
    })
    status_counts = aggregate["status_counts"]
    non_answered = len(per_case) - status_counts.get("answered", 0)
    api_failures = sum(
        count for status, count in status_counts.items()
        if status in _API_FAILURE_STATUSES
    )
    timeout_count = sum(
        1 for case in per_case
        if "timed out" in (case.get("error") or "").lower()
        or "timeout" in (case.get("error") or "").lower()
    )
    usage = aggregate["model_usage"]
    return {
        "cases": len(per_case),
        "correct": aggregate["correct_count"],
        "answer_accuracy": _rate(
            aggregate["correct_count"], len(per_case)
        ),
        "verified_answer_accuracy": _rate(
            answer_quality["verified_answer_accuracy"]["numerator"],
            answer_quality["verified_answer_accuracy"]["denominator"],
        ),
        "answerable_accuracy": _rate(
            answer_quality["answerable_accuracy"]["numerator"],
            answer_quality["answerable_accuracy"]["denominator"],
        ),
        "unanswerable_handling": _rate(
            answer_quality["unanswerable_handling"]["numerator"],
            answer_quality["unanswerable_handling"]["denominator"],
        ),
        "evidence_recall_mean": aggregate["mean_evidence_recall"],
        "read_evidence_recall_mean": aggregate["mean_read_evidence_recall"],
        "multi_hop_success_rate": aggregate["multi_hop_success_rate"],
        "verified_multi_hop_success_rate": aggregate[
            "verified_multi_hop_success_rate"
        ],
        "source_precision_all": _mean([
            case["source_precision"] for case in per_case
        ]),
        "source_precision_answerable": _mean([
            case["source_precision"] for case in answerable
        ]),
        "model_requests": usage["total_model_requests"],
        "total_tokens": usage["total_tokens"],
        "usage_missing_calls": len(missing_usage_calls),
        "usage_missing_case_ids": missing_usage_cases,
        "mean_latency_seconds": usage["mean_latency_seconds"],
        "total_latency_seconds": usage["total_latency_seconds"],
        "failure_rate": aggregate["failure_rate"],
        "non_answered_count": non_answered,
        "api_failure_count": api_failures,
        "protocol_error_count": status_counts.get("protocol_error", 0),
        "model_error_count": status_counts.get("model_error", 0),
        "timeout_count": timeout_count,
        "status_counts": status_counts,
    }


_ACROSS_ROUND_METRICS = (
    "answer_accuracy",
    "verified_answer_accuracy",
    "answerable_accuracy",
    "unanswerable_handling",
    "evidence_recall_mean",
    "read_evidence_recall_mean",
    "multi_hop_success_rate",
    "verified_multi_hop_success_rate",
    "source_precision_all",
    "source_precision_answerable",
    "model_requests",
    "total_tokens",
    "usage_missing_calls",
    "mean_latency_seconds",
    "failure_rate",
    "non_answered_count",
    "api_failure_count",
    "protocol_error_count",
    "model_error_count",
    "timeout_count",
)


def _metric_value(metrics: dict, name: str):
    value = metrics[name]
    return value["value"] if isinstance(value, dict) else value


def _case_map(payload: dict, group: str) -> dict[str, dict]:
    return {
        case["id"]: case for case in payload["results"][group]["per_case"]
    }


def _per_case_consistency(round_payloads: list[dict], group: str) -> dict:
    first = round_payloads[0]["results"][group]["per_case"]
    ids = [case["id"] for case in first]
    maps = [_case_map(payload, group) for payload in round_payloads]
    rows = []
    for case_id in ids:
        correct = [bool(m[case_id]["answer_correct"]) for m in maps]
        statuses = [m[case_id]["status"] for m in maps]
        non_answered = [status != "answered" for status in statuses]
        api_failure = [
            status in _API_FAILURE_STATUSES for status in statuses
        ]
        rows.append({
            "case_id": case_id,
            "correct_by_round": correct,
            "correct_count": sum(correct),
            "correct_consistent": len(set(correct)) == 1,
            "status_by_round": statuses,
            "status_consistent": len(set(statuses)) == 1,
            "non_answered_by_round": non_answered,
            "api_failure_by_round": api_failure,
        })
    always_correct = [r["case_id"] for r in rows if all(r["correct_by_round"])]
    always_wrong = [
        r["case_id"] for r in rows if not any(r["correct_by_round"])
    ]
    unstable = [
        r["case_id"] for r in rows if not r["correct_consistent"]
    ]
    non_answered_union = sorted({
        r["case_id"] for r in rows if any(r["non_answered_by_round"])
    })
    non_answered_intersection = sorted({
        r["case_id"] for r in rows if all(r["non_answered_by_round"])
    })
    api_union = sorted({
        r["case_id"] for r in rows if any(r["api_failure_by_round"])
    })
    api_intersection = sorted({
        r["case_id"] for r in rows if all(r["api_failure_by_round"])
    })
    return {
        "cases": len(rows),
        "always_correct": len(always_correct),
        "always_wrong": len(always_wrong),
        "unstable": len(unstable),
        "consistent_rate": (
            (len(always_correct) + len(always_wrong)) / len(rows)
            if rows else None
        ),
        "unstable_case_ids": unstable,
        "always_correct_case_ids": always_correct,
        "always_wrong_case_ids": always_wrong,
        "non_answered_union_case_ids": non_answered_union,
        "non_answered_intersection_case_ids": non_answered_intersection,
        "api_failure_union_case_ids": api_union,
        "api_failure_intersection_case_ids": api_intersection,
        "per_case": rows,
    }


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _paired_summary(round_payloads: list[dict]) -> dict:
    rag_maps = [_case_map(payload, "traditional_rag") for payload in round_payloads]
    agent_maps = [_case_map(payload, "wiki_agent") for payload in round_payloads]
    case_ids = list(rag_maps[0])
    rows = []
    for case_id in case_ids:
        diffs = [
            int(agent_maps[i][case_id]["answer_correct"])
            - int(rag_maps[i][case_id]["answer_correct"])
            for i in range(len(round_payloads))
        ]
        rows.append({
            "case_id": case_id,
            "diff_by_round": diffs,
            "mean_diff": sum(diffs) / len(diffs),
            "agent_wins": sum(diff > 0 for diff in diffs),
            "rag_wins": sum(diff < 0 for diff in diffs),
            "ties": sum(diff == 0 for diff in diffs),
            "flipped": len({_sign(diff) for diff in diffs}) > 1,
        })
    mean_diffs = [row["mean_diff"] for row in rows]
    mean_diff = _mean(mean_diffs)
    sd = statistics.stdev(mean_diffs) if len(mean_diffs) > 1 else None
    ci95 = None
    if sd is not None:
        half = Z_95 * sd / math.sqrt(len(mean_diffs))
        ci95 = {
            "low": mean_diff - half,
            "high": mean_diff + half,
            "method": "normal approximation over per-case mean differences",
        }
    return {
        "per_round": [
            payload["results"]["paired_summary"]
            for payload in round_payloads
        ],
        "cases": len(rows),
        "agent_only_correct": sum(row["agent_wins"] for row in rows),
        "rag_only_correct": sum(row["rag_wins"] for row in rows),
        "ties": sum(row["ties"] for row in rows),
        "mean_agent_minus_rag": mean_diff,
        "mean_agent_minus_rag_ci95": ci95,
        "flipped_case_ids": [
            row["case_id"] for row in rows if row["flipped"]
        ],
        "consistent_agent_win_case_ids": [
            row["case_id"] for row in rows
            if row["agent_wins"] == len(round_payloads)
        ],
        "consistent_rag_win_case_ids": [
            row["case_id"] for row in rows
            if row["rag_wins"] == len(round_payloads)
        ],
        "consistent_tie_case_ids": [
            row["case_id"] for row in rows
            if row["ties"] == len(round_payloads)
        ],
        "per_case": rows,
    }


def build_summary(
    round_payloads: list[dict],
    *,
    batch_id: str,
    rounds: int,
    config: m10.Stage10Config,
    dataset: dict,
    dataset_sha256: str,
) -> dict:
    expected = _expected_metadata(config, dataset_sha256, rounds)
    group_payloads = {}
    for group in GROUPS:
        per_round = [
            _round_metrics(payload, group) for payload in round_payloads
        ]
        group_payloads[group] = {
            "per_round": per_round,
            "across_rounds": {
                name: _summary_stats([
                    _metric_value(metrics, name) for metrics in per_round
                ])
                for name in _ACROSS_ROUND_METRICS
            },
            "per_case_consistency": _per_case_consistency(
                round_payloads, group
            ),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "batch_id": batch_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rounds": rounds,
        "run_ids": [
            payload.get("run_id") for payload in round_payloads
        ],
        "dataset": {
            "path": "benchmark/datasets/multihop_wiki.json",
            "version": dataset.get("version", "unknown"),
            "sha256": dataset_sha256,
            "sha256_16": dataset_sha256[:16],
        },
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "rag_answer_corrections": config.rag_answer_corrections,
        "limit": config.limit,
        "stage10_config_hash": expected["stage10_config_hash"],
        "stage12_config_hash": expected["stage12_config_hash"],
        "stage12_source_sha256_16": STAGE12_SOURCE_SHA256_16,
        "git_sha": expected["git_sha"],
        "source_file_sha256_16": expected["source_file_sha256_16"],
        "prompts": expected["prompt_hashes"],
        "scoring": {
            "version": m10.SCORER_VERSION,
            **expected["scoring_hashes"],
            "manual_review_case_ids": [
                case["id"] for case in dataset["cases"]
                if case.get("manual_review")
            ],
            "manual_review_note": (
                "manual_review_case_ids are flagged for audit, but no human "
                "verdicts are stored; summary metrics use automatic scores only"
            ),
        },
        "groups": group_payloads,
        "paired": _paired_summary(round_payloads),
        "statistics": {
            "proportion_ci95": (
                "Wilson score interval per round; see groups.*.per_round"
            ),
            "across_rounds_ci95": (
                "mean +/- t(0.975, rounds-1) * sample_sd / sqrt(rounds)"
            ),
            "paired_difference_ci95": (
                "mean +/- 1.96 * sample_sd / sqrt(cases) over per-case "
                "mean differences (normal approximation)"
            ),
            "usage_note": (
                "A model call with total_tokens=0 is counted as missing usage; "
                "total_tokens is a known lower bound, never zero-imputed."
            ),
            "denominator_note": (
                "Every round uses the full case list; terminal failures are "
                "retained in accuracy denominators."
            ),
        },
    }


def run_batch(
    dataset: dict,
    cases: list[dict],
    index,
    config: m10.Stage10Config,
    *,
    rounds: int,
    batch_id: str,
    dataset_sha256: str,
    results_dir: Path,
    resume: bool = False,
    transports: list[tuple[Any, Any]] | None = None,
    logger=None,
) -> dict:
    summary_path = results_dir / f"{batch_id}_summary.json"
    expected = _expected_metadata(config, dataset_sha256, rounds)
    if summary_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing batch summary: {summary_path}"
            )
        for round_index in range(1, rounds + 1):
            _, output_path, _ = _round_paths(
                results_dir, batch_id, round_index
            )
            if not output_path.exists():
                raise RuntimeError(
                    f"summary exists but round output is missing: {output_path}"
                )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("batch_id") != batch_id:
            raise RuntimeError("existing summary batch_id mismatch")
        if summary.get("rounds") != rounds:
            raise RuntimeError("existing summary round count mismatch")
        if (
            summary.get("stage12_config_hash")
            != expected["stage12_config_hash"]
        ):
            raise RuntimeError("existing summary stage12 config hash mismatch")
        if logger:
            logger(f"Batch already complete: {summary_path}")
        return summary

    if transports is not None and len(transports) != rounds:
        raise ValueError(
            f"transports must contain {rounds} (rag, agent) pairs"
        )
    round_payloads = []
    for round_index in range(1, rounds + 1):
        rag_transport, agent_transport = (
            transports[round_index - 1]
            if transports is not None else (None, None)
        )
        if logger:
            logger(f"=== round {round_index}/{rounds} ===")
        payload = run_round(
            dataset,
            cases,
            index,
            config,
            round_index=round_index,
            rounds=rounds,
            batch_id=batch_id,
            dataset_sha256=dataset_sha256,
            results_dir=results_dir,
            resume=resume,
            rag_transport=rag_transport,
            agent_transport=agent_transport,
            logger=logger,
        )
        _verify_round_payload(
            payload,
            expected,
            payload.get("run_id", ""),
            batch_id=batch_id,
            round_index=round_index,
            rounds=rounds,
        )
        round_payloads.append(payload)

    summary = build_summary(
        round_payloads,
        batch_id=batch_id,
        rounds=rounds,
        config=config,
        dataset=dataset,
        dataset_sha256=dataset_sha256,
    )
    summary["outputs"] = [
        {
            "round": index,
            "run_id": payload["run_id"],
            "output": str(
                _round_paths(results_dir, batch_id, index)[1]
            ),
            "checkpoint": str(
                _round_paths(results_dir, batch_id, index)[2]
            ),
            "checkpoint_owner": str(
                _owner_path(_round_paths(results_dir, batch_id, index)[2])
            ),
        }
        for index, payload in enumerate(round_payloads, 1)
    ]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _load_index():
    dataset, pages, index, dataset_sha256 = m10._load_index()
    if len(pages) != 9:
        raise RuntimeError(f"expected 9 Wiki pages, got {len(pages)}")
    return dataset, pages, index, dataset_sha256


def _api_key(argument: str | None) -> str | None:
    return m10._api_key(argument)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 12: repeated Stage 10 runs for stability and per-case "
            "consistency"
        )
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Actually call the model endpoint. Omit for offline validation.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=m10.DEFAULT_BASE_URL)
    parser.add_argument("--model", default=m10.DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--top-k", type=int, default=m10.DEFAULT_TOP_K)
    parser.add_argument("--max-steps", type=int, default=m10.DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)

    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    results_dir = (
        Path(args.output_dir) if args.output_dir
        else ROOT / "benchmark" / "results"
    )
    batch_id = _batch_id(args.batch_id)
    dataset, pages, index, dataset_sha256 = _load_index()
    cases = dataset["cases"]
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(cases):
            parser.error(f"--limit must be between 1 and {len(cases)}")
        cases = cases[:args.limit]
    config = m10.Stage10Config(
        base_url=args.base_url,
        model=args.model,
        api_key=None,
        top_k=args.top_k,
        max_steps=args.max_steps,
        limit=args.limit,
    )
    expected = _expected_metadata(config, dataset_sha256, args.rounds)
    print(f"experiment_id={EXPERIMENT_ID}")
    print(f"batch_id={batch_id} rounds={args.rounds}")
    print(f"dataset_sha256={dataset_sha256}")
    print(f"pages={len(pages)} cases={len(cases)}")
    print(f"stage10_config_hash={expected['stage10_config_hash']}")
    print(f"stage12_config_hash={expected['stage12_config_hash']}")
    print(f"agent_prompt_sha256_16={expected['prompt_hashes']['agent_prompt_sha256_16']}")
    print(f"rag_prompt_sha256_16={expected['prompt_hashes']['rag_prompt_sha256_16']}")
    for round_index in range(1, args.rounds + 1):
        run_id, output_path, checkpoint_path = _round_paths(
            results_dir, batch_id, round_index
        )
        print(
            f"round {round_index} run_id={run_id} "
            f"output={output_path} checkpoint={checkpoint_path}"
        )
    if not args.live:
        print("DRY RUN: no model request sent.")
        return

    api_key = _api_key(args.api_key)
    if api_key is None:
        parser.error("No API key provided (--api-key or ZMWIKI_API_KEY).")
    if args.model != m10.DEFAULT_MODEL:
        parser.error(
            f"Stage 12 is frozen to model={m10.DEFAULT_MODEL!r}; "
            f"got {args.model!r}"
        )
    if args.base_url.rstrip("/") != m10.DEFAULT_BASE_URL:
        parser.error(
            f"Stage 12 is frozen to base_url={m10.DEFAULT_BASE_URL!r}; "
            f"got {args.base_url!r}"
        )
    if args.top_k != m10.DEFAULT_TOP_K:
        parser.error(f"Stage 12 is frozen to top_k={m10.DEFAULT_TOP_K}")
    if args.max_steps != m10.DEFAULT_MAX_STEPS:
        parser.error(
            f"Stage 12 is frozen to max_steps={m10.DEFAULT_MAX_STEPS}"
        )
    config = m10.Stage10Config(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        top_k=args.top_k,
        max_steps=args.max_steps,
        limit=args.limit,
    )
    expected = _expected_metadata(config, dataset_sha256, args.rounds)
    log_path = results_dir / f"{batch_id}.log"
    summary_path = results_dir / f"{batch_id}_summary.json"
    logger = m10.RunLogger(log_path, resume=args.resume)
    try:
        logger(f"=== {EXPERIMENT_ID} ===")
        logger(f"batch_id={batch_id} rounds={args.rounds}")
        logger(f"dataset_sha256={dataset_sha256}")
        logger(f"stage10_config_hash={expected['stage10_config_hash']}")
        logger(f"stage12_config_hash={expected['stage12_config_hash']}")
        summary = run_batch(
            dataset,
            cases,
            index,
            config,
            rounds=args.rounds,
            batch_id=batch_id,
            dataset_sha256=dataset_sha256,
            results_dir=results_dir,
            resume=args.resume,
            logger=logger,
        )
        logger(f"summary={summary_path}")
        for group in GROUPS:
            across = summary["groups"][group]["across_rounds"]
            logger(
                f"{group} answer_accuracy "
                f"mean={across['answer_accuracy']['mean']:.4f} "
                f"std={across['answer_accuracy']['std']}"
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
