"""Stage 11: Follow Links ablation inside the current WikiAgent.

Full:     the current WikiAgent with search / read / follow_links.
No-Links: the same WikiAgent with search / read / answer only; follow_links
          is removed from the system prompt and rejected at the action guard.

Both groups use the same model, dataset, temperature, top-k, max-steps,
answer constraints, action parser, and frozen score_answer implementation.

This module is offline-safe by default. ``--live`` is required before any
model request is sent. No API key is written to results or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK_DIR))

import run_multihop_evaluation as m4  # noqa: E402
import run_stage10_comparison as m10  # noqa: E402
from zmwiki.wiki import wiki_agent  # noqa: E402
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402

EXPERIMENT_ID = "stage11_follow_links_ablation"
FULL_SYSTEM_PROMPT = wiki_agent._SYSTEM_PROMPT
FULL_ALLOWED_ACTIONS = frozenset({"search", "read", "follow_links", "answer"})
NO_LINKS_ALLOWED_ACTIONS = frozenset({"search", "read", "answer"})
FULL_PROMPT_VERSION = "wiki_agent.system_prompt.frozen"
NO_LINKS_PROMPT_VERSION = "wiki_agent.no_follow_links.v1"
SCORER_VERSION = "stage9_4_frozen"


def _derive_no_links_prompt(full_prompt: str) -> str:
    """Remove only the follow_links action line and renumber the answer action."""
    lines = full_prompt.splitlines(keepends=True)
    kept = [line for line in lines if '"action": "follow_links"' not in line]
    if len(kept) != len(lines) - 1:
        raise RuntimeError("could not locate exactly one follow_links prompt line")
    text = "".join(kept)
    if '4. {"action": "answer"' not in text:
        raise RuntimeError("could not locate the answer action line for renumbering")
    text = text.replace('4. {"action": "answer"', '3. {"action": "answer"', 1)
    if '"action": "follow_links"' in text:
        raise RuntimeError("follow_links still present in No-Links prompt")
    return text


NO_LINKS_SYSTEM_PROMPT = _derive_no_links_prompt(FULL_SYSTEM_PROMPT)
FULL_PROMPT_SHA256_16 = hashlib.sha256(FULL_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
NO_LINKS_PROMPT_SHA256_16 = hashlib.sha256(
    NO_LINKS_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()[:16]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class Stage11Config:
    base_url: str = m10.DEFAULT_BASE_URL
    model: str = m10.DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    temperature: float = 0.0
    top_k: int = 5
    max_steps: int = 8
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty str")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty str")
        if not isinstance(self.temperature, (int, float)) or self.temperature != 0.0:
            raise ValueError("Stage 11 requires temperature=0.0")
        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k"))
        object.__setattr__(self, "max_steps", _positive_int(self.max_steps, "max_steps"))
        if self.limit is not None:
            object.__setattr__(self, "limit", _positive_int(self.limit, "limit"))


@dataclass
class Stage11CaseResult(m10.Stage10CaseResult):
    follow_links_calls: int = 0
    follow_links_attempts: int = 0
    follow_links_pages: list[str] = field(default_factory=list)
    tool_disabled_attempts: int = 0

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload.update({
            "follow_links_calls": self.follow_links_calls,
            "follow_links_attempts": self.follow_links_attempts,
            "follow_links_pages": self.follow_links_pages,
            "tool_disabled_attempts": self.tool_disabled_attempts,
        })
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Stage11CaseResult":
        base_names = {f.name for f in fields(m10.Stage10CaseResult)}
        base = {name: payload[name] for name in base_names if name in payload}
        return cls(
            **base,
            follow_links_calls=payload.get("follow_links_calls", 0),
            follow_links_attempts=payload.get("follow_links_attempts", 0),
            follow_links_pages=payload.get("follow_links_pages", []),
            tool_disabled_attempts=payload.get("tool_disabled_attempts", 0),
        )

    @classmethod
    def from_base(
        cls,
        base: m10.Stage10CaseResult,
        *,
        follow_links_calls: int,
        follow_links_attempts: int,
        follow_links_pages: list[str],
        tool_disabled_attempts: int,
    ) -> "Stage11CaseResult":
        kwargs = {
            f.name: getattr(base, f.name) for f in fields(m10.Stage10CaseResult)
        }
        return cls(
            **kwargs,
            follow_links_calls=follow_links_calls,
            follow_links_attempts=follow_links_attempts,
            follow_links_pages=follow_links_pages,
            tool_disabled_attempts=tool_disabled_attempts,
        )


def _follow_link_metrics(step_details: list[dict]) -> tuple[int, int, list[str], int]:
    calls = 0
    attempts = 0
    pages: set[str] = set()
    disabled = 0
    for step in step_details:
        action = step.get("action")
        if action == "follow_links":
            calls += 1
            attempts += 1
            pages.update(step.get("read_page_ids", ()) or ())
        elif action == "tool_disabled":
            disabled += 1
            if (step.get("params") or {}).get("action") == "follow_links":
                attempts += 1
    return calls, attempts, sorted(pages), disabled


def _run_agent_case(
    case: dict,
    index,
    config: Stage11Config,
    *,
    transport=None,
    system_prompt: str | None,
    allowed_actions: frozenset[str] | None,
) -> Stage11CaseResult:
    """Run one agent case with explicit tool availability."""
    started = time.perf_counter()
    recorder = m10.RecordingTransport(transport)
    monitor = m4.MonitoringWikiIndex(index)
    try:
        agent = WikiAgent(
            monitor,
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            transport=recorder,
            max_steps=config.max_steps,
            search_top_k=config.top_k,
            system_prompt=system_prompt,
            allowed_actions=allowed_actions,
        )
        agent_result = agent.answer(case["question"])
    except Exception as exc:
        elapsed = time.perf_counter() - started
        base = m10._make_result(
            case, answer=None, status="runtime_error", sources=[],
            accessed=set(monitor.accessed_page_ids), read=[], retrieval=[],
            steps=0, step_details=[],
            error=f"{type(exc).__name__}: {exc}",
            link_source="manual", calls=recorder.calls,
            citation_correction_used=False, wall_seconds=elapsed,
        )
        return Stage11CaseResult.from_base(
            base, follow_links_calls=0, follow_links_attempts=0,
            follow_links_pages=[], tool_disabled_attempts=0,
        )

    source_ids = m4._source_ids_from_paths(agent_result.sources, index)
    search_hits = sorted({
        page_id
        for step in agent_result.steps
        for page_id in getattr(step, "search_hit_page_ids", ())
    })
    read_ids = sorted(set(getattr(agent_result, "read_page_ids", ()) or ()))
    step_details = [
        {
            "step": step.step,
            "action": step.action,
            "params": getattr(step, "params", {}),
            "error": step.error,
            "result_summary": step.result_summary,
            "raw_output": step.raw_output,
            "read_page_ids": list(getattr(step, "read_page_ids", ())),
            "search_hit_page_ids": list(
                getattr(step, "search_hit_page_ids", ())
            ),
        }
        for step in agent_result.steps
    ]
    correction_used = any(
        step.action == "answer" and step.error for step in agent_result.steps
    )
    elapsed = time.perf_counter() - started
    base = m10._make_result(
        case,
        answer=agent_result.answer,
        status=agent_result.status,
        sources=source_ids,
        accessed=set(monitor.accessed_page_ids),
        read=read_ids,
        retrieval=search_hits,
        steps=len(agent_result.steps),
        step_details=step_details,
        error=agent_result.error,
        link_source="manual",
        calls=recorder.calls,
        citation_correction_used=correction_used,
        wall_seconds=elapsed,
    )
    calls, attempts, pages, disabled = _follow_link_metrics(step_details)
    return Stage11CaseResult.from_base(
        base,
        follow_links_calls=calls,
        follow_links_attempts=attempts,
        follow_links_pages=pages,
        tool_disabled_attempts=disabled,
    )


def run_full_case(
    case: dict, index, config: Stage11Config, *, transport=None
) -> Stage11CaseResult:
    return _run_agent_case(
        case, index, config, transport=transport,
        system_prompt=None, allowed_actions=None,
    )


def run_no_links_case(
    case: dict, index, config: Stage11Config, *, transport=None
) -> Stage11CaseResult:
    return _run_agent_case(
        case, index, config, transport=transport,
        system_prompt=NO_LINKS_SYSTEM_PROMPT,
        allowed_actions=NO_LINKS_ALLOWED_ACTIONS,
    )


def _config_payload(config: Stage11Config, dataset_sha256: str) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_sha256": dataset_sha256,
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "limit": config.limit,
        "full_prompt_sha256_16": FULL_PROMPT_SHA256_16,
        "no_links_prompt_sha256_16": NO_LINKS_PROMPT_SHA256_16,
        "scorer_version": SCORER_VERSION,
        "score_answer_source_sha256_16": m10._sha16(
            inspect.getsource(m4.score_answer)
        ),
        "aggregate_results_source_sha256_16": m10._sha16(
            inspect.getsource(m4.aggregate_results)
        ),
    }


def config_hash(config: Stage11Config, dataset_sha256: str) -> str:
    payload = _config_payload(config, dataset_sha256)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_checkpoint(
    path: Path, dataset_sha256: str, expected_config_hash: str
) -> dict[str, tuple[Stage11CaseResult, Stage11CaseResult]]:
    completed: dict[str, tuple[Stage11CaseResult, Stage11CaseResult]] = {}
    if not path.exists():
        return completed
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid checkpoint line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid checkpoint line {line_no}: not an object")
        if record.get("dataset_sha256") != dataset_sha256:
            raise RuntimeError(
                f"checkpoint dataset hash mismatch at line {line_no}: "
                f"{record.get('dataset_sha256')!r} != {dataset_sha256!r}"
            )
        if record.get("config_hash") != expected_config_hash:
            raise RuntimeError(f"checkpoint config hash mismatch at line {line_no}")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"checkpoint line {line_no}: missing case_id")
        if case_id in completed:
            raise RuntimeError(f"duplicate checkpoint case_id: {case_id}")
        try:
            full = Stage11CaseResult.from_dict(record["full"])
            no_links = Stage11CaseResult.from_dict(record["no_links"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"invalid checkpoint result for {case_id}: {exc}"
            ) from exc
        completed[case_id] = (full, no_links)
    return completed


def _append_checkpoint(
    path: Path,
    case: dict,
    full: Stage11CaseResult,
    no_links: Stage11CaseResult,
    dataset_sha256: str,
    expected_config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case["id"],
        "dataset_sha256": dataset_sha256,
        "config_hash": expected_config_hash,
        "full": full.to_dict(),
        "no_links": no_links.to_dict(),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_stage11(
    cases: list[dict],
    index,
    config: Stage11Config,
    *,
    dataset_sha256: str,
    full_transport=None,
    no_links_transport=None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    logger=None,
) -> list[tuple[dict, Stage11CaseResult, Stage11CaseResult]]:
    expected = config_hash(config, dataset_sha256)
    completed = {}
    if checkpoint_path is not None:
        if checkpoint_path.exists() and not resume:
            raise FileExistsError(
                f"Checkpoint exists; use --resume or choose a new output: {checkpoint_path}"
            )
        if resume:
            completed = load_checkpoint(checkpoint_path, dataset_sha256, expected)
            if logger:
                logger(f"Resuming with {len(completed)} completed case(s)")
    paired = []
    for case in cases:
        if case["id"] in completed:
            full, no_links = completed[case["id"]]
            paired.append((case, full, no_links))
            continue
        if logger:
            logger(f"[{case['id']}] Full")
        full = run_full_case(case, index, config, transport=full_transport)
        if logger:
            logger(
                f"[{case['id']}] Full status={full.status} "
                f"correct={full.answer_correct} link_calls={full.follow_links_calls}"
            )
            logger(f"[{case['id']}] No-Links")
        no_links = run_no_links_case(
            case, index, config, transport=no_links_transport
        )
        if logger:
            logger(
                f"[{case['id']}] No-Links status={no_links.status} "
                f"correct={no_links.answer_correct} "
                f"disabled={no_links.tool_disabled_attempts}"
            )
        paired.append((case, full, no_links))
        if checkpoint_path is not None:
            _append_checkpoint(
                checkpoint_path, case, full, no_links, dataset_sha256, expected
            )
    return paired


def _group_results(
    paired: list[tuple[dict, Stage11CaseResult, Stage11CaseResult]],
    index: int,
) -> list[Stage11CaseResult]:
    return [pair[index] for pair in paired]


def aggregate_group(cases: list[dict], results: list[Stage11CaseResult]) -> dict:
    aggregate = m10.aggregate_group(cases, results)
    aggregate["answer_quality_metrics"]["note"] = (
        "Stage 11: Full and No-Links use the same model, dataset, answer rules, "
        "action parser, and frozen scorer; only follow_links availability differs."
    )
    aggregate["retrieval_evidence_metrics"]["note"] = (
        "Stage 11: read_evidence_recall uses pages actually read. Full may obtain "
        "additional pages through follow_links; No-Links cannot."
    )
    unique_link_pages = sorted({
        page_id for result in results for page_id in result.follow_links_pages
    })
    aggregate["follow_links_metrics"] = {
        "total_calls": sum(result.follow_links_calls for result in results),
        "total_attempts": sum(result.follow_links_attempts for result in results),
        "cases_with_calls": sum(
            1 for result in results if result.follow_links_calls > 0
        ),
        "unique_pages_obtained": len(unique_link_pages),
        "unique_page_ids": unique_link_pages,
        "tool_disabled_attempts": sum(
            result.tool_disabled_attempts for result in results
        ),
    }
    return aggregate


def paired_summary(
    paired: list[tuple[dict, Stage11CaseResult, Stage11CaseResult]]
) -> dict:
    # m10.paired_summary's "agent" is Full; its "rag" is No-Links.
    summary = m10.paired_summary([
        (case, no_links, full) for case, full, no_links in paired
    ])
    return {
        "cases": summary["cases"],
        "full_only_correct": summary["agent_only_correct"],
        "no_links_only_correct": summary["rag_only_correct"],
        "both_correct": summary["both_correct"],
        "both_wrong": summary["both_wrong"],
        "mean_full_minus_no_links": summary["mean_agent_minus_rag"],
    }


def _paired_per_case(
    paired: list[tuple[dict, Stage11CaseResult, Stage11CaseResult]]
) -> list[dict]:
    payload = []
    for case, full, no_links in paired:
        payload.append({
            "case_id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "hops": case["hops"],
            "full": full.to_dict(),
            "no_links": no_links.to_dict(),
            "diff": {
                "answer_correct": int(full.answer_correct) - int(no_links.answer_correct),
                "grounded_correct": (
                    int(full.grounded_correct) - int(no_links.grounded_correct)
                ),
                "evidence_recall": (
                    None if full.evidence_recall is None or no_links.evidence_recall is None
                    else full.evidence_recall - no_links.evidence_recall
                ),
                "read_evidence_recall": (
                    None if full.read_evidence_recall is None
                    or no_links.read_evidence_recall is None
                    else full.read_evidence_recall - no_links.read_evidence_recall
                ),
                "source_precision": full.source_precision - no_links.source_precision,
                "model_requests": full.model_requests - no_links.model_requests,
                "total_tokens": full.total_tokens - no_links.total_tokens,
                "latency_seconds": full.latency_seconds - no_links.latency_seconds,
                "steps": full.steps - no_links.steps,
                "follow_links_calls": (
                    full.follow_links_calls - no_links.follow_links_calls
                ),
                "follow_links_pages": (
                    len(full.follow_links_pages) - len(no_links.follow_links_pages)
                ),
            },
        })
    return payload


def _provenance(dataset: dict, config: Stage11Config, dataset_sha256: str) -> dict:
    source_hashes = m4._source_file_hashes()
    source_hashes["benchmark/run_stage11_ablation.py"] = m10._file_sha256(
        Path(__file__)
    )[:16]
    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "benchmark/datasets/multihop_wiki.json",
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_sha256": dataset_sha256,
        "dataset_sha256_16": dataset_sha256[:16],
        "dataset_sha256_expected": m10.EXPECTED_DATASET_SHA256,
        "config_hash": config_hash(config, dataset_sha256),
        "git_sha": m4._git_sha(),
        "python_version": platform.python_version(),
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "api_key_recorded": False,
        "working_tree_status": m4._working_tree_status(),
        "source_file_sha256_16": source_hashes,
        "prompts": {
            "full_version": FULL_PROMPT_VERSION,
            "full_sha256_16": FULL_PROMPT_SHA256_16,
            "no_links_version": NO_LINKS_PROMPT_VERSION,
            "no_links_sha256_16": NO_LINKS_PROMPT_SHA256_16,
            "shared_answer_rules": m10.SHARED_ANSWER_RULES,
            "answer_schema": (
                '{"action": "answer", "text": "...", "sources": ["page_id"]}'
            ),
            "tool_difference": "Full includes follow_links; No-Links removes its "
                               "prompt line and rejects it at the action guard.",
        },
        "scoring": {
            "version": SCORER_VERSION,
            "rule": dataset.get("scoring_rule", "unknown"),
            "score_answer_source_sha256_16": m10._sha16(
                inspect.getsource(m4.score_answer)
            ),
            "aggregate_results_source_sha256_16": m10._sha16(
                inspect.getsource(m4.aggregate_results)
            ),
            "manual_review_case_ids": [
                case["id"] for case in dataset["cases"]
                if case.get("manual_review")
            ],
        },
        "config": {
            "full": {
                "agent": "WikiAgent",
                "tools": sorted(FULL_ALLOWED_ACTIONS),
                "system_prompt_version": FULL_PROMPT_VERSION,
                "system_prompt_sha256_16": FULL_PROMPT_SHA256_16,
            },
            "no_links": {
                "agent": "WikiAgent with allowed_actions=search/read/answer",
                "tools": sorted(NO_LINKS_ALLOWED_ACTIONS),
                "system_prompt_version": NO_LINKS_PROMPT_VERSION,
                "system_prompt_sha256_16": NO_LINKS_PROMPT_SHA256_16,
                "follow_links": "disabled and rejected before execution",
            },
            "shared": {
                "model": config.model,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "top_k": config.top_k,
                "max_steps": config.max_steps,
                "dataset": "benchmark/datasets/multihop_wiki.json",
                "scoring": "run_multihop_evaluation.score_answer",
            },
        },
        "comparability_notes": {
            "fairness": "Full and No-Links use the same model, dataset, temperature, "
                        "top-k, max-steps, answer rules, citation rules, action parser, "
                        "and frozen score_answer implementation.",
            "difference": "Only follow_links availability and its necessary prompt/action "
                         "guard differ.",
            "failure_handling": "Protocol, model, insufficient-evidence, disabled-tool, "
                               "and budget outcomes are retained; no successful-only filtering.",
            "steps_definition": "steps are agent action iterations. follow_links_calls "
                               "counts executed follow_links actions; "
                               "follow_links_attempts also counts disabled attempts.",
        },
    }


def _ensure_new(path: Path) -> None:
    m10._ensure_new(path)


def serialize_stage11(
    dataset: dict,
    config: Stage11Config,
    paired: list[tuple[dict, Stage11CaseResult, Stage11CaseResult]],
    output_path: Path,
    *,
    dataset_sha256: str,
    checkpoint_path: Path | None = None,
) -> dict:
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    run_info_path = output_path.with_name(output_path.stem + "_run_info.json")
    for path in (output_path, metrics_path, run_info_path):
        _ensure_new(path)

    cases = [pair[0] for pair in paired]
    full_results = _group_results(paired, 1)
    no_links_results = _group_results(paired, 2)
    full_aggregate = aggregate_group(cases, full_results)
    no_links_aggregate = aggregate_group(cases, no_links_results)
    summary = paired_summary(paired)
    provenance = _provenance(dataset, config, dataset_sha256)

    payload = {
        **provenance,
        "results": {
            "full": {
                "per_case": [result.to_dict() for result in full_results],
                "aggregate": full_aggregate,
            },
            "no_links": {
                "per_case": [result.to_dict() for result in no_links_results],
                "aggregate": no_links_aggregate,
            },
            "paired_per_case": _paired_per_case(paired),
            "paired_summary": summary,
        },
    }
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_sha256": dataset_sha256,
        "dataset_sha256_16": dataset_sha256[:16],
        "dataset_sha256_expected": m10.EXPECTED_DATASET_SHA256,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "full": full_aggregate,
        "no_links": no_links_aggregate,
        "paired_summary": summary,
        "scoring": provenance["scoring"],
    }
    run_info = {
        key: provenance[key] for key in (
            "experiment_id", "timestamp", "dataset", "dataset_version",
            "dataset_sha256", "dataset_sha256_16", "dataset_sha256_expected",
            "config_hash", "git_sha", "python_version", "model", "base_url",
            "temperature", "top_k", "max_steps", "api_key_recorded",
            "working_tree_status", "source_file_sha256_16", "prompts",
            "scoring", "config", "comparability_notes",
        )
    }
    run_info["config_hash"] = config_hash(config, dataset_sha256)
    run_info["checkpoint"] = str(checkpoint_path) if checkpoint_path else None
    run_info["output"] = str(output_path)
    run_info["metrics"] = str(metrics_path)
    run_info["case_count"] = len(paired)
    run_info["status"] = "completed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_path.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _load_index() -> tuple[dict, list[WikiPage], Any, str]:
    dataset, pages, index, dataset_sha256 = m10._load_index()
    if dataset_sha256 != m10.EXPECTED_DATASET_SHA256:
        raise RuntimeError("dataset hash mismatch for Stage 11")
    ids = [case["id"] for case in dataset["cases"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("dataset case IDs are not unique")
    if len(ids) != 50:
        raise RuntimeError(f"expected 50 cases, got {len(ids)}")
    return dataset, pages, index, dataset_sha256


def _api_key(argument: str | None) -> str | None:
    return m10._api_key(argument)


class RunLogger(m10.RunLogger):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 11: WikiAgent Follow Links ablation"
    )
    parser.add_argument("--live", action="store_true",
                        help="Actually call the model endpoint. Omit for offline validation.")
    parser.add_argument("--base-url", default=m10.DEFAULT_BASE_URL)
    parser.add_argument("--model", default=m10.DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None,
                        help="API key (or set ZMWIKI_API_KEY). Never written to outputs.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset, pages, index, dataset_sha256 = _load_index()
    print(f"dataset_sha256={dataset_sha256}")
    print(f"dataset_sha256_16={dataset_sha256[:16]}")
    print(f"cases={len(dataset['cases'])} pages={len(pages)}")
    print(f"full_tools={sorted(FULL_ALLOWED_ACTIONS)}")
    print(f"no_links_tools={sorted(NO_LINKS_ALLOWED_ACTIONS)}")
    print(f"full_prompt_sha256_16={FULL_PROMPT_SHA256_16}")
    print(f"no_links_prompt_sha256_16={NO_LINKS_PROMPT_SHA256_16}")
    if not args.live:
        print("Offline validation OK; no model request sent.")
        print("Use --live (and an API key) to run the Stage 11 ablation.")
        return

    api_key = _api_key(args.api_key)
    if api_key is None:
        parser.error("No API key provided (--api-key or ZMWIKI_API_KEY).")
    if args.model != m10.DEFAULT_MODEL:
        parser.error(
            f"Stage 11 is frozen to model={m10.DEFAULT_MODEL!r}; got {args.model!r}"
        )
    if args.base_url.rstrip("/") != m10.DEFAULT_BASE_URL:
        parser.error(
            f"Stage 11 is frozen to base_url={m10.DEFAULT_BASE_URL!r}; got {args.base_url!r}"
        )
    if args.top_k != 5:
        parser.error("Stage 11 is frozen to top_k=5")
    if args.max_steps != 8:
        parser.error("Stage 11 is frozen to max_steps=8")

    cases = dataset["cases"]
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(cases):
            parser.error(f"--limit must be between 1 and {len(cases)}")
        cases = cases[:args.limit]
    config = Stage11Config(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        top_k=args.top_k,
        max_steps=args.max_steps,
        limit=args.limit,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output) if args.output
        else ROOT / "benchmark" / "results" / f"stage11_ablation_{timestamp}.json"
    )
    checkpoint_path = output_path.with_name(output_path.stem + ".checkpoint.jsonl")
    log_path = output_path.with_suffix(".log")
    resumable = {checkpoint_path, log_path} if args.resume else set()
    for path in (
        output_path,
        checkpoint_path,
        output_path.with_name(output_path.stem + "_metrics.json"),
        output_path.with_name(output_path.stem + "_run_info.json"),
        log_path,
    ):
        if path.exists() and path not in resumable:
            parser.error(f"Refusing to overwrite existing file: {path}")
    if checkpoint_path.exists() and not args.resume:
        parser.error(
            f"Checkpoint exists; use --resume or choose a new output: {checkpoint_path}"
        )

    logger = RunLogger(log_path, resume=args.resume)
    try:
        logger(f"=== {EXPERIMENT_ID} ===")
        logger(f"dataset_sha256={dataset_sha256}")
        logger(f"model={config.model} base_url={config.base_url} "
               f"temperature={config.temperature} top_k={config.top_k} "
               f"max_steps={config.max_steps}")
        logger(f"full_prompt_sha256_16={FULL_PROMPT_SHA256_16}")
        logger(f"no_links_prompt_sha256_16={NO_LINKS_PROMPT_SHA256_16}")
        logger(f"output={output_path}")
        logger(f"checkpoint={checkpoint_path}")
        paired = run_stage11(
            cases, index, config,
            dataset_sha256=dataset_sha256,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            logger=logger,
        )
        serialize_stage11(
            dataset, config, paired, output_path,
            dataset_sha256=dataset_sha256,
            checkpoint_path=checkpoint_path,
        )
        full = aggregate_group([p[0] for p in paired], _group_results(paired, 1))
        no_links = aggregate_group(
            [p[0] for p in paired], _group_results(paired, 2)
        )
        logger(f"full answer_accuracy={full['answer_accuracy']:.3f} "
               f"({full['correct_count']}/{full['total_cases']}) "
               f"requests={full['model_usage']['total_model_requests']}")
        logger(f"no_links answer_accuracy={no_links['answer_accuracy']:.3f} "
               f"({no_links['correct_count']}/{no_links['total_cases']}) "
               f"requests={no_links['model_usage']['total_model_requests']}")
        logger(f"results={output_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
