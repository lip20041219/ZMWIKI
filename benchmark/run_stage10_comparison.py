"""Stage 10: same-model Traditional RAG vs WikiAgent comparison.

Traditional RAG:
    one wiki_search(top_k) -> wiki_read(only those top-k pages) ->
    one answer generation (plus at most one citation correction).

WikiAgent:
    the existing WikiAgent with search/read/follow_links, max_steps=8,
    search_top_k=5, and its existing evidence constraints.

Both groups use the same model endpoint, temperature, question set, shared
answer rules, answer-action JSON schema, and the frozen ``score_answer``
implementation from run_multihop_evaluation.py.

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
from zmwiki import Chunk, Context, OpenAICompatibleGenerator  # noqa: E402
from zmwiki.llm_generators import HttpResponse, urllib_transport  # noqa: E402
from zmwiki.wiki import wiki_agent  # noqa: E402
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent  # noqa: E402
from zmwiki.wiki.wiki_retriever import wiki_read, wiki_search  # noqa: E402
from zmwiki.wiki.wiki_schema import WikiPage  # noqa: E402

EXPERIMENT_ID = "stage10_traditional_rag_vs_wiki_agent"
EXPECTED_DATASET_SHA256 = (
    "f12127421c762ce191252f6f82559dc6b60397f7b603101bc1b034ee3299cce8"
)
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOP_K = 5
DEFAULT_MAX_STEPS = 8
TEMPERATURE = 0.0
RAG_PROMPT_VERSION = "stage10.traditional_rag.v1"
AGENT_PROMPT_VERSION = "wiki_agent.system_prompt.frozen"
SCORER_VERSION = "stage9_4_frozen"
MAX_RAW_CONTENT_CHARS = 4000

# These lines are copied from WikiAgent._SYSTEM_PROMPT's answer rules. The
# Traditional RAG prompt includes the same block; only the tool protocol differs.
SHARED_ANSWER_RULES = (
    "You MUST read at least one page before answering.\n"
    "Base your answer ONLY on content from pages you have read.\n"
    "If evidence is insufficient, answer with text explaining what you could "
    "not confirm, and list the page IDs of the pages you did read as sources.\n"
    "Sources must be the exact page IDs shown in [square brackets] in the "
    "search/read results. Never use titles, paths, filenames, or source "
    "document IDs."
)

RAG_SYSTEM_PROMPT = (
    "You are a Wiki research assistant. You answer a question in a single pass "
    "from a fixed Top-K page set. You cannot search, read additional pages, or "
    "follow links.\n\n"
    "Available action — output ONLY exactly one JSON object for the answer action:\n"
    '{"action": "answer", "text": "...", "sources": ["page_id1", "page_id2"]}\n\n'
    "Rules:\n"
    f"{SHARED_ANSWER_RULES}\n"
    "- The Top-K pages below are the pages you have read; do not assume any "
    "other page exists.\n"
    "- Do not output search, read, or follow_links actions.\n"
    "- Output only the JSON action object — no markdown fencing, no extra text."
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha16(text: str) -> str:
    return _sha256(text)[:16]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class Stage10Config:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = field(default=None, repr=False)
    temperature: float = TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    max_steps: int = DEFAULT_MAX_STEPS
    rag_answer_corrections: int = 1
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty str")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty str")
        if not isinstance(self.temperature, (int, float)) or self.temperature != 0.0:
            raise ValueError("Stage 10 requires temperature=0.0")
        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k"))
        object.__setattr__(self, "max_steps", _positive_int(self.max_steps, "max_steps"))
        object.__setattr__(
            self,
            "rag_answer_corrections",
            _positive_int(self.rag_answer_corrections, "rag_answer_corrections"),
        )
        if self.limit is not None:
            object.__setattr__(self, "limit", _positive_int(self.limit, "limit"))


@dataclass
class Stage10CaseResult(m4.CaseResult):
    """One group's result with model-usage and retrieval diagnostics."""

    model_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    latency_seconds: float = 0.0
    wall_seconds: float = 0.0
    retrieval_page_ids: list[str] = field(default_factory=list)
    citation_correction_used: bool = False
    failure_type: str | None = None
    model_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload.update({
            "model_requests": self.model_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "latency_seconds": self.latency_seconds,
            "wall_seconds": self.wall_seconds,
            "retrieval_page_ids": self.retrieval_page_ids,
            "retrieval_page_count": len(self.retrieval_page_ids),
            "citation_correction_used": self.citation_correction_used,
            "failure_type": self.failure_type,
            "protocol_error": self.status == AgentStatus.PROTOCOL_ERROR,
            "model_error": self.status == AgentStatus.MODEL_ERROR,
            "model_calls": self.model_calls,
        })
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Stage10CaseResult":
        base_names = {f.name for f in fields(m4.CaseResult)}
        base = {name: payload[name] for name in base_names if name in payload}
        extras = {
            "model_requests": payload.get("model_requests", 0),
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
            "total_tokens": payload.get("total_tokens", 0),
            "cache_hit_tokens": payload.get("cache_hit_tokens", 0),
            "cache_miss_tokens": payload.get("cache_miss_tokens", 0),
            "latency_seconds": payload.get("latency_seconds", 0.0),
            "wall_seconds": payload.get("wall_seconds", 0.0),
            "retrieval_page_ids": payload.get("retrieval_page_ids", []),
            "citation_correction_used": payload.get(
                "citation_correction_used", False
            ),
            "failure_type": payload.get("failure_type"),
            "model_calls": payload.get("model_calls", []),
        }
        return cls(**base, **extras)


class RecordingTransport:
    """Wrap a transport and record request/response metadata and usage.

    Only response-side content and usage are stored. Authorization headers and
    the API key are never copied into a record.
    """

    def __init__(self, inner=None) -> None:
        self.inner = inner if inner is not None else urllib_transport
        self.calls: list[dict] = []

    def __call__(self, request):
        started = time.perf_counter()
        body = _request_body(request)
        model = body.get("model")
        temperature = body.get("temperature")
        messages = body.get("messages", [])
        call: dict[str, Any] = {
            "index": len(self.calls),
            "url": request.url,
            "model": model,
            "temperature": temperature,
            "message_count": len(messages) if isinstance(messages, list) else None,
            "request_sha256_16": _sha16(request.body.decode("utf-8", errors="replace")),
            "status_code": None,
            "raw_content": None,
            "raw_content_truncated": False,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
            },
            "latency_seconds": 0.0,
            "error": None,
        }
        try:
            response = self.inner(request)
        except Exception as exc:
            call["latency_seconds"] = time.perf_counter() - started
            call["error"] = f"{type(exc).__name__}: {exc}"
            self.calls.append(call)
            raise
        call["latency_seconds"] = time.perf_counter() - started
        call["status_code"] = getattr(response, "status", None)
        if isinstance(response, HttpResponse):
            _record_response(call, response)
        self.calls.append(call)
        return response


def _request_body(request) -> dict:
    try:
        value = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_response(call: dict, response: HttpResponse) -> None:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        call["error"] = f"response_parse_error: {exc}"
        return
    if not isinstance(payload, dict):
        call["error"] = "response JSON is not an object"
        return
    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in call["usage"]:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                call["usage"][key] = value
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                call["raw_content"] = content[:MAX_RAW_CONTENT_CHARS]
                call["raw_content_truncated"] = len(content) > MAX_RAW_CONTENT_CHARS


def _usage(calls: list[dict]) -> dict:
    total = {
        "model_requests": len(calls),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "latency_seconds": 0.0,
    }
    for call in calls:
        usage = call.get("usage", {})
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        total["total_tokens"] += usage.get("total_tokens", 0)
        total["cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
        total["cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0)
        total["latency_seconds"] += call.get("latency_seconds", 0.0)
    return total


def _make_result(
    case: dict,
    *,
    answer: str | None,
    status: str,
    sources: list[str],
    accessed: set[str],
    read: list[str],
    retrieval: list[str],
    steps: int,
    step_details: list[dict],
    error: str | None,
    link_source: str,
    calls: list[dict],
    citation_correction_used: bool,
    wall_seconds: float,
) -> Stage10CaseResult:
    lexical_match = m4.score_answer(answer, case)
    correct = lexical_match and status == AgentStatus.ANSWERED
    grounded = correct and bool(sources)
    recall, precision, rec_num, rec_den = m4._compute_evidence_metrics(
        accessed, case["required_page_ids"], sources,
        case.get("alternative_evidence_sets"),
    )
    read_recall, read_num, read_den = m4._compute_recall(
        set(read), case["required_page_ids"], case.get("alternative_evidence_sets")
    )
    usage = _usage(calls)
    failure_type = status if error is not None else None
    return Stage10CaseResult(
        id=case["id"],
        question=case["question"],
        answer=answer,
        answer_correct=correct,
        lexical_match=lexical_match,
        grounded_correct=grounded,
        unsupported_correct=lexical_match and not grounded,
        status=status,
        sources=sources,
        accessed_pages=sorted(accessed),
        required_pages=case["required_page_ids"],
        evidence_recall=recall,
        source_precision=precision,
        steps=steps,
        pages_read=len(read),
        answerable=case["answerable"],
        scoring_rule="keyword_group_match",
        link_source=link_source,
        evidence_recall_numerator=rec_num,
        evidence_recall_denominator=rec_den,
        error=error,
        step_details=step_details,
        search_hit_page_ids=retrieval,
        read_page_ids=read,
        access_recall=recall,
        access_recall_numerator=rec_num,
        access_recall_denominator=rec_den,
        read_evidence_recall=read_recall,
        read_evidence_recall_numerator=read_num,
        read_evidence_recall_denominator=read_den,
        model_requests=usage["model_requests"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        cache_hit_tokens=usage["cache_hit_tokens"],
        cache_miss_tokens=usage["cache_miss_tokens"],
        latency_seconds=usage["latency_seconds"],
        wall_seconds=wall_seconds,
        retrieval_page_ids=retrieval,
        citation_correction_used=citation_correction_used,
        failure_type=failure_type,
        model_calls=calls,
    )


def _build_rag_context(pages: list[WikiPage]) -> Context:
    entries = [
        f"--- Page [{page.id}] {page.title} (path: {page.path}) ---\n{page.content}"
        for page in pages
    ]
    text = (
        "Top-K pages (this is the complete evidence set you have read):\n\n"
        + "\n\n".join(entries)
    )
    chunks = tuple(
        Chunk(
            id=f"{page.id}::stage10",
            document_id=page.id,
            text=page.content,
            position=0,
        )
        for page in pages
    )
    return Context(text=text, chunks=chunks)


def _parse_answer(
    raw: str, read_pages: dict[str, WikiPage]
) -> tuple[str | None, list[str], str, str | None]:
    """Parse one RAG response with the WikiAgent action parser."""
    try:
        action = wiki_agent._parse_action(raw)
    except wiki_agent.AgentProtocolError as exc:
        return None, [], AgentStatus.PROTOCOL_ERROR, str(exc)
    if action.get("action") != "answer":
        return (
            None, [], AgentStatus.PROTOCOL_ERROR,
            f"traditional RAG returned non-answer action {action.get('action')!r}",
        )
    text = action.get("text", "")
    if not isinstance(text, str) or not text.strip():
        text = ""
    valid = wiki_agent._valid_source_ids(action.get("sources", []), read_pages)
    if not valid:
        return (
            text or None, [], AgentStatus.INSUFFICIENT_EVIDENCE,
            wiki_agent._ANSWER_SOURCES_ERROR,
        )
    return text or None, valid, AgentStatus.ANSWERED, None


def run_traditional_rag_case(
    case: dict,
    index,
    config: Stage10Config,
    *,
    transport=None,
) -> Stage10CaseResult:
    """Run one single-pass Top-K RAG case."""
    started = time.perf_counter()
    recorder = RecordingTransport(transport)
    retrieved: list[str] = []
    pages: list[WikiPage] = []
    step_details: list[dict] = []
    answer: str | None = None
    sources: list[str] = []
    status = AgentStatus.INSUFFICIENT_EVIDENCE
    error = "no pages retrieved"
    correction_used = False

    try:
        results = wiki_search(index, case["question"], top_k=config.top_k)
        retrieved = [result.page_id for result in results]
        if retrieved:
            pages = wiki_read(index, retrieved)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        step_details.append({
            "step": 0,
            "action": "retrieve",
            "params": {"query": case["question"], "top_k": config.top_k},
            "error": f"{type(exc).__name__}: {exc}",
            "result_summary": "retrieval failed",
            "read_page_ids": [],
            "search_hit_page_ids": [],
        })
        return _make_result(
            case, answer=None, status="retrieval_error", sources=[],
            accessed=set(), read=[], retrieval=[], steps=1,
            step_details=step_details, error=f"{type(exc).__name__}: {exc}",
            link_source="none", calls=recorder.calls,
            citation_correction_used=False, wall_seconds=elapsed,
        )  # RAG steps=1: one retrieval/generation round; model_requests counts calls.

    read_pages = {page.id: page for page in pages}
    step_details.append({
        "step": 0,
        "action": "retrieve",
        "params": {"query": case["question"], "top_k": config.top_k},
        "error": None,
        "result_summary": f"retrieved {len(retrieved)} pages",
        "read_page_ids": list(read_pages),
        "search_hit_page_ids": retrieved,
    })

    if pages:
        try:
            generator = OpenAICompatibleGenerator(
                base_url=config.base_url,
                model=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
                transport=recorder,
                system_prompt=RAG_SYSTEM_PROMPT,
            )
            context = _build_rag_context(pages)
            raw = generator.generate(case["question"], context)
            answer, sources, status, error = _parse_answer(raw, read_pages)
            if (
                status == AgentStatus.INSUFFICIENT_EVIDENCE
                and read_pages
                and not correction_used
                and config.rag_answer_corrections > 0
            ):
                correction_used = True
                correction = wiki_agent._source_correction_msg(read_pages)
                raw = generator.generate(
                    f"{case['question']}\n\n{correction}", context
                )
                answer, sources, status, error = _parse_answer(raw, read_pages)
        except Exception as exc:
            status = AgentStatus.MODEL_ERROR
            error = f"{type(exc).__name__}: {exc}"
            answer = None
            sources = []
    else:
        status = AgentStatus.INSUFFICIENT_EVIDENCE
        error = "no pages retrieved"
        answer = None
        sources = []

    step_details.append({
        "step": len(step_details),
        "action": "answer",
        "params": {"text": (answer or "")[:200], "sources": sources},
        "error": error,
        "result_summary": f"answer status={status}",
        "read_page_ids": list(read_pages),
        "search_hit_page_ids": retrieved,
    })
    elapsed = time.perf_counter() - started
    return _make_result(
        case,
        answer=answer,
        status=status,
        sources=sources,
        accessed=set(read_pages),
        read=list(read_pages),
        retrieval=retrieved,
        steps=1,
        step_details=step_details,
        error=error,
        link_source="none",
        calls=recorder.calls,
        citation_correction_used=correction_used,
        wall_seconds=elapsed,
    )


def run_wiki_agent_case(
    case: dict,
    index,
    config: Stage10Config,
    *,
    transport=None,
) -> Stage10CaseResult:
    """Run one current WikiAgent case, preserving its tools and constraints."""
    started = time.perf_counter()
    recorder = RecordingTransport(transport)
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
        )
        agent_result = agent.answer(case["question"])
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return _make_result(
            case, answer=None, status="runtime_error", sources=[],
            accessed=set(monitor.accessed_page_ids), read=[], retrieval=[],
            steps=0, step_details=[], error=f"{type(exc).__name__}: {exc}",
            link_source="manual", calls=recorder.calls,
            citation_correction_used=False, wall_seconds=elapsed,
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
        step.action == "answer" and step.error
        for step in agent_result.steps
    )
    elapsed = time.perf_counter() - started
    return _make_result(
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


def _config_payload(config: Stage10Config, dataset_sha256: str) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_sha256": dataset_sha256,
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "rag_answer_corrections": config.rag_answer_corrections,
        "limit": config.limit,
        "rag_prompt_version": RAG_PROMPT_VERSION,
        "rag_prompt_sha256_16": _sha16(RAG_SYSTEM_PROMPT),
        "agent_prompt_version": AGENT_PROMPT_VERSION,
        "agent_prompt_sha256_16": _sha16(wiki_agent._SYSTEM_PROMPT),
        "scorer_version": SCORER_VERSION,
        "score_answer_source_sha256_16": _sha16(
            inspect.getsource(m4.score_answer)
        ),
        "aggregate_results_source_sha256_16": _sha16(
            inspect.getsource(m4.aggregate_results)
        ),
    }


def config_hash(config: Stage10Config, dataset_sha256: str) -> str:
    payload = _config_payload(config, dataset_sha256)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical)


def load_checkpoint(
    path: Path, dataset_sha256: str, expected_config_hash: str
) -> dict[str, tuple[Stage10CaseResult, Stage10CaseResult]]:
    completed: dict[str, tuple[Stage10CaseResult, Stage10CaseResult]] = {}
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
            raise RuntimeError(
                f"checkpoint config hash mismatch at line {line_no}"
            )
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"checkpoint line {line_no}: missing case_id")
        if case_id in completed:
            raise RuntimeError(f"duplicate checkpoint case_id: {case_id}")
        try:
            rag = Stage10CaseResult.from_dict(record["rag"])
            agent = Stage10CaseResult.from_dict(record["agent"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"invalid checkpoint result for {case_id}: {exc}"
            ) from exc
        completed[case_id] = (rag, agent)
    return completed


def _append_checkpoint(
    path: Path,
    case: dict,
    rag: Stage10CaseResult,
    agent: Stage10CaseResult,
    dataset_sha256: str,
    expected_config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case["id"],
        "dataset_sha256": dataset_sha256,
        "config_hash": expected_config_hash,
        "rag": rag.to_dict(),
        "agent": agent.to_dict(),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_stage10(
    cases: list[dict],
    index,
    config: Stage10Config,
    *,
    dataset_sha256: str,
    rag_transport=None,
    agent_transport=None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    logger=None,
) -> list[tuple[dict, Stage10CaseResult, Stage10CaseResult]]:
    """Run or resume the paired Stage 10 evaluation."""
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
            rag, agent = completed[case["id"]]
            paired.append((case, rag, agent))
            continue
        if logger:
            logger(f"[{case['id']}] traditional RAG")
        rag = run_traditional_rag_case(
            case, index, config, transport=rag_transport
        )
        if logger:
            logger(
                f"[{case['id']}] RAG status={rag.status} "
                f"correct={rag.answer_correct} requests={rag.model_requests}"
            )
            logger(f"[{case['id']}] WikiAgent")
        agent = run_wiki_agent_case(
            case, index, config, transport=agent_transport
        )
        if logger:
            logger(
                f"[{case['id']}] Agent status={agent.status} "
                f"correct={agent.answer_correct} requests={agent.model_requests}"
            )
        paired.append((case, rag, agent))
        if checkpoint_path is not None:
            _append_checkpoint(
                checkpoint_path, case, rag, agent, dataset_sha256, expected
            )
    return paired


def _group_results(
    paired: list[tuple[dict, Stage10CaseResult, Stage10CaseResult]],
    index: int,
) -> list[Stage10CaseResult]:
    return [pair[index] for pair in paired]


def aggregate_group(cases: list[dict], results: list[Stage10CaseResult]) -> dict:
    aggregate = m4.aggregate_results(cases, results)
    aggregate["answer_quality_metrics"]["note"] = (
        "Stage 10: both groups use the same LLM, temperature, question, answer "
        "schema, citation rules, and frozen scorer. Answer-quality differences "
        "are attributable to retrieval/tool policy, not generator identity."
    )
    aggregate["retrieval_evidence_metrics"]["note"] = (
        "Stage 10: read_evidence_recall uses only pages actually read; "
        "Traditional RAG reads only its Top-K pages, WikiAgent reads pages "
        "returned by search/read/follow_links."
    )
    aggregate["model_usage"] = {
        "mean_model_requests": (
            sum(r.model_requests for r in results) / len(results)
            if results else 0.0
        ),
        "total_model_requests": sum(r.model_requests for r in results),
        "mean_prompt_tokens": (
            sum(r.prompt_tokens for r in results) / len(results)
            if results else 0.0
        ),
        "mean_completion_tokens": (
            sum(r.completion_tokens for r in results) / len(results)
            if results else 0.0
        ),
        "mean_total_tokens": (
            sum(r.total_tokens for r in results) / len(results)
            if results else 0.0
        ),
        "total_tokens": sum(r.total_tokens for r in results),
        "total_cache_hit_tokens": sum(r.cache_hit_tokens for r in results),
        "total_cache_miss_tokens": sum(r.cache_miss_tokens for r in results),
        "mean_latency_seconds": (
            sum(r.latency_seconds for r in results) / len(results)
            if results else 0.0
        ),
        "total_latency_seconds": sum(r.latency_seconds for r in results),
        "mean_wall_seconds": (
            sum(r.wall_seconds for r in results) / len(results)
            if results else 0.0
        ),
        "mean_retrieval_pages": (
            sum(len(r.retrieval_page_ids) for r in results) / len(results)
            if results else 0.0
        ),
        "citation_correction_count": sum(
            1 for r in results if r.citation_correction_used
        ),
        "protocol_error_count": sum(
            1 for r in results if r.status == AgentStatus.PROTOCOL_ERROR
        ),
        "model_error_count": sum(
            1 for r in results if r.status == AgentStatus.MODEL_ERROR
        ),
        "insufficient_evidence_count": sum(
            1 for r in results if r.status == AgentStatus.INSUFFICIENT_EVIDENCE
        ),
        "retrieval_error_count": sum(
            1 for r in results if r.status == "retrieval_error"
        ),
        "runtime_error_count": sum(
            1 for r in results if r.status == "runtime_error"
        ),
    }
    return aggregate


def _mean_delta(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def paired_summary(
    pairs: list[tuple[dict, Stage10CaseResult, Stage10CaseResult]]
) -> dict:
    wins = ties = losses = both_wrong = 0
    deltas: dict[str, list[float | None]] = {
        "answer_correct": [],
        "grounded_correct": [],
        "evidence_recall": [],
        "read_evidence_recall": [],
        "source_precision": [],
        "model_requests": [],
        "total_tokens": [],
        "latency_seconds": [],
        "steps": [],
    }
    for _, rag, agent in pairs:
        if agent.answer_correct and not rag.answer_correct:
            wins += 1
        elif rag.answer_correct and not agent.answer_correct:
            losses += 1
        elif agent.answer_correct and rag.answer_correct:
            ties += 1
        else:
            both_wrong += 1
        deltas["answer_correct"].append(
            int(agent.answer_correct) - int(rag.answer_correct)
        )
        deltas["grounded_correct"].append(
            int(agent.grounded_correct) - int(rag.grounded_correct)
        )
        for key, attr in (
            ("evidence_recall", "evidence_recall"),
            ("read_evidence_recall", "read_evidence_recall"),
            ("source_precision", "source_precision"),
        ):
            rag_value = getattr(rag, attr)
            agent_value = getattr(agent, attr)
            deltas[key].append(
                None if rag_value is None or agent_value is None
                else agent_value - rag_value
            )
        deltas["model_requests"].append(
            agent.model_requests - rag.model_requests
        )
        deltas["total_tokens"].append(agent.total_tokens - rag.total_tokens)
        deltas["latency_seconds"].append(
            agent.latency_seconds - rag.latency_seconds
        )
        deltas["steps"].append(agent.steps - rag.steps)
    return {
        "cases": len(pairs),
        "agent_only_correct": wins,
        "rag_only_correct": losses,
        "both_correct": ties,
        "both_wrong": both_wrong,
        "mean_agent_minus_rag": {
            key: _mean_delta(values) for key, values in deltas.items()
        },
    }


def _paired_per_case(
    pairs: list[tuple[dict, Stage10CaseResult, Stage10CaseResult]]
) -> list[dict]:
    payload = []
    for case, rag, agent in pairs:
        payload.append({
            "case_id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "hops": case["hops"],
            "traditional_rag": rag.to_dict(),
            "wiki_agent": agent.to_dict(),
            "diff": {
                "answer_correct": int(agent.answer_correct) - int(rag.answer_correct),
                "grounded_correct": (
                    int(agent.grounded_correct) - int(rag.grounded_correct)
                ),
                "evidence_recall": (
                    None if rag.evidence_recall is None or agent.evidence_recall is None
                    else agent.evidence_recall - rag.evidence_recall
                ),
                "read_evidence_recall": (
                    None if rag.read_evidence_recall is None
                    or agent.read_evidence_recall is None
                    else agent.read_evidence_recall - rag.read_evidence_recall
                ),
                "source_precision": agent.source_precision - rag.source_precision,
                "model_requests": agent.model_requests - rag.model_requests,
                "total_tokens": agent.total_tokens - rag.total_tokens,
                "latency_seconds": (
                    agent.latency_seconds - rag.latency_seconds
                ),
                "steps": agent.steps - rag.steps,
            },
        })
    return payload


def _provenance(dataset: dict, config: Stage10Config, dataset_sha256: str) -> dict:
    source_hashes = m4._source_file_hashes()
    source_hashes["benchmark/run_stage10_comparison.py"] = _file_sha256(
        Path(__file__)
    )[:16]
    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "benchmark/datasets/multihop_wiki.json",
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_sha256": dataset_sha256,
        "dataset_sha256_16": dataset_sha256[:16],
        "dataset_sha256_expected": EXPECTED_DATASET_SHA256,
        "config_hash": config_hash(config, dataset_sha256),
        "git_sha": m4._git_sha(),
        "python_version": platform.python_version(),
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "rag_answer_corrections": config.rag_answer_corrections,
        "api_key_recorded": False,
        "working_tree_status": m4._working_tree_status(),
        "source_file_sha256_16": source_hashes,
        "answer_generation": {
            "shared_answer_rules": SHARED_ANSWER_RULES,
            "answer_schema": (
                '{"action": "answer", "text": "...", "sources": ["page_id"]}'
            ),
            "rag_prompt_version": RAG_PROMPT_VERSION,
            "rag_prompt_sha256_16": _sha16(RAG_SYSTEM_PROMPT),
            "agent_prompt_version": AGENT_PROMPT_VERSION,
            "agent_prompt_sha256_16": _sha16(wiki_agent._SYSTEM_PROMPT),
        },
        "scoring": {
            "version": SCORER_VERSION,
            "rule": dataset.get("scoring_rule", "unknown"),
            "score_answer_source_sha256_16": _sha16(
                inspect.getsource(m4.score_answer)
            ),
            "aggregate_results_source_sha256_16": _sha16(
                inspect.getsource(m4.aggregate_results)
            ),
            "manual_review_case_ids": [
                case["id"] for case in dataset["cases"]
                if case.get("manual_review")
            ],
        },
        "config": {
            "traditional_rag": {
                "retrieval": "wiki_search(top_k) then wiki_read(only those pages)",
                "tools": [],
                "max_context_pages": config.top_k,
                "answer_calls": "1 single-shot answer + up to 1 citation correction",
                "link_source": "none",
            },
            "wiki_agent": {
                "agent": "WikiAgent",
                "tools": ["search", "read", "follow_links"],
                "max_steps": config.max_steps,
                "search_top_k": config.top_k,
                "link_source": "manual",
                "fail_closed": True,
            },
            "shared": {
                "model": config.model,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "dataset": "benchmark/datasets/multihop_wiki.json",
                "scoring": "run_multihop_evaluation.score_answer",
                "answer_schema": "WikiAgent answer action JSON",
            },
        },
        "comparability_notes": {
            "fairness": "Both groups use the same model, base URL, temperature, "
                        "question set, answer rules, citation requirements, action "
                        "parser, and score_answer implementation.",
            "difference": "Traditional RAG may only use the first Top-K search "
                         "result set; WikiAgent may perform additional search/read/"
                         "follow_links calls. This is the system difference under test.",
            "failure_handling": "Protocol, model, retrieval, insufficient-evidence, "
                               "and budget failures are retained as failed samples; "
                               "no successful-only filtering.",
            "steps_definition": "Traditional RAG steps=1 per single retrieval/generation "
                               "round; model_requests counts actual model calls, including "
                               "at most one citation correction. WikiAgent steps=action "
                               "iterations.",
            "manual_review": "mh46/mh47 and other manual_review cases keep automatic "
                            "scores; manual review is not merged into answer_correct.",
        },
    }


def _ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def serialize_stage10(
    dataset: dict,
    config: Stage10Config,
    pairs: list[tuple[dict, Stage10CaseResult, Stage10CaseResult]],
    output_path: Path,
    *,
    dataset_sha256: str,
    checkpoint_path: Path | None = None,
) -> dict:
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    run_info_path = output_path.with_name(output_path.stem + "_run_info.json")
    for path in (output_path, metrics_path, run_info_path):
        _ensure_new(path)

    cases = [pair[0] for pair in pairs]
    rag_results = _group_results(pairs, 1)
    agent_results = _group_results(pairs, 2)
    rag_aggregate = aggregate_group(cases, rag_results)
    agent_aggregate = aggregate_group(cases, agent_results)
    summary = paired_summary(pairs)
    provenance = _provenance(dataset, config, dataset_sha256)

    payload = {
        **provenance,
        "results": {
            "traditional_rag": {
                "per_case": [result.to_dict() for result in rag_results],
                "aggregate": rag_aggregate,
            },
            "wiki_agent": {
                "per_case": [result.to_dict() for result in agent_results],
                "aggregate": agent_aggregate,
            },
            "paired_per_case": _paired_per_case(pairs),
            "paired_summary": summary,
        },
    }
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_sha256": dataset_sha256,
        "dataset_sha256_16": dataset_sha256[:16],
        "dataset_sha256_expected": EXPECTED_DATASET_SHA256,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "max_steps": config.max_steps,
        "traditional_rag": rag_aggregate,
        "wiki_agent": agent_aggregate,
        "paired_summary": summary,
        "scoring": provenance["scoring"],
    }
    run_info = {
        key: provenance[key] for key in (
            "experiment_id", "timestamp", "dataset", "dataset_version",
            "dataset_sha256", "dataset_sha256_16", "dataset_sha256_expected",
            "git_sha", "python_version",
            "model", "base_url", "temperature", "top_k", "max_steps",
            "rag_answer_corrections", "api_key_recorded", "working_tree_status",
            "source_file_sha256_16", "answer_generation", "scoring", "config",
            "comparability_notes",
        )
    }
    run_info["config_hash"] = config_hash(config, dataset_sha256)
    run_info["checkpoint"] = str(checkpoint_path) if checkpoint_path else None
    run_info["output"] = str(output_path)
    run_info["metrics"] = str(metrics_path)
    run_info["case_count"] = len(pairs)
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
    dataset = m4.load_dataset()
    dataset_sha256 = _file_sha256(m4.DATASET_PATH)
    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "dataset SHA-256 does not match the frozen Stage 10 hash: "
            f"expected {EXPECTED_DATASET_SHA256}, got {dataset_sha256}. "
            "Stop and report; do not update the baseline hash."
        )
    documents = m4.load_documents()
    pages = m4.build_wiki_pages(documents)
    index = m4.build_wiki_index(pages)
    return dataset, pages, index, dataset_sha256


def _api_key(argument: str | None) -> str | None:
    if argument is not None:
        return argument
    value = os.environ.get("ZMWIKI_API_KEY", "").strip()
    return value or None


class RunLogger:
    def __init__(self, path: Path, *, resume: bool) -> None:
        mode = "a" if resume else "x"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, mode, encoding="utf-8")
        self.path = path

    def __call__(self, message: str) -> None:
        print(message)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 10: same-model Traditional RAG vs WikiAgent comparison"
    )
    parser.add_argument("--live", action="store_true",
                        help="Actually call the model endpoint. Omit for offline validation.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None,
                        help="API key (or set ZMWIKI_API_KEY). Never written to outputs.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset, pages, index, dataset_sha256 = _load_index()
    if len(pages) != 9:
        raise RuntimeError(f"expected 9 Wiki pages, got {len(pages)}")
    print(f"dataset_sha256={dataset_sha256}")
    print(f"dataset_sha256_16={dataset_sha256[:16]}")
    print(f"pages={len(pages)}")
    if not args.live:
        print("Offline validation OK; no model request sent.")
        print("Use --live (and an API key) to run the Stage 10 comparison.")
        return

    api_key = _api_key(args.api_key)
    if api_key is None:
        parser.error("No API key provided (--api-key or ZMWIKI_API_KEY).")
    if args.model != DEFAULT_MODEL:
        parser.error(
            f"Stage 10 is frozen to model={DEFAULT_MODEL!r}; got {args.model!r}"
        )
    if args.base_url.rstrip("/") != DEFAULT_BASE_URL:
        parser.error(
            f"Stage 10 is frozen to base_url={DEFAULT_BASE_URL!r}; got {args.base_url!r}"
        )
    if args.top_k != DEFAULT_TOP_K:
        parser.error(f"Stage 10 is frozen to top_k={DEFAULT_TOP_K}")
    if args.max_steps != DEFAULT_MAX_STEPS:
        parser.error(f"Stage 10 is frozen to max_steps={DEFAULT_MAX_STEPS}")

    cases = dataset["cases"]
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(cases):
            parser.error(f"--limit must be between 1 and {len(cases)}")
        cases = cases[:args.limit]
    config = Stage10Config(
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
        else m4.ROOT / "benchmark" / "results" /
             f"stage10_comparison_{timestamp}.json"
    )
    checkpoint_path = output_path.with_name(
        output_path.stem + ".checkpoint.jsonl"
    )
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
        parser.error(f"Checkpoint exists; use --resume or choose a new output: {checkpoint_path}")

    logger = RunLogger(log_path, resume=args.resume)
    try:
        logger(f"=== {EXPERIMENT_ID} ===")
        logger(f"dataset_sha256={dataset_sha256}")
        logger(f"model={config.model} base_url={config.base_url} "
               f"temperature={config.temperature} top_k={config.top_k} "
               f"max_steps={config.max_steps}")
        logger(f"output={output_path}")
        logger(f"checkpoint={checkpoint_path}")
        pairs = run_stage10(
            cases, index, config,
            dataset_sha256=dataset_sha256,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            logger=logger,
        )
        serialize_stage10(
            dataset, config, pairs, output_path,
            dataset_sha256=dataset_sha256,
            checkpoint_path=checkpoint_path,
        )
        rag = aggregate_group([p[0] for p in pairs], _group_results(pairs, 1))
        agent = aggregate_group([p[0] for p in pairs], _group_results(pairs, 2))
        logger(f"rag answer_accuracy={rag['answer_accuracy']:.3f} "
               f"({rag['correct_count']}/{rag['total_cases']}) "
               f"requests={rag['model_usage']['total_model_requests']}")
        logger(f"agent answer_accuracy={agent['answer_accuracy']:.3f} "
               f"({agent['correct_count']}/{agent['total_cases']}) "
               f"requests={agent['model_usage']['total_model_requests']}")
        logger(f"results={output_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
