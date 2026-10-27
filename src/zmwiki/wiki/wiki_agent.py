"""ZMWiki minimal multi-step retrieval Agent.

WikiAgent answers a user question by iteratively choosing actions (search,
read, follow_links, answer) through an LLM, executing each validated action
against a WikiIndex, and accumulating evidence from actually-read pages.

Design constraints (M3):
  - Reuses the existing Transport / HttpRequest / HttpResponse boundary and
    configuration shape. No second HTTP client.
  - Actions are model-suggested but program-validated: the model can never
    execute arbitrary code, read arbitrary files, or call undefined actions.
  - Evidence is strictly tiered: search candidates are NOT evidence; only
    pages actually fetched via wiki_read() count.
  - Budget control is enforced in code (max steps, duplicate-action dedup,
    consecutive protocol-error limit), not left to prompt compliance.
  - Stdlib only. No new runtime dependencies. No Agent framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm_generators import (
    HttpRequest,
    HttpResponse,
    Transport,
    urllib_transport,
)
from ..llm_generators import (
    _check_base_url,
    _check_temperature,
    _check_timeout,
    _required_str,
    _resolve_api_key,
)
from .wiki_index import WikiIndex
from .wiki_retriever import WikiSearchResult, wiki_read, wiki_search
from .wiki_schema import WikiPage

__all__ = [
    "WikiAgent",
    "AgentResult",
    "AgentStep",
    "AgentStatus",
    "AgentProtocolError",
]

#: Default maximum number of action iterations before forced stop.
_DEFAULT_MAX_STEPS = 8

#: Default number of candidates returned per search.
_DEFAULT_SEARCH_TOP_K = 5

#: Default maximum consecutive protocol errors before forced stop.
_DEFAULT_MAX_CONSECUTIVE_ERRORS = 2

#: Maximum characters of page content included in a tool-result message.
_READ_CONTENT_CAP = 3000

_SYSTEM_PROMPT = (
    "You are a Wiki research assistant. You answer questions by searching and "
    "reading pages from a Wiki knowledge base.\n\n"
    "Available actions — output ONLY a JSON object for one action each turn:\n"
    '1. {"action": "search", "query": "..."} — Search for relevant pages. '
    "Returns candidate titles and paths.\n"
    '2. {"action": "read", "paths": ["path1"], "offset": 0} — Read pages by '
    "path. Optional offset (characters) continues a long page. Long pages "
    "include a continuation offset in the result.\n"
    '3. {"action": "follow_links", "from_page": "page_id"} — Follow links from '
    "a page you already read. Reads all linked pages.\n"
    '4. {"action": "answer", "text": "...", "sources": ["page_id1", "page_id2"]} '
    "— Final answer. Sources must be page IDs of pages you actually read.\n\n"
    "Rules:\n"
    "- You MUST read at least one page before answering.\n"
    "- Base your answer ONLY on content from pages you have read.\n"
    "- If a page result is truncated, use its continuation offset before "
    "answering if the missing part may contain the evidence.\n"
    "- If evidence is insufficient, answer with text explaining what you could "
    "not confirm, and list the pages you did read as sources.\n"
    "- Do not repeat the exact same search or read.\n"
    "- Output only the JSON action object — no markdown fencing, no extra text."
)

_VALID_ACTIONS = frozenset({"search", "read", "follow_links", "answer"})

#: Protocol-error diagnostics keep at most this many characters of model output.
_MAX_RAW_OUTPUT_CHARS = 2000


# ---------------------------------------------------------------------------
# Status & errors
# ---------------------------------------------------------------------------


class AgentStatus:
    """Terminal status strings for AgentResult."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"
    PROTOCOL_ERROR = "protocol_error"


class AgentProtocolError(Exception):
    """Raised when the model output cannot be parsed as a valid action."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentStep:
    """Record of one action iteration.

    Attributes
    ----------
    step: 0-based iteration number.
    action: the action type executed, or "protocol_error" / "duplicate".
    params: normalized action parameters (for diagnostics); search steps also
        record the effective top_k.
    result_summary: short description of what happened.
    error: error message if this step failed, else None.
    raw_output: source model text when parsing failed (only on protocol_error
        steps), truncated to _MAX_RAW_OUTPUT_CHARS; None on success steps.
    read_page_ids: page ids actually returned by this step's read or
        follow_links action; empty for other actions and failed reads.
    search_hit_page_ids: page ids returned by this step's search action;
        empty for other actions and searches with no hits.
    """

    step: int
    action: str
    params: dict[str, Any]
    result_summary: str
    error: str | None = None
    raw_output: str | None = None
    read_page_ids: tuple[str, ...] = ()
    search_hit_page_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    """Final result of an Agent answer() call.

    Attributes
    ----------
    answer: the model's answer text, or None if no answer was produced.
    sources: page paths of pages actually read and cited as sources.
    status: one of AgentStatus constants.
    steps: full action history for diagnostics.
    error: terminal error message, or None.
    pages_read: number of distinct pages actually read.
    read_page_ids: ids of the pages actually read, for diagnostics.
    """

    answer: str | None
    sources: tuple[str, ...]
    status: str
    steps: tuple[AgentStep, ...]
    error: str | None = None
    pages_read: int = 0
    read_page_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class WikiAgent:
    """Multi-step retrieval Agent over a WikiIndex.

    Parameters mirror WikiCompiler / OpenAICompatibleGenerator so the same
    endpoint config works across generation, compilation, and agent retrieval.
    """

    def __init__(
        self,
        index: WikiIndex,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        transport: Transport | None = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        search_top_k: int = _DEFAULT_SEARCH_TOP_K,
        max_consecutive_errors: int = _DEFAULT_MAX_CONSECUTIVE_ERRORS,
    ) -> None:
        if not isinstance(index, WikiIndex):
            raise TypeError(
                f"index must be a WikiIndex, got {type(index).__name__}"
            )
        self.index = index
        self.base_url = _check_base_url(base_url)
        self.model = _required_str(model, "model")
        self._api_key = _resolve_api_key(api_key)
        self.temperature = _check_temperature(temperature)
        self.timeout = _check_timeout(timeout)
        self.transport: Transport = (
            transport if transport is not None else urllib_transport
        )
        if not callable(self.transport):
            raise TypeError(
                f"transport must be callable, got {type(self.transport).__name__}"
            )
        self.max_steps = _check_positive_int(max_steps, "max_steps")
        self.search_top_k = _check_positive_int(search_top_k, "search_top_k")
        self.max_consecutive_errors = _check_positive_int(
            max_consecutive_errors, "max_consecutive_errors"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> AgentResult:
        """Run the multi-step retrieval loop and return an AgentResult.

        The loop: build messages → call model → parse/validate action →
        execute action (search/read/follow_links/answer) → append tool result
        → repeat until answer, budget exhaustion, or error.
        """
        if not isinstance(question, str) or not question.strip():
            raise TypeError("question must be a non-empty str")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        read_pages: dict[str, WikiPage] = {}
        seen_page_ids: set[str] = set()
        seen_actions: set[str] = set()
        consecutive_errors = 0

        for step_num in range(self.max_steps):
            # --- Model call ---
            try:
                raw = self._call_model(messages)
            except Exception as exc:
                steps.append(AgentStep(
                    step=step_num, action="model_error", params={},
                    result_summary="model call failed", error=str(exc),
                ))
                return AgentResult(
                    answer=None, sources=(), status=AgentStatus.MODEL_ERROR,
                    steps=tuple(steps), error=str(exc), pages_read=len(read_pages),
                    read_page_ids=tuple(sorted(read_pages)),
                )

            # --- Parse action ---
            try:
                action_data = _parse_action(raw)
            except AgentProtocolError as exc:
                consecutive_errors += 1
                raw_output = (
                    raw if len(raw) <= _MAX_RAW_OUTPUT_CHARS
                    else raw[:_MAX_RAW_OUTPUT_CHARS] + "\n...[truncated]"
                )
                steps.append(AgentStep(
                    step=step_num, action="protocol_error", params={},
                    result_summary="invalid model output", error=str(exc),
                    raw_output=raw_output,
                ))
                if consecutive_errors >= self.max_consecutive_errors:
                    return AgentResult(
                        answer=None, sources=(), status=AgentStatus.PROTOCOL_ERROR,
                        steps=tuple(steps),
                        error=f"{consecutive_errors} consecutive protocol errors",
                        pages_read=len(read_pages),
                        read_page_ids=tuple(sorted(read_pages)),
                    )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _protocol_error_msg(exc)})
                continue

            consecutive_errors = 0
            action = action_data["action"]

            # --- Duplicate action detection ---
            action_key = _action_key(action, action_data)
            if action_key in seen_actions and action != "answer":
                steps.append(AgentStep(
                    step=step_num, action="duplicate", params=action_data,
                    result_summary=f"duplicate action ignored: {action}",
                ))
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": f"You already performed action {action!r} with the same "
                    "parameters. Choose a different action or answer with what you have.",
                })
                continue
            seen_actions.add(action_key)

            # --- Execute action ---
            if action == "search":
                step, tool_msg = self._do_search(step_num, action_data, seen_page_ids)
            elif action == "read":
                step, tool_msg, new_pages = self._do_read(step_num, action_data, read_pages)
                for p in new_pages:
                    read_pages[p.id] = p
                    seen_page_ids.add(p.id)
            elif action == "follow_links":
                step, tool_msg, new_pages = self._do_follow_links(
                    step_num, action_data, read_pages
                )
                for p in new_pages:
                    read_pages[p.id] = p
                    seen_page_ids.add(p.id)
            elif action == "answer":
                return self._do_answer(step_num, action_data, read_pages, steps)
            else:
                # Should be unreachable due to _parse_action validation.
                step = AgentStep(
                    step=step_num, action="protocol_error", params=action_data,
                    result_summary=f"unknown action: {action}",
                    error=f"unknown action {action!r}",
                )
                tool_msg = f"Unknown action {action!r}. Valid actions: {sorted(_VALID_ACTIONS)}"

            steps.append(step)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": tool_msg})

        # --- Budget exhausted ---
        return AgentResult(
            answer=None, sources=(), status=AgentStatus.BUDGET_EXHAUSTED,
            steps=tuple(steps),
            error=f"reached max_steps={self.max_steps} without an answer",
            pages_read=len(read_pages),
            read_page_ids=tuple(sorted(read_pages)),
        )

    # ------------------------------------------------------------------
    # Model call
    # ------------------------------------------------------------------

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        """Send the conversation to the model and return the assistant content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = HttpRequest(
            url=f"{self.base_url}/chat/completions",
            method="POST",
            headers=headers,
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout,
        )
        response = self.transport(request)
        if not isinstance(response, HttpResponse):
            raise RuntimeError(
                f"transport must return HttpResponse, got {type(response).__name__}"
            )
        return _extract_content(response)

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    def _do_search(
        self, step_num: int, action_data: dict[str, Any], seen_page_ids: set[str]
    ) -> tuple[AgentStep, str]:
        query = action_data["query"]
        results = wiki_search(self.index, query, top_k=self.search_top_k)
        new_count = sum(1 for r in results if r.page_id not in seen_page_ids)
        summary = f"search returned {len(results)} candidates ({new_count} new)"
        step = AgentStep(
            step=step_num, action="search",
            params={"query": query, "top_k": self.search_top_k},
            result_summary=summary,
            search_hit_page_ids=tuple(r.page_id for r in results),
        )
        tool_msg = _format_search_results(results)
        return step, tool_msg

    def _do_read(
        self, step_num: int, action_data: dict[str, Any], read_pages: dict[str, WikiPage]
    ) -> tuple[AgentStep, str, list[WikiPage]]:
        paths = action_data["paths"]
        offset = action_data.get("offset", 0)
        try:
            pages = wiki_read(self.index, paths)
        except KeyError as exc:
            params = {"paths": paths}
            if offset:
                params["offset"] = offset
            step = AgentStep(
                step=step_num, action="read", params=params,
                result_summary="read failed: paths not found", error=str(exc),
            )
            return step, f"Read error: {exc}", []
        new_pages = [p for p in pages if p.id not in read_pages]
        summary = f"read {len(pages)} pages ({len(new_pages)} new)"
        params = {"paths": paths}
        if offset:
            params["offset"] = offset
            summary += f" at offset {offset}"
        step = AgentStep(
            step=step_num, action="read", params=params,
            result_summary=summary,
            read_page_ids=tuple(p.id for p in pages),
        )
        tool_msg = _format_read_pages(pages, offset=offset)
        return step, tool_msg, new_pages

    def _do_follow_links(
        self, step_num: int, action_data: dict[str, Any], read_pages: dict[str, WikiPage]
    ) -> tuple[AgentStep, str, list[WikiPage]]:
        from_page_id = action_data["from_page"]
        if from_page_id not in read_pages:
            step = AgentStep(
                step=step_num, action="follow_links",
                params={"from_page": from_page_id},
                result_summary="follow_links failed: source page not read",
                error=f"page {from_page_id!r} has not been read yet",
            )
            return step, (
                f"follow_links error: page {from_page_id!r} has not been read. "
                "You can only follow links from pages you have already read."
            ), []
        source_page = read_pages[from_page_id]
        target_ids = [tid for tid in source_page.links_to if tid not in read_pages]
        if not target_ids:
            step = AgentStep(
                step=step_num, action="follow_links",
                params={"from_page": from_page_id},
                result_summary="no new links to follow",
            )
            return step, (
                f"Page {source_page.title!r} has no unread linked pages."
            ), []
        new_pages: list[WikiPage] = []
        missing: list[str] = []
        for tid in target_ids:
            target = self.index.get(tid)
            if target is not None:
                new_pages.append(target)
            else:
                missing.append(tid)
        summary = (
            f"followed {len(target_ids)} links: read {len(new_pages)} new pages, "
            f"{len(missing)} missing"
        )
        step = AgentStep(
            step=step_num, action="follow_links",
            params={"from_page": from_page_id}, result_summary=summary,
            error=f"missing target pages: {missing}" if missing else None,
            read_page_ids=tuple(p.id for p in new_pages),
        )
        tool_msg = _format_read_pages(new_pages)
        if missing:
            tool_msg += f"\n\nNote: {len(missing)} linked page(s) do not exist in the index: {missing}"
        return step, tool_msg, new_pages

    def _do_answer(
        self,
        step_num: int,
        action_data: dict[str, Any],
        read_pages: dict[str, WikiPage],
        steps: list[AgentStep],
    ) -> AgentResult:
        text = action_data.get("text", "")
        source_ids = action_data.get("sources", [])
        if not isinstance(text, str) or not text.strip():
            text = ""
        if not isinstance(source_ids, list):
            source_ids = []
        # Only count sources that were actually read.
        valid_ids = [sid for sid in source_ids if isinstance(sid, str) and sid in read_pages]
        source_paths = tuple(read_pages[sid].path for sid in valid_ids)

        if not read_pages:
            status = AgentStatus.INSUFFICIENT_EVIDENCE
            error = "answered without reading any pages"
        elif not valid_ids:
            status = AgentStatus.INSUFFICIENT_EVIDENCE
            error = "answer cited no read pages as sources"
        else:
            status = AgentStatus.ANSWERED
            error = None

        steps.append(AgentStep(
            step=step_num, action="answer",
            params={"text": text[:200], "sources": valid_ids},
            result_summary=f"answer produced ({len(valid_ids)} sources)",
            error=error,
        ))
        return AgentResult(
            answer=text if text else None,
            sources=source_paths,
            status=status,
            steps=tuple(steps),
            error=error,
            pages_read=len(read_pages),
            read_page_ids=tuple(sorted(read_pages)),
        )


# ---------------------------------------------------------------------------
# Parsing & formatting helpers
# ---------------------------------------------------------------------------


def _parse_action(raw: str) -> dict[str, Any]:
    """Parse model output into a validated action dict.

    Raises AgentProtocolError for any parsing or validation failure.

    A model occasionally appends prose or a second block after the action
    object (observed with real endpoints). The first complete JSON object is
    the action; anything after it is ignored, while output with no complete
    leading object still fails as a protocol error.
    """
    text = _strip_code_fence(raw).strip()
    if not text:
        raise AgentProtocolError("model returned empty content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        data = _decode_leading_object(text)
        if data is None:
            raise AgentProtocolError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentProtocolError(f"output must be a JSON object, got {type(data).__name__}")

    action = data.get("action")
    if not isinstance(action, str) or action not in _VALID_ACTIONS:
        raise AgentProtocolError(
            f"invalid action {action!r}; must be one of {sorted(_VALID_ACTIONS)}"
        )

    if action == "search":
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise AgentProtocolError("search action requires non-empty 'query'")
        data["query"] = query.strip()

    elif action == "read":
        paths = data.get("paths")
        if not isinstance(paths, list) or not paths:
            raise AgentProtocolError("read action requires non-empty 'paths' list")
        for i, p in enumerate(paths):
            if not isinstance(p, str) or not p.strip():
                raise AgentProtocolError(f"read paths[{i}] must be a non-empty string")
        offset = data.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AgentProtocolError(
                "read 'offset' must be a non-negative integer"
            )
        data["paths"] = [p.strip() for p in paths]

    elif action == "follow_links":
        from_page = data.get("from_page")
        if not isinstance(from_page, str) or not from_page.strip():
            raise AgentProtocolError("follow_links action requires non-empty 'from_page'")
        data["from_page"] = from_page.strip()

    elif action == "answer":
        text = data.get("text")
        if not isinstance(text, str):
            raise AgentProtocolError("answer action requires 'text' string")
        sources = data.get("sources", [])
        if not isinstance(sources, list):
            raise AgentProtocolError("answer 'sources' must be a list")
        data["text"] = text
        data["sources"] = sources

    return data


def _action_key(action: str, data: dict[str, Any]) -> str:
    """Deterministic hash key for duplicate-action detection."""
    if action == "search":
        return f"search:{data['query'].lower()}"
    if action == "read":
        return (
            f"read:{data.get('offset', 0)}:"
            f"{','.join(sorted(p.lower() for p in data['paths']))}"
        )
    if action == "follow_links":
        return f"follow_links:{data['from_page']}"
    return "answer"


def _format_search_results(results: list[WikiSearchResult]) -> str:
    if not results:
        return "Search returned no results."
    lines = ["Search results (candidates only — read pages to see content):"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"  {i}. [{r.page_id}] {r.title} (path: {r.path})"
            f" — score={r.score:.2f}"
        )
        if r.description:
            lines.append(f"     {r.description[:120]}")
    return "\n".join(lines)


def _format_read_pages(
    pages: list[WikiPage], *, offset: int = 0
) -> str:
    if not pages:
        return "No pages were read."
    parts = []
    for p in pages:
        total = len(p.content)
        start = min(offset, total)
        end = min(total, start + _READ_CONTENT_CAP)
        content = p.content[start:end]
        if start >= total:
            content = f"[offset {offset} is at or past the end of this page]"
        elif end < total:
            content += (
                f"\n... [chars {start}-{end} of {total}; "
                f"use offset={end} to continue]"
            )
        aliases = f", aliases: {list(p.aliases)}" if p.aliases else ""
        tags = f", tags: {list(p.tags)}" if p.tags else ""
        links_to = f", links_to: {list(p.links_to)}" if p.links_to else ""
        parts.append(
            f"--- Page [{p.id}] {p.title} (path: {p.path}){aliases}{tags}{links_to} ---\n"
            f"{content}"
        )
    return "\n\n".join(parts)


def _protocol_error_msg(exc: AgentProtocolError) -> str:
    return (
        f"Error: {exc}\n"
        "Output ONLY a valid JSON action object with one of these actions: "
        f"{sorted(_VALID_ACTIONS)}. "
        "Do not include markdown fencing or extra text."
    )


def _decode_leading_object(text: str) -> dict[str, Any] | None:
    """Decode one leading JSON object when followed by non-JSON prose.

    Multiple adjacent JSON values are deliberately NOT accepted: that output
    is ambiguous and stays a protocol error.
    """
    start = text.find("{")
    if start == -1:
        return None
    try:
        data, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    trailing = text[start + end:].lstrip()
    if trailing.startswith(("{", "[", '"')):
        return None
    return data if isinstance(data, dict) else None


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _extract_content(response: HttpResponse) -> str:
    """Extract choices[0].message.content from an OpenAI-compatible response."""
    if not 200 <= response.status < 300:
        detail = response.body.decode("utf-8", errors="replace")
        raise RuntimeError(f"model returned HTTP {response.status}: {detail[:500]}")
    try:
        text = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"response body is not valid UTF-8: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"response body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"response JSON must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("response has no non-empty 'choices' list")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("choices[0] has no 'message' object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("choices[0].message.content must be a non-empty str")
    return content


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value
