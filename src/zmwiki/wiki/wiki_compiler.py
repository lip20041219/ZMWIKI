"""ZMWiki minimal compilation engine.

WikiCompiler takes an existing Document and produces one structured WikiPage
by calling an OpenAI-compatible chat completions endpoint through an injectable
Transport. The model is asked for a single JSON object; the response is
parsed, validated, and converted into a WikiPage with a stable id/path and
the source document id preserved in ``sources``.

Design constraints (M2):
  - Single-document compilation only. No batch framework.
  - No second model call for link resolution. Candidate links from the model
    are stored in metadata, never written to WikiPage.links_to, so a freshly
    compiled page never carries a broken link.
  - Reuses the existing Transport / HttpRequest / HttpResponse boundary and
    the same configuration shape as OpenAICompatibleGenerator.
  - Stdlib only. No new runtime dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..llm_generators import (
    GeneratorConfigError,
    GeneratorResponseError,
    GeneratorTransportError,
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
from ..schema import Document, _nonempty_str
from .wiki_schema import WikiPage

__all__ = [
    "WikiCompiler",
    "WikiCompilerError",
    "CompilerTransportError",
    "CompilerResponseError",
    "CompilerValidationError",
    "CompiledPage",
]

_CHAT_COMPLETIONS_PATH = "/chat/completions"

_SYSTEM_PROMPT = (
    "You are a wiki compiler. Read the document and produce exactly one "
    "structured wiki page summarizing its key knowledge.\n"
    "Return ONLY a JSON object with these fields:\n"
    '- title (string, required): concise page title\n'
    '- aliases (array of strings, optional): alternative names or abbreviations\n'
    '- tags (array of strings, optional): categorical topic tags\n'
    '- description (string, optional): one-sentence summary of the page\n'
    '- content (string, required): full page body in plain text or markdown\n'
    '- links_to (array of strings, optional): titles of other wiki pages this '
    'page conceptually references\n'
    "Do not include any text, explanation, or markdown fencing outside the JSON object."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WikiCompilerError(Exception):
    """Base class for every failure raised by WikiCompiler."""


class CompilerTransportError(WikiCompilerError):
    """The HTTP exchange never completed (network, DNS, timeout)."""


class CompilerResponseError(WikiCompilerError):
    """The model answered but the response cannot be used (non-2xx, bad JSON,
    missing content)."""


class CompilerValidationError(WikiCompilerError):
    """The model returned parseable JSON but it fails field validation
    (missing required fields, wrong types, empty title/content)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledPage:
    """A compilation result: the WikiPage plus diagnostic metadata.

    Attributes
    ----------
    page: the validated WikiPage, ready for WikiIndex.upsert().
    candidate_links: link titles the model suggested but were NOT written to
        page.links_to (because their targets cannot be verified at compile
        time). Stored here and in page.metadata for downstream link
        resolution.
    raw_response: the raw model content string, for debugging.
    """

    page: WikiPage
    candidate_links: tuple[str, ...]
    raw_response: str


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class WikiCompiler:
    """Compile a single Document into one WikiPage via an LLM.

    Parameters mirror OpenAICompatibleGenerator so the same endpoint config
    works for both generation and compilation.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, document: Document) -> CompiledPage:
        """Compile one Document into one WikiPage.

        Raises
        ------
        TypeError: if document is not a Document.
        CompilerTransportError: if the HTTP request fails.
        CompilerResponseError: if the model response is unusable.
        CompilerValidationError: if the JSON lacks required fields or has
            wrong types.
        """
        if not isinstance(document, Document):
            raise TypeError(
                f"document must be a Document, got {type(document).__name__}"
            )

        request = self._build_request(document)
        try:
            response = self.transport(request)
        except Exception as exc:  # transport may raise any exception
            raise CompilerTransportError(
                f"compilation request failed for document {document.id!r}: {exc}"
            ) from exc

        if not isinstance(response, HttpResponse):
            raise CompilerResponseError(
                "transport must return an HttpResponse, got "
                f"{type(response).__name__}"
            )

        raw = self._extract_content(response)
        data = self._parse_json(raw)
        page, candidate_links = self._build_page(document, data)
        return CompiledPage(
            page=page,
            candidate_links=tuple(candidate_links),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(self, document: Document) -> HttpRequest:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._user_prompt(document)},
            ],
            "temperature": self.temperature,
        }
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return HttpRequest(
            url=f"{self.base_url}{_CHAT_COMPLETIONS_PATH}",
            method="POST",
            headers=headers,
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout,
        )

    @staticmethod
    def _user_prompt(document: Document) -> str:
        return (
            f"Document ID: {document.id}\n"
            f"Document content:\n{document.content}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(response: HttpResponse) -> str:
        """Pull choices[0].message.content from an OpenAI-compatible response."""
        if not 200 <= response.status < 300:
            detail = response.body.decode("utf-8", errors="replace")
            raise CompilerResponseError(
                f"model returned HTTP {response.status}: {_brief(detail)}"
            )
        try:
            text = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CompilerResponseError(
                f"response body is not valid UTF-8: {exc}"
            ) from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CompilerResponseError(
                f"response body is not valid JSON: {_brief(text)}"
            ) from exc
        if not isinstance(payload, dict):
            raise CompilerResponseError(
                f"response JSON must be an object, got {type(payload).__name__}"
            )
        if payload.get("error") is not None:
            raise CompilerResponseError(
                f"model error payload: {_brief(json.dumps(payload['error']))}"
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise CompilerResponseError("response has no non-empty 'choices' list")
        first = choices[0]
        if not isinstance(first, dict):
            raise CompilerResponseError("choices[0] must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise CompilerResponseError("choices[0] has no 'message' object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CompilerResponseError(
                "choices[0].message.content must be a non-empty str, got "
                f"{type(content).__name__}"
            )
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse model content as JSON, stripping markdown code fences."""
        text = _strip_code_fence(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CompilerValidationError(
                f"model output is not valid JSON: {exc}. "
                f"Raw output: {_brief(text)}"
            ) from exc
        if not isinstance(data, dict):
            raise CompilerValidationError(
                f"model JSON must be an object, got {type(data).__name__}"
            )
        return data

    # ------------------------------------------------------------------
    # Page construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_page(
        cls, document: Document, data: dict[str, Any]
    ) -> tuple[WikiPage, list[str]]:
        """Validate model JSON and construct a WikiPage.

        Returns (page, candidate_links). candidate_links are stored in
        metadata but NOT written to page.links_to (targets unverified).
        """
        title = _require_str_field(data, "title")
        content = _require_str_field(data, "content")

        aliases = _optional_str_list(data, "aliases")
        tags = _optional_str_list(data, "tags")
        description = _optional_str(data, "description")
        candidate_links = _optional_str_list(data, "links_to")

        page_id = cls._generate_page_id(document)
        path = cls._generate_path(title, document)

        metadata: dict[str, Any] = {}
        if candidate_links:
            metadata["candidate_links"] = list(candidate_links)
        metadata["source_document_id"] = document.id

        page = WikiPage(
            id=page_id,
            path=path,
            title=title,
            content=content,
            aliases=tuple(aliases),
            tags=tuple(tags),
            description=description,
            sources=(document.id,),
            links_to=(),
            links_from=(),
            metadata=metadata,
        )
        return page, candidate_links

    @staticmethod
    def _generate_page_id(document: Document) -> str:
        """Stable, repeatable page id derived from the document id.

        Format: ``page-{document.id}``. Same document + same code -> same id
        on every run. No randomness, no timestamp.
        """
        return f"page-{document.id}"

    @staticmethod
    def _generate_path(title: str, document: Document) -> str:
        """Stable page path: slugified title, falling back to document id."""
        slug = _slugify(title)
        if slug:
            return slug
        # Title slugified to empty (e.g. all punctuation): fall back.
        return f"page-{document.id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_str_field(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CompilerValidationError(
            f"field {key!r} must be a non-empty string, got {value!r}"
        )
    return value.strip()


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CompilerValidationError(
            f"field {key!r} must be a string or null, got {type(value).__name__}"
        )
    stripped = value.strip()
    return stripped if stripped else None


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise CompilerValidationError(
            f"field {key!r} must be an array of strings, got {type(value).__name__}"
        )
    result: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CompilerValidationError(
                f"field {key!r}[{i}] must be a non-empty string, got {item!r}"
            )
        result.append(item.strip())
    return result


def _slugify(text: str) -> str:
    """Convert text to a path-safe slug.

    Lowercase; keep alphanumerics and CJK characters; replace everything else
    with hyphens; collapse consecutive hyphens; strip leading/trailing.
    """
    chars: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fencing if present."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _brief(text: str, limit: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}... [{len(collapsed)} chars total]"
