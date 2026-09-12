"""Real LLM generation: a provider-compatible Generator over stdlib HTTP.

The first Generator in this package that actually calls a model. It is an
*adapter*: it plugs into the ``Generator`` contract ``generators.py`` already
defines -- ``generate(query, context) -> str`` -- and changes nothing else. The
pipeline still knows no provider, the ContextBuilder still packs, the
DeterministicGenerator still exists, and citations are still projected from the
Context (``citations.py``), never generated: this module never sees, builds, or
returns a citation.

Why "OpenAI-compatible" rather than an SDK per vendor: a long list of servers
expose the same ``POST {base_url}/chat/completions`` shape (OpenAI, DeepSeek,
Groq, Together, Mistral, vLLM, llama.cpp, Ollama, LM Studio, LiteLLM, ...). One
adapter against that shape reaches all of them. Binding to any vendor SDK would
add a runtime dependency and pick a winner this package has no business picking.

Dependencies: **stdlib only** (``urllib.request`` + ``json``). ``dependencies``
stays empty -- no openai / anthropic / requests / httpx / langchain. Nothing
runs at import time, no module-level client is created, and no test in this
repository talks to a network: the HTTP call goes through an injectable
``Transport``, and the suite injects a fake one.

What this module deliberately does not do: streaming, retries/backoff,
provider-specific parameters, function calling, multi-provider routing, token
counting, prompt engineering beyond a fixed system instruction, and citation
generation. Failures are raised, never smoothed into an empty answer -- an
answer that silently says nothing is worse than an exception that says why.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from .context import Context
from .generators import Generator, build_prompt
from .schema import _nonempty_str

__all__ = [
    "GeneratorError",
    "GeneratorConfigError",
    "GeneratorTransportError",
    "GeneratorResponseError",
    "HttpRequest",
    "HttpResponse",
    "OpenAICompatibleGenerator",
    "Transport",
    "urllib_transport",
]

#: Seconds a single request may take before it is abandoned. Bounded on
#: purpose: no unbounded wait, and no retry loop to make a test flaky.
_DEFAULT_TIMEOUT = 30.0

#: Sampling temperature default. 0.0 (greedy) matches the package's
#: deterministic-first posture wherever a provider honors it.
_DEFAULT_TEMPERATURE = 0.0

#: The one environment variable consulted, and only when no ``api_key``
#: argument was given. No vendor-specific variable (OPENAI_API_KEY,
#: DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...) is read or required.
_ENV_API_KEY = "ZMRAG_API_KEY"

#: Path appended to ``base_url``. ``base_url`` is the API root -- for OpenAI
#: that is ``https://api.openai.com/v1``, not the full endpoint.
_CHAT_COMPLETIONS_PATH = "/chat/completions"

#: The whole system prompt. Deliberately one fixed sentence pair rather than a
#: template system: this adapter carries no prompt-engineering layer, and a
#: caller who wants different wording belongs behind the same ``Generator``
#: contract rather than inside this transport adapter.
_SYSTEM_INSTRUCTIONS = (
    "Answer the user's question using the provided context. "
    "If the context does not contain enough information, say that the "
    "information is insufficient rather than inventing facts."
)

#: Upper bound on how much provider text is copied into an exception message.
_MAX_ERROR_DETAIL = 500


class GeneratorError(Exception):
    """Base class for every failure this adapter raises."""


class GeneratorConfigError(GeneratorError, ValueError):
    """The adapter was given a value it cannot work with.

    A ``ValueError`` as well, so callers that already catch the builtin keep
    working. Raised at construction (bad URL, model, key, timeout, temperature,
    max_tokens) -- never after a request has been sent.
    """


class GeneratorTransportError(GeneratorError):
    """The HTTP exchange never completed: unreachable host, DNS, TLS, timeout.

    The underlying ``urllib`` / socket exception is chained as ``__cause__``,
    so the low-level reason is recoverable without every caller having to know
    about ``urllib``.
    """


class GeneratorResponseError(GeneratorError):
    """The provider answered, but the answer cannot be used.

    Covers a non-2xx status, an error payload, a body that is not valid
    UTF-8 / JSON, a payload that is not an object, missing or empty
    ``choices``, a missing ``message``, and a ``content`` that is not a string.
    """


#: Header names whose values are never rendered by a repr. A repr ends up in
#: logs, tracebacks, and debuggers -- the same places a leaked key is found.
_REDACTED_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "api-key", "x-api-key"}
)


@dataclass(frozen=True)
class HttpRequest:
    """One fully described request, ready for a transport to send.

    The transport sees nothing else: no provider SDK, no auth helper, no
    client object. ``headers`` is a plain mapping that the transport must treat
    as read-only (the same shallow-immutability trade-off ``Metadata`` makes).
    """

    url: str
    method: str
    headers: dict[str, str]
    body: bytes
    timeout: float

    def __repr__(self) -> str:
        """Redacted on purpose: this request carries the API key in a header.

        The default dataclass repr would print ``Authorization: Bearer ...``
        into any log line that touches a request, so credential-bearing headers
        are masked and the body is bounded before rendering.
        """
        headers = {
            name: ("***" if name.lower() in _REDACTED_HEADERS else value)
            for name, value in self.headers.items()
        }
        body = _brief(self.body.decode("utf-8", errors="replace"))
        return (
            f"HttpRequest(url={self.url!r}, method={self.method!r}, "
            f"headers={headers!r}, body={body!r}, timeout={self.timeout!r})"
        )


@dataclass(frozen=True)
class HttpResponse:
    """One raw response: status plus undecoded body.

    The body stays ``bytes`` so decoding and JSON parsing happen in exactly one
    place (``_parse_completion``), where they can fail loudly.
    """

    status: int
    body: bytes


#: The injectable HTTP boundary. A transport takes one request and returns one
#: response; anything else (auth, retries, proxies) is a transport's business,
#: not this adapter's.
Transport = Callable[[HttpRequest], HttpResponse]


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """Default transport: one blocking stdlib HTTP request.

    Non-2xx is *returned*, not raised: the provider usually explains itself in
    the body, and that explanation is what the caller needs to see. Only a
    genuinely failed exchange (never reached the server, DNS, TLS, timeout)
    raises :class:`GeneratorTransportError`.
    """
    http_request = urllib.request.Request(
        request.url,
        data=request.body,
        headers=dict(request.headers),
        method=request.method,
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
            return HttpResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as error:
        # HTTPError *is* a URLError (and an OSError), so it must come first.
        return HttpResponse(status=error.code, body=error.read())
    except (urllib.error.URLError, OSError) as error:
        # ponytail: one blocking attempt, no retry/backoff -- swap in a
        # retrying transport here if a provider turns out to need it.
        raise GeneratorTransportError(
            f"HTTP request to {request.url} failed: {error}"
        ) from error


class OpenAICompatibleGenerator(Generator):
    """Generator backed by an OpenAI-compatible chat completions endpoint.

    Each :meth:`generate` call builds one request from the query and the
    Context, sends it through the injected transport, and returns the model's
    message content. The Generator owns prompt construction, the HTTP shape,
    response parsing, and the translation of provider failures into Python
    exceptions; the pipeline owns none of that and never sees the wire format.

    The Context is read, never modified: ``context.text`` is embedded verbatim
    by :func:`zmrag.generators.build_prompt`, and ``context.chunks`` is not
    touched at all. This class never retrieves, reranks, packs, or cites.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = _check_base_url(base_url)
        self.model = _required_str(model, "model")
        self._api_key = _resolve_api_key(api_key)
        self.temperature = _check_temperature(temperature)
        self.max_tokens = _check_max_tokens(max_tokens)
        self.timeout = _check_timeout(timeout)
        self.transport = transport if transport is not None else urllib_transport
        if not callable(self.transport):
            raise TypeError(
                f"transport must be callable, got {type(self.transport).__name__}"
            )

    @property
    def has_api_key(self) -> bool:
        """Whether a key was supplied. Never returns the key itself."""
        return self._api_key is not None

    def __repr__(self) -> str:
        """Redacted on purpose: a repr lands in logs, tracebacks, and debuggers.

        The API key is never rendered -- only whether one is present -- so an
        accidentally logged generator cannot leak a credential.
        """
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"model={self.model!r}, temperature={self.temperature!r}, "
            f"max_tokens={self.max_tokens!r}, timeout={self.timeout!r}, "
            f"api_key={'set' if self._api_key else 'unset'})"
        )

    def generate(self, query: str, context: Context) -> str:
        """Return the model's answer for ``query`` given ``context``."""
        _nonempty_str(query, "query")
        if not isinstance(context, Context):
            raise TypeError(f"context must be a Context, got {type(context).__name__}")
        response = self.transport(self._build_request(query, context))
        if not isinstance(response, HttpResponse):
            raise TypeError(
                "transport must return an HttpResponse, got "
                f"{type(response).__name__}"
            )
        return _parse_completion(response)

    def _build_request(self, query: str, context: Context) -> HttpRequest:
        """Assemble the single request this generator knows how to send.

        The system instruction states the grounding policy at the system level,
        while ``build_prompt`` supplies the delimited evidence + question block
        that carries the same rule inline -- so a server that drops or ignores
        the system role still receives a self-contained instruction.
        """
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": build_prompt(query, context)},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            # Omitted rather than defaulted: a server's own limit is a better
            # default than a number invented here.
            payload["max_tokens"] = self.max_tokens
        headers = {
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


def _parse_completion(response: HttpResponse) -> str:
    """Pull ``choices[0].message.content`` out of one response, or say why not.

    Every failure mode is a distinct, diagnosable error message; nothing here
    degrades into an empty answer or a default string.
    """
    if not 200 <= response.status < 300:
        detail = response.body.decode("utf-8", errors="replace")
        raise GeneratorResponseError(
            f"provider returned HTTP {response.status}: {_brief(detail)}"
        )

    try:
        text = response.body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GeneratorResponseError(
            f"response body is not valid UTF-8: {error}"
        ) from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise GeneratorResponseError(
            f"response body is not valid JSON: {_brief(text)}"
        ) from error

    if not isinstance(payload, dict):
        raise GeneratorResponseError(
            f"response JSON must be an object, got {type(payload).__name__}"
        )

    error_payload = payload.get("error")
    if error_payload is not None:
        # Some servers answer 200 with an error object; that is still a failure.
        raise GeneratorResponseError(f"provider error payload: {_error_detail(error_payload)}")

    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise GeneratorResponseError("response has no 'choices' list")
    if not choices:
        raise GeneratorResponseError("response 'choices' is empty")

    first = choices[0]
    if not isinstance(first, dict):
        raise GeneratorResponseError(
            f"choices[0] must be an object, got {type(first).__name__}"
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise GeneratorResponseError("choices[0] has no 'message' object")

    content = message.get("content")
    if not isinstance(content, str):
        raise GeneratorResponseError(
            "choices[0].message.content must be a str, got "
            f"{type(content).__name__}"
        )
    return content


def _brief(text: str) -> str:
    """Bound provider text before it reaches an exception message."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_ERROR_DETAIL:
        return collapsed
    return f"{collapsed[:_MAX_ERROR_DETAIL]}... [{len(collapsed)} chars total]"


def _error_detail(payload: object) -> str:
    """Best-effort readable form of an ``error`` payload."""
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return _brief(message)
    try:
        return _brief(json.dumps(payload))
    except (TypeError, ValueError):
        return _brief(repr(payload))


def _required_str(value: object, name: str) -> str:
    """Type errors raise ``TypeError``; unusable values raise a config error."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise GeneratorConfigError(f"{name} must be a non-empty str")
    return value


def _check_base_url(base_url: object) -> str:
    """Validate the API root. The trailing slash is absorbed, not required."""
    url = _required_str(base_url, "base_url")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise GeneratorConfigError(
            f"base_url must be an absolute http(s) URL, got {base_url!r}"
        )
    return url.rstrip("/")


def _resolve_api_key(api_key: object) -> str | None:
    """Explicit argument first, then ``ZMRAG_API_KEY``, then no key at all.

    A key is read once, here, so a running generator can never pick one up
    mid-flight. ``None`` means "send no Authorization header" -- local servers
    legitimately need none, and a server that does want a key will answer 401
    with a message that reaches the caller as a response error.
    """
    if api_key is not None:
        return _required_str(api_key, "api_key")
    from_env = os.environ.get(_ENV_API_KEY)
    if from_env is not None and from_env.strip():
        return from_env
    return None


def _check_timeout(timeout: object) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a number, got {type(timeout).__name__}")
    if timeout <= 0:
        raise GeneratorConfigError(f"timeout must be > 0 seconds, got {timeout!r}")
    return float(timeout)


def _check_temperature(temperature: object) -> float:
    """Range is the OpenAI-compatible convention; no vendor-specific extras."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(
            f"temperature must be a number, got {type(temperature).__name__}"
        )
    if not 0.0 <= temperature <= 2.0:
        raise GeneratorConfigError(
            f"temperature must be in [0, 2], got {temperature!r}"
        )
    return float(temperature)


def _check_max_tokens(max_tokens: object) -> int | None:
    if max_tokens is None:
        return None
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError(
            f"max_tokens must be an int or None, got {type(max_tokens).__name__}"
        )
    if max_tokens < 1:
        raise GeneratorConfigError(f"max_tokens must be >= 1, got {max_tokens}")
    return max_tokens
