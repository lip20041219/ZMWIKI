"""Wikipedia search and article reading over the public MediaWiki API.

This module is a network adapter, not knowledge. It fetches public Wikipedia
data and returns plain Python values; it never builds ``WikiPage`` objects,
never touches a ``WikiIndex`` or workspace, and never writes to disk. Turning
a fetched article into local knowledge is a later phase's job.

Content licensing: Wikipedia text is available under CC BY-SA 4.0 (see
https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use). Fetched text is
*not* original ZMWiki content -- every result and article carries its source
URL so attribution can be preserved if it is ever imported. Wikimedia's API
usage policy asks for a descriptive User-Agent and no aggressive scraping:
this module sends one sequential request per call, with a timeout, and no
retry loop.

Only two endpoint shapes are ever contacted, both built here from constants:
``action=query&list=search`` for search and ``action=query&prop=extracts``
for article text. Caller-supplied URLs are never fetched -- a URL argument is
parsed only for its article title, and its host must be a wikipedia.org host.

The HTTP exchange reuses the package's existing stdlib transport boundary
(:class:`zmwiki.llm_generators.HttpRequest` / ``Transport``), so tests inject
a fake transport and no test in this repository touches the network.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlencode, urlsplit

from zmwiki.llm_generators import (
    GeneratorTransportError,
    HttpRequest,
    Transport,
    urllib_transport,
)

__all__ = [
    "API_URL",
    "ARTICLE_BASE_URL",
    "DEFAULT_LIMIT",
    "DEFAULT_TIMEOUT",
    "MAX_LIMIT",
    "MAX_LINK_REQUESTS",
    "MAX_QUERY_LENGTH",
    "USER_AGENT",
    "WikipediaArticle",
    "WikipediaConnector",
    "WikipediaError",
    "WikipediaLink",
    "WikipediaNotFoundError",
    "WikipediaResponseError",
    "WikipediaSearchResult",
    "WikipediaTransportError",
]

#: English Wikipedia's Action API. A configurable language is deliberately
#: left out until a caller needs one; ``en`` keeps the contract fixed.
API_URL = "https://en.wikipedia.org/w/api.php"

#: Human-readable article URL prefix, stored on every result for attribution.
ARTICLE_BASE_URL = "https://en.wikipedia.org/wiki/"

#: Wikimedia's User-Agent policy asks for a descriptive, project-identifying
#: agent. Sent on every request; carries no credentials.
USER_AGENT = "ZMWiki/0.1 (+https://github.com/lip20041219/ZMWiki)"

#: Seconds a single request may take before it is abandoned.
DEFAULT_TIMEOUT = 10.0

#: Result count defaults and bounds.
DEFAULT_LIMIT = 5
MAX_LIMIT = 20

#: Upper bound on a search term, in characters.
MAX_QUERY_LENGTH = 300

#: Upper bound on API pages followed when listing one article's links.
#: 10 x gpllimit=max (500 for logged-in callers) is far above a normal
#: article; hitting it means the listing is incomplete and we say so.
MAX_LINK_REQUESTS = 10

#: Upper bound on how much API error text is copied into an exception.
_MAX_ERROR_DETAIL = 200

_TAG_RE = re.compile(r"<[^>]*>")


class WikipediaError(Exception):
    """Base class for every failure this connector raises."""


class WikipediaTransportError(WikipediaError):
    """The HTTP exchange never completed: unreachable host, DNS, TLS, timeout.

    The underlying transport exception is chained as ``__cause__``.
    """


class WikipediaResponseError(WikipediaError):
    """Wikipedia answered, but the answer cannot be used.

    Covers a non-2xx status, a body that is not a JSON object, and an ``error``
    payload returned by the API.
    """


class WikipediaNotFoundError(WikipediaError):
    """The requested page does not exist or has no readable text."""


@dataclass(frozen=True)
class WikipediaSearchResult:
    """One search hit, with enough information to cite it and read it later."""

    title: str
    page_id: int
    url: str
    summary: str = ""


@dataclass(frozen=True)
class WikipediaArticle:
    """One fetched article: plain text plus the provenance needed to cite it."""

    title: str
    text: str
    url: str
    page_id: int | None = None
    revision_id: int | None = None


@dataclass(frozen=True)
class WikipediaLink:
    """One internal link target: its Wikipedia page id and canonical title."""

    title: str
    page_id: int


class WikipediaConnector:
    """Searches Wikipedia and reads article text through an injectable transport."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Transport | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be a number, got {type(timeout).__name__}")
        if not timeout > 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        self.timeout = float(timeout)
        self.transport = transport if transport is not None else urllib_transport
        if not callable(self.transport):
            raise TypeError(
                f"transport must be callable, got {type(self.transport).__name__}"
            )

    def search(
        self, query: str, *, limit: int = DEFAULT_LIMIT
    ) -> list[WikipediaSearchResult]:
        """Return up to ``limit`` Wikipedia search hits for ``query``.

        No hits is a normal outcome and yields an empty list.
        """
        term = _check_query(query)
        count = _check_limit(limit)
        payload = self._get(
            {
                "list": "search",
                "srsearch": term,
                "srlimit": str(count),
                "utf8": "1",
            }
        )
        query_object = payload.get("query")
        items = query_object.get("search") if isinstance(query_object, dict) else None
        if items is None:
            return []
        if not isinstance(items, list):
            raise WikipediaResponseError("Wikipedia search response is malformed")
        return [_to_search_result(item) for item in items]

    def read_article(
        self, *, page_id: int | str | None = None, url: str | None = None
    ) -> WikipediaArticle:
        """Read one article by page id or by a wikipedia.org article URL.

        Exactly one identifier must be given. The URL is parsed locally for its
        title and is never fetched, so only wikipedia.org hosts are accepted.
        """
        if (page_id is None) == (url is None):
            raise ValueError("provide exactly one of page_id or url")

        params = {"prop": "extracts|info", "explaintext": "1", "redirects": "1"}
        if page_id is not None:
            params["pageids"] = str(_check_page_id(page_id))
        else:
            params["titles"] = _title_from_url(url)

        payload = self._get(params)
        query_object = payload.get("query")
        pages = query_object.get("pages") if isinstance(query_object, dict) else None
        if not isinstance(pages, list) or not pages:
            raise WikipediaNotFoundError("Wikipedia page not found")

        page = pages[0]
        if not isinstance(page, dict) or page.get("missing"):
            raise WikipediaNotFoundError("Wikipedia page not found")

        title = page.get("title")
        if not isinstance(title, str) or not title:
            raise WikipediaResponseError("Wikipedia page response is missing a title")
        extract = page.get("extract")
        if not isinstance(extract, str) or not extract.strip():
            raise WikipediaNotFoundError(
                f"Wikipedia page {title!r} has no readable text"
            )

        return WikipediaArticle(
            title=title,
            text=extract.strip(),
            url=_article_url(title),
            page_id=_optional_int(page.get("pageid")),
            revision_id=_optional_int(page.get("lastrevid")),
        )

    def read_links(self, page_id: int | str) -> list[WikipediaLink]:
        """Return the main-namespace articles linked from ``page_id``.

        Uses ``generator=links`` so each result already carries a page id;
        ``redirects=1`` makes MediaWiki return canonical target pages, and
        duplicates introduced by redirects collapse on page id. Continuation
        is followed until exhausted or the request budget is spent -- the
        latter raises instead of returning a silently partial link set.

        Category pages, external/interwiki links and other namespaces are
        excluded by ``gplnamespace=0`` and by filtering on ``ns``.
        """
        source_id = _check_page_id(page_id)
        links: dict[int, WikipediaLink] = {}
        continuation: dict[str, str] = {}

        for _ in range(MAX_LINK_REQUESTS):
            params = {
                "generator": "links",
                "gplnamespace": "0",
                "gpllimit": "max",
                "prop": "info",
                "pageids": str(source_id),
                "redirects": "1",
            }
            params.update(continuation)
            payload = self._get(params)
            for link in _extract_links(payload, source_id):
                links.setdefault(link.page_id, link)

            next_continue = payload.get("continue")
            if not isinstance(next_continue, dict) or not next_continue:
                return sorted(
                    links.values(), key=lambda link: (link.title.lower(), link.page_id)
                )
            continuation = {
                str(key): str(value)
                for key, value in next_continue.items()
                if isinstance(value, (str, int))
            }
            if not continuation:
                return sorted(
                    links.values(), key=lambda link: (link.title.lower(), link.page_id)
                )

        raise WikipediaResponseError(
            f"Wikipedia link listing for page {source_id} exceeded the request budget"
        )

    def _get(self, params: dict[str, str]) -> dict:
        """Send one GET to the fixed API endpoint and return the JSON object."""
        query = {"action": "query", "format": "json", "formatversion": "2"}
        query.update(params)
        request = HttpRequest(
            url=API_URL + "?" + urlencode(query),
            method="GET",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            body=b"",
            timeout=self.timeout,
        )
        try:
            response = self.transport(request)
        except GeneratorTransportError as error:
            raise WikipediaTransportError(f"request to Wikipedia failed: {error}") from error
        if not 200 <= response.status < 300:
            # The body is deliberately not included: error bodies are unbounded
            # and this message ends up in logs.
            raise WikipediaResponseError(
                f"Wikipedia API returned HTTP {response.status}"
            )
        return _parse_payload(response.body)


def _check_query(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"query must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError("query must not be empty")
    if len(text) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query must be at most {MAX_QUERY_LENGTH} characters, got {len(text)}"
        )
    return text


def _check_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"limit must be an int, got {type(value).__name__}")
    if not 1 <= value <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {value}")
    return value


def _check_page_id(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("page_id must be a positive integer, not bool")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"page_id must be positive, got {value}")
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text.isascii() or not text.isdigit() or int(text) <= 0:
            raise ValueError(f"page_id must be a positive integer, got {value!r}")
        return int(text)
    raise TypeError(f"page_id must be an int or str, got {type(value).__name__}")


def _title_from_url(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"url must be a str, got {type(value).__name__}")
    parts = urlsplit(value.strip())
    if parts.scheme != "https":
        raise ValueError("article url must use https")
    host = (parts.hostname or "").lower()
    if not (host == "wikipedia.org" or host.endswith(".wikipedia.org")):
        raise ValueError(f"article url host is not allowed: {host or '<empty>'}")
    prefix = "/wiki/"
    if not parts.path.startswith(prefix):
        raise ValueError("article url must point at /wiki/<title>")
    title = unquote(parts.path[len(prefix):]).replace("_", " ").strip()
    if not title:
        raise ValueError("article url does not contain a title")
    return title


def _article_url(title: str) -> str:
    return ARTICLE_BASE_URL + quote(title.replace(" ", "_"), safe="/")


def _to_search_result(item: object) -> WikipediaSearchResult:
    if not isinstance(item, dict):
        raise WikipediaResponseError("Wikipedia search result is malformed")
    title = item.get("title")
    page_id = item.get("pageid")
    if not isinstance(title, str) or not title:
        raise WikipediaResponseError("Wikipedia search result is missing a title")
    if isinstance(page_id, bool) or not isinstance(page_id, int):
        raise WikipediaResponseError("Wikipedia search result is missing a page id")
    snippet = item.get("snippet")
    return WikipediaSearchResult(
        title=title,
        page_id=page_id,
        url=_article_url(title),
        summary=_plain_text(snippet) if isinstance(snippet, str) else "",
    )


def _plain_text(fragment: str) -> str:
    """Turn the API's highlighted HTML snippet into readable text."""
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def _extract_links(payload: dict, source_page_id: int) -> list[WikipediaLink]:
    """Pull valid main-namespace link targets out of a generator=links page."""
    query_object = payload.get("query")
    pages = query_object.get("pages") if isinstance(query_object, dict) else None
    if pages is None:
        return []
    if not isinstance(pages, list):
        raise WikipediaResponseError("Wikipedia link response is malformed")

    links: list[WikipediaLink] = []
    for page in pages:
        if not isinstance(page, dict):
            raise WikipediaResponseError("Wikipedia link entry is malformed")
        if page.get("missing") or page.get("invalid"):
            continue
        if page.get("ns") not in (0, None):
            continue  # categories, files, templates, ...
        page_id = page.get("pageid")
        title = page.get("title")
        if isinstance(page_id, bool) or not isinstance(page_id, int) or page_id <= 0:
            continue
        if page_id == source_page_id:
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        links.append(WikipediaLink(title=title, page_id=page_id))
    return links


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_payload(body: bytes) -> dict:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WikipediaResponseError("Wikipedia API returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise WikipediaResponseError("Wikipedia API response is not a JSON object")
    if "error" in payload:
        raise WikipediaResponseError(_api_error_message(payload["error"]))
    return payload


def _api_error_message(error: object) -> str:
    if not isinstance(error, dict):
        return "Wikipedia API returned an error"
    code = error.get("code")
    info = error.get("info")
    detail = " ".join(info.split())[:_MAX_ERROR_DETAIL] if isinstance(info, str) else ""
    prefix = f"Wikipedia API error {code!r}" if isinstance(code, str) else "Wikipedia API error"
    return f"{prefix}: {detail}" if detail else prefix
