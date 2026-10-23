"""ZMWiki search and read interfaces.

Provides wiki_search (metadata + content BM25 fusion with exact-match
boosting) and wiki_read (batch path-based page lookup). Both operate on a
WikiIndex instance and never modify it.

Design notes
------------
- wiki_search returns WikiSearchResult objects containing metadata only --
  never the full page content. Callers use wiki_read to fetch content.
- Scoring is a weighted fusion of metadata BM25 and content BM25, plus
  discrete bonuses for exact title/alias token overlap. Weights and bonuses
  are configurable with documented defaults; sorting is always deterministic
  (-score, then page_id).
- wiki_read preserves input order and raises KeyError listing every missing
  path -- it never silently drops unknown paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .wiki_index import WikiIndex
from .wiki_schema import WikiPage

__all__ = ["WikiSearchResult", "wiki_search", "wiki_read"]


# Default scoring parameters. These are documented defaults, not claims of
# universal optimality -- tests prove specific ordering behaviours.
_DEFAULT_METADATA_WEIGHT = 2.0
_DEFAULT_CONTENT_WEIGHT = 1.0
_DEFAULT_EXACT_TITLE_BONUS = 10.0
_DEFAULT_EXACT_ALIAS_BONUS = 5.0


@dataclass(frozen=True)
class WikiSearchResult:
    """A candidate page returned by wiki_search.

    Contains metadata and score only -- intentionally excludes ``content``.
    Use wiki_read to fetch the full page.
    """

    page_id: str
    path: str
    title: str
    aliases: tuple[str, ...]
    tags: tuple[str, ...]
    description: str | None
    score: float


def wiki_search(
    index: WikiIndex,
    query: str,
    *,
    top_k: int = 5,
    metadata_weight: float = _DEFAULT_METADATA_WEIGHT,
    content_weight: float = _DEFAULT_CONTENT_WEIGHT,
    exact_title_bonus: float = _DEFAULT_EXACT_TITLE_BONUS,
    exact_alias_bonus: float = _DEFAULT_EXACT_ALIAS_BONUS,
) -> list[WikiSearchResult]:
    """Search the wiki index for pages relevant to ``query``.

    Strategy
    --------
    1. BM25 search over metadata (title + aliases + tags + description).
    2. BM25 search over page content.
    3. Weighted fusion: metadata_score * metadata_weight + content_score *
       content_weight.
    4. Exact-match bonuses: if any query token appears exactly in the page
       title, add exact_title_bonus; similarly for aliases.
    5. Deterministic sort by (-score, page_id), then truncate to top_k.

    Parameters
    ----------
    index: the WikiIndex to search.
    query: search query string. Empty or whitespace-only queries return [].
    top_k: maximum number of results. Must be a positive int.
    metadata_weight, content_weight: BM25 fusion weights.
    exact_title_bonus, exact_alias_bonus: discrete bonuses for exact token
        overlap with title / aliases.

    Returns
    -------
    list[WikiSearchResult] sorted by relevance. Empty list for no matches or
    empty query. Never returns pages without at least one positive score
    component.
    """
    if not isinstance(index, WikiIndex):
        raise TypeError(
            f"index must be a WikiIndex, got {type(index).__name__}"
        )
    if not isinstance(query, str):
        raise TypeError(f"query must be a str, got {type(query).__name__}")
    _check_top_k(top_k)
    _check_weight(metadata_weight, "metadata_weight")
    _check_weight(content_weight, "content_weight")
    _check_weight(exact_title_bonus, "exact_title_bonus")
    _check_weight(exact_alias_bonus, "exact_alias_bonus")

    if not query.strip() or index.count() == 0:
        return []

    # Fetch raw BM25 results from both indexes. Use a wider candidate pool
    # than top_k so fusion doesn't prematurely cut pages that score well on
    # only one index.
    candidate_k = max(top_k * 3, 10)
    meta_raw = index.search_metadata(query, top_k=candidate_k)
    content_raw = index.search_content(query, top_k=candidate_k)

    # Weighted fusion.
    scores: dict[str, float] = {}
    for page_id, score in meta_raw:
        scores[page_id] = scores.get(page_id, 0.0) + score * metadata_weight
    for page_id, score in content_raw:
        scores[page_id] = scores.get(page_id, 0.0) + score * content_weight

    # Exact-match bonuses.
    query_terms = set(index.tokenizer.tokenize(query))
    if query_terms:
        for page_id in list(scores.keys()):
            page = index.get(page_id)
            if page is None:
                continue
            title_terms = set(index.tokenizer.tokenize(page.title))
            if query_terms & title_terms:
                scores[page_id] += exact_title_bonus
            alias_terms: set[str] = set()
            for alias in page.aliases:
                alias_terms.update(index.tokenizer.tokenize(alias))
            if query_terms & alias_terms:
                scores[page_id] += exact_alias_bonus

    # Build results and sort deterministically.
    results: list[WikiSearchResult] = []
    for page_id, score in scores.items():
        page = index.get(page_id)
        if page is None:
            continue
        results.append(
            WikiSearchResult(
                page_id=page.id,
                path=page.path,
                title=page.title,
                aliases=page.aliases,
                tags=page.tags,
                description=page.description,
                score=score,
            )
        )
    results.sort(key=lambda r: (-r.score, r.page_id))
    return results[:top_k]


def wiki_read(
    index: WikiIndex,
    paths: str | list[str] | tuple[str, ...],
) -> list[WikiPage]:
    """Read one or more pages by path, in input order.

    Parameters
    ----------
    index: the WikiIndex to read from.
    paths: a single path string or a list/tuple of path strings. Each path
        is normalized before lookup (case-insensitive, slash-collapsed).

    Returns
    -------
    list[WikiPage] in the same order as ``paths``. Full page objects
    (including content, sources, links).

    Raises
    ------
    TypeError: if index is not a WikiIndex, paths is not a str/list/tuple,
        or any individual path is not a str.
    KeyError: if any path does not exist in the index. The error message
        lists every missing path. No partial results are returned.
    """
    if not isinstance(index, WikiIndex):
        raise TypeError(
            f"index must be a WikiIndex, got {type(index).__name__}"
        )
    if isinstance(paths, str):
        path_list = [paths]
    elif isinstance(paths, (list, tuple)):
        path_list = list(paths)
    else:
        raise TypeError(
            f"paths must be a str or list/tuple of str, got {type(paths).__name__}"
        )

    results: list[WikiPage] = []
    missing: list[str] = []
    for p in path_list:
        if not isinstance(p, str):
            raise TypeError(
                f"each path must be a str, got {type(p).__name__} for {p!r}"
            )
        page = index.get_by_path(p)
        if page is None:
            missing.append(p)
        else:
            results.append(page)

    if missing:
        raise KeyError(f"paths not found in wiki index: {missing}")
    return results


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def _check_top_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"top_k must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"top_k must be >= 1, got {value}")
    return value


def _check_weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value
