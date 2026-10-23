"""ZMWiki candidate link resolution and bidirectional link construction.

Takes a set of already-compiled WikiPages, resolves the candidate links
stored in each page's ``metadata["candidate_links"]`` against the existing
page set, and produces new WikiPage objects with validated, bidirectionally
consistent ``links_to`` / ``links_from``.

Design constraints (M4.1):
  - Deterministic exact matching only (page_id, normalized path, title,
    aliases). No fuzzy matching, no LLM calls.
  - Three outcomes per candidate: uniquely resolved, unresolved (no match),
    ambiguous (multiple matches). Ambiguous candidates are never silently
    assigned to one target.
  - Bidirectional consistency: if A links to B, B must link from A.
  - Idempotent: re-running on the same page set produces the same links.
  - Original pages are never modified (WikiPage is frozen; we use
    dataclasses.replace to produce new objects).
  - Stdlib only. No new runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .wiki_index import _normalize_path
from .wiki_schema import WikiPage

__all__ = [
    "LinkResolutionReport",
    "ResolvedLink",
    "resolve_candidate_links",
]


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedLink:
    """A successfully resolved candidate link."""

    source_page_id: str
    candidate: str
    target_page_id: str
    match_method: str  # "page_id" | "path" | "title" | "alias"


@dataclass(frozen=True)
class LinkResolutionReport:
    """Diagnostic report from a candidate link resolution pass.

    All counts are over candidate links (not pages). A single page may
    contribute multiple candidates.
    """

    pages_processed: int
    candidates_total: int
    resolved_count: int
    unresolved_count: int
    ambiguous_count: int
    self_link_skipped: int
    broken_existing_links: int
    inconsistent_existing_links: int
    resolved_links: tuple[ResolvedLink, ...]
    unresolved_candidates: tuple[tuple[str, str], ...]  # (page_id, candidate)
    ambiguous_candidates: tuple[tuple[str, str, tuple[str, ...]], ...]  # (page_id, candidate, matched_ids)
    broken_existing: tuple[tuple[str, str], ...]  # (source, target)
    inconsistent_existing: tuple[tuple[str, str], ...]  # (page_a, page_b)

    @property
    def is_clean(self) -> bool:
        """True when no unresolved, ambiguous, broken, or inconsistent items."""
        return (
            self.unresolved_count == 0
            and self.ambiguous_count == 0
            and self.broken_existing_links == 0
            and self.inconsistent_existing_links == 0
        )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _build_match_index(pages: list[WikiPage]) -> dict[str, dict[str, list[str]]]:
    """Build lookup tables for candidate matching.

    Returns a dict with keys: "page_id", "path", "title", "alias".
    Each maps a normalized key -> list of page_ids (multiple pages may share
    a title or alias, which is how ambiguity is detected).
    """
    by_id: dict[str, list[str]] = {}
    by_path: dict[str, list[str]] = {}
    by_title: dict[str, list[str]] = {}
    by_alias: dict[str, list[str]] = {}

    for page in pages:
        by_id.setdefault(page.id, []).append(page.id)
        norm_path = _normalize_path(page.path)
        if norm_path:
            by_path.setdefault(norm_path, []).append(page.id)
        title_key = page.title.strip().lower()
        if title_key:
            by_title.setdefault(title_key, []).append(page.id)
        for alias in page.aliases:
            alias_key = alias.strip().lower()
            if alias_key:
                by_alias.setdefault(alias_key, []).append(page.id)

    return {"page_id": by_id, "path": by_path, "title": by_title, "alias": by_alias}


def _match_candidate(
    candidate: str, match_index: dict[str, dict[str, list[str]]]
) -> tuple[str | None, str | None]:
    """Match a single candidate string against the lookup tables.

    Returns (target_page_id, match_method) if exactly one page matches across
    all methods, or (None, None) if zero or multiple pages match.

    Matching collects results from ALL methods (page_id, path, title, alias)
    and accepts only when the union has exactly one page.
    """
    candidate = candidate.strip()
    if not candidate:
        return None, None

    matches: dict[str, str] = {}  # page_id -> first method that matched

    def _collect(key: str, table: dict[str, list[str]], method: str) -> None:
        for pid in table.get(key, []):
            matches.setdefault(pid, method)

    # 1. page_id exact match
    _collect(candidate, match_index["page_id"], "page_id")
    # 2. normalized path match
    norm = _normalize_path(candidate)
    if norm:
        _collect(norm, match_index["path"], "path")
    # 3. title exact match (case-insensitive)
    title_key = candidate.lower()
    _collect(title_key, match_index["title"], "title")
    # 4. alias exact match (case-insensitive)
    _collect(title_key, match_index["alias"], "alias")

    if len(matches) == 1:
        target_id, method = next(iter(matches.items()))
        return target_id, method
    return None, None


def _find_all_matches(
    candidate: str, match_index: dict[str, dict[str, list[str]]]
) -> list[str]:
    """Return ALL page_ids that match a candidate (for ambiguity reporting)."""
    candidate = candidate.strip()
    if not candidate:
        return []
    matched: set[str] = set()
    matched.update(match_index["page_id"].get(candidate, []))
    norm = _normalize_path(candidate)
    if norm:
        matched.update(match_index["path"].get(norm, []))
    title_key = candidate.lower()
    matched.update(match_index["title"].get(title_key, []))
    matched.update(match_index["alias"].get(title_key, []))
    return sorted(matched)


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------


def resolve_candidate_links(pages: list[WikiPage]) -> tuple[list[WikiPage], LinkResolutionReport]:
    """Resolve candidate links and build bidirectionally consistent pages.

    Parameters
    ----------
    pages : list[WikiPage]
        The set of compiled pages. Each page may carry
        ``metadata["candidate_links"]`` (a list of str).

    Returns
    -------
    (resolved_pages, report)
        resolved_pages: new WikiPage objects with validated links_to /
            links_from. Original pages are unmodified.
        report: LinkResolutionReport with full diagnostics.

    The function is idempotent: calling it again on the output produces
    the same link set (candidate_links remain in metadata but re-matching
    yields the same results, and duplicate links are not added).
    """
    if not isinstance(pages, list):
        raise TypeError(f"pages must be a list, got {type(pages).__name__}")
    for p in pages:
        if not isinstance(p, WikiPage):
            raise TypeError(f"each page must be a WikiPage, got {type(p).__name__}")

    page_ids = {p.id for p in pages}
    match_index = _build_match_index(pages)

    # --- Phase 1: resolve candidates ---
    resolved_links: list[ResolvedLink] = []
    unresolved: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str, tuple[str, ...]]] = []
    self_skipped = 0

    for page in pages:
        candidates = page.metadata.get("candidate_links", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            target_id, method = _match_candidate(candidate, match_index)
            if target_id is not None:
                if target_id == page.id:
                    self_skipped += 1
                    continue
                resolved_links.append(ResolvedLink(
                    source_page_id=page.id,
                    candidate=candidate,
                    target_page_id=target_id,
                    match_method=method,
                ))
            else:
                all_matches = _find_all_matches(candidate, match_index)
                if len(all_matches) > 1:
                    ambiguous.append((page.id, candidate, tuple(all_matches)))
                else:
                    unresolved.append((page.id, candidate))

    # --- Phase 2: build link graph from existing + resolved links ---
    # links_to_map: source_page_id -> set of target_page_ids
    # links_from_map: target_page_id -> set of source_page_ids
    links_to_map: dict[str, set[str]] = {pid: set() for pid in page_ids}
    links_from_map: dict[str, set[str]] = {pid: set() for pid in page_ids}
    broken_existing: list[tuple[str, str]] = []
    inconsistent_existing: set[tuple[str, str]] = set()

    # 2a. Process existing links (preserve valid ones, report broken/inconsistent).
    for page in pages:
        for target in page.links_to:
            if target not in page_ids:
                broken_existing.append((page.id, target))
            else:
                links_to_map[page.id].add(target)
        for source in page.links_from:
            if source not in page_ids:
                broken_existing.append((source, page.id))
            else:
                links_from_map[page.id].add(source)

    # Detect existing inconsistencies (A→B but B doesn't have A in links_from).
    for page in pages:
        for target in page.links_to:
            if target in page_ids and page.id not in links_from_map.get(target, set()):
                inconsistent_existing.add(tuple(sorted((page.id, target))))  # type: ignore[arg-type]
        for source in page.links_from:
            if source in page_ids and page.id not in links_to_map.get(source, set()):
                inconsistent_existing.add(tuple(sorted((page.id, source))))  # type: ignore[arg-type]

    # 2b. Add resolved candidate links.
    for rl in resolved_links:
        links_to_map[rl.source_page_id].add(rl.target_page_id)
        links_from_map[rl.target_page_id].add(rl.source_page_id)

    # 2c. Ensure bidirectional consistency for ALL links in the graph.
    # (This also fixes existing inconsistencies by adding the missing side.)
    for source_id, targets in links_to_map.items():
        for target_id in targets:
            links_from_map[target_id].add(source_id)
    for target_id, sources in links_from_map.items():
        for source_id in sources:
            links_to_map[source_id].add(target_id)

    # --- Phase 3: produce new page objects ---
    resolved_pages: list[WikiPage] = []
    for page in pages:
        new_links_to = tuple(sorted(links_to_map.get(page.id, set())))
        new_links_from = tuple(sorted(links_from_map.get(page.id, set())))
        # Only create a new object if links actually changed.
        if new_links_to == page.links_to and new_links_from == page.links_from:
            resolved_pages.append(page)
        else:
            resolved_pages.append(replace(
                page, links_to=new_links_to, links_from=new_links_from
            ))

    # --- Phase 4: build report ---
    report = LinkResolutionReport(
        pages_processed=len(pages),
        candidates_total=len(resolved_links) + len(unresolved) + len(ambiguous) + self_skipped,
        resolved_count=len(resolved_links),
        unresolved_count=len(unresolved),
        ambiguous_count=len(ambiguous),
        self_link_skipped=self_skipped,
        broken_existing_links=len(broken_existing),
        inconsistent_existing_links=len(inconsistent_existing),
        resolved_links=tuple(resolved_links),
        unresolved_candidates=tuple(unresolved),
        ambiguous_candidates=tuple(ambiguous),
        broken_existing=tuple(sorted(broken_existing)),
        inconsistent_existing=tuple(sorted(inconsistent_existing)),
    )

    return resolved_pages, report
