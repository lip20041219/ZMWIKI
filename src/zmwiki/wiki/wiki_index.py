"""ZMWiki in-memory page index.

WikiIndex stores registered WikiPage objects and maintains:
  - an id -> page lookup
  - a normalized-path -> page_id lookup
  - two BM25 indexes (metadata text and body text) via the existing
    InMemoryLexicalIndex, reused without modification
  - link integrity checking (broken links + bidirectional inconsistency)

Mutations (upsert/delete) rebuild the BM25 indexes automatically, matching
the existing InMemoryLexicalIndex consistency strategy. No persistence,
concurrency, or caching is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..lexical import InMemoryLexicalIndex
from ..schema import Chunk
from ..tokenizers import SimpleTokenizer, Tokenizer
from .wiki_schema import WikiPage

__all__ = ["WikiIndex", "LinkCheckResult"]


def _normalize_path(path: str) -> str:
    """Normalize a wiki page path for lookup.

    Rules: strip whitespace, lowercase, strip leading/trailing slashes,
    collapse consecutive slashes. The empty string is a valid normalized
    result (callers decide whether to reject it).
    """
    p = path.strip().lower().strip("/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


@dataclass(frozen=True)
class LinkCheckResult:
    """Result of a link integrity check over all registered pages.

    Attributes
    ----------
    broken_links: (source_page_id, target_page_id) pairs where the source
        page exists in the index but the target does not. Covers both
        links_to targets and links_from sources that are missing.
    inconsistent_links: sorted (page_a_id, page_b_id) pairs where both
        pages exist but their bidirectional link relation disagrees --
        either A.links_to contains B without B.links_from containing A,
        or vice versa. Each unordered pair is reported once.
    """

    broken_links: tuple[tuple[str, str], ...] = ()
    inconsistent_links: tuple[tuple[str, str], ...] = ()

    @property
    def is_clean(self) -> bool:
        """True when there are no broken or inconsistent links."""
        return not self.broken_links and not self.inconsistent_links


class WikiIndex:
    """In-memory index of WikiPage objects with BM25-backed search.

    Parameters
    ----------
    tokenizer: optional Tokenizer for BM25 indexing. Defaults to
        SimpleTokenizer. The same instance is shared by both the metadata
        and content BM25 indexes so tokenization stays consistent.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        if not isinstance(self._tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be a Tokenizer, got {type(self._tokenizer).__name__}"
            )
        self._pages: dict[str, WikiPage] = {}
        self._path_index: dict[str, str] = {}  # normalized_path -> page_id
        self._metadata_index = InMemoryLexicalIndex(tokenizer=self._tokenizer)
        self._content_index = InMemoryLexicalIndex(tokenizer=self._tokenizer)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def upsert(self, page: WikiPage) -> None:
        """Insert or replace a page keyed by ``page.id``.

        If a page with the same id already exists, it is replaced and all
        indexes (path, BM25) are updated consistently. If the new page's
        normalized path collides with a *different* page's path, raises
        ValueError.
        """
        if not isinstance(page, WikiPage):
            raise TypeError(
                f"page must be a WikiPage, got {type(page).__name__}"
            )
        norm_path = _normalize_path(page.path)
        if not norm_path:
            raise ValueError(
                f"page path must not be empty after normalization, got {page.path!r}"
            )

        # If updating an existing page, release its old path first.
        existing = self._pages.get(page.id)
        if existing is not None:
            old_norm = _normalize_path(existing.path)
            if old_norm in self._path_index and self._path_index[old_norm] == page.id:
                del self._path_index[old_norm]

        # Check path collision with a *different* page.
        occupant = self._path_index.get(norm_path)
        if occupant is not None and occupant != page.id:
            raise ValueError(
                f"path {page.path!r} (normalized: {norm_path!r}) is already "
                f"used by page {occupant!r}"
            )

        self._pages[page.id] = page
        self._path_index[norm_path] = page.id

        # Update BM25 indexes. InMemoryLexicalIndex.upsert rebuilds fully,
        # so stale entries for this page_id are automatically replaced.
        self._metadata_index.upsert(self._metadata_chunk(page))
        self._content_index.upsert(self._content_chunk(page))

    def delete(self, page_id: str) -> bool:
        """Remove a page by id. Returns True if one was removed."""
        if not isinstance(page_id, str):
            raise TypeError(
                f"page_id must be a str, got {type(page_id).__name__}"
            )
        page = self._pages.pop(page_id, None)
        if page is None:
            return False
        norm_path = _normalize_path(page.path)
        self._path_index.pop(norm_path, None)
        self._metadata_index.delete(page_id)
        self._content_index.delete(page_id)
        return True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, page_id: str) -> WikiPage | None:
        """Return the page for ``page_id``, or None."""
        return self._pages.get(page_id)

    def get_by_path(self, path: str) -> WikiPage | None:
        """Return the page whose normalized path matches, or None."""
        if not isinstance(path, str):
            raise TypeError(f"path must be a str, got {type(path).__name__}")
        page_id = self._path_index.get(_normalize_path(path))
        if page_id is None:
            return None
        return self._pages.get(page_id)

    def list_pages(self) -> list[WikiPage]:
        """Return all registered pages, sorted by page_id for determinism."""
        return [self._pages[pid] for pid in sorted(self._pages)]

    def count(self) -> int:
        return len(self._pages)

    # ------------------------------------------------------------------
    # BM25 raw search (consumed by wiki_retriever.wiki_search)
    # ------------------------------------------------------------------

    def search_metadata(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 search over page metadata (title + aliases + tags + description).

        Returns a list of (page_id, score) tuples sorted by (-score, page_id).
        Returns [] for empty/whitespace queries or empty index.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")
        results = self._metadata_index.search(query, top_k=top_k)
        return [(r.chunk.id, r.score) for r in results]

    def search_content(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 search over page body content.

        Returns a list of (page_id, score) tuples sorted by (-score, page_id).
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")
        results = self._content_index.search(query, top_k=top_k)
        return [(r.chunk.id, r.score) for r in results]

    # ------------------------------------------------------------------
    # Link integrity
    # ------------------------------------------------------------------

    def check_links(self) -> LinkCheckResult:
        """Check all registered pages for broken links and bidirectional
        inconsistency.

        Broken link: a links_to target or links_from source that does not
        exist in the index.

        Inconsistent link: both pages exist but A.links_to contains B while
        B.links_from does not contain A (or vice versa). Each unordered pair
        is reported once.

        The original pages are never modified by this check.
        """
        broken: set[tuple[str, str]] = set()
        inconsistent: set[tuple[str, str]] = set()

        for page_id, page in self._pages.items():
            # Check outgoing links.
            for target in page.links_to:
                if target not in self._pages:
                    broken.add((page_id, target))
                elif page_id not in self._pages[target].links_from:
                    inconsistent.add(tuple(sorted((page_id, target))))  # type: ignore[return-value]
            # Check incoming links.
            for source in page.links_from:
                if source not in self._pages:
                    broken.add((source, page_id))
                elif page_id not in self._pages[source].links_to:
                    inconsistent.add(tuple(sorted((page_id, source))))  # type: ignore[return-value]

        return LinkCheckResult(
            broken_links=tuple(sorted(broken)),
            inconsistent_links=tuple(sorted(inconsistent)),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_text(page: WikiPage) -> str:
        """Concatenate title, aliases, tags, and description into one text
        block for BM25 indexing. Spaces separate fields so tokenization
        treats them as distinct token runs."""
        parts: list[str] = [page.title]
        parts.extend(page.aliases)
        parts.extend(page.tags)
        if page.description is not None:
            parts.append(page.description)
        return " ".join(parts)

    @classmethod
    def _metadata_chunk(cls, page: WikiPage) -> Chunk:
        return Chunk(
            id=page.id,
            document_id="__wiki_metadata__",
            text=cls._metadata_text(page),
            position=0,
        )

    @staticmethod
    def _content_chunk(page: WikiPage) -> Chunk:
        return Chunk(
            id=page.id,
            document_id="__wiki_content__",
            text=page.content,
            position=0,
        )
