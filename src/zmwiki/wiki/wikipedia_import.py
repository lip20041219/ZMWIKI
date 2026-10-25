"""Import a fetched Wikipedia article into the local Wiki index.

This module sits between the network connector (:mod:`zmwiki.wiki.wikipedia`)
and the in-memory index (:class:`zmwiki.wiki.wiki_index.WikiIndex`). It
converts one ``WikipediaArticle`` into a ``WikiPage``, adds that page to an
index without ever creating a duplicate or clobbering a page the user may
have edited, and reconciles local links from the article's Wikipedia links.

It does **not** fetch, save, or schedule anything. Persistence stays with the
caller, who passes the index to ``zmwiki.web.workspace.save_workspace`` --
that function already refuses to overwrite an existing file unless
``overwrite=True`` is passed, so this module has no save flag of its own::

    article = WikipediaConnector().read_article(page_id=11675)
    result = import_wikipedia_article(index, article)
    save_workspace(index, "workspace.json")            # refuses to overwrite
    save_workspace(index, "workspace.json", overwrite=True)

Provenance is kept twice on purpose: the article URL goes into
``WikiPage.sources`` and the same URL plus the Wikipedia page id and revision
id go into ``metadata``. Nothing from the source is pasted into ``content``.

Identity: the page id is ``wikipedia-<page_id>`` and the path is
``wikipedia/<page_id>``. Both derive from Wikipedia's own page id, so the same
article always maps to the same page and two similarly titled articles never
collide. Duplicate detection reads the stored ``wikipedia_page_id`` metadata,
not the title; an existing page is returned untouched (it may carry user
edits), and the caller is told it was already there.

Links: fetched internal targets are stored in ``metadata["wikipedia_links"]``
(title + page id) and, as deterministic local ids, in
``metadata["candidate_links"]``. The existing resolver
(:func:`zmwiki.wiki.wiki_link_resolver.resolve_candidate_links`) turns those
into bidirectionally consistent ``links_to`` / ``links_from``; targets that
are not imported stay candidates and never become pages. On a duplicate
import, only this link metadata is refreshed -- title, content and other
fields are kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from zmwiki.wiki.wikipedia import WikipediaArticle, WikipediaLink
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import resolve_candidate_links
from zmwiki.wiki.wiki_schema import WikiPage

__all__ = [
    "ImportResult",
    "WikipediaImportError",
    "article_to_page",
    "import_wikipedia_article",
]

#: Metadata key marking a page as imported rather than hand-authored.
_SOURCE_MARKER = "wikipedia"


class WikipediaImportError(Exception):
    """An article cannot be imported without conflicting with an existing page."""


@dataclass(frozen=True)
class ImportResult:
    """Outcome of one import.

    ``created`` is False when the article was already present; ``page`` is then
    the existing page, exactly as stored (including any user edits).

    ``links_resolved`` counts this article's link targets that already exist in
    the local index. ``links_updated`` counts pages whose link fields changed
    during reconciliation (including pages that gained a backlink).
    """

    page: WikiPage
    created: bool
    links_resolved: int = 0
    links_updated: int = 0


def article_to_page(
    article: WikipediaArticle, *, links: Iterable[WikipediaLink] | None = None
) -> WikiPage:
    """Convert one fetched article into a WikiPage. Pure, no side effects.

    ``links`` is the article's internal link list, when it was fetched. Targets
    are stored by Wikipedia page id; each also becomes a ``candidate_links``
    entry pointing at the deterministic local id ``wikipedia-<page_id>``, which
    the existing link resolver matches exactly (never by title). Targets that
    are not imported stay candidates -- no page is fabricated for them.
    """
    if not isinstance(article, WikipediaArticle):
        raise TypeError(
            f"article must be a WikipediaArticle, got {type(article).__name__}"
        )
    if isinstance(article.page_id, bool) or not isinstance(article.page_id, int):
        raise ValueError("article must carry a Wikipedia page id")
    if article.page_id <= 0:
        raise ValueError(f"article page id must be positive, got {article.page_id}")
    if article.revision_id is not None:
        if isinstance(article.revision_id, bool) or not isinstance(
            article.revision_id, int
        ):
            raise TypeError("article revision id must be an int or None")

    title = _nonempty(article.title, "article title")
    content = _nonempty(article.text, "article text")
    url = _nonempty(article.url, "article url")

    metadata = {
        "source": _SOURCE_MARKER,
        "wikipedia_page_id": article.page_id,
        "wikipedia_revision_id": article.revision_id,
        "wikipedia_url": url,
    }
    link_metadata = _link_metadata(article.page_id, links)
    if link_metadata is not None:
        metadata["wikipedia_links"], metadata["candidate_links"] = link_metadata

    return WikiPage(
        id=f"wikipedia-{article.page_id}",
        path=f"wikipedia/{article.page_id}",
        title=title,
        content=content,
        sources=(url,),
        metadata=metadata,
    )


def import_wikipedia_article(
    index: WikiIndex,
    article: WikipediaArticle,
    *,
    links: Iterable[WikipediaLink] | None = None,
) -> ImportResult:
    """Add ``article`` to ``index`` unless the same Wikipedia page is present.

    Duplicate policy: skip and return the existing page (``created=False``).
    An existing page is never overwritten -- content, title and user edits are
    kept; only the stored Wikipedia link metadata is refreshed when ``links``
    is provided. Same-title articles with different page ids are distinct.

    After either path the existing link resolver reconciles the whole index
    from ``candidate_links``: the new article links to already-imported
    targets, and those pages gain the matching backlink. All other page fields
    are preserved.
    """
    if not isinstance(index, WikiIndex):
        raise TypeError(f"index must be a WikiIndex, got {type(index).__name__}")
    page = article_to_page(article, links=links)

    existing = _find_imported(index, article.page_id)
    created = existing is None
    if existing is not None:
        if links is not None:
            metadata = dict(existing.metadata)
            metadata["wikipedia_links"] = page.metadata["wikipedia_links"]
            metadata["candidate_links"] = page.metadata["candidate_links"]
            existing = replace(existing, metadata=metadata)
            index.upsert(existing)
    else:
        occupant = index.get(page.id)
        if occupant is not None:
            raise WikipediaImportError(
                f"page id {page.id!r} is already used by a page that did not "
                "come from this Wikipedia article"
            )
        if index.get_by_path(page.path) is not None:
            raise WikipediaImportError(
                f"page path {page.path!r} is already used by a different page"
            )
        index.upsert(page)
        existing = page

    links_resolved, links_updated = _resolve_links(index)
    current = index.get(existing.id)
    own_targets = _stored_link_ids(current)
    local_targets = _local_wikipedia_ids(index)
    return ImportResult(
        page=current,
        created=created,
        links_resolved=sum(
            1
            for target in own_targets
            if target != article.page_id and target in local_targets
        ),
        links_updated=links_updated,
    )


def _find_imported(index: WikiIndex, wikipedia_page_id: int) -> WikiPage | None:
    for page in index.list_pages():
        if _wikipedia_page_id(page) == wikipedia_page_id:
            return page
    return None


def _link_metadata(
    source_page_id: int, links: Iterable[WikipediaLink] | None
) -> tuple[list[dict], list[str]] | None:
    """Return (wikipedia_links, candidate_links), or None when not fetched."""
    if links is None:
        return None
    if isinstance(links, (str, bytes)):
        raise TypeError("links must be an iterable of WikipediaLink, not a str")
    by_page_id: dict[int, WikipediaLink] = {}
    for link in links:
        if not isinstance(link, WikipediaLink):
            raise TypeError(
                f"links must contain WikipediaLink, got {type(link).__name__}"
            )
        if isinstance(link.page_id, bool) or not isinstance(link.page_id, int):
            raise TypeError("link page id must be an int")
        if link.page_id <= 0:
            raise ValueError(f"link page id must be positive, got {link.page_id}")
        _nonempty(link.title, "link title")
        if link.page_id == source_page_id:
            continue
        by_page_id.setdefault(link.page_id, link)

    ordered = sorted(
        by_page_id.values(), key=lambda link: (link.title.lower(), link.page_id)
    )
    return (
        [{"page_id": link.page_id, "title": link.title} for link in ordered],
        [f"wikipedia-{link.page_id}" for link in ordered],
    )


def _resolve_links(index: WikiIndex) -> tuple[int, int]:
    """Reconcile the index's links; return (resolved candidates, pages updated)."""
    pages = index.list_pages()
    resolved, report = resolve_candidate_links(pages)
    updated = 0
    for page in resolved:
        if index.get(page.id) is not page:
            index.upsert(page)
            updated += 1
    return report.resolved_count, updated


def _stored_link_ids(page: WikiPage) -> set[int]:
    stored = page.metadata.get("wikipedia_links")
    if not isinstance(stored, list):
        return set()
    ids: set[int] = set()
    for item in stored:
        if isinstance(item, dict):
            value = item.get("page_id")
            if isinstance(value, int) and not isinstance(value, bool):
                ids.add(value)
    return ids


def _local_wikipedia_ids(index: WikiIndex) -> set[int]:
    return {
        page_id
        for page in index.list_pages()
        if (page_id := _wikipedia_page_id(page)) is not None
    }


def _wikipedia_page_id(page: WikiPage) -> int | None:
    if page.metadata.get("source") != _SOURCE_MARKER:
        return None
    value = page.metadata.get("wikipedia_page_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text
