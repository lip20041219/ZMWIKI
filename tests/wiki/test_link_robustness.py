"""Phase 11: link-resolution robustness against a hand-built ground truth.

Scope and honesty:
  * Every page, candidate and expected edge here is SYNTHETIC. Nothing in
    this file is real Wikipedia data, and no network or LLM call is made.
    Results validate the resolver's exact-matching contract only.
  * Ground truth is an explicit directed edge set. ``links_to`` carries the
    resolved directed edges; ``links_from`` is the matching reverse index
    (which pages link here), not an invented reverse edge.
  * Precision/recall are measured over UNIQUE (source, target) edges.
    ``LinkResolutionReport`` counters are measured over candidate
    occurrences, so one page can contribute several occurrences of the same
    edge (e.g. matched by both path and alias); both are asserted.

Metric definitions used below:
  precision = |resolved ∩ expected| / |resolved|            (0 if no resolved)
  recall    = |resolved ∩ expected| / |expected|            (0 if no expected)
  wrong     = |resolved - expected|                         (false positives)
  unresolved / ambiguous are taken from the resolver report.
"""

import pytest

from zmwiki.wiki.wikipedia import WikipediaArticle, WikipediaLink
from zmwiki.wiki.wikipedia_import import import_wikipedia_article
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import resolve_candidate_links
from zmwiki.wiki.wiki_schema import WikiPage

# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------

#: Directed candidate edges the resolver SHOULD produce.
EXPECTED_EDGES = frozenset({
    ("alice-smith", "bob-jones"),          # by title "Bob Jones"
    ("bob-jones", "alice-smith"),          # by path AND by alias (2 occurrences)
    ("project-phoenix", "alice-smith"),    # by page_id
    ("mercury-element", "mercury-planet"), # by unique title
    ("mercury-planet", "mercury"),         # by unique title (disambiguation page)
    ("mercury", "mercury-element"),        # by unique title
    ("java", "project-phoenix"),           # by page_id
    ("java-island", "java"),               # by path
})

#: One deliberately mislinked candidate: the compiler claimed Alice links to
#: Mercury's disambiguation page. Exact matching cannot know it is wrong, so
#: it resolves and must show up as a precision loss, not be hidden.
INJECTED_WRONG_EDGES = frozenset({("alice-smith", "mercury")})

EXPECTED_UNRESOLVED = frozenset({
    ("alice-smith", "Mars"),
    ("bob-jones", "wikipedia-999"),
})

EXPECTED_AMBIGUOUS = frozenset({
    ("bob-jones", "Java"),   # title "Java" + alias "Java" on another page
    ("alice-smith", "Ceres") # two pages share the title "Ceres"
})


def _page(page_id, path, title, aliases=(), candidates=()):
    return WikiPage(
        id=page_id,
        path=path,
        title=title,
        content=f"Body of {title}.",
        aliases=tuple(aliases),
        metadata={"candidate_links": list(candidates)},
    )


def _synthetic_pages():
    return [
        _page("alice-smith", "people/alice-smith", "Alice Smith",
              aliases=("A. Smith",),
              candidates=("Bob Jones", "Mercury", "Ceres", "Mars")),
        _page("bob-jones", "people/bob-jones", "Bob Jones",
              candidates=("people/alice-smith", "A. Smith", "Java",
                          "wikipedia-999")),
        _page("project-phoenix", "projects/phoenix", "Project Phoenix",
              aliases=("Phoenix Project",), candidates=("alice-smith",)),
        _page("mercury", "science/mercury", "Mercury",
              candidates=("Mercury", "Mercury (element)")),  # first is self
        _page("mercury-element", "science/mercury-element", "Mercury (element)",
              candidates=("Mercury (planet)",)),
        _page("mercury-planet", "science/mercury-planet", "Mercury (planet)",
              candidates=("Mercury",)),
        _page("java", "tech/java", "Java", candidates=("project-phoenix",)),
        _page("java-island", "geo/java", "Java Island", aliases=("Java",),
              candidates=("tech/java",)),
        _page("ceres-astronomy", "science/ceres-astronomy", "Ceres"),
        _page("ceres-mythology", "myth/ceres", "Ceres"),
    ]


def _final_edges(pages):
    """All directed edges carried by links_to."""
    return {(page.id, target) for page in pages for target in page.links_to}


def _metrics(report, expected):
    resolved = {(rl.source_page_id, rl.target_page_id) for rl in report.resolved_links}
    correct = resolved & set(expected)
    return {
        "resolved_unique": len(resolved),
        "correct": len(correct),
        "wrong": len(resolved - set(expected)),
        "precision": len(correct) / len(resolved) if resolved else 0.0,
        "recall": len(correct) / len(expected) if expected else 0.0,
    }


# ---------------------------------------------------------------------------
# Resolver: correctness against explicit ground truth
# ---------------------------------------------------------------------------


def test_synthetic_graph_matches_ground_truth_and_metrics():
    resolved, report = resolve_candidate_links(_synthetic_pages())

    # Exact final graph: expected edges plus the injected mislink.
    all_resolved_edges = set(EXPECTED_EDGES) | set(INJECTED_WRONG_EDGES)
    assert _final_edges(resolved) == all_resolved_edges

    # Report sets are exact, not just counts.
    assert report.resolved_count == 10  # occurrences: bob->alice twice
    assert set(report.unresolved_candidates) == EXPECTED_UNRESOLVED
    assert {(p, c) for p, c, _ids in report.ambiguous_candidates} == EXPECTED_AMBIGUOUS
    assert report.self_link_skipped == 1
    assert report.candidates_total == 15
    assert report.broken_existing_links == 0
    assert report.inconsistent_existing_links == 0
    assert report.is_clean is False  # unresolved + ambiguous remain

    # Ambiguous candidates never produce an edge; direction is not invented.
    edges = _final_edges(resolved)
    assert ("bob-jones", "java") not in edges
    assert ("bob-jones", "java-island") not in edges
    assert not any(target.startswith("ceres") for _, target in edges)
    assert ("mercury", "alice-smith") not in edges  # only the backlink exists

    metrics = _metrics(report, EXPECTED_EDGES)
    assert metrics["resolved_unique"] == 9
    assert metrics["correct"] == 8
    assert metrics["wrong"] == 1
    assert metrics["precision"] == pytest.approx(8 / 9)
    assert metrics["recall"] == 1.0

    # Graph is internally consistent and has no dangling targets.
    index = WikiIndex()
    for page in resolved:
        index.upsert(page)
    assert index.check_links().is_clean


def test_resolution_is_idempotent_and_order_independent():
    once, report_once = resolve_candidate_links(_synthetic_pages())
    twice, report_twice = resolve_candidate_links(list(once))

    assert _final_edges(twice) == _final_edges(once)
    for field in ("candidates_total", "resolved_count", "unresolved_count",
                  "ambiguous_count", "self_link_skipped"):
        assert getattr(report_twice, field) == getattr(report_once, field)

    shuffled_pages, report_shuffled = resolve_candidate_links(
        list(reversed(_synthetic_pages()))
    )

    assert _final_edges(shuffled_pages) == _final_edges(once)
    assert set(report_shuffled.unresolved_candidates) == EXPECTED_UNRESOLVED
    assert {
        (p, c) for p, c, _ids in report_shuffled.ambiguous_candidates
    } == EXPECTED_AMBIGUOUS


# ---------------------------------------------------------------------------
# Wikipedia import: duplicate imports and import order
# ---------------------------------------------------------------------------


def _article(page_id, title):
    return WikipediaArticle(
        title=title,
        text=f"Body of {title}.",
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        page_id=page_id,
        revision_id=page_id * 10,
    )


A = _article(101, "Alpha")
B = _article(102, "Beta")
G = _article(103, "Gamma")

# Ground truth for the import test: A and B link each other; G links to A.
A_LINKS = (WikipediaLink("Beta", 102), WikipediaLink("Unimported", 999))
B_LINKS = (WikipediaLink("Alpha", 101),)
G_LINKS = (WikipediaLink("Alpha", 101),)


def _import_all(articles_and_links):
    index = WikiIndex()
    created = [import_wikipedia_article(index, article, links=links).created
               for article, links in articles_and_links]
    graph = {
        page.id: (page.links_to, page.links_from) for page in index.list_pages()
    }
    return index, created, graph


EXPECTED_IMPORT_GRAPH = {
    "wikipedia-101": (("wikipedia-102",), ("wikipedia-102", "wikipedia-103")),
    "wikipedia-102": (("wikipedia-101",), ("wikipedia-101",)),
    "wikipedia-103": (("wikipedia-101",), ()),
}


def test_duplicate_imports_are_order_independent_and_idempotent():
    orders = [
        [(A, A_LINKS), (B, B_LINKS), (G, G_LINKS)],
        [(G, G_LINKS), (B, B_LINKS), (A, A_LINKS)],
        [(B, B_LINKS), (G, G_LINKS), (A, A_LINKS)],
    ]
    graphs = []
    for batch in orders:
        index, created, graph = _import_all(batch)
        assert created == [True, True, True]
        assert index.check_links().is_clean
        graphs.append(graph)
    assert graphs[0] == graphs[1] == graphs[2] == EXPECTED_IMPORT_GRAPH

    # Re-import the same articles in yet another order: nothing is created,
    # duplicated, or relinked.
    index, _created, graph = _import_all(orders[0])
    for article, links in reversed(orders[0]):
        result = import_wikipedia_article(index, article, links=links)
        assert result.created is False
    assert {p.id: (p.links_to, p.links_from) for p in index.list_pages()} == EXPECTED_IMPORT_GRAPH
    assert index.count() == 3
    assert index.check_links().is_clean


def test_reimport_refreshes_candidates_but_keeps_existing_resolved_links():
    """Known limitation, pinned so a future change is deliberate.

    The import contract keeps a page's link fields on duplicate import; only
    Wikipedia link metadata is refreshed. A link resolved by an earlier pass
    therefore survives even when the new link list no longer contains it.
    The graph stays consistent and idempotent, but a stale edge cannot be
    retracted through re-import.
    """
    index = WikiIndex()
    import_wikipedia_article(index, A, links=A_LINKS)
    import_wikipedia_article(index, B, links=B_LINKS)
    assert index.get("wikipedia-101").links_to == ("wikipedia-102",)

    again = import_wikipedia_article(index, A, links=())

    assert again.created is False
    alpha = index.get("wikipedia-101")
    assert alpha.metadata["candidate_links"] == []
    assert alpha.links_to == ("wikipedia-102",)  # stale edge preserved
    assert index.get("wikipedia-102").links_from == ("wikipedia-101",)
    assert index.check_links().is_clean

    # Repeating the same re-import changes nothing (no oscillation).
    before = {p.id: (p.links_to, p.links_from) for p in index.list_pages()}
    import_wikipedia_article(index, A, links=())
    after = {p.id: (p.links_to, p.links_from) for p in index.list_pages()}
    assert after == before
