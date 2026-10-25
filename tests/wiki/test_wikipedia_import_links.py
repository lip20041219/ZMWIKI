"""Wikipedia import -> local bidirectional link association tests."""

import dataclasses

import pytest

from zmwiki.web.workspace import load_workspace, save_workspace
from zmwiki.wiki.wikipedia import WikipediaArticle, WikipediaLink
from zmwiki.wiki.wikipedia_import import (
    article_to_page,
    import_wikipedia_article,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


def _article(page_id, title):
    return WikipediaArticle(
        title=title,
        text=f"Body of {title}.",
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        page_id=page_id,
        revision_id=page_id * 10,
    )


A = _article(1, "Alpha")
B = _article(2, "Beta")
C = _article(3, "Gamma")

A_LINKS = (WikipediaLink(title="Beta", page_id=2), WikipediaLink(title="Unimported", page_id=999))
B_LINKS = (WikipediaLink(title="Alpha", page_id=1),)
C_LINKS = (WikipediaLink(title="Alpha", page_id=1),)


def _links(index) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        page.id: (page.links_to, page.links_from) for page in index.list_pages()
    }


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_two_imported_articles_get_bidirectional_links():
    index = WikiIndex()
    import_wikipedia_article(index, A, links=A_LINKS)
    assert index.get("wikipedia-1").links_to == ()

    result = import_wikipedia_article(index, B, links=B_LINKS)

    alpha = index.get("wikipedia-1")
    beta = index.get("wikipedia-2")
    assert alpha.links_to == ("wikipedia-2",)
    assert beta.links_from == ("wikipedia-1",)
    assert beta.links_to == ("wikipedia-1",)
    assert alpha.links_from == ("wikipedia-2",)
    assert result.links_resolved == 1
    assert result.links_updated >= 2


def test_import_order_does_not_change_the_final_graph():
    forward = WikiIndex()
    import_wikipedia_article(forward, A, links=A_LINKS)
    import_wikipedia_article(forward, B, links=B_LINKS)

    backward = WikiIndex()
    import_wikipedia_article(backward, B, links=B_LINKS)
    import_wikipedia_article(backward, A, links=A_LINKS)

    assert _links(forward) == _links(backward)


def test_repeated_import_and_resolution_are_idempotent():
    index = WikiIndex()
    import_wikipedia_article(index, A, links=A_LINKS)
    import_wikipedia_article(index, B, links=B_LINKS)
    before = _links(index)

    again_a = import_wikipedia_article(index, A, links=A_LINKS)
    again_b = import_wikipedia_article(index, B, links=B_LINKS)

    assert again_a.created is False
    assert again_b.created is False
    assert _links(index) == before
    assert index.count() == 2


def test_similar_titles_do_not_link_without_matching_page_id():
    index = WikiIndex()
    other = _article(42, "Mercury")
    import_wikipedia_article(index, other, links=())

    mercury_planet = _article(7, "Mercury (planet)")
    result = import_wikipedia_article(
        index,
        mercury_planet,
        links=(WikipediaLink(title="Mercury", page_id=999),),
    )

    page = index.get("wikipedia-7")
    assert page.links_to == ()
    assert index.get("wikipedia-42").links_from == ()
    assert result.links_resolved == 0


def test_unimported_targets_are_not_fabricated():
    index = WikiIndex()
    result = import_wikipedia_article(index, A, links=A_LINKS)

    assert index.count() == 1
    assert index.get("wikipedia-999") is None
    assert index.get_by_path("wikipedia/999") is None
    assert result.links_resolved == 0

    page = index.get("wikipedia-1")
    stored = page.metadata["wikipedia_links"]
    assert {"page_id": 999, "title": "Unimported"} in stored
    assert "wikipedia-999" in page.metadata["candidate_links"]


def test_reimport_refreshes_links_but_keeps_user_content():
    index = WikiIndex()
    import_wikipedia_article(index, C, links=())
    first = import_wikipedia_article(index, A, links=(WikipediaLink("Beta", 2),))
    edited = dataclasses.replace(first.page, title="My Alpha", content="Hand notes.")
    index.upsert(edited)

    second = import_wikipedia_article(index, A, links=(WikipediaLink("Gamma", 3),))

    assert second.created is False
    page = index.get("wikipedia-1")
    assert page.title == "My Alpha"
    assert page.content == "Hand notes."
    assert page.links_to == ("wikipedia-3",)
    assert index.get("wikipedia-3").links_from == ("wikipedia-1",)
    stored_ids = [item["page_id"] for item in page.metadata["wikipedia_links"]]
    assert stored_ids == [3]


def test_self_links_are_dropped():
    index = WikiIndex()
    import_wikipedia_article(
        index, A, links=(WikipediaLink("Alpha", 1), WikipediaLink("Beta", 2))
    )

    page = index.get("wikipedia-1")
    assert page.links_to == ()
    assert page.metadata["wikipedia_links"] == [{"page_id": 2, "title": "Beta"}]


def test_manual_local_links_are_preserved():
    index = WikiIndex()
    index.upsert(WikiPage(id="p", path="notes/p", title="P", content="P",
                          links_to=("q",)))
    index.upsert(WikiPage(id="q", path="notes/q", title="Q", content="Q",
                          links_from=("p",)))

    import_wikipedia_article(index, A, links=A_LINKS)

    assert index.get("p").links_to == ("q",)
    assert index.get("q").links_from == ("p",)


def test_links_survive_save_and_reload(tmp_path):
    index = WikiIndex()
    import_wikipedia_article(index, A, links=A_LINKS)
    import_wikipedia_article(index, B, links=B_LINKS)
    workspace = tmp_path / "wiki.json"

    save_workspace(index, workspace)
    reloaded = load_workspace(workspace)

    assert _links(reloaded) == _links(index)
    alpha = reloaded.get("wikipedia-1")
    assert alpha.links_to == ("wikipedia-2",)
    assert alpha.metadata["wikipedia_links"] == [
        {"page_id": 2, "title": "Beta"},
        {"page_id": 999, "title": "Unimported"},
    ]


def test_import_without_links_still_imports_content():
    index = WikiIndex()

    result = import_wikipedia_article(index, A, links=None)

    assert result.created is True
    assert result.links_resolved == 0
    page = index.get("wikipedia-1")
    assert page.content == "Body of Alpha."
    assert "candidate_links" not in page.metadata


def test_invalid_link_input_is_rejected():
    with pytest.raises(TypeError):
        article_to_page(A, links=["not-a-link"])
    with pytest.raises(ValueError):
        article_to_page(A, links=(WikipediaLink("Bad", -1),))
