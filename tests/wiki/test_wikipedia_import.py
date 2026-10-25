"""Wikipedia -> WikiPage import tests -- no network, temp workspaces only."""

import dataclasses

import pytest

from zmwiki.web.workspace import WorkspaceError, load_workspace, save_workspace
from zmwiki.wiki.wikipedia import WikipediaArticle
from zmwiki.wiki.wikipedia_import import (
    ImportResult,
    WikipediaImportError,
    article_to_page,
    import_wikipedia_article,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage

PAGE_ID = 11675
REVISION_ID = 123456
ARTICLE_URL = "https://en.wikipedia.org/wiki/Alan_Turing"
ARTICLE_TEXT = "Alan Mathison Turing was an English mathematician and logician."


def _article(**overrides) -> WikipediaArticle:
    values = {
        "title": "Alan Turing",
        "text": ARTICLE_TEXT,
        "url": ARTICLE_URL,
        "page_id": PAGE_ID,
        "revision_id": REVISION_ID,
    }
    values.update(overrides)
    return WikipediaArticle(**values)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def test_article_converts_to_a_wiki_page():
    page = article_to_page(_article())

    assert isinstance(page, WikiPage)
    assert page.title == "Alan Turing"
    assert page.content == ARTICLE_TEXT
    assert page.id == f"wikipedia-{PAGE_ID}"
    assert page.path == f"wikipedia/{PAGE_ID}"


def test_source_url_and_revision_are_kept_out_of_the_body():
    page = article_to_page(_article())

    assert page.sources == (ARTICLE_URL,)
    assert page.metadata["source"] == "wikipedia"
    assert page.metadata["wikipedia_page_id"] == PAGE_ID
    assert page.metadata["wikipedia_revision_id"] == REVISION_ID
    assert page.metadata["wikipedia_url"] == ARTICLE_URL
    assert ARTICLE_URL not in page.content
    assert str(PAGE_ID) not in page.content
    assert str(REVISION_ID) not in page.content


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"title": "   "}, ValueError),
        ({"title": 123}, TypeError),
        ({"text": ""}, ValueError),
        ({"url": ""}, ValueError),
        ({"page_id": None}, ValueError),
        ({"page_id": -1}, ValueError),
        ({"page_id": True}, ValueError),
        ({"revision_id": "123"}, TypeError),
        ({"revision_id": True}, TypeError),
    ],
)
def test_invalid_articles_are_rejected(overrides, error):
    with pytest.raises(error):
        article_to_page(_article(**overrides))


def test_non_article_input_is_rejected():
    with pytest.raises(TypeError):
        article_to_page({"title": "Alan Turing"})


# ---------------------------------------------------------------------------
# Import and duplicate handling
# ---------------------------------------------------------------------------


def test_imported_page_is_searchable_and_readable():
    index = WikiIndex()

    result = import_wikipedia_article(index, _article())

    assert isinstance(result, ImportResult)
    assert result.created is True
    assert index.count() == 1
    assert index.get(f"wikipedia-{PAGE_ID}") == result.page
    hits = index.search_content("mathematician logician")
    assert [page_id for page_id, _ in hits] == [result.page.id]


def test_reimporting_the_same_page_id_does_not_duplicate():
    index = WikiIndex()
    first = import_wikipedia_article(index, _article())
    second = import_wikipedia_article(index, _article(title="Alan Turing (changed)"))

    assert first.created is True
    assert second.created is False
    assert second.page == first.page
    assert index.count() == 1


def test_reimport_does_not_overwrite_user_edits():
    index = WikiIndex()
    first = import_wikipedia_article(index, _article())
    edited = dataclasses.replace(first.page, content="Hand-edited notes.")
    index.upsert(edited)

    second = import_wikipedia_article(index, _article())

    assert second.created is False
    assert index.get(first.page.id).content == "Hand-edited notes."


def test_similar_titles_with_different_page_ids_are_distinct_sources():
    index = WikiIndex()

    first = import_wikipedia_article(index, _article())
    second = import_wikipedia_article(index, _article(page_id=999, revision_id=1))

    assert first.created and second.created
    assert index.count() == 2
    assert first.page.id != second.page.id


def test_conflicting_page_id_is_reported_not_overwritten():
    index = WikiIndex()
    index.upsert(
        WikiPage(
            id=f"wikipedia-{PAGE_ID}",
            path="notes/elsewhere",
            title="Local notes",
            content="Not from Wikipedia.",
        )
    )

    with pytest.raises(WikipediaImportError):
        import_wikipedia_article(index, _article())

    assert index.count() == 1
    assert index.get(f"wikipedia-{PAGE_ID}").content == "Not from Wikipedia."


def test_conflicting_path_is_reported():
    index = WikiIndex()
    index.upsert(
        WikiPage(
            id="other-page",
            path=f"wikipedia/{PAGE_ID}",
            title="Local notes",
            content="Not from Wikipedia.",
        )
    )

    with pytest.raises(WikipediaImportError):
        import_wikipedia_article(index, _article())

    assert index.count() == 1


def test_import_does_not_modify_the_article():
    index = WikiIndex()
    article = _article()
    before = dataclasses.astuple(article)

    import_wikipedia_article(index, article)

    assert dataclasses.astuple(article) == before


# ---------------------------------------------------------------------------
# Workspace save / reload (existing save_workspace semantics)
# ---------------------------------------------------------------------------


def test_saved_and_reloaded_workspace_keeps_every_field(tmp_path):
    index = WikiIndex()
    import_wikipedia_article(index, _article())
    workspace = tmp_path / "wiki.json"

    save_workspace(index, workspace)
    reloaded = load_workspace(workspace)

    page = reloaded.get(f"wikipedia-{PAGE_ID}")
    original = index.get(f"wikipedia-{PAGE_ID}")
    assert page == original
    assert page.title == "Alan Turing"
    assert page.content == ARTICLE_TEXT
    assert page.sources == (ARTICLE_URL,)
    assert page.metadata["wikipedia_page_id"] == PAGE_ID
    assert page.metadata["wikipedia_revision_id"] == REVISION_ID
    assert reloaded.search_content("mathematician")[0][0] == page.id


def test_import_does_not_save_anything_by_itself(tmp_path):
    workspace = tmp_path / "wiki.json"

    import_wikipedia_article(WikiIndex(), _article())

    assert not workspace.exists()


def test_existing_workspace_is_not_overwritten_by_default(tmp_path):
    workspace = tmp_path / "wiki.json"
    workspace.write_text('{"version": 1, "pages": []}', encoding="utf-8")
    index = WikiIndex()
    import_wikipedia_article(index, _article())

    with pytest.raises(WorkspaceError):
        save_workspace(index, workspace)

    assert workspace.read_text(encoding="utf-8") == '{"version": 1, "pages": []}'


def test_explicit_overwrite_persists_the_imported_page(tmp_path):
    workspace = tmp_path / "wiki.json"
    workspace.write_text('{"version": 1, "pages": []}', encoding="utf-8")
    index = WikiIndex()
    import_wikipedia_article(index, _article())

    save_workspace(index, workspace, overwrite=True)

    assert load_workspace(workspace).get(f"wikipedia-{PAGE_ID}") is not None


def test_save_failure_is_reported_not_swallowed(tmp_path):
    index = WikiIndex()
    import_wikipedia_article(index, _article())
    target = tmp_path / "a-directory"
    target.mkdir()

    with pytest.raises(WorkspaceError):
        save_workspace(index, target, overwrite=True)
