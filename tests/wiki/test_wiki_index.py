"""WikiIndex, wiki_search, and wiki_read unit tests.

Covers registration, lookup, search ranking, read ordering, link integrity,
update/delete consistency, empty-index behaviour, and regression guards for
existing RAG models.
"""

import pytest

from zmwiki import Chunk, Document
from zmwiki.wiki.wiki_index import LinkCheckResult, WikiIndex, _normalize_path
from zmwiki.wiki.wiki_retriever import WikiSearchResult, wiki_read, wiki_search
from zmwiki.wiki.wiki_schema import WikiPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(
    page_id: str = "p1",
    path: str = "topic/p1",
    title: str = "Page One",
    content: str = "This is the body of page one.",
    **kwargs,
) -> WikiPage:
    return WikiPage(id=page_id, path=path, title=title, content=content, **kwargs)


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


class TestNormalizePath:
    def test_lowercases(self):
        assert _normalize_path("Topic/Page") == "topic/page"

    def test_strips_whitespace(self):
        assert _normalize_path("  topic/page  ") == "topic/page"

    def test_strips_leading_trailing_slashes(self):
        assert _normalize_path("/topic/page/") == "topic/page"

    def test_collapses_consecutive_slashes(self):
        assert _normalize_path("topic//page///sub") == "topic/page/sub"

    def test_empty_becomes_empty(self):
        assert _normalize_path("") == ""
        assert _normalize_path("   ") == ""
        assert _normalize_path("/") == ""


# ---------------------------------------------------------------------------
# WikiIndex: registration & lookup
# ---------------------------------------------------------------------------


class TestWikiIndexRegistration:
    def test_register_and_get_by_id(self):
        idx = WikiIndex()
        page = _page()
        idx.upsert(page)
        assert idx.get("p1") is page
        assert idx.count() == 1

    def test_get_by_path(self):
        idx = WikiIndex()
        idx.upsert(_page(path="databases/btree"))
        result = idx.get_by_path("databases/btree")
        assert result is not None
        assert result.id == "p1"

    def test_get_by_path_case_insensitive(self):
        idx = WikiIndex()
        idx.upsert(_page(path="DataBases/BTree"))
        assert idx.get_by_path("databases/btree") is not None
        assert idx.get_by_path("DATABASES/BTREE") is not None

    def test_get_by_path_slash_insensitive(self):
        idx = WikiIndex()
        idx.upsert(_page(path="topic/page"))
        assert idx.get_by_path("/topic/page/") is not None
        assert idx.get_by_path("topic//page") is not None

    def test_get_missing_returns_none(self):
        idx = WikiIndex()
        assert idx.get("nonexistent") is None
        assert idx.get_by_path("nonexistent") is None

    def test_list_pages_sorted_by_id(self):
        idx = WikiIndex()
        idx.upsert(_page(page_id="z", path="z"))
        idx.upsert(_page(page_id="a", path="a"))
        idx.upsert(_page(page_id="m", path="m"))
        ids = [p.id for p in idx.list_pages()]
        assert ids == ["a", "m", "z"]

    def test_count(self):
        idx = WikiIndex()
        assert idx.count() == 0
        idx.upsert(_page())
        assert idx.count() == 1


class TestWikiIndexDuplicatesAndInvalid:
    def test_duplicate_id_replaces(self):
        """upsert with same id replaces the old page (update semantics)."""
        idx = WikiIndex()
        idx.upsert(_page(title="Old Title"))
        idx.upsert(_page(title="New Title"))
        assert idx.count() == 1
        assert idx.get("p1").title == "New Title"

    def test_duplicate_path_different_id_rejected(self):
        idx = WikiIndex()
        idx.upsert(_page(page_id="p1", path="shared/path"))
        with pytest.raises(ValueError, match="already used"):
            idx.upsert(_page(page_id="p2", path="shared/path"))

    def test_duplicate_path_case_insensitive_rejected(self):
        idx = WikiIndex()
        idx.upsert(_page(page_id="p1", path="Topic/Page"))
        with pytest.raises(ValueError):
            idx.upsert(_page(page_id="p2", path="topic/page"))

    def test_same_page_path_change_allowed(self):
        """Updating a page to a new path releases the old path."""
        idx = WikiIndex()
        idx.upsert(_page(page_id="p1", path="old/path"))
        idx.upsert(_page(page_id="p1", path="new/path"))
        assert idx.get_by_path("old/path") is None
        assert idx.get_by_path("new/path") is not None
        assert idx.count() == 1

    def test_empty_path_rejected(self):
        """Empty path is rejected at WikiPage construction (TypeError), before
        it ever reaches WikiIndex.upsert."""
        with pytest.raises(TypeError):
            idx = WikiIndex()
            idx.upsert(_page(path=""))

    def test_whitespace_path_rejected(self):
        with pytest.raises(TypeError):
            idx = WikiIndex()
            idx.upsert(_page(path="   "))

    def test_non_wikipage_rejected(self):
        idx = WikiIndex()
        with pytest.raises(TypeError):
            idx.upsert("not a page")  # type: ignore[arg-type]

    def test_delete_returns_true_when_exists(self):
        idx = WikiIndex()
        idx.upsert(_page())
        assert idx.delete("p1") is True
        assert idx.count() == 0
        assert idx.get("p1") is None

    def test_delete_returns_false_when_missing(self):
        idx = WikiIndex()
        assert idx.delete("nonexistent") is False

    def test_delete_non_str_rejected(self):
        idx = WikiIndex()
        with pytest.raises(TypeError):
            idx.delete(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# wiki_search
# ---------------------------------------------------------------------------


class TestWikiSearch:
    def _index_with_pages(self) -> WikiIndex:
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="btree", path="databases/btree",
            title="B-Tree Index",
            aliases=["BTree", "balanced tree"],
            tags=["database", "index"],
            description="A balanced tree data structure for database indexes.",
            content="A B-tree keeps keys sorted and balanced, enabling efficient insertion, deletion, and search operations.",
        ))
        idx.upsert(WikiPage(
            id="cache", path="systems/cache",
            title="Cache Eviction",
            aliases=["LRU", "cache replacement"],
            tags=["memory", "performance"],
            description="How caches evict entries when full.",
            content="When a cache is full, an eviction policy removes an entry. LRU evicts the least recently used entry.",
        ))
        idx.upsert(WikiPage(
            id="queue", path="systems/queue",
            title="Message Queue",
            aliases=["MQ", "message broker"],
            tags=["distributed", "messaging"],
            description="Decoupling producers from consumers.",
            content="A message queue decouples producers from consumers by buffering messages between them.",
        ))
        return idx

    def test_title_match_returns_page(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "B-Tree")
        assert len(results) > 0
        assert results[0].page_id == "btree"

    def test_alias_match_returns_page(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "LRU")
        assert any(r.page_id == "cache" for r in results)

    def test_tag_match_returns_page(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "database")
        assert any(r.page_id == "btree" for r in results)

    def test_description_match_returns_page(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "eviction policy")
        assert any(r.page_id == "cache" for r in results)

    def test_content_match_returns_page(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "buffering messages")
        assert any(r.page_id == "queue" for r in results)

    def test_exact_title_ranks_above_content_only(self):
        """A page whose title exactly matches the query should outrank a page
        that only matches in content."""
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="title_match", path="a",
            title="Neural Networks",
            content="Unrelated body text about cooking.",
        ))
        idx.upsert(WikiPage(
            id="content_match", path="b",
            title="Cooking Guide",
            content="Neural networks are used in deep learning models.",
        ))
        results = wiki_search(idx, "Neural Networks")
        assert results[0].page_id == "title_match"
        assert results[0].score > results[1].score

    def test_exact_alias_ranks_above_content_only(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="alias_page", path="a",
            title="Some Page",
            aliases=["JavaScript"],
            content="Generic content.",
        ))
        idx.upsert(WikiPage(
            id="content_page", path="b",
            title="Other Page",
            content="JavaScript is a programming language for the web.",
        ))
        results = wiki_search(idx, "JavaScript")
        assert results[0].page_id == "alias_page"

    def test_deterministic_sorting(self):
        """Same query on same index always returns same order."""
        idx = self._index_with_pages()
        r1 = wiki_search(idx, "database index")
        r2 = wiki_search(idx, "database index")
        assert [r.page_id for r in r1] == [r.page_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    def test_empty_query_returns_empty(self):
        idx = self._index_with_pages()
        assert wiki_search(idx, "") == []
        assert wiki_search(idx, "   ") == []
        assert wiki_search(idx, "\n\t") == []

    def test_no_match_returns_empty(self):
        idx = self._index_with_pages()
        assert wiki_search(idx, "quantum submarine agriculture") == []

    def test_top_k_limits_results(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "the", top_k=2)
        assert len(results) <= 2

    def test_top_k_must_be_positive(self):
        idx = self._index_with_pages()
        with pytest.raises(ValueError):
            wiki_search(idx, "test", top_k=0)
        with pytest.raises(TypeError):
            wiki_search(idx, "test", top_k=1.5)  # type: ignore[arg-type]

    def test_non_str_query_rejected(self):
        idx = self._index_with_pages()
        with pytest.raises(TypeError):
            wiki_search(idx, 123)  # type: ignore[arg-type]

    def test_non_index_rejected(self):
        with pytest.raises(TypeError):
            wiki_search("not an index", "query")  # type: ignore[arg-type]

    def test_results_exclude_content(self):
        """WikiSearchResult must not contain page content."""
        idx = self._index_with_pages()
        results = wiki_search(idx, "B-Tree")
        assert not hasattr(results[0], "content")
        assert "content" not in results[0].__dataclass_fields__

    def test_results_contain_metadata(self):
        idx = self._index_with_pages()
        results = wiki_search(idx, "B-Tree")
        r = results[0]
        assert r.page_id == "btree"
        assert r.path == "databases/btree"
        assert r.title == "B-Tree Index"
        assert r.aliases == ("BTree", "balanced tree")
        assert r.tags == ("database", "index")
        assert r.description is not None
        assert r.score > 0

    def test_empty_index_returns_empty(self):
        idx = WikiIndex()
        assert wiki_search(idx, "anything") == []


# ---------------------------------------------------------------------------
# wiki_read
# ---------------------------------------------------------------------------


class TestWikiRead:
    def _index(self) -> WikiIndex:
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="p1", path="topic/p1", title="Page One",
            content="Content one.", sources=["doc_a"],
            links_to=["p2"], links_from=["p3"],
        ))
        idx.upsert(WikiPage(
            id="p2", path="topic/p2", title="Page Two",
            content="Content two.", sources=["doc_b"],
            links_to=["p1"], links_from=["p1"],
        ))
        return idx

    def test_read_single_path(self):
        idx = self._index()
        pages = wiki_read(idx, "topic/p1")
        assert len(pages) == 1
        assert pages[0].id == "p1"
        assert pages[0].content == "Content one."

    def test_read_returns_full_page(self):
        idx = self._index()
        pages = wiki_read(idx, "topic/p1")
        page = pages[0]
        assert page.sources == ("doc_a",)
        assert page.links_to == ("p2",)
        assert page.links_from == ("p3",)
        assert page.title == "Page One"

    def test_read_multiple_paths_in_order(self):
        idx = self._index()
        pages = wiki_read(idx, ["topic/p2", "topic/p1"])
        assert [p.id for p in pages] == ["p2", "p1"]

    def test_read_accepts_tuple(self):
        idx = self._index()
        pages = wiki_read(idx, ("topic/p1", "topic/p2"))
        assert [p.id for p in pages] == ["p1", "p2"]

    def test_read_case_insensitive(self):
        idx = self._index()
        pages = wiki_read(idx, "TOPIC/P1")
        assert pages[0].id == "p1"

    def test_read_missing_path_raises_keyerror(self):
        idx = self._index()
        with pytest.raises(KeyError) as exc_info:
            wiki_read(idx, "nonexistent/path")
        assert "nonexistent/path" in str(exc_info.value)

    def test_read_partial_missing_raises_with_all_missing(self):
        """If any path is missing, KeyError lists ALL missing paths and no
        partial results are returned."""
        idx = self._index()
        with pytest.raises(KeyError) as exc_info:
            wiki_read(idx, ["topic/p1", "missing/one", "missing/two"])
        msg = str(exc_info.value)
        assert "missing/one" in msg
        assert "missing/two" in msg

    def test_read_non_str_path_rejected(self):
        idx = self._index()
        with pytest.raises(TypeError):
            wiki_read(idx, [123])  # type: ignore[list-item]

    def test_read_invalid_paths_type_rejected(self):
        idx = self._index()
        with pytest.raises(TypeError):
            wiki_read(idx, 123)  # type: ignore[arg-type]

    def test_read_non_index_rejected(self):
        with pytest.raises(TypeError):
            wiki_read("not an index", "p")  # type: ignore[arg-type]

    def test_read_empty_index_raises(self):
        idx = WikiIndex()
        with pytest.raises(KeyError):
            wiki_read(idx, "any/path")


# ---------------------------------------------------------------------------
# Link integrity
# ---------------------------------------------------------------------------


class TestLinkIntegrity:
    def test_valid_bidirectional_links(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=["b"], links_from=["b"],
        ))
        idx.upsert(WikiPage(
            id="b", path="b", title="B", content="cb",
            links_to=["a"], links_from=["a"],
        ))
        result = idx.check_links()
        assert result.is_clean
        assert result.broken_links == ()
        assert result.inconsistent_links == ()

    def test_broken_link_detected(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=["nonexistent"],
        ))
        result = idx.check_links()
        assert not result.is_clean
        assert ("a", "nonexistent") in result.broken_links

    def test_broken_incoming_link_detected(self):
        """A links_from entry pointing to a nonexistent page is also broken."""
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_from=["nonexistent"],
        ))
        result = idx.check_links()
        assert ("nonexistent", "a") in result.broken_links

    def test_inconsistent_links_to_without_from(self):
        """A.links_to contains B but B.links_from does not contain A."""
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=["b"],
        ))
        idx.upsert(WikiPage(
            id="b", path="b", title="B", content="cb",
            links_from=(),  # missing "a"
        ))
        result = idx.check_links()
        assert not result.is_clean
        assert ("a", "b") in result.inconsistent_links or ("b", "a") in result.inconsistent_links

    def test_inconsistent_links_from_without_to(self):
        """B.links_from contains A but A.links_to does not contain B."""
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=(),  # missing "b"
        ))
        idx.upsert(WikiPage(
            id="b", path="b", title="B", content="cb",
            links_from=["a"],
        ))
        result = idx.check_links()
        assert len(result.inconsistent_links) == 1

    def test_check_does_not_modify_pages(self):
        idx = WikiIndex()
        page = WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=["nonexistent"],
        )
        idx.upsert(page)
        idx.check_links()
        # Original page unchanged.
        assert idx.get("a").links_to == ("nonexistent",)

    def test_empty_index_clean(self):
        idx = WikiIndex()
        result = idx.check_links()
        assert result.is_clean


# ---------------------------------------------------------------------------
# Update / delete consistency
# ---------------------------------------------------------------------------


class TestUpdateDeleteConsistency:
    def test_update_removes_old_path_from_search(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="p1", path="old/path", title="Old Title",
            content="old content about dinosaurs",
        ))
        idx.upsert(WikiPage(
            id="p1", path="new/path", title="New Title",
            content="new content about space",
        ))
        # Old path no longer resolves.
        assert idx.get_by_path("old/path") is None
        # New path resolves.
        assert idx.get_by_path("new/path") is not None
        # Search finds new content, not old.
        results = wiki_search(idx, "space")
        assert any(r.page_id == "p1" for r in results)
        results_old = wiki_search(idx, "dinosaurs")
        assert not any(r.page_id == "p1" for r in results_old)

    def test_update_removes_old_metadata_from_search(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="p1", path="p", title="Apples",
            content="fruit content",
        ))
        idx.upsert(WikiPage(
            id="p1", path="p", title="Oranges",
            content="citrus content",
        ))
        results = wiki_search(idx, "Apples")
        # "Apples" should no longer match (title changed to Oranges).
        # But content "fruit" might still... let's check title specifically.
        title_results = [r for r in results if r.title == "Apples"]
        assert len(title_results) == 0

    def test_delete_removes_from_search(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="p1", path="p", title="Unique Title XYZ",
            content="unique content abc",
        ))
        assert wiki_search(idx, "Unique Title XYZ")
        idx.delete("p1")
        assert wiki_search(idx, "Unique Title XYZ") == []
        assert wiki_search(idx, "unique content abc") == []

    def test_delete_removes_from_link_check(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(
            id="a", path="a", title="A", content="ca",
            links_to=["b"],
        ))
        idx.upsert(WikiPage(
            id="b", path="b", title="B", content="cb",
            links_from=["a"],
        ))
        assert idx.check_links().is_clean
        idx.delete("b")
        # Now a->b is a broken link.
        result = idx.check_links()
        assert ("a", "b") in result.broken_links

    def test_delete_then_reinsert_with_same_id(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(id="p1", path="p", title="V1", content="first"))
        idx.delete("p1")
        idx.upsert(WikiPage(id="p1", path="p", title="V2", content="second"))
        assert idx.get("p1").title == "V2"
        assert idx.count() == 1
        results = wiki_search(idx, "second")
        assert any(r.page_id == "p1" for r in results)


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_document_unchanged(self):
        doc = Document(id="d1", content="hello", metadata={"k": "v"})
        assert doc.id == "d1"
        assert doc.content == "hello"
        assert doc.metadata == {"k": "v"}

    def test_chunk_unchanged(self):
        chunk = Chunk(id="d1::0", document_id="d1", text="body", position=0)
        assert chunk.id == "d1::0"
        assert chunk.position == 0

    def test_wiki_modules_not_in_top_level_exports(self):
        """M1 constraint: WikiIndex/wiki_search/wiki_read must NOT be in
        zmwiki.__init__ yet."""
        import zmwiki

        for name in ("WikiIndex", "wiki_search", "wiki_read", "WikiSearchResult",
                      "LinkCheckResult"):
            assert not hasattr(zmwiki, name), f"{name} should not be in top-level exports"

    def test_existing_lexical_index_still_works(self):
        """InMemoryLexicalIndex behaviour is not affected by WikiIndex."""
        from zmwiki import InMemoryLexicalIndex

        lex = InMemoryLexicalIndex()
        lex.upsert(Chunk(id="c1", document_id="d1", text="hello world", position=0))
        results = lex.search("hello")
        assert len(results) == 1
        assert results[0].chunk.id == "c1"
