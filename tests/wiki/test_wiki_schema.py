"""WikiPage data model unit tests.

Covers construction, defaults, immutability, aliasing, serialization
round-trip, validation failures, and regression guard for existing models.
"""

import json

import pytest

from zmwiki import Chunk, Document
from zmwiki.wiki.wiki_schema import WikiPage


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_minimal_page():
    page = WikiPage(id="p1", path="topic/p1", title="Page One", content="body")
    assert page.id == "p1"
    assert page.path == "topic/p1"
    assert page.title == "Page One"
    assert page.content == "body"
    assert page.aliases == ()
    assert page.tags == ()
    assert page.description is None
    assert page.sources == ()
    assert page.links_to == ()
    assert page.links_from == ()
    assert page.metadata == {}


def test_full_page():
    page = WikiPage(
        id="btree-overview",
        path="databases/btree/overview",
        title="B-Tree Overview",
        aliases=["BTree", "b-tree"],
        tags=["database", "index"],
        description="A balanced tree data structure for indexes.",
        content="A B-tree keeps keys sorted and balanced...",
        sources=["doc_a", "doc_b"],
        links_to=["btree-insertion", "btree-deletion"],
        links_from=["database-indexes"],
        metadata={"compiled_at": "2026-10-21", "confidence": 0.9},
    )
    assert page.id == "btree-overview"
    assert page.aliases == ("BTree", "b-tree")
    assert page.tags == ("database", "index")
    assert page.description == "A balanced tree data structure for indexes."
    assert page.sources == ("doc_a", "doc_b")
    assert page.links_to == ("btree-insertion", "btree-deletion")
    assert page.links_from == ("database-indexes",)
    assert page.metadata == {"compiled_at": "2026-10-21", "confidence": 0.9}


def test_sequences_normalized_to_tuple():
    """List inputs are normalized to immutable tuples."""
    page = WikiPage(
        id="p", path="p", title="t", content="c",
        aliases=["a", "b"], tags=["t1"], sources=["d1"],
        links_to=["p2"], links_from=["p3"],
    )
    assert isinstance(page.aliases, tuple)
    assert isinstance(page.tags, tuple)
    assert isinstance(page.sources, tuple)
    assert isinstance(page.links_to, tuple)
    assert isinstance(page.links_from, tuple)


def test_generator_accepted_for_sequences():
    page = WikiPage(
        id="p", path="p", title="t", content="c",
        aliases=(x for x in ["a", "b"]),
    )
    assert page.aliases == ("a", "b")


# ---------------------------------------------------------------------------
# Immutability & no aliasing
# ---------------------------------------------------------------------------


def test_frozen_prevents_assignment():
    page = WikiPage(id="p", path="p", title="t", content="c")
    with pytest.raises(Exception):  # FrozenInstanceError -> AttributeError
        page.content = "changed"  # type: ignore[misc]


def test_metadata_copied_not_aliased():
    bag = {"k": "v"}
    page = WikiPage(id="p", path="p", title="t", content="c", metadata=bag)
    bag["k"] = "mutated"
    assert page.metadata == {"k": "v"}


def test_sequence_inputs_not_aliased():
    """Mutating the caller's list after construction does not affect the page."""
    aliases = ["a", "b"]
    page = WikiPage(id="p", path="p", title="t", content="c", aliases=aliases)
    aliases.append("c")
    assert page.aliases == ("a", "b")


def test_independent_pages_do_not_share_defaults():
    """Two pages get independent mutable containers; empty tuples are
    interned by Python (immutable, no pollution risk) so we check non-empty
    sequences and the mutable metadata dict instead."""
    a = WikiPage(id="a", path="a", title="A", content="ca",
                  aliases=["a1"], tags=["t1"], metadata={"k": "v"})
    b = WikiPage(id="b", path="b", title="B", content="cb",
                  aliases=["b1"], tags=["t2"], metadata={"k": "w"})
    # Mutable metadata dict must be independent.
    assert a.metadata is not b.metadata
    # Non-empty tuple fields are distinct objects per instance.
    assert a.aliases is not b.aliases
    assert a.tags is not b.tags
    # Values are correct and not cross-contaminated.
    assert a.aliases == ("a1",)
    assert b.aliases == ("b1",)
    assert a.metadata == {"k": "v"}
    assert b.metadata == {"k": "w"}


def test_equality_by_value():
    a = WikiPage(id="p", path="p", title="t", content="c", tags=["x"])
    b = WikiPage(id="p", path="p", title="t", content="c", tags=["x"])
    assert a == b
    c = WikiPage(id="p", path="p", title="t", content="c", tags=["y"])
    assert a != c


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_to_dict_returns_json_compatible():
    page = WikiPage(
        id="p", path="p", title="t", content="c",
        aliases=["a"], tags=["t1"], description="d",
        sources=["d1"], links_to=["p2"], links_from=["p3"],
        metadata={"nested": {"list": [1, 2, 3]}},
    )
    data = page.to_dict()
    # Must be JSON-serializable.
    serialized = json.dumps(data)
    restored = json.loads(serialized)
    assert restored == data


def test_round_trip_minimal():
    original = WikiPage(id="p", path="p", title="t", content="c")
    restored = WikiPage.from_dict(original.to_dict())
    assert restored == original
    assert restored.aliases == ()
    assert restored.metadata == {}


def test_round_trip_full():
    original = WikiPage(
        id="btree",
        path="databases/btree",
        title="B-Tree",
        aliases=["BTree", "b-tree"],
        tags=["database", "index", "data-structure"],
        description="Balanced tree index structure.",
        content="A B-tree keeps keys sorted...",
        sources=["doc_a", "doc_c"],
        links_to=["btree-insert", "btree-delete", "btree-search"],
        links_from=["index-types"],
        metadata={"compiled_at": "2026-10-21", "confidence": 0.95, "nested": {"a": 1}},
    )
    restored = WikiPage.from_dict(original.to_dict())
    assert restored == original
    assert restored.aliases == original.aliases
    assert restored.sources == original.sources
    assert restored.links_to == original.links_to
    assert restored.metadata == original.metadata


def test_round_trip_nested_metadata():
    """Nested dict/list values in metadata survive serialization."""
    original = WikiPage(
        id="p", path="p", title="t", content="c",
        metadata={"sections": [{"name": "intro", "length": 100}, {"name": "body"}]},
    )
    restored = WikiPage.from_dict(original.to_dict())
    assert restored.metadata == {
        "sections": [{"name": "intro", "length": 100}, {"name": "body"}]
    }


def test_from_dict_accepts_lists_for_sequence_fields():
    """to_dict emits lists; from_dict must accept them."""
    data = {
        "id": "p", "path": "p", "title": "t", "content": "c",
        "aliases": ["a", "b"], "tags": ["t1"], "sources": ["d1"],
        "links_to": ["p2"], "links_from": ["p3"], "metadata": {},
    }
    page = WikiPage.from_dict(data)
    assert page.aliases == ("a", "b")
    assert page.sources == ("d1",)


def test_from_dict_optional_fields_default_when_absent():
    data = {"id": "p", "path": "p", "title": "t", "content": "c"}
    page = WikiPage.from_dict(data)
    assert page.description is None
    assert page.aliases == ()
    assert page.metadata == {}


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_missing_required_fields_rejected():
    for field_name in ("id", "path", "title", "content"):
        kwargs = dict(id="p", path="p", title="t", content="c")
        del kwargs[field_name]
        with pytest.raises(TypeError):
            WikiPage(**kwargs)  # type: ignore[call-arg]


def test_empty_id_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="", path="p", title="t", content="c")
    with pytest.raises(TypeError):
        WikiPage(id="   ", path="p", title="t", content="c")


def test_non_str_id_rejected():
    with pytest.raises(TypeError):
        WikiPage(id=123, path="p", title="t", content="c")  # type: ignore[arg-type]


def test_id_with_newline_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="page\nid", path="p", title="t", content="c")
    with pytest.raises(TypeError):
        WikiPage(id="page\rid", path="p", title="t", content="c")


def test_empty_path_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="", title="t", content="c")


def test_empty_title_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="", content="c")


def test_empty_content_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="t", content="")


def test_empty_description_rejected_when_provided():
    """description may be None, but if provided as a string it must be non-empty."""
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="t", content="c", description="")


def test_description_none_allowed():
    page = WikiPage(id="p", path="p", title="t", content="c", description=None)
    assert page.description is None


def test_bare_str_rejected_for_sequence_fields():
    """A single str is iterable but semantically wrong for sequence fields."""
    for field_name in ("aliases", "tags", "sources", "links_to", "links_from"):
        with pytest.raises(TypeError):
            WikiPage(id="p", path="p", title="t", content="c", **{field_name: "not-a-list"})  # type: ignore[arg-type]


def test_empty_string_in_sequence_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="t", content="c", aliases=["ok", ""])


def test_non_iterable_sequence_rejected():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="t", content="c", aliases=42)  # type: ignore[arg-type]


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        WikiPage(id="p", path="p", title="t", content="c", metadata=[("a", 1)])  # type: ignore[arg-type]


def test_from_dict_non_dict_rejected():
    with pytest.raises(TypeError):
        WikiPage.from_dict("not a dict")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        WikiPage.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_from_dict_missing_required_field_rejected():
    for field_name in ("id", "path", "title", "content"):
        data = dict(id="p", path="p", title="t", content="c")
        del data[field_name]
        with pytest.raises(TypeError, match=field_name):
            WikiPage.from_dict(data)


# ---------------------------------------------------------------------------
# Regression guard: existing models unaffected
# ---------------------------------------------------------------------------


def test_document_contract_unchanged():
    """WikiPage addition must not alter Document behavior."""
    doc = Document(id="d1", content="hello", metadata={"k": "v"})
    assert doc.id == "d1"
    assert doc.content == "hello"
    assert doc.metadata == {"k": "v"}
    assert doc.source is None
    assert doc.content_type == "text/plain"
    # Frozen
    with pytest.raises(Exception):
        doc.content = "changed"  # type: ignore[misc]


def test_chunk_contract_unchanged():
    """WikiPage addition must not alter Chunk behavior."""
    chunk = Chunk(id="d1::0", document_id="d1", text="body", position=0)
    assert chunk.id == "d1::0"
    assert chunk.document_id == "d1"
    assert chunk.text == "body"
    assert chunk.position == 0
    assert chunk.metadata == {}
    with pytest.raises(Exception):
        chunk.text = "changed"  # type: ignore[misc]


def test_wikipage_not_in_top_level_exports():
    """M1-A constraint: WikiPage must NOT be in zmwiki.__init__ yet."""
    import zmwiki

    assert not hasattr(zmwiki, "WikiPage")
    assert "WikiPage" not in getattr(zmwiki, "__all__", [])
