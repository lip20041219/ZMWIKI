"""Chunk schema unit tests."""

import pytest

from zmrag import Chunk, Document


def test_minimal_chunk():
    chunk = Chunk(id="c1", document_id="d1", text="text", position=0)
    assert chunk.id == "c1"
    assert chunk.document_id == "d1"
    assert chunk.text == "text"
    assert chunk.position == 0
    assert chunk.metadata == {}


def test_full_chunk():
    chunk = Chunk(
        id="c2",
        document_id="d2",
        text="full chunk",
        position=3,
        metadata={"page": 2, "filename": "a.txt"},
    )
    assert chunk.id == "c2"
    assert chunk.document_id == "d2"
    assert chunk.position == 3
    assert chunk.metadata == {"page": 2, "filename": "a.txt"}


def test_document_chunk_link_by_id_only():
    # Chunk references Document by id; it must not need the Document instance.
    doc = Document(id="d1", content="hello world")
    chunk = Chunk(id="c1", document_id=doc.id, text="hello", position=0)
    assert chunk.document_id == doc.id
    assert not hasattr(chunk, "document")


def test_empty_document_id_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="", text="t", position=0)


def test_empty_chunk_id_rejected():
    with pytest.raises(TypeError):
        Chunk(id="", document_id="d", text="t", position=0)
    with pytest.raises(TypeError):
        Chunk(id="  ", document_id="d", text="t", position=0)


def test_non_str_ids_rejected():
    with pytest.raises(TypeError):
        Chunk(id=1, document_id="d", text="t", position=0)
    with pytest.raises(TypeError):
        Chunk(id="c", document_id=None, text="t", position=0)  # type: ignore[arg-type]


def test_empty_text_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="", position=0)


def test_whitespace_text_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="   \n\t", position=0)


def test_non_str_text_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text=42, position=0)  # type: ignore[arg-type]


def test_negative_position_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="t", position=-1)


def test_non_int_position_rejected():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="t", position="0")  # type: ignore[arg-type]


def test_bool_position_rejected():
    # True is an int subclass but a nonsensical position.
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="t", position=True)  # type: ignore[arg-type]


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        Chunk(id="c", document_id="d", text="t", position=0, metadata="nope")


def test_metadata_is_copied_not_aliased():
    bag = {"page": 1}
    chunk = Chunk(id="c", document_id="d", text="t", position=0, metadata=bag)
    bag["page"] = 99
    assert chunk.metadata == {"page": 1}


def test_frozen_prevents_assignment():
    chunk = Chunk(id="c", document_id="d", text="t", position=0)
    with pytest.raises(Exception):
        chunk.text = "changed"  # type: ignore[misc]


def test_equality_by_value():
    a = Chunk(id="c", document_id="d", text="t", position=1, metadata={"k": 1})
    b = Chunk(id="c", document_id="d", text="t", position=1, metadata={"k": 1})
    assert a == b
    c = Chunk(id="c", document_id="d", text="t", position=2, metadata={"k": 1})
    assert a != c


def test_position_orders_chunks():
    chunks = [
        Chunk(id="c0", document_id="d", text="first", position=0),
        Chunk(id="c1", document_id="d", text="second", position=1),
    ]
    assert [c.position for c in chunks] == [0, 1]
    # Sorted positions reproduce original document order deterministically.
    assert sorted(chunks, key=lambda c: c.position) == chunks


def test_public_import():
    from zmrag import __all__

    assert {"Chunk", "Document", "Source", "Metadata"} <= set(__all__)
    assert Chunk.__name__ == "Chunk"
