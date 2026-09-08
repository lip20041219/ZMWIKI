"""Document schema unit tests."""

import pytest

from zmrag import Document, Metadata, Source


class FileSource(Source):
    """Minimal concrete Source used only to exercise Document typing."""

    @property
    def kind(self) -> str:
        return "file"


def test_minimal_document():
    doc = Document(id="d1", content="hello")
    assert doc.id == "d1"
    assert doc.content == "hello"
    assert doc.metadata == {}
    assert doc.source is None
    assert doc.content_type == "text/plain"


def test_full_document():
    doc = Document(
        id="d2",
        content="full body",
        metadata={"filename": "a.txt", "lang": "en"},
        source=FileSource(),
        content_type="text/markdown",
    )
    assert doc.id == "d2"
    assert doc.metadata == {"filename": "a.txt", "lang": "en"}
    assert doc.source.kind == "file"
    assert doc.content_type == "text/markdown"


def test_metadata_is_copied_not_aliased():
    bag = {"filename": "x.txt"}
    doc = Document(id="d", content="c", metadata=bag)
    bag["filename"] = "mutated.txt"
    assert doc.metadata == {"filename": "x.txt"}


def test_metadata_accepts_user_fields():
    doc = Document(
        id="d", content="c", metadata={"custom_key": 42, "nested": {"a": [1, 2]}}
    )
    assert doc.metadata["custom_key"] == 42


def test_source_none_is_allowed():
    assert Document(id="d", content="c").source is None


def test_empty_id_rejected():
    with pytest.raises(TypeError):
        Document(id="", content="c")
    with pytest.raises(TypeError):
        Document(id="   ", content="c")


def test_non_str_id_rejected():
    with pytest.raises(TypeError):
        Document(id=123, content="c")


def test_empty_content_rejected():
    with pytest.raises(TypeError):
        Document(id="d", content="")
    with pytest.raises(TypeError):
        Document(id="d", content="   ")


def test_metadata_must_be_dict():
    with pytest.raises(TypeError):
        Document(id="d", content="c", metadata=[("a", 1)])


def test_source_type_enforced():
    with pytest.raises(TypeError):
        Document(id="d", content="c", source="file.txt")


def test_empty_content_type_rejected():
    with pytest.raises(TypeError):
        Document(id="d", content="c", content_type="")


def test_equality_by_value():
    a = Document(id="d", content="c", metadata={"k": 1})
    b = Document(id="d", content="c", metadata={"k": 1})
    assert a == b
    assert not (a != b)
    # Different dict instances, same value -> equal (copy makes this robust).
    c = Document(id="d", content="c", metadata={"k": 2})
    assert a != c


def test_frozen_prevents_assignment():
    doc = Document(id="d", content="c")
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        doc.content = "changed"  # type: ignore[misc]


def test_public_import():
    from zmrag import __all__

    assert {"Document", "Source", "Metadata"} <= set(__all__)
    bag: Metadata = {"k": 1}
    assert isinstance(bag, dict)
    assert isinstance(FileSource(), Source)


def test_source_is_abstract():
    # Abstraction only: no concrete sources shipped yet, Source cannot be
    # instantiated directly.
    with pytest.raises(TypeError):
        Source()  # type: ignore[abstract]
