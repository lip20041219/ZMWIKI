"""TextIngestor / PlainTextIngestor unit tests."""

import pytest

from zmwiki import Document, PlainTextIngestor, TextIngestor


def _ingest(text="hello", document_id="doc-001", metadata=None):
    return PlainTextIngestor().ingest(
        document_id, text, metadata=metadata
    )


# --- basic ---


def test_ingest_returns_document():
    doc = _ingest()
    assert isinstance(doc, Document)


def test_document_id_preserved():
    doc = _ingest("hello", document_id="doc-001")
    assert doc.id == "doc-001"


def test_content_round_trips_plain_text():
    assert _ingest("hello world").content == "hello world"


def test_content_type_is_text_plain():
    assert _ingest().content_type == "text/plain"


def test_source_defaults_to_none():
    assert _ingest().source is None


def test_metadata_passed_through():
    doc = _ingest(metadata={"title": "example"})
    assert doc.metadata == {"title": "example"}


def test_metadata_defaults_to_empty():
    assert _ingest().metadata == {}


# --- validation (delegated to Document) ---


@pytest.mark.parametrize("bad", [123, None, b"bytes", ["t"], 2.5, True])
def test_text_non_str_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(text=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_text_empty_or_whitespace_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(text=bad)


@pytest.mark.parametrize("bad", [123, None, b"id", 1.5, True])
def test_document_id_non_str_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(document_id=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_document_id_empty_or_whitespace_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(document_id=bad)


@pytest.mark.parametrize("bad", ["nope", 7, [1], ("t",)])
def test_metadata_non_dict_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(metadata=bad)  # type: ignore[arg-type]


# --- determinism ---


def test_same_input_same_document():
    a = _ingest("hello", document_id="doc-001", metadata={"title": "x"})
    b = _ingest("hello", document_id="doc-001", metadata={"title": "x"})
    assert a == b


def test_no_random_or_auto_generated_id():
    assert _ingest(document_id="doc-001").id == "doc-001"
    assert _ingest(document_id="doc-002").id == "doc-002"


def test_no_hidden_metadata_injected():
    # no timestamp / hash / source auto-added; keys are exactly what caller gave
    doc = _ingest("hello")
    assert doc.metadata == {}
    doc2 = _ingest("hello", metadata={"title": "x"})
    assert doc2.metadata == {"title": "x"}


# --- immutability / no aliasing ---


def test_ingestor_does_not_mutate_input_metadata():
    bag = {"title": "hello"}
    _ingest(metadata=bag)
    assert bag == {"title": "hello"}


def test_later_mutation_of_input_metadata_does_not_leak():
    bag = {"title": "hello"}
    doc = _ingest(metadata=bag)
    bag["title"] = "changed"
    assert doc.metadata == {"title": "hello"}
    assert doc.metadata is not bag


def test_repeated_ingest_does_not_share_metadata():
    a = _ingest(metadata={"n": 1})
    b = _ingest(metadata={"n": 1})
    assert a.metadata is not b.metadata


# --- text preservation (verbatim, no cleaning) ---


@pytest.mark.parametrize(
    "text",
    [
        "  leading spaces",
        "trailing spaces  ",
        "  both sides  ",
        "line1\nline2\n",
        "\n\t  mixed whitespace \n",
        "héllo wörld — ünïcode",
        "中文内容保留 😀",
        "中😀ok",
        "Café naïve — 100%",
        "!@#$%^&*() punctuation ,.;:'\"",
    ],
)
def test_text_preserved_verbatim(text):
    assert _ingest(text).content == text


def test_emoji_preserved():
    text = "😀a😀b😀"
    assert _ingest(text).content == text


# --- architecture ---


def test_ingestor_is_abstract():
    with pytest.raises(TypeError):
        TextIngestor()  # type: ignore[abstract]


def test_ingestion_module_has_no_stage_coupling():
    import zmwiki.ingestion as ing

    for name in (
        "Chunker",
        "FixedSizeChunker",
        "EmbeddingProvider",
        "VectorStore",
        "Retriever",
        "LexicalIndex",
        "ContextBuilder",
    ):
        assert not hasattr(ing, name), f"ingestion must not import {name}"


def test_public_import():
    from zmwiki import __all__

    assert {"TextIngestor", "PlainTextIngestor", "Document"} <= set(__all__)


def test_output_is_document_not_dict_or_str():
    out = _ingest()
    assert not isinstance(out, (dict, str))
    assert isinstance(out, Document)
