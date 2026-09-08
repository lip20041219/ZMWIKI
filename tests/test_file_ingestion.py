"""PlainTextFileIngestor / FileIngestor / FileSource unit tests."""

import pytest
from pathlib import Path

from zmrag import (
    Document,
    FileIngestor,
    FileSource,
    PlainTextFileIngestor,
    Source,
)


def _write(tmp_path, name, content: str):
    """Write exact UTF-8 bytes (no platform newline translation)."""
    p = tmp_path / name
    p.write_bytes(content.encode("utf-8"))
    return p


def _ingest(p, document_id="doc-1", metadata=None):
    return PlainTextFileIngestor().ingest(
        p, document_id=document_id, metadata=metadata
    )


# --- basic reads ---


def test_reads_ascii_txt(tmp_path):
    p = _write(tmp_path, "hello.txt", "hello world")
    doc = _ingest(p)
    assert isinstance(doc, Document)
    assert doc.id == "doc-1"
    assert doc.content == "hello world"


def test_reads_chinese(tmp_path):
    p = _write(tmp_path, "cn.txt", "中文内容保留 测试")
    assert _ingest(p).content == "中文内容保留 测试"


def test_reads_emoji(tmp_path):
    p = _write(tmp_path, "emoji.txt", "😀a😀b😀")
    assert _ingest(p).content == "😀a😀b😀"


def test_reads_multiline(tmp_path):
    text = "line1\nline2\n\nline4\n"
    p = _write(tmp_path, "multi.txt", text)
    assert _ingest(p).content == text


# --- content preservation (verbatim) ---


@pytest.mark.parametrize(
    "content",
    [
        "  leading spaces",
        "trailing spaces  ",
        "  both sides  ",
        "a  b   c",  # interior runs kept
        "line1\nline2",
        "para1\n\npara2",
        "héllo wörld — ünïcode",
        "中文😀混合 text 内容",
        "!@#$%^&*() punctuation ,.;:'\"",
        "\tindented\n  spaced",
    ],
)
def test_content_preserved_verbatim(tmp_path, content):
    p = _write(tmp_path, "v.txt", content)
    doc = _ingest(p)
    assert doc.content == content
    # equals what a plain read_text produces (read is faithful)
    assert doc.content == p.read_text(encoding="utf-8")


# --- str and Path both accepted ---


def test_str_and_path_equal(tmp_path):
    p = _write(tmp_path, "both.txt", "some text\n")
    from_str = _ingest(str(p))
    from_path = _ingest(Path(p))
    assert from_str == from_path


def test_uppercase_extension_accepted(tmp_path):
    for name in ("up.TXT", "mix.Txt"):
        p = _write(tmp_path, name, "case")
        assert _ingest(p).content == "case"


# --- document fields ---


def test_document_fields(tmp_path):
    p = _write(tmp_path, "f.txt", "body")
    doc = _ingest(p, document_id="doc-9", metadata={"title": "t"})
    assert doc.id == "doc-9"
    assert doc.content_type == "text/plain"
    assert doc.metadata == {"title": "t"}


def test_metadata_defaults_to_empty(tmp_path):
    assert _ingest(_write(tmp_path, "m.txt", "x")).metadata == {}


def test_source_is_file_source(tmp_path):
    p = _write(tmp_path, "src.txt", "x")
    src = _ingest(p).source
    assert isinstance(src, FileSource)
    assert isinstance(src, Source)
    assert src.kind == "file"
    assert src.path == Path(p)


def test_source_path_normalized_to_path(tmp_path):
    p = _write(tmp_path, "norm.txt", "x")
    assert _ingest(str(p)).source == FileSource(Path(p))


# --- metadata immutability / no aliasing ---


def test_input_metadata_not_mutated(tmp_path):
    bag = {"title": "hello"}
    _ingest(_write(tmp_path, "md.txt", "x"), metadata=bag)
    assert bag == {"title": "hello"}


def test_later_mutation_of_input_metadata_does_not_leak(tmp_path):
    bag = {"title": "hello"}
    doc = _ingest(_write(tmp_path, "md2.txt", "x"), metadata=bag)
    bag["title"] = "changed"
    assert doc.metadata == {"title": "hello"}
    assert doc.metadata is not bag


def test_no_auto_metadata_fields(tmp_path):
    # no timestamp / hash / filename / size injected
    doc = _ingest(_write(tmp_path, "plain.txt", "x"))
    assert doc.metadata == {}
    assert doc.source.kind == "file"  # provenance in source, not metadata


# --- determinism ---


def test_same_file_same_document(tmp_path):
    p = _write(tmp_path, "d.txt", "stable content")
    a = _ingest(p, document_id="doc-1", metadata={"title": "x"})
    b = _ingest(p, document_id="doc-1", metadata={"title": "x"})
    assert a == b


# --- error handling: path type ---


@pytest.mark.parametrize("bad", [None, 5, 1.5, ["a"], {"a": 1}, b"bytes", True])
def test_invalid_path_type_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(bad)  # type: ignore[arg-type]


# --- error handling: filesystem ---


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ingest(tmp_path / "missing.txt")


def test_directory_raises_is_a_directory_error(tmp_path):
    with pytest.raises(IsADirectoryError):
        _ingest(tmp_path)


def test_empty_txt_file_rejected_like_document(tmp_path):
    # empty content violates Document's non-empty contract -> TypeError
    p = _write(tmp_path, "empty.txt", "")
    with pytest.raises(TypeError):
        _ingest(p)


def test_whitespace_only_txt_rejected_like_document(tmp_path):
    p = _write(tmp_path, "ws.txt", "   \n\t ")
    with pytest.raises(TypeError):
        _ingest(p)


def test_bad_utf8_raises_unicode_decode_error(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"ok \xff\xfe not utf-8")
    with pytest.raises(UnicodeDecodeError):
        _ingest(p)


# --- unsupported formats refused (nothing parsed) ---


@pytest.mark.parametrize(
    "name",
    ["a.pdf", "a.md", "a.json", "a.csv", "a.html", "a.docx", "noext", "a.txt.bak"],
)
def test_unsupported_extension_rejected(tmp_path, name):
    p = _write(tmp_path, name, "some content")
    with pytest.raises(ValueError):
        _ingest(p)


# --- invalid document_id / metadata delegated to Document ---


@pytest.mark.parametrize("bad", ["", "   "])
def test_document_id_empty_rejected(tmp_path, bad):
    p = _write(tmp_path, "id.txt", "x")
    with pytest.raises(TypeError):
        _ingest(p, document_id=bad)


def test_metadata_non_dict_rejected(tmp_path):
    p = _write(tmp_path, "meta.txt", "x")
    with pytest.raises(TypeError):
        _ingest(p, metadata="nope")  # type: ignore[arg-type]


# --- architecture ---


def test_file_ingestor_is_abstract():
    with pytest.raises(TypeError):
        FileIngestor()  # type: ignore[abstract]


def test_file_ingestion_module_has_no_stage_coupling():
    import zmrag.file_ingestion as fi

    for name in (
        "Chunker",
        "EmbeddingProvider",
        "VectorStore",
        "Retriever",
        "LexicalIndex",
        "ContextBuilder",
        "PlainTextIngestor",
    ):
        assert not hasattr(fi, name), f"file_ingestion must not import {name}"


def test_public_import():
    from zmrag import __all__

    assert {"FileIngestor", "FileSource", "PlainTextFileIngestor"} <= set(__all__)


def test_document_not_mutated(tmp_path):
    p = _write(tmp_path, "imm.txt", "  keep me  \n")
    before = p.read_text(encoding="utf-8")
    _ingest(p)
    assert p.read_text(encoding="utf-8") == before
