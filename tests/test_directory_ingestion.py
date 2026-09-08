"""PlainTextDirectoryIngestor / DirectoryIngestor unit tests."""

import pytest
from pathlib import Path

from zmrag import (
    DirectoryIngestor,
    Document,
    FileSource,
    PlainTextDirectoryIngestor,
    PlainTextFileIngestor,
)


def _dir(tmp_path, files):
    """Build tmp_path/docs with {name: content}; bytes values are written raw."""
    d = tmp_path / "docs"
    d.mkdir()
    for name, content in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return d


def _ingest(directory, prefix="", metadata=None):
    return PlainTextDirectoryIngestor().ingest(
        directory, document_id_prefix=prefix, metadata=metadata
    )


class RecordingFileIngestor(PlainTextFileIngestor):
    """Spy proving delegation (no reading/Document logic in the directory ingestor)."""

    def __init__(self):
        self.calls = []

    def ingest(self, path, *, document_id, metadata=None):
        self.calls.append((str(path), document_id))
        return super().ingest(path, document_id=document_id, metadata=metadata)


# --- basic ---


def test_ingests_all_txt_in_directory(tmp_path):
    d = _dir(tmp_path, {"a.txt": "alpha", "b.txt": "beta"})
    docs = _ingest(d)
    assert isinstance(docs, list) and len(docs) == 2
    assert all(isinstance(x, Document) for x in docs)
    assert [x.content for x in docs] == ["alpha", "beta"]


def test_non_txt_files_ignored(tmp_path):
    d = _dir(tmp_path, {"a.txt": "only", "b.md": "x", "c.pdf": "y", "d.json": "{}", "e.docx": "z"})
    docs = _ingest(d)
    assert len(docs) == 1
    assert docs[0].content == "only"


def test_non_txt_files_not_even_opened(tmp_path):
    # garbage bytes in an ignored file must never be read
    d = _dir(tmp_path, {"a.txt": "ok", "b.md": b"\xff\xfe broken"})
    spy = RecordingFileIngestor()
    docs = PlainTextDirectoryIngestor(spy).ingest(d)
    assert len(docs) == 1
    assert [p for p, _ in spy.calls] == [str(d / "a.txt")]


def test_extension_case_insensitive(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.TXT": "2", "c.Txt": "3"})
    docs = _ingest(d)
    assert {x.content for x in docs} == {"1", "2", "3"}
    assert len(docs) == 3


def test_current_directory_only_no_recursion(tmp_path):
    d = _dir(tmp_path, {"a.txt": "top"})
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("nested", encoding="utf-8")
    spy = RecordingFileIngestor()
    docs = PlainTextDirectoryIngestor(spy).ingest(d)
    assert [x.content for x in docs] == ["top"]
    assert len(spy.calls) == 1  # sub/b.txt never reached


# --- deterministic ordering ---


def test_output_sorted_regardless_of_creation_order(tmp_path):
    d = _dir(tmp_path, {"c.txt": "3", "a.txt": "1", "b.txt": "2"})
    docs = _ingest(d)
    assert [x.id for x in docs] == ["a.txt", "b.txt", "c.txt"]


def test_repeated_ingestion_same_order_and_documents(tmp_path):
    d = _dir(tmp_path, {"c.txt": "3", "a.txt": "1", "b.txt": "2"})
    a, b = _ingest(d), _ingest(d)
    assert a == b
    assert [x.id for x in a] == ["a.txt", "b.txt", "c.txt"]


def test_sorting_is_case_insensitive(tmp_path):
    # uppercase names sort by their casefolded key (a, b, c) -- not ASCII order
    d = _dir(tmp_path, {"B.txt": "2", "a.txt": "1", "C.txt": "3"})
    docs = PlainTextDirectoryIngestor().ingest(d)
    assert [x.id for x in docs] == ["a.txt", "B.txt", "C.txt"]


# --- deterministic ids ---


def test_prefix_document_ids(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2", "c.txt": "3"})
    docs = _ingest(d, prefix="docs")
    assert [x.id for x in docs] == ["docs/a.txt", "docs/b.txt", "docs/c.txt"]


def test_empty_prefix_document_ids(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2"})
    assert [x.id for x in _ingest(d)] == ["a.txt", "b.txt"]


def test_ids_independent_of_absolute_path(tmp_path):
    root1 = _dir(tmp_path, {"a.txt": "same"})
    # build a second, distinct directory with the same file
    d2 = tmp_path / "other"
    d2.mkdir()
    (d2 / "a.txt").write_bytes(b"same")
    ids1 = [x.id for x in _ingest(root1, prefix="docs")]
    ids2 = [x.id for x in _ingest(d2, prefix="docs")]
    assert ids1 == ids2 == ["docs/a.txt"]
    # ids never contain the temp (absolute) root
    assert all("/" + d.name not in i for d in [root1, d2] for i in ids1)


def test_document_fields(tmp_path):
    d = _dir(tmp_path, {"a.txt": "body"})
    doc = _ingest(d, prefix="docs")[0]
    assert doc.id == "docs/a.txt"
    assert doc.content_type == "text/plain"
    assert doc.content == "body"


def test_does_not_rely_on_iterdir_order_between_runs(tmp_path):
    d = _dir(tmp_path, {"zzz.txt": "1", "aaa.txt": "2"})
    first = [x.id for x in _ingest(d)]
    # simulate a different physical order by re-listing through a fresh ingestor
    second = [x.id for x in _ingest(d)]
    assert first == second == ["aaa.txt", "zzz.txt"]


# --- str / Path inputs ---


def test_str_and_path_directory_equal(tmp_path):
    d = _dir(tmp_path, {"a.txt": "x", "b.txt": "y"})
    assert _ingest(str(d)) == _ingest(Path(d))


# --- metadata ---


def test_metadata_forwarded_to_each_document(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2"})
    docs = _ingest(d, metadata={"source": "manual"})
    assert all(x.metadata == {"source": "manual"} for x in docs)


def test_metadata_not_mutated_and_independent_copies(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2"})
    bag = {"source": "test"}
    docs = _ingest(d, metadata=bag)
    bag["source"] = "changed"
    assert docs[0].metadata == {"source": "test"}  # no leak
    assert docs[1].metadata == {"source": "test"}
    assert all(x.metadata is not bag for x in docs)
    assert docs[0].metadata is not docs[1].metadata


def test_no_auto_metadata_fields(tmp_path):
    d = _dir(tmp_path, {"a.txt": "x"})
    doc = _ingest(d, prefix="docs")[0]
    assert doc.metadata == {}  # nothing auto-added


# --- source ---


def test_each_document_has_file_source(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2"})
    docs = _ingest(d, prefix="docs")
    for name, doc in zip(("a.txt", "b.txt"), docs):
        assert isinstance(doc.source, FileSource)
        assert doc.source.kind == "file"
        assert doc.source.path == d / name


# --- content preservation ---


@pytest.mark.parametrize(
    "content",
    [
        "中文内容保留 测试",
        "😀a😀b😀",
        "  leading and trailing  ",
        "a  b   c",
        "line1\nline2",
        "para1\n\npara2\n",
    ],
)
def test_content_preserved_verbatim(tmp_path, content):
    d = _dir(tmp_path, {"x.txt": content})
    assert _ingest(d)[0].content == content


# --- empty results ---


def test_empty_directory_returns_empty(tmp_path):
    d = _dir(tmp_path, {})
    assert _ingest(d) == []


def test_directory_with_no_txt_returns_empty(tmp_path):
    d = _dir(tmp_path, {"only.md": "x", "only.pdf": "y"})
    assert _ingest(d) == []


# --- error handling ---


def test_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ingest(tmp_path / "nope")


def test_path_is_a_file_raises_not_a_directory_error(tmp_path):
    p = _dir(tmp_path, {"f.txt": "x"}) / "f.txt"
    with pytest.raises(NotADirectoryError):
        _ingest(p)


@pytest.mark.parametrize("bad", [None, 5, 1.5, b"dir", ["a"], {"a": 1}, True])
def test_invalid_directory_type_rejected(bad):
    with pytest.raises(TypeError):
        _ingest(bad)  # type: ignore[arg-type]


def test_utf8_failure_propagates_not_swallowed(tmp_path):
    d = _dir(tmp_path, {"good.txt": "ok", "bad.txt": b"ok \xff\xfe not utf-8"})
    with pytest.raises(UnicodeDecodeError):
        _ingest(d)


def test_empty_txt_propagates_document_validation(tmp_path):
    d = _dir(tmp_path, {"good.txt": "ok", "empty.txt": ""})
    with pytest.raises(TypeError):
        _ingest(d)


def test_whitespace_only_txt_propagates_document_validation(tmp_path):
    d = _dir(tmp_path, {"ws.txt": "   \n\t "})
    with pytest.raises(TypeError):
        _ingest(d)


# --- delegation / architecture ---


def test_directory_ingestor_is_abstract():
    with pytest.raises(TypeError):
        DirectoryIngestor()  # type: ignore[abstract]


def test_delegates_reading_to_file_ingestor(tmp_path):
    d = _dir(tmp_path, {"a.txt": "1", "b.txt": "2"})
    spy = RecordingFileIngestor()
    docs = PlainTextDirectoryIngestor(spy).ingest(d, document_id_prefix="docs")
    assert [p for p, _ in spy.calls] == [str(d / "a.txt"), str(d / "b.txt")]
    assert [i for _, i in spy.calls] == ["docs/a.txt", "docs/b.txt"]
    assert len(docs) == 2


def test_directory_ingestion_module_has_no_stage_coupling():
    import zmrag.directory_ingestion as di

    for name in (
        "Chunker",
        "EmbeddingProvider",
        "VectorStore",
        "Retriever",
        "LexicalIndex",
        "ContextBuilder",
    ):
        assert not hasattr(di, name), f"directory_ingestion must not import {name}"


def test_public_import():
    from zmrag import __all__

    assert {"DirectoryIngestor", "PlainTextDirectoryIngestor"} <= set(__all__)
