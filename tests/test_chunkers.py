"""FixedSizeChunker unit tests."""

import pytest

from zmwiki import Chunk, Chunker, Document, FixedSizeChunker


def _chunk(doc, chunk_size, overlap=0):
    return FixedSizeChunker(chunk_size, overlap).chunk(doc)


# --- basic length behaviour ---


def test_short_text_single_chunk():
    doc = Document(id="d", content="hi")  # 2 chars < chunk_size
    chunks = _chunk(doc, chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hi"
    assert chunks[0].position == 0


def test_text_equal_to_chunk_size_single_chunk():
    doc = Document(id="d", content="0123456789")  # len 10 == chunk_size
    chunks = _chunk(doc, chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0].text == "0123456789"


def test_text_longer_than_chunk_size_multiple_chunks():
    doc = Document(id="d", content="0123456789abcdef")  # len 16
    chunks = _chunk(doc, chunk_size=4)
    assert len(chunks) == 4
    assert [c.text for c in chunks] == ["0123", "4567", "89ab", "cdef"]


# --- overlap ---


def test_no_overlap_chunks_are_contiguous_and_preserve_text():
    text = "hello  world   "  # deliberate interior + trailing spaces
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=5)
    assert "".join(c.text for c in chunks) == text  # slice only, never rewrites
    assert [c.text for c in chunks] == ["hello", "  wor", "ld   "]


def test_overlap_windows_are_step_apart():
    # size 4, overlap 1 -> step 3. Windows start at 0,3,6; the last reaches the
    # text end exactly, so no extra tail chunk is emitted (it would be fully
    # inside the previous window).
    text = "0123456789"  # len 10
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=4, overlap=1)
    assert [c.position for c in chunks] == [0, 1, 2]  # consecutive, 0-based
    assert [c.text for c in chunks] == ["0123", "3456", "6789"]
    # each chunk starts `step` chars after the previous one
    assert chunks[1].text == text[3:7]
    assert chunks[2].text == text[6:10]


def test_overlap_with_tail_remainder_keeps_partial_last_chunk():
    text = "0123456789a"  # len 11: windows 0:4, 3:7, 6:10, 9:11 (partial tail)
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=4, overlap=1)
    assert [c.text for c in chunks] == ["0123", "3456", "6789", "9a"]
    # tail is covered, and the shared char between the last two windows is 1
    assert len(chunks[-1].text) < 4


def test_overlap_smallest_step_finite():
    text = "0123456789"
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=4, overlap=3)  # step 1, max overlap
    # 10 chars, size 4, step 1 -> starts 0..6 = 7 chunks; terminates (no infinite loop)
    assert len(chunks) == 7
    assert [c.text for c in chunks] == [text[s : s + 4] for s in range(7)]


# --- parameter validation ---


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_chunk_size_non_positive_rejected(bad):
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=bad)


@pytest.mark.parametrize("bad", [2.5, "40", None, True])
def test_chunk_size_wrong_type_rejected(bad):
    with pytest.raises(TypeError):
        FixedSizeChunker(chunk_size=bad)


@pytest.mark.parametrize("bad", [-1, -5])
def test_overlap_negative_rejected(bad):
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=bad)


@pytest.mark.parametrize("bad", [10, 11, 100])
def test_overlap_at_least_chunk_size_rejected(bad):
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=bad)


@pytest.mark.parametrize("bad", [2.5, "1", None, True])
def test_overlap_wrong_type_rejected(bad):
    with pytest.raises(TypeError):
        FixedSizeChunker(chunk_size=10, overlap=bad)


# --- data relationships ---


def test_document_id_and_positions_consistent():
    doc = Document(id="doc-7", content="0123456789")
    chunks = _chunk(doc, chunk_size=4)
    assert all(c.document_id == "doc-7" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    doc = Document(id="d", content="0123456789")
    a = _chunk(doc, chunk_size=4)
    b = _chunk(doc, chunk_size=4)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.id for c in a] == ["d::0", "d::1", "d::2"]


def test_chunk_ids_do_not_collide_across_documents():
    docs = [Document(id=f"doc-{i}", content="0123456789") for i in range(3)]
    ids = {c.id for d in docs for c in _chunk(d, chunk_size=4)}
    assert len(ids) == 3 * 3  # 3 docs x 3 chunks, all distinct


def test_metadata_inherited_but_not_shared():
    bag = {"filename": "a.txt", "lang": "en"}
    doc = Document(id="d", content="0123456789", metadata=bag)
    chunks = _chunk(doc, chunk_size=4)
    assert all(c.metadata == {"filename": "a.txt", "lang": "en"} for c in chunks)
    # no auto-injected chunk fields
    assert all(c.metadata.keys() == {"filename", "lang"} for c in chunks)
    # independent dict objects, none aliasing the document's
    assert all(c.metadata is not doc.metadata for c in chunks)
    assert len({id(c.metadata) for c in chunks}) == len(chunks)


def test_chunker_does_not_mutate_document():
    doc = Document(id="d", content="0123456789", metadata={"k": "v"})
    _chunk(doc, chunk_size=4)
    assert doc.content == "0123456789"
    assert doc.metadata == {"k": "v"}


def test_chunker_is_abstract():
    with pytest.raises(TypeError):
        Chunker()  # type: ignore[abstract]


def test_rejects_non_document():
    with pytest.raises(TypeError):
        FixedSizeChunker(10).chunk("not a document")  # type: ignore[arg-type]


# --- unicode / code-point slicing ---


def test_chinese_text_splits_by_code_point():
    text = "中文字符串测试"  # 8 code points
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=3)
    assert "".join(c.text for c in chunks) == text
    assert [c.text for c in chunks] == ["中文字", "符串测", "试"]


def test_emoji_not_split_by_bytes():
    text = "😀a😀b😀"  # 5 code points (emoji are single code points in py3)
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=2)
    assert "".join(c.text for c in chunks) == text
    assert [c.text for c in chunks] == ["😀a", "😀b", "😀"]


def test_mixed_text_preserved():
    text = "hi中😀ok"
    doc = Document(id="d", content=text)
    chunks = _chunk(doc, chunk_size=2)
    assert "".join(c.text for c in chunks) == text
    assert [c.text for c in chunks] == ["hi", "中😀", "ok"]


def test_output_is_chunks():
    doc = Document(id="d", content="0123456789")
    chunks = _chunk(doc, chunk_size=4)
    assert chunks and all(isinstance(c, Chunk) for c in chunks)


def test_public_import():
    from zmwiki import __all__

    assert {"Chunker", "FixedSizeChunker", "Chunk", "Document"} <= set(__all__)
