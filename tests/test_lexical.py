"""Lexical retrieval tests: tokenizer, BM25 index, LexicalRetriever."""

import pytest

from zmrag import (
    Chunk,
    InMemoryLexicalIndex,
    LexicalIndex,
    LexicalRetriever,
    Retriever,
    SearchResult,
    SimpleTokenizer,
    Tokenizer,
)


def _chunk(cid, text="t"):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


# --- tokenizer ---


def test_tokenizer_english():
    assert SimpleTokenizer().tokenize("Hello World") == ["hello", "world"]
    assert SimpleTokenizer().tokenize("The Quick BROWN fox") == [
        "the",
        "quick",
        "brown",
        "fox",
    ]


def test_tokenizer_numbers():
    assert SimpleTokenizer().tokenize("RAG 2024 v2") == ["rag", "2024", "v2"]


def test_tokenizer_punctuation_dropped():
    assert SimpleTokenizer().tokenize("hello, world! (test) [x]") == [
        "hello",
        "world",
        "test",
        "x",
    ]


def test_tokenizer_chinese_character_level():
    # character-level, deterministic, not semantic segmentation
    assert SimpleTokenizer().tokenize("苹果很好吃") == ["苹", "果", "很", "好", "吃"]


def test_tokenizer_unicode_and_emoji_dropped():
    toks = SimpleTokenizer().tokenize("中文 mix 😀 café")
    assert "😀" not in toks
    assert toks == ["中", "文", "mix", "caf", "é"]


def test_tokenizer_deterministic():
    t = SimpleTokenizer()
    for text in ["Hello world", "苹果", "a b a", "2024-01-01"]:
        assert t.tokenize(text) == t.tokenize(text)


def test_tokenizer_rejects_non_string():
    with pytest.raises(TypeError):
        SimpleTokenizer().tokenize(42)  # type: ignore[arg-type]


# --- index basics ---


def test_empty_index():
    idx = InMemoryLexicalIndex()
    assert idx.count() == 0
    assert idx.get("x") is None
    assert idx.delete("x") is False
    assert idx.search("anything") == []


def test_upsert_and_count_and_get():
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("a", text="apple banana"))
    idx.upsert(_chunk("b", text="orange"))
    assert idx.count() == 2
    assert idx.get("a").text == "apple banana"
    assert idx.get("zz") is None


def test_delete_removes_chunk_and_postings():
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("a", text="apple banana"))
    idx.upsert(_chunk("b", text="apple pie"))
    assert idx.delete("a") is True
    assert idx.delete("missing") is False
    assert idx.count() == 1
    hits = idx.search("apple")
    assert [h.chunk.id for h in hits] == ["b"]  # a's apple gone


# --- upsert replacement ---


def test_upsert_same_id_replaces_text():
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("a", text="apple banana"))
    idx.upsert(_chunk("a", text="orange"))
    assert idx.count() == 1
    assert [h.chunk.id for h in idx.search("apple")] == []  # old term gone
    assert [h.chunk.id for h in idx.search("orange")] == ["a"]


# --- BM25 search behaviour ---


def _index_for(*texts):
    idx = InMemoryLexicalIndex()
    for i, text in enumerate(texts):
        idx.upsert(_chunk(f"d{i}", text=text))
    return idx


def test_exact_term_match_has_positive_score():
    idx = _index_for("apple pie", "banana split")
    hits = idx.search("apple")
    assert len(hits) == 1
    assert hits[0].chunk.id == "d0"
    assert hits[0].score > 0


def test_no_match_returns_empty():
    idx = _index_for("apple pie", "banana split")
    assert idx.search("zebra") == []


def test_repeated_query_term_not_double_counted():
    idx = _index_for("apple pie", "banana split")
    single = idx.search("apple")[0].score
    repeated = idx.search("apple apple apple")[0].score
    assert repeated == pytest.approx(single)  # query terms deduped


def test_punctuation_query():
    idx = _index_for("apple pie")
    assert len(idx.search("apple!")) == 1
    assert idx.search("!!,") == []  # no tokens -> no results, no exception


def test_empty_query_returns_empty():
    idx = _index_for("apple pie")
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_document_frequency_effect_rare_term_higher_idf():
    # rare term appears in 1/4 docs, common term in 3/4. Equal length, tf=1.
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("r", text="rare"))
    for i in range(3):
        idx.upsert(_chunk(f"c{i}", text="common"))
    rare_score = idx.search("rare")[0].score
    common_score = idx.search("common")[0].score
    assert rare_score > common_score


def test_length_normalization_shorter_wins():
    # both docs have exactly one "alpha"; the short doc should score higher.
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("short", text="alpha"))
    long_text = "alpha " + " ".join(["beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"])
    idx.upsert(_chunk("long", text=long_text))
    hits = idx.search("alpha")
    assert hits[0].chunk.id == "short"
    assert hits[0].score > hits[1].score


def test_term_frequency_increases_score():
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("tf1", text="alpha beta gamma delta"))
    idx.upsert(_chunk("tf2", text="alpha alpha beta gamma delta"))
    hits = idx.search("alpha")
    by_id = {h.chunk.id: h.score for h in hits}
    assert by_id["tf2"] > by_id["tf1"]


def test_k1_parameter_controls_saturation():
    # b=0 isolates k1: with tf 5 vs tf 1, a larger k1 gives a larger relative
    # advantage to the high-tf doc.
    def ratio(k1):
        idx = InMemoryLexicalIndex(k1=k1, b=0.0)
        idx.upsert(_chunk("low", text="alpha beta gamma delta"))
        idx.upsert(_chunk("high", text="alpha alpha alpha alpha alpha beta gamma delta"))
        s = {h.chunk.id: h.score for h in idx.search("alpha")}
        return s["high"] / s["low"]

    assert ratio(k1=10.0) > ratio(k1=0.2)


def test_b_parameter_controls_length_normalization():
    # b=0: no length penalty -> equal-length-irrelevant: long == short tf=1.
    def scores(b):
        idx = InMemoryLexicalIndex(k1=1.2, b=b)
        idx.upsert(_chunk("short", text="alpha"))
        filler = " ".join(["beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi"])
        idx.upsert(_chunk("long", text="alpha " + filler))
        s = {h.chunk.id: h.score for h in idx.search("alpha")}
        return s["short"], s["long"]

    short0, long0 = scores(b=0.0)
    assert short0 == pytest.approx(long0)
    short1, long1 = scores(b=1.0)
    assert short1 > long1


def test_ranking_descending_and_top_k():
    idx = InMemoryLexicalIndex()
    idx.upsert(_chunk("apple", text="apple pie"))
    idx.upsert(_chunk("banana", text="banana apple crisp"))
    idx.upsert(_chunk("cherry", text="cherry tomato"))
    hits = idx.search("apple", top_k=2)
    assert len(hits) == 2
    assert [h.chunk.id for h in hits] == ["apple", "banana"]
    assert hits[0].score >= hits[1].score


def test_top_k_beyond_count_returns_all():
    idx = _index_for("apple pie", "banana pie")
    assert len(idx.search("pie", top_k=100)) == 2


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "3"])
def test_invalid_top_k_rejected(bad):
    idx = _index_for("apple pie")
    with pytest.raises((ValueError, TypeError)):
        idx.search("apple", top_k=bad)


def test_deterministic_tie_break_by_chunk_id():
    idx = InMemoryLexicalIndex()
    # identical texts -> identical BM25 scores -> tie-break by chunk id asc
    idx.upsert(_chunk("b2", text="alpha"))
    idx.upsert(_chunk("a1", text="alpha"))
    first = [h.chunk.id for h in idx.search("alpha")]
    second = [h.chunk.id for h in idx.search("alpha")]
    assert first == ["a1", "b2"]
    assert first == second


def test_repeated_search_stable_and_score_nonnegative():
    idx = _index_for("apple pie", "apple apple banana", "banana")
    hits = idx.search("apple banana")
    assert all(h.score >= 0 for h in hits)
    again = [h.chunk.id for h in idx.search("apple banana")]
    assert again == [h.chunk.id for h in hits]


# --- parameter validation ---


def test_invalid_k1_and_b():
    with pytest.raises(TypeError):
        InMemoryLexicalIndex(k1=True)
    with pytest.raises(TypeError):
        InMemoryLexicalIndex(b="0.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        InMemoryLexicalIndex(k1=-1)
    with pytest.raises(ValueError):
        InMemoryLexicalIndex(b=1.5)


def test_index_rejects_non_chunk():
    idx = InMemoryLexicalIndex()
    with pytest.raises(TypeError):
        idx.upsert("apple")  # type: ignore[arg-type]


def test_tokenizer_injected():
    class Upper(Tokenizer):
        def tokenize(self, text):
            return text.upper().split()

    idx = InMemoryLexicalIndex(tokenizer=Upper())
    idx.upsert(_chunk("a", text="Hello World"))
    assert idx.search("HELLO")[0].chunk.id == "a"


def test_lexical_index_abstract_and_public_import():
    with pytest.raises(TypeError):
        LexicalIndex()  # type: ignore[abstract]
    from zmrag import __all__

    assert {"Tokenizer", "SimpleTokenizer", "LexicalIndex", "InMemoryLexicalIndex"} <= set(__all__)


# --- LexicalRetriever ---


class RecordingIndex(LexicalIndex):
    def __init__(self, preset=None, error=None):
        self.preset = preset or []
        self.error = error
        self.calls = 0
        self.last_query = None
        self.last_top_k = None

    def upsert(self, chunk):
        raise NotImplementedError

    def get(self, chunk_id):
        raise NotImplementedError

    def delete(self, chunk_id):
        raise NotImplementedError

    def count(self):
        return len(self.preset)

    def search(self, query, top_k=10):
        self.calls += 1
        self.last_query = query
        self.last_top_k = top_k
        if self.error is not None:
            raise self.error
        return list(self.preset)


def test_lexical_retriever_is_retriever():
    assert issubclass(LexicalRetriever, Retriever)


def test_lexical_retriever_injection():
    retriever = LexicalRetriever(InMemoryLexicalIndex())
    assert isinstance(retriever, LexicalRetriever)
    with pytest.raises(TypeError):
        LexicalRetriever("nope")  # type: ignore[arg-type]


def test_lexical_retriever_preserves_index_results_and_order():
    a = _chunk("a", "apple")
    b = _chunk("b", "banana")
    preset = [SearchResult(chunk=a, score=0.2), SearchResult(chunk=b, score=0.9)]
    index = RecordingIndex(preset=preset)
    retriever = LexicalRetriever(index)
    out = retriever.retrieve("q", top_k=5)
    assert out == preset  # untouched order (no re-sort by retriever)
    assert [r.chunk.id for r in out] == ["a", "b"]
    assert index.calls == 1
    assert index.last_query == "q"
    assert index.last_top_k == 5


def test_lexical_retriever_default_top_k_and_empty():
    index = RecordingIndex()
    retriever = LexicalRetriever(index)
    assert retriever.retrieve("q") == []
    assert index.last_top_k == 10
    with pytest.raises(TypeError):
        retriever.retrieve("")
    with pytest.raises(TypeError):
        retriever.retrieve(42)  # type: ignore[arg-type]


def test_lexical_retriever_exception_propagates():
    boom = RuntimeError("index boom")
    retriever = LexicalRetriever(RecordingIndex(error=boom))
    with pytest.raises(RuntimeError) as exc:
        retriever.retrieve("q")
    assert exc.value is boom
