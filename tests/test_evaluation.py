"""Evaluation contract tests: metrics, validation, determinism, pipeline use.

Covers the Phase 8A RetrievalEvaluator: Hit Rate / Recall / Precision as a
macro average over queries, strict input validation (types, empty ground
truth, duplicate ids, positional matching), id-only identity, determinism, and
an end-to-end proof that DeterministicRAGPipeline output can be consumed by
the evaluation layer.
"""

import pytest

from zmrag import (
    Chunk,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    EvaluationCase,
    EvaluationPrediction,
    EvaluationResult,
    Evaluator,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalRetriever,
    RetrievalEvaluator,
    SearchResult,
    WeightedScoreFusion,
)


def _chunk(cid, text=None):
    return Chunk(
        id=cid, document_id="doc", text=text or cid, position=0
    )


def _sr(cid, score=0.0, text=None):
    return SearchResult(chunk=_chunk(cid, text=text), score=score)


def _case(query, *ids):
    return EvaluationCase(query=query, relevant_chunk_ids=ids)


def _pred(query, *results):
    return EvaluationPrediction(query=query, results=results)


def _evaluate(cases, predictions):
    return RetrievalEvaluator().evaluate(cases, predictions)


# --- A. Perfect retrieval ---


def test_perfect_retrieval_metrics_are_one():
    result = _evaluate([_case("q", "A")], [_pred("q", _sr("A"))])
    assert result.hit_rate == 1.0
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert (result.total_cases, result.evaluated_cases) == (1, 1)


# --- B. Partial recall ---


def test_partial_recall_half():
    result = _evaluate([_case("q", "A", "B")], [_pred("q", _sr("A"), _sr("C"))])
    assert result.recall == 0.5
    assert result.precision == 0.5
    assert result.hit_rate == 1.0  # at least one relevant chunk surfaced


# --- C. Zero hit ---


def test_zero_hit():
    result = _evaluate([_case("q", "A")], [_pred("q", _sr("B"), _sr("C"))])
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.hit_rate == 0.0


# --- D. Empty retrieval ---


def test_empty_retrieval_is_zero_not_error():
    result = _evaluate([_case("q", "A")], [_pred("q")])
    assert result.recall == 0.0
    assert result.precision == 0.0  # defined as 0.0 over an empty candidate set
    assert result.hit_rate == 0.0


# --- E. Multiple queries -> macro average ---


def test_macro_average_over_queries():
    cases = [
        _case("q1", "A"),          # recall 1.0, precision 1.0, hit 1
        _case("q2", "A", "B"),     # recall 0.5, precision 0.5, hit 1
        _case("q3", "A"),          # recall 0.0, precision 0.0, hit 0
    ]
    predictions = [
        _pred("q1", _sr("A")),
        _pred("q2", _sr("A"), _sr("C")),
        _pred("q3", _sr("B"), _sr("C")),
    ]
    result = _evaluate(cases, predictions)
    assert result.recall == pytest.approx((1.0 + 0.5 + 0.0) / 3)  # 0.5
    assert result.precision == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.hit_rate == pytest.approx(2.0 / 3.0)
    assert (result.total_cases, result.evaluated_cases) == (3, 3)


# --- F. Duplicate prediction ids ---


def test_duplicate_prediction_ids_rejected():
    with pytest.raises(ValueError):
        _pred("q", _sr("A"), _sr("A"), _sr("B"))


# --- G. Empty ground truth ---


@pytest.mark.parametrize("empty", [[], (), set(), frozenset()])
def test_empty_relevant_ids_rejected(empty):
    with pytest.raises(ValueError):
        EvaluationCase(query="q", relevant_chunk_ids=empty)


def test_duplicate_relevant_ids_rejected():
    with pytest.raises(ValueError):
        EvaluationCase(query="q", relevant_chunk_ids=["A", "B", "A"])


# --- H. Mismatched case/prediction count ---


def test_case_prediction_count_mismatch_rejected():
    with pytest.raises(ValueError):
        _evaluate([_case("q1", "A"), _case("q2", "B")], [_pred("q1", _sr("A"))])


# --- I. Determinism ---


def test_evaluation_deterministic_across_calls_and_instances():
    cases = [_case("q1", "A"), _case("q2", "A", "B"), _case("q3", "A")]
    predictions = [
        _pred("q1", _sr("A")),
        _pred("q2", _sr("A"), _sr("C")),
        _pred("q3", _sr("B")),
    ]
    first = RetrievalEvaluator().evaluate(cases, predictions)
    second = RetrievalEvaluator().evaluate(cases, predictions)
    assert first == second
    assert first == _evaluate(cases, predictions)


# --- J. Identity is chunk.id, never text / object / score ---


def test_identity_uses_chunk_id_not_text():
    # Same text, different id: a hit on the wrong id must count as a miss.
    case = _case("q", "needle::0")
    wrong_text_same_id = _pred("q", _sr("needle::0", text="identical body text"))
    wrong_id_same_text = _pred("q", _sr("needle::1", text="identical body text"))
    assert _evaluate([case], [wrong_text_same_id]).recall == 1.0
    assert _evaluate([case], [wrong_id_same_text]).recall == 0.0


def test_score_is_never_read():
    cases = [_case("q", "A", "B")]
    a = _evaluate(cases, [_pred("q", _sr("A", score=0.99), _sr("B", score=-5.0))])
    b = _evaluate(cases, [_pred("q", _sr("A", score=0.0), _sr("B", score=0.0))])
    assert a == b


# --- input validation ---


@pytest.mark.parametrize("bad_query", ["", "   \n\t", 42, None])
def test_case_rejects_invalid_query(bad_query):
    with pytest.raises(TypeError):
        EvaluationCase(query=bad_query, relevant_chunk_ids=["A"])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["A", 5, b"A", {"A": 1}])
def test_case_rejects_non_collection_relevant_ids(bad):
    with pytest.raises(TypeError):
        EvaluationCase(query="q", relevant_chunk_ids=bad)  # type: ignore[arg-type]


def test_case_rejects_non_str_relevant_id():
    with pytest.raises(TypeError):
        EvaluationCase(query="q", relevant_chunk_ids=["A", 7])  # type: ignore[list-item]


def test_case_rejects_empty_string_id():
    with pytest.raises(ValueError):
        EvaluationCase(query="q", relevant_chunk_ids=["A", " "])


def test_case_normalizes_relevant_ids_order_and_container():
    from_set = EvaluationCase(query="q", relevant_chunk_ids={"B", "A"})
    from_reversed_list = EvaluationCase(query="q", relevant_chunk_ids=["B", "A"])
    assert from_set == from_reversed_list
    assert from_set.relevant_chunk_ids == ("A", "B")  # canonical sorted tuple


@pytest.mark.parametrize("bad_query", ["", None, 42])
def test_prediction_rejects_invalid_query(bad_query):
    with pytest.raises(TypeError):
        EvaluationPrediction(query=bad_query, results=[])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_results", ["A", _sr("A"), {"x": 1}])
def test_prediction_rejects_non_list_results(bad_results):
    with pytest.raises(TypeError):
        EvaluationPrediction(query="q", results=bad_results)  # type: ignore[arg-type]


def test_prediction_rejects_non_searchresult_element():
    with pytest.raises(TypeError):
        EvaluationPrediction(query="q", results=[_sr("A"), "not a result"])  # type: ignore[list-item]


def test_evaluate_rejects_non_list_arguments():
    with pytest.raises(TypeError):
        _evaluate("q", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _evaluate([_case("q", "A")], "q")  # type: ignore[arg-type]


def test_evaluate_rejects_non_case_and_non_prediction_elements():
    with pytest.raises(TypeError):
        _evaluate([_case("q", "A")], ["not a prediction"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        _evaluate(["not a case"], [_pred("q")])  # type: ignore[list-item]


def test_empty_cases_contract_returns_zero_aggregates():
    result = _evaluate([], [])
    assert isinstance(result, EvaluationResult)
    assert result.total_cases == 0
    assert result.evaluated_cases == 0
    assert result.hit_rate == 0.0
    assert result.recall == 0.0
    assert result.precision == 0.0


# --- End-to-end: pipeline output is consumable by the evaluator ---


def _needle_pipeline():
    doc = Document(
        id="e2e",
        content="needle " + "filler " * 8 + "trailing ordinary words",
    )
    chunker = FixedSizeChunker(chunk_size=12)
    embedder = DeterministicEmbedder(dimension=8)
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    dense = DenseRetriever(embedder, vstore)
    lexical = LexicalRetriever(lindex)
    hybrid = HybridRetriever(dense, lexical, WeightedScoreFusion())
    pipe = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    pipe.ingest(doc)
    return doc, chunker, pipe


def test_end_to_end_pipeline_context_feed_evaluator():
    doc, chunker, pipe = _needle_pipeline()
    ctx = pipe.query("needle", top_k=10, candidate_k=10, max_chars=2000)
    assert ctx.chunks  # pipeline returned a real answer

    # Ground truth: the chunk that genuinely contains the query term.
    needle_id = next(c.id for c in chunker.chunk(doc) if "needle" in c.text)
    case = _case("needle", needle_id)
    prediction = _pred(
        "needle",
        *[SearchResult(chunk=c, score=0.0) for c in ctx.chunks],  # score unused
    )
    result = _evaluate([case], [prediction])
    # Not a quality claim -- only that pipeline output feeds evaluation cleanly.
    assert isinstance(result, EvaluationResult)
    assert (result.total_cases, result.evaluated_cases) == (1, 1)
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.precision <= 1.0
    # The needle chunk is in the full top-k pool, so it must register as a hit.
    assert result.hit_rate == 1.0
    assert result.recall == 1.0
    assert result == _evaluate([case], [prediction])  # deterministic


def test_public_import():
    from zmrag import __all__

    assert {
        "EvaluationCase",
        "EvaluationPrediction",
        "EvaluationResult",
        "Evaluator",
        "RetrievalEvaluator",
    } <= set(__all__)
    assert issubclass(RetrievalEvaluator, Evaluator)
